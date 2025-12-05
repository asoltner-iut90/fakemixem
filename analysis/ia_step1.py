import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pandas as pd
import numpy as np
import ast
import joblib
from datetime import timedelta
import random
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. CHARGEMENT ET PREP ---
def load_and_prep_data(df_or_path, limit=300):
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
        df['categories'] = df['categories'].apply(ast.literal_eval)
        df['tags'] = df['tags'].apply(ast.literal_eval)
        df['upload_date'] = pd.to_datetime(df['upload_date'], format='%Y%m%d')
    else:
        df = df_or_path.copy()
        
    df = df.sort_values('upload_date').reset_index(drop=True)
    df['days_since_last'] = df['upload_date'].diff().dt.days.fillna(0)
    
    # Features Cycliques
    day_of_week = df['upload_date'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    month = df['upload_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # Rolling
    # Correction Warning: fillna(method='bfill') -> bfill()
    df['rolling_views'] = df['view_count'].shift(1).rolling(window=5).mean().bfill()
    
    # Logs
    df['log_views'] = np.log1p(df['view_count'])
    df['log_likes'] = np.log1p(df['likes'])
    df['log_rolling_views'] = np.log1p(df['rolling_views'])
    
    # --- CLASSIFICATION DURÉE ---
    def get_duration_class(seconds):
        mins = seconds / 60
        if mins < 26: return 0 # Short
        elif mins < 48: return 1 # Standard
        else: return 2 # Long
    df['duration_class'] = df['duration'].apply(get_duration_class)
    
    # --- JOUR DE LA SEMAINE CIBLE (0-6) ---
    df['target_dow'] = df['upload_date'].dt.dayofweek
    
    # Feature utile pour le modèle : distance au dimanche
    df['days_until_sunday'] = (6 - df['upload_date'].dt.dayofweek) % 7
    
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df

# --- 2. ENCODAGE ---
def encode_labels(df):
    mlb_tags = MultiLabelBinarizer(sparse_output=False)
    tags_encoded = mlb_tags.fit_transform(df['tags'])
    
    mlb_cats = MultiLabelBinarizer(sparse_output=False)
    cats_encoded = mlb_cats.fit_transform(df['categories'])
    
    duration_encoded = to_categorical(df['duration_class'], num_classes=3)
    dow_encoded = to_categorical(df['target_dow'], num_classes=7)
    
    return tags_encoded, cats_encoded, duration_encoded, dow_encoded, mlb_tags, mlb_cats

# --- 3. SÉQUENCES ---
def create_multi_output_sequences(X_input, y_delay, y_dur_cls, y_dow, y_tags, y_cats, seq_len=10):
    X, Y_delay, Y_dur, Y_dow, Y_tags, Y_cats = [], [], [], [], [], []
    for i in range(len(X_input) - seq_len):
        X.append(X_input[i:i+seq_len])
        Y_delay.append(y_delay[i+seq_len])
        Y_dur.append(y_dur_cls[i+seq_len])
        Y_dow.append(y_dow[i+seq_len])
        Y_tags.append(y_tags[i+seq_len])
        Y_cats.append(y_cats[i+seq_len])
        
    return (np.array(X), 
            [np.array(Y_delay), np.array(Y_dur), np.array(Y_dow), np.array(Y_tags), np.array(Y_cats)])

# --- 4. MODÈLE ---
def build_planner_model(seq_len, n_features, n_tags, n_cats):
    inp = Input(shape=(seq_len, n_features))
    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    
    delay_out = Dense(1, name='delay_out')(Dense(32, activation='relu')(x))
    dur_out = Dense(3, activation='softmax', name='dur_out')(Dense(32, activation='relu')(x))
    dow_out = Dense(7, activation='softmax', name='dow_out')(Dense(32, activation='relu')(x)) 
    tags_out = Dense(n_tags, activation='sigmoid', name='tags_out')(Dense(64, activation='relu')(x))
    cats_out = Dense(n_cats, activation='sigmoid', name='cats_out')(Dense(32, activation='relu')(x))
    
    model = Model(inputs=inp, outputs=[delay_out, dur_out, dow_out, tags_out, cats_out])
    
    losses = {
        'delay_out': 'mse',
        'dur_out': 'categorical_crossentropy',
        'dow_out': 'categorical_crossentropy',
        'tags_out': 'binary_crossentropy',
        'cats_out': 'binary_crossentropy'
    }
    # Poids ajustés pour privilégier la régularité du jour
    loss_weights = {'delay_out': 1.0, 'dur_out': 1.2, 'dow_out': 3.0, 'tags_out': 0.5, 'cats_out': 0.5}
    
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
    return model

# --- 5. LOGIQUE INTELLIGENTE (TagManager & Predict) ---

class TagManager:
    def __init__(self, df, mlb_tags):
        self.mlb = mlb_tags
        self.tag_names = mlb_tags.classes_
        self.n_tags = len(self.tag_names)
        
        # 1. Matrice de Co-occurrence
        tags_matrix = mlb_tags.transform(df['tags'])
        self.co_occurrence = np.dot(tags_matrix.T, tags_matrix)
        
        # Normalisation
        diag = np.diag(self.co_occurrence)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.correlation = self.co_occurrence / diag[:, None]
        self.correlation = np.nan_to_num(self.correlation) 

        # 2. Liens Tags <-> Durée
        self.tag_duration_profile = {}
        durations = df['duration_class'].values
        for i, tag in enumerate(self.tag_names):
            idxs = np.where(tags_matrix[:, i] == 1)[0]
            if len(idxs) > 0:
                self.tag_duration_profile[i] = np.mean(durations[idxs])
            else:
                self.tag_duration_profile[i] = 1.0 

    def get_coherent_tags(self, raw_probs, forced_duration_class=None):
        anchor_idx = np.argmax(raw_probs)
        anchor_prob = raw_probs[anchor_idx]
        
        if anchor_prob < 0.1: return [] 
        
        chosen_indices = [anchor_idx]
        potential_friends = np.where(raw_probs > 0.2)[0]
        
        for idx in potential_friends:
            if idx == anchor_idx: continue
            
            friendship_score = self.correlation[anchor_idx][idx]
            
            if friendship_score > 0.1: 
                if forced_duration_class is not None:
                    tag_avg_dur = self.tag_duration_profile[idx]
                    # Si on veut du court (0) et que le tag est typiquement long (>1.5), on évite
                    if forced_duration_class == 0 and tag_avg_dur > 1.5:
                        continue
                chosen_indices.append(idx)
                
        return [self.tag_names[x] for x in chosen_indices]

def predict_hierarchical(model, initial_seq, scaler_delay, scaler_dur, mlb_tags, mlb_cats, df_train, start_date, n_steps=6):
    # Initialisation du Manager
    tag_manager = TagManager(df_train, mlb_tags)
    
    current_seq = initial_seq.copy()
    current_date = start_date
    dow_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    print(f"\n{'='*60}")
    print(f" PLANNING HIÉRARCHIQUE (LOGIQUE & COHÉRENCE V5)")
    print(f"{'='*60}")

    for i in range(n_steps):
        # --- 1. PRÉDICTION BRUTE ---
        preds = model.predict(current_seq, verbose=0)
        pred_delay, pred_dur_probs, pred_dow_probs, pred_tags_probs, pred_cats = preds
        
        # --- 2. LOGIQUE TEMPORELLE (Le "Quand") ---
        raw_days = scaler_delay.inverse_transform(pred_delay)[0][0]
        # On s'assure d'avancer d'au moins 1 jour
        days_to_add = max(1, int(round(raw_days)))
        approx_date = current_date + timedelta(days=days_to_add)
        
        # Gravité Dimanche
        prob_sunday = pred_dow_probs[0][6]
        # CORRECTION ICI: .dayofweek est une propriété, pas une méthode ()
        dist_to_sunday = (6 - approx_date.dayofweek) % 7
        if dist_to_sunday > 3: dist_to_sunday -= 7
        closest_sunday = approx_date + timedelta(days=dist_to_sunday)
        
        is_bonus_video = False
        
        # Si on tombe proche d'un dimanche ou que l'IA est sûre que c'est dimanche
        if abs((closest_sunday - approx_date).days) <= 1 or prob_sunday > 0.4:
            final_date = closest_sunday
            target_dow = 6 # Dimanche
        else:
            # C'est une vidéo bonus (Semaine)
            is_bonus_video = True
            target_dow = np.argmax(pred_dow_probs[0][:6]) # Max hors dimanche
            # CORRECTION ICI: .dayofweek sans parenthèses
            diff = target_dow - approx_date.dayofweek
            final_date = approx_date + timedelta(days=diff)

        if final_date <= current_date: final_date = current_date + timedelta(days=1)
        real_wait = (final_date - current_date).days
        current_date = final_date

        # --- 3. LOGIQUE FORMAT (Le "Combien de temps") ---
        if is_bonus_video:
            # Bonus = Court ou Standard, jamais Long
            final_dur_class = np.random.choice([0, 1], p=[0.7, 0.3])
            origin_str = "(Forcé: Bonus)"
        else:
            # Dimanche = L'IA décide
            final_dur_class = np.random.choice([0, 1, 2], p=pred_dur_probs[0])
            origin_str = "(Choix IA)"
            
        if final_dur_class == 0: duration_min = random.randint(15, 24)
        elif final_dur_class == 1: duration_min = random.randint(25, 40)
        else: duration_min = random.randint(45, 70)
        
        gen_duration_sec = duration_min * 60

        # --- 4. LOGIQUE CONTENU (Le "Quoi") ---
        clean_tags = tag_manager.get_coherent_tags(pred_tags_probs[0], forced_duration_class=final_dur_class)
        if not clean_tags: clean_tags = ["Vrac"]
        
        cat_str = mlb_cats.classes_[np.argmax(pred_cats[0])]

        # --- AFFICHAGE ---
        icon = "🌟" if not is_bonus_video else "🎁"
        print(f"\n[{i+1}] {final_date.strftime('%A %d %B')} (+{real_wait}j)")
        print(f"    {icon} Slot: {dow_names[target_dow]} | {origin_str}")
        print(f"    ⏱️  Durée: {duration_min} min ({['Court','Standard','Long'][final_dur_class]})")
        print(f"    🏷️  Tags: {', '.join(clean_tags)}")
        
        # --- 5. REBOUCLAGE ---
        d_sin = np.sin(2 * np.pi * target_dow / 7)
        d_cos = np.cos(2 * np.pi * target_dow / 7)
        m_sin = np.sin(2 * np.pi * final_date.month / 12)
        m_cos = np.cos(2 * np.pi * final_date.month / 12)
        
        s_views = current_seq[0, -1, 2] 
        s_likes = current_seq[0, -1, 3]
        s_roll = current_seq[0, -1, 8]
        
        s_dur_scaled = scaler_dur.transform([[gen_duration_sec]])[0][0]
        s_day_scaled = scaler_delay.transform([[real_wait]])[0][0]
        
        new_row = np.array([[
            s_dur_scaled, s_day_scaled, s_views, s_likes, d_sin, d_cos, m_sin, m_cos, s_roll
        ]]).reshape(1, 1, 9) # Reshape important
        
        current_seq = np.concatenate([current_seq[:, 1:, :], new_row], axis=1)

# --- 6. MAIN ---
def main():
    # Adapter le chemin selon ton environnement
    csv_path = '../datasets/amixem_20251023.csv'
    
    # 1. Chargement
    df = load_and_prep_data(csv_path, limit=300)
    
    # 2. Encodage
    tags_enc, cats_enc, dur_enc, dow_enc, mlb_t, mlb_c = encode_labels(df)
    
    # 3. Scalers
    scaler_dur = StandardScaler().fit(df[['duration']])
    scaler_delay = StandardScaler().fit(df[['days_since_last']])
    
    rest_cols = ['log_views', 'log_likes', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'log_rolling_views']
    scaler_rest = StandardScaler().fit(df[rest_cols])
    
    # 4. Construction X
    X_dur = scaler_dur.transform(df[['duration']])
    X_del = scaler_delay.transform(df[['days_since_last']])
    X_rest = scaler_rest.transform(df[rest_cols])
    X_scaled = np.hstack([X_dur, X_del, X_rest])
    
    # Targets
    Y_delay = scaler_delay.transform(df[['days_since_last']])
    
    # 5. Séquences
    SEQ_LEN = 10
    X_seq, Y_all = create_multi_output_sequences(X_scaled, Y_delay, dur_enc, dow_enc, tags_enc, cats_enc, SEQ_LEN)
    
    # 6. Entraînement
    print("Entraînement du modèle V5 (Hierarchical Logic)...")
    # Note : Assure-toi que X_scaled a bien 9 features comme attendu dans le reshape du predict
    # (1 dur + 1 delay + 7 rest = 9 features -> OK)
    model = build_planner_model(SEQ_LEN, 9, tags_enc.shape[1], cats_enc.shape[1])
    
    early = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_seq, Y_all, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early], verbose=0)
    
    # 7. Sauvegarde
    model.save('../models/planner_v5_hierarchical.keras')
    joblib.dump(scaler_dur, '../models/scaler_dur.pkl')
    joblib.dump(scaler_delay, '../models/scaler_delay.pkl')
    joblib.dump(mlb_t, '../models/mlb_tags.pkl')
    joblib.dump(mlb_c, '../models/mlb_cats.pkl')
    
    # 8. Test Prédiction Hiérarchique
    # On passe 'df' (l'historique complet) pour que le TagManager puisse calculer les matrices de corrélation
    predict_hierarchical(
        model=model, 
        initial_seq=X_seq[-1:], 
        scaler_delay=scaler_delay, 
        scaler_dur=scaler_dur, 
        mlb_tags=mlb_t, 
        mlb_cats=mlb_c, 
        df_train=df,  # <--- Ajouté ici
        start_date=df['upload_date'].iloc[-1], 
        n_steps=8
    )

if __name__ == "__main__":
    main()