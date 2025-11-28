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

# --- 1. CHARGEMENT ---
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
    df['rolling_views'] = df['view_count'].shift(1).rolling(window=5).mean().fillna(method='bfill')
    
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
    loss_weights = {'delay_out': 1.0, 'dur_out': 1.2, 'dow_out': 1.5, 'tags_out': 0.5, 'cats_out': 0.5}
    
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
    return model

# --- 5. LOGIQUE DE "SNAP" (CORRIGÉE) ---
def snap_date_to_dow(approx_date, target_dow_idx):
    """
    Ajuste une date approximative pour qu'elle tombe sur le jour de la semaine désiré.
    """
    approx_dow = approx_date.dayofweek
    # target_dow_idx est souvent un numpy.int64, on force le calcul
    diff = target_dow_idx - approx_dow
    
    if diff > 3: diff -= 7
    if diff < -3: diff += 7
    
    # --- CORRECTION CRITIQUE ICI : int() ---
    return approx_date + timedelta(days=int(diff))

def generate_duration(class_idx):
    if class_idx == 0: return random.randint(18 * 60, 26 * 60) 
    elif class_idx == 1: return random.randint(28 * 60, 45 * 60)
    else: return random.randint(50 * 60, 75 * 60)

def predict_smart(model, initial_seq, scaler_delay, scaler_dur_feat, mlb_tags, mlb_cats, start_date, n_steps=6):
    current_seq = initial_seq.copy() 
    current_date = start_date
    dow_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    
    print(f"\n{'='*50}")
    print(f" PLANNING FINAL (DATES & FORMATS CORRIGÉS)")
    print(f"{'='*50}")
    
    for i in range(n_steps):
        # 1. Prédiction
        preds = model.predict(current_seq, verbose=0)
        pred_delay, pred_dur_probs, pred_dow_probs, pred_tags, pred_cats = preds
        
        # 2. Gestion du TEMPS
        raw_days_wait = scaler_delay.inverse_transform(pred_delay)[0][0]
        
        # --- CORRECTION CRITIQUE ICI : int() ---
        days_to_add = int(round(max(1, raw_days_wait)))
        approx_date = current_date + timedelta(days=days_to_add)
        
        target_dow_idx = np.random.choice(range(7), p=pred_dow_probs[0])
        
        final_date = snap_date_to_dow(approx_date, target_dow_idx)
        
        if final_date <= current_date:
            final_date = current_date + timedelta(days=1)
        
        real_days_wait = (final_date - current_date).days
        current_date = final_date 
        
        # 3. Gestion du CONTENU
        pred_dur_class = np.random.choice([0, 1, 2], p=pred_dur_probs[0])
        gen_duration = generate_duration(pred_dur_class)
        
        cat_str = mlb_cats.classes_[np.argmax(pred_cats[0])]
        tags_idx = np.where(pred_tags[0] > 0.25)[0]
        if len(tags_idx) > 0:
            tags_str = [mlb_tags.classes_[x] for x in tags_idx]
        else:
            tags_str = ["Vrac"]
        
        # 4. Affichage
        print(f"\n[{i+1}] {final_date.strftime('%A %d %B')} (+{real_days_wait}j)")
        print(f"    📅 Cible IA: {dow_names[target_dow_idx]} (Prob: {pred_dow_probs[0][target_dow_idx]:.2f})")
        print(f"    ⏱️  Durée: {int(gen_duration/60)} min ({['Court','Standard','Long'][pred_dur_class]})")
        print(f"    📂 {cat_str}")
        print(f"    🏷️  {tags_str[:3]}")

        # 5. REBOUCLAGE
        d_sin = np.sin(2 * np.pi * final_date.dayofweek / 7)
        d_cos = np.cos(2 * np.pi * final_date.dayofweek / 7)
        m_sin = np.sin(2 * np.pi * final_date.month / 12)
        m_cos = np.cos(2 * np.pi * final_date.month / 12)
        
        s_views = np.mean(current_seq[0, :, 2])
        s_likes = np.mean(current_seq[0, :, 3])
        s_roll = np.mean(current_seq[0, :, 8])
        
        # Scaling manuel
        s_dur = scaler_dur_feat.transform([[gen_duration]])[0][0]
        s_day = scaler_delay.transform([[real_days_wait]])[0][0]
        
        new_row = np.array([[
            s_dur, s_day, s_views, s_likes, d_sin, d_cos, m_sin, m_cos, s_roll
        ]]).reshape(1, 1, 9)
        
        current_seq = np.concatenate([current_seq[:, 1:, :], new_row], axis=1)

# --- 6. MAIN ---
def main():
    csv_path = '../Datasets/amixem_20251023.csv'
    df = load_and_prep_data(csv_path, limit=300)
    
    tags_enc, cats_enc, dur_enc, dow_enc, mlb_t, mlb_c = encode_labels(df)
    
    # Scalers
    scaler_dur = StandardScaler().fit(df[['duration']])
    scaler_delay = StandardScaler().fit(df[['days_since_last']])
    
    rest_cols = ['log_views', 'log_likes', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'log_rolling_views']
    scaler_rest = StandardScaler().fit(df[rest_cols])
    
    # Construction X
    X_dur = scaler_dur.transform(df[['duration']])
    X_del = scaler_delay.transform(df[['days_since_last']])
    X_rest = scaler_rest.transform(df[rest_cols])
    X_scaled = np.hstack([X_dur, X_del, X_rest])
    
    # Targets
    Y_delay = scaler_delay.transform(df[['days_since_last']])
    
    # Séquences
    SEQ_LEN = 10
    X_seq, Y_all = create_multi_output_sequences(X_scaled, Y_delay, dur_enc, dow_enc, tags_enc, cats_enc, SEQ_LEN)
    
    # Entraînement
    print("Entraînement du modèle V4 (Hybrid Date/Time)...")
    model = build_planner_model(SEQ_LEN, 9, tags_enc.shape[1], cats_enc.shape[1])
    
    early = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_seq, Y_all, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early], verbose=0)
    
    # Sauvegarde
    model.save('planner_v4_hybrid.keras')
    joblib.dump(scaler_dur, 'scaler_dur.pkl')
    joblib.dump(scaler_delay, 'scaler_delay.pkl')
    joblib.dump(mlb_t, 'mlb_tags.pkl')
    joblib.dump(mlb_c, 'mlb_cats.pkl')
    
    # Test
    predict_smart(model, X_seq[-1:], scaler_delay, scaler_dur, mlb_t, mlb_c, df['upload_date'].iloc[-1], n_steps=8)

if __name__ == "__main__":
    main()