import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import ast
import joblib
import random
import warnings
from datetime import timedelta
from tensorflow.keras.models import load_model

# Ignore les avertissements de Scikit-Learn pour les noms de features manquants
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- CLASSES ET UTILITAIRES ---
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
    
    day_of_week = df['upload_date'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    month = df['upload_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    df['rolling_views'] = df['view_count'].shift(1).rolling(window=5).mean().bfill()
    df['log_views'] = np.log1p(df['view_count'])
    df['log_likes'] = np.log1p(df['likes'])
    df['log_rolling_views'] = np.log1p(df['rolling_views'])
    
    def get_duration_class(seconds):
        mins = seconds / 60
        if mins < 26: return 0
        elif mins < 48: return 1
        else: return 2
    
    df['duration_class'] = df['duration'].apply(get_duration_class)
    
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df

class TagManager:
    def __init__(self, df, mlb_tags):
        self.mlb = mlb_tags
        self.tag_names = mlb_tags.classes_
        
        tags_matrix = mlb_tags.transform(df['tags'])
        self.co_occurrence = np.dot(tags_matrix.T, tags_matrix)
        
        diag = np.diag(self.co_occurrence)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.correlation = self.co_occurrence / diag[:, None]
        self.correlation = np.nan_to_num(self.correlation) 

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
                    if forced_duration_class == 0 and tag_avg_dur > 1.5:
                        continue
                chosen_indices.append(idx)
                
        return [self.tag_names[x] for x in chosen_indices]

# --- FONCTION DE PRÉDICTION ---
def generate_schedule(model_path, data_path, models_dir, steps=6):
    # Chargement
    model = load_model(model_path)
    scaler_dur = joblib.load(os.path.join(models_dir, 'scaler_dur.pkl'))
    scaler_delay = joblib.load(os.path.join(models_dir, 'scaler_delay.pkl'))
    scaler_rest = joblib.load(os.path.join(models_dir, 'scaler_rest.pkl'))
    mlb_tags = joblib.load(os.path.join(models_dir, 'mlb_tags.pkl'))
    mlb_cats = joblib.load(os.path.join(models_dir, 'mlb_cats.pkl'))
    
    # Préparation
    df = load_and_prep_data(data_path, limit=300)
    tag_manager = TagManager(df, mlb_tags)
    
    X_dur = scaler_dur.transform(df[['duration']])
    X_del = scaler_delay.transform(df[['days_since_last']])
    rest_cols = ['log_views', 'log_likes', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'log_rolling_views']
    X_rest = scaler_rest.transform(df[rest_cols])
    
    X_final = np.hstack([X_dur, X_del, X_rest])
    current_seq = np.array([X_final[-10:]]) 
    
    current_date = df['upload_date'].iloc[-1]
    dow_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    
    predictions_list = []

    # Boucle de génération
    for i in range(steps):
        preds = model.predict(current_seq, verbose=0)
        pred_delay, pred_dur_probs, pred_dur_scalar, pred_dow_probs, pred_tags_probs, pred_cats = preds
        
        # Logique temporelle
        raw_days = scaler_delay.inverse_transform(pred_delay)[0][0]
        days_to_add = max(1, int(round(raw_days)))
        approx_date = current_date + timedelta(days=days_to_add)
        
        prob_sunday = pred_dow_probs[0][6]
        dist_to_sunday = (6 - approx_date.dayofweek) % 7
        if dist_to_sunday > 3: dist_to_sunday -= 7
        closest_sunday = approx_date + timedelta(days=dist_to_sunday)
        
        is_bonus_video = False
        if abs((closest_sunday - approx_date).days) <= 1 or prob_sunday > 0.4:
            final_date = closest_sunday
            target_dow = 6 
        else:
            is_bonus_video = True
            target_dow = np.argmax(pred_dow_probs[0][:6]) 
            diff = target_dow - approx_date.dayofweek
            final_date = approx_date + timedelta(days=diff)

        if final_date <= current_date: final_date = current_date + timedelta(days=1)
        real_wait = (final_date - current_date).days
        current_date = final_date

        # Logique format
        raw_duration_scaled = pred_dur_scalar[0][0]
        pred_dur_minutes = scaler_dur.inverse_transform(
            pd.DataFrame([[raw_duration_scaled]], columns=['duration'])
        )[0][0] / 60
        
        probs = pred_dur_probs[0]
        if np.max(probs) < 0.7:
            final_dur_class = np.random.choice([0, 1, 2], p=probs)
        else:
            final_dur_class = np.argmax(probs)

        bounds = {0: (18, 24), 1: (25, 42), 2: (43, 75)}
        min_b, max_b = bounds[final_dur_class]
        
        if min_b <= pred_dur_minutes <= max_b:
            base_duration = pred_dur_minutes
        else:
            base_duration = random.uniform(min_b, max_b)
            
        noise = np.random.normal(0, base_duration * 0.05)
        final_duration = base_duration + noise
        
        duration_min = int(max(min_b, min(final_duration, max_b)))
        
        gen_duration_sec = duration_min * 60
        format_label = ['Court','Standard','Long'][final_dur_class]

        # Logique contenu
        clean_tags = tag_manager.get_coherent_tags(pred_tags_probs[0], forced_duration_class=final_dur_class)
        if not clean_tags: clean_tags = ["Vrac"]
        cat_str = mlb_cats.classes_[np.argmax(pred_cats[0])]
        format_label = ['Court','Standard','Long'][final_dur_class]

        video_data = {
            "date_str": final_date.strftime('%Y-%m-%d'),
            "day_name": dow_names[target_dow],
            "duration_min": str(duration_min),
            "format_class": str(format_label),
            "tags": str(clean_tags),
            "category": str(cat_str)
        }
        predictions_list.append(video_data)

        # Mise à jour séquence
        d_sin = np.sin(2 * np.pi * target_dow / 7)
        d_cos = np.cos(2 * np.pi * target_dow / 7)
        m_sin = np.sin(2 * np.pi * final_date.month / 12)
        m_cos = np.cos(2 * np.pi * final_date.month / 12)
        
        s_views = current_seq[0, -1, 2] 
        s_likes = current_seq[0, -1, 3]
        s_roll = current_seq[0, -1, 8]
        
        s_dur_scaled = scaler_dur.transform(pd.DataFrame([[gen_duration_sec]], columns=['duration']))[0][0]
        s_day_scaled = scaler_delay.transform(pd.DataFrame([[real_wait]], columns=['days_since_last']))[0][0]
        
        new_row = np.array([[
            s_dur_scaled, s_day_scaled, s_views, s_likes, d_sin, d_cos, m_sin, m_cos, s_roll
        ]]).reshape(1, 1, 9)
        
        current_seq = np.concatenate([current_seq[:, 1:, :], new_row], axis=1)

    return predictions_list

def predict_one() -> dict:
    '''
    Fonction pour prédire le dictionnaire des informations de la prochaine vidéo
    '''
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'models', 'planner_v5_hierarchical.keras')
    models_dir = os.path.join(BASE_DIR, 'models')
    data_path = os.path.join(BASE_DIR, 'datasets', 'amixem_20251023.csv')
    
    return generate_schedule(model_path, data_path, models_dir, steps=1)[0]

def predict_multiple(nb) -> list:
    '''
    Fonction pour prédire une liste de dictionnaires des informations des prochaines vidéos
    '''
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'models', 'planner_v5_hierarchical.keras')
    models_dir = os.path.join(BASE_DIR, 'models')
    data_path = os.path.join(BASE_DIR, 'datasets', 'amixem_20251023.csv')
    
    return generate_schedule(model_path, data_path, models_dir, steps=nb)

if __name__ == "__main__":
    try:
        schedule = predict_multiple(6)
        print("\n--- RÉSULTAT RETOURNÉ ---")
        for video in schedule:
            print(video)
    except Exception as e:
        print(f"Erreur : {e}")