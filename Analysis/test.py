# =============================================================================
# PRÉDICTEUR IA YOUTUBE - ARCHITECTURE HYBRIDE
# LSTM (Génération Features) + Random Forest (Performance)
# =============================================================================

import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. CHARGEMENT ET NETTOYAGE
# =============================================================================

def load_and_clean_data(filepath):
    """Charge et nettoie les données YouTube"""
    df = pd.read_csv(filepath)
    
    # Parser les listes
    df['categories'] = df['categories'].apply(ast.literal_eval)
    df['tags'] = df['tags'].apply(ast.literal_eval)
    
    # Convertir dates
    df['upload_date'] = pd.to_datetime(df['upload_date'], format='%Y%m%d')
    
    # Trier par date
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """Crée toutes les features nécessaires"""
    
    # Features temporelles
    df['day_of_week'] = df['upload_date'].dt.dayofweek
    df['hour'] = df['upload_date'].dt.hour
    df['month'] = df['upload_date'].dt.month
    df['quarter'] = df['upload_date'].dt.quarter
    
    # Features cycliques
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Contexte historique (fenêtre glissante)
    for window in [3, 5, 10]:
        df[f'avg_views_last_{window}'] = df['view_count'].rolling(window).mean()
        df[f'avg_likes_last_{window}'] = df['likes'].rolling(window).mean()
        df[f'avg_comments_last_{window}'] = df['comment_count'].rolling(window).mean()
    
    # Jours depuis dernière vidéo
    df['days_since_last'] = df['timestamp'].diff() / 86400
    
    # Tendance (pente régression linéaire sur 10 dernières)
    df['trend_views'] = df['view_count'].rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
    )
    
    # Engagement rate
    df['engagement_rate'] = (df['likes'] + df['comment_count']) / df['view_count']
    
    # Log transform pour distribution normale
    df['log_views'] = np.log1p(df['view_count'])
    df['log_likes'] = np.log1p(df['likes'])
    
    return df

def encode_tags_categories(df, top_n=50):
    """Encode tags et catégories"""
    
    # Top-N tags les plus fréquents
    from collections import Counter
    all_tags = [tag for tags in df['tags'] for tag in tags]
    top_tags = [tag for tag, _ in Counter(all_tags).most_common(top_n)]
    
    # Filter tags
    df['tags_filtered'] = df['tags'].apply(
        lambda x: [t for t in x if t in top_tags]
    )
    
    # Multi-label encoding
    mlb_tags = MultiLabelBinarizer()
    tags_encoded = mlb_tags.fit_transform(df['tags_filtered'])
    
    mlb_cat = MultiLabelBinarizer()
    cat_encoded = mlb_cat.fit_transform(df['categories'])
    
    return tags_encoded, cat_encoded, mlb_tags, mlb_cat

# =============================================================================
# 3. PRÉPARATION SÉQUENCES LSTM
# =============================================================================

def create_sequences(df, sequence_length=10):
    """Crée des séquences pour LSTM"""
    
    # Features pour séquence
    feature_cols = [
        'duration', 'day_of_week', 'hour', 'month',
        'log_views', 'log_likes', 'days_since_last'
    ]
    
    X, y = [], []
    
    for i in range(len(df) - sequence_length):
        # Séquence de N vidéos
        seq = df.iloc[i:i+sequence_length][feature_cols].values
        
        # Target = prochaine vidéo
        target = df.iloc[i+sequence_length][feature_cols].values
        
        X.append(seq)
        y.append(target)
    
    return np.array(X), np.array(y)

# =============================================================================
# 4. MODÈLE LSTM (PHASE 1)
# =============================================================================

def build_lstm_model(sequence_length, n_features, n_outputs):
    """Construit le modèle LSTM"""
    
    model = Sequential([
        LSTM(128, return_sequences=True, 
             input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(n_outputs)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_lstm(model, X_train, y_train, X_val, y_val):
    """Entraîne le modèle LSTM"""
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1
    )
    
    return history

def main_pipeline(csv_path):
    """Pipeline complet d'entraînement et prédiction"""
    
    print("🚀 Démarrage du pipeline...")
    
    # 1. Chargement
    print("\n📊 Chargement des données...")
    df = load_and_clean_data(csv_path)
    print(f"   {len(df)} vidéos chargées")
    
    # 2. Feature Engineering
    print("\n🔧 Feature Engineering...")
    df = engineer_features(df)
    tags_enc, cat_enc, mlb_tags, mlb_cat = encode_tags_categories(df)
    
    # 3. Split temporel
    print("\n✂️ Split temporel...")
    n = len(df)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    train = df[:train_idx].copy()
    val = df[train_idx:val_idx].copy()
    test = df[val_idx:].copy()
    
    print(f"   Train: {len(train)} vidéos")
    print(f"   Val: {len(val)} vidéos")
    print(f"   Test: {len(test)} vidéos")
    
    # 4. Préparation LSTM
    print("\n🧠 Préparation séquences LSTM...")
    sequence_length = 10
    
    X_train_seq, y_train_seq = create_sequences(train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(val, sequence_length)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    scaler.fit(X_train_flat)
    
    X_train_scaled = scaler.transform(
        X_train_seq.reshape(-1, X_train_seq.shape[-1])
    ).reshape(X_train_seq.shape)
    
    X_val_scaled = scaler.transform(
        X_val_seq.reshape(-1, X_val_seq.shape[-1])
    ).reshape(X_val_seq.shape)
    
    # 5. Entraînement LSTM
    print("\n🔥 Entraînement LSTM...")
    lstm = build_lstm_model(
        sequence_length, 
        X_train_seq.shape[-1], 
        y_train_seq.shape[-1]
    )
    
    history = train_lstm(
        lstm, 
        X_train_scaled, y_train_seq,
        X_val_scaled, y_val_seq
    )