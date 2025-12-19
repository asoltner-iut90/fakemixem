import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

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
    
    # Features cycliques
    day_of_week = df['upload_date'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    month = df['upload_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # Rolling & Logs
    df['rolling_views'] = df['view_count'].shift(1).rolling(window=5).mean().bfill()
    df['log_views'] = np.log1p(df['view_count'])
    df['log_likes'] = np.log1p(df['likes'])
    df['log_rolling_views'] = np.log1p(df['rolling_views'])
    
    # Classification des durées
    def get_duration_class(seconds):
        mins = seconds / 60
        if mins < 26: return 0 
        elif mins < 48: return 1 
        else: return 2 
    df['duration_class'] = df['duration'].apply(get_duration_class)
    df['target_dow'] = df['upload_date'].dt.dayofweek
    
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df

def encode_labels(df):
    mlb_tags = MultiLabelBinarizer(sparse_output=False)
    tags_encoded = mlb_tags.fit_transform(df['tags'])
    
    mlb_cats = MultiLabelBinarizer(sparse_output=False)
    cats_encoded = mlb_cats.fit_transform(df['categories'])
    
    duration_encoded = to_categorical(df['duration_class'], num_classes=3)
    dow_encoded = to_categorical(df['target_dow'], num_classes=7)
    return tags_encoded, cats_encoded, duration_encoded, dow_encoded, mlb_tags, mlb_cats

def create_multi_output_sequences(X_input, y_delay, y_dur_cls, y_dur_scalar, y_dow, y_tags, y_cats, seq_len=10):
    X, Y_delay, Y_dur_c, Y_dur_s, Y_dow, Y_tags, Y_cats = [], [], [], [], [], [], []
    
    for i in range(len(X_input) - seq_len):
        X.append(X_input[i:i+seq_len])
        Y_delay.append(y_delay[i+seq_len])
        Y_dur_c.append(y_dur_cls[i+seq_len])  
        Y_dur_s.append(y_dur_scalar[i+seq_len])
        Y_dow.append(y_dow[i+seq_len])
        Y_tags.append(y_tags[i+seq_len])
        Y_cats.append(y_cats[i+seq_len])
        
    return (np.array(X), 
            [np.array(Y_delay), np.array(Y_dur_c), np.array(Y_dur_s), 
             np.array(Y_dow), np.array(Y_tags), np.array(Y_cats)])

def build_planner_model(seq_len, n_features, n_tags, n_cats):
    inp = Input(shape=(seq_len, n_features))
    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    
    delay_out = Dense(1, name='delay_out')(Dense(32, activation='relu')(x))
    
    # Classification (Court/Moyen/Long) - Utile pour la cohérence des tags
    dur_class_out = Dense(3, activation='softmax', name='dur_class_out')(Dense(32, activation='relu')(x))
    
    dur_scalar_out = Dense(1, name='dur_scalar_out')(Dense(32, activation='relu')(x))
    
    dow_out = Dense(7, activation='softmax', name='dow_out')(Dense(32, activation='relu')(x)) 
    tags_out = Dense(n_tags, activation='sigmoid', name='tags_out')(Dense(64, activation='relu')(x))
    cats_out = Dense(n_cats, activation='sigmoid', name='cats_out')(Dense(32, activation='relu')(x))
    
    model = Model(inputs=inp, outputs=[delay_out, dur_class_out, dur_scalar_out, dow_out, tags_out, cats_out])
    
    losses = {
        'delay_out': 'mse',
        'dur_class_out': 'categorical_crossentropy',
        'dur_scalar_out': 'mae',
        'dow_out': 'categorical_crossentropy',
        'tags_out': 'binary_crossentropy',
        'cats_out': 'binary_crossentropy'
    }
    
    loss_weights = {
        'delay_out': 1.0, 
        'dur_class_out': 1.0, 
        'dur_scalar_out': 2.0,
        'dow_out': 3.0, 
        'tags_out': 0.5, 
        'cats_out': 0.5
    }
    
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
    return model

if __name__ == "__main__":
    csv_path = '../datasets/amixem_20251219.csv'
    
    print("Chargement des données...")
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
    
    Y_delay = scaler_delay.transform(df[['days_since_last']])
    Y_dur_scalar = scaler_dur.transform(df[['duration']])
    
    SEQ_LEN = 10
    X_seq, Y_all = create_multi_output_sequences(X_scaled, Y_delay, dur_enc, Y_dur_scalar, dow_enc, tags_enc, cats_enc, SEQ_LEN)
    print("Entraînement du modèle V5...")
    model = build_planner_model(SEQ_LEN, 9, tags_enc.shape[1], cats_enc.shape[1])
    
    early = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_seq, Y_all, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early], verbose=1)
    
    # Sauvegarde
    print("Sauvegarde des modèles dans ../models/ ...")
    model.save('../models/planner_v5_hierarchical.keras')
    joblib.dump(scaler_dur, '../models/scaler_dur.pkl')
    joblib.dump(scaler_delay, '../models/scaler_delay.pkl')
    joblib.dump(scaler_rest, '../models/scaler_rest.pkl')
    joblib.dump(mlb_t, '../models/mlb_tags.pkl')
    joblib.dump(mlb_c, '../models/mlb_cats.pkl')
    print("Terminé.")