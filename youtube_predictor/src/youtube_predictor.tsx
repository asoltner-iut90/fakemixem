import React, { useState } from 'react';
import { Upload, Brain, TrendingUp, Calendar, Tag, Eye, ThumbsUp, MessageSquare, Play, ChevronRight, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

const YouTubePredictorDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [analysisStep, setAnalysisStep] = useState(0);

  const architectureSteps = [
    {
      title: "Phase 1 : Modèle Temporel (LSTM)",
      icon: <Calendar className="w-6 h-6" />,
      color: "blue",
      inputs: ["Historique 10-20 dernières vidéos", "Dates, durées, catégories, tags"],
      outputs: ["upload_date prédite", "duration", "categories", "tags"],
      model: "LSTM Seq2Seq multi-output"
    },
    {
      title: "Phase 2 : Random Forest (Performance)",
      icon: <TrendingUp className="w-6 h-6" />,
      color: "green",
      inputs: ["Features Phase 1", "Contexte historique", "Features cycliques"],
      outputs: ["view_count", "likes", "comment_count", "dislikes"],
      model: "RandomForestRegressor multi-target"
    }
  ];

  const dataInsights = [
    { label: "Cadence publication", value: "~7 jours", icon: <Calendar className="w-5 h-5" />, trend: "stable" },
    { label: "Durée moyenne", value: "42 min", icon: <Play className="w-5 h-5" />, trend: "up" },
    { label: "Vues moyennes", value: "3.8M", icon: <Eye className="w-5 h-5" />, trend: "up" },
    { label: "Engagement moyen", value: "5.2%", icon: <ThumbsUp className="w-5 h-5" />, trend: "stable" }
  ];

  const pipeline = [
    {
      step: 1,
      title: "Chargement & Nettoyage",
      tasks: [
        "Parser les listes (categories, tags)",
        "Convertir dates en datetime",
        "Gérer valeurs manquantes",
        "Supprimer outliers extrêmes"
      ],
      code: `import pandas as pd
import ast

df = pd.read_csv('youtube_data.csv')
df['categories'] = df['categories'].apply(ast.literal_eval)
df['tags'] = df['tags'].apply(ast.literal_eval)
df['upload_date'] = pd.to_datetime(df['upload_date'], format='%Y%m%d')
df = df.sort_values('timestamp')`
    },
    {
      step: 2,
      title: "Feature Engineering",
      tasks: [
        "Extraire features temporelles",
        "Calculer moyennes mobiles",
        "Encoder catégories/tags",
        "Créer features cycliques"
      ],
      code: `# Features temporelles
df['day_of_week'] = df['upload_date'].dt.dayofweek
df['hour'] = df['upload_date'].dt.hour
df['month'] = df['upload_date'].dt.month

# Features cycliques
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Contexte historique
df['avg_views_last_5'] = df['view_count'].rolling(5).mean()
df['days_since_last'] = df['timestamp'].diff() / 86400`
    },
    {
      step: 3,
      title: "Encodage Multi-Label",
      tasks: [
        "MultiLabelBinarizer pour tags",
        "Garder top-50 tags fréquents",
        "TF-IDF optionnel",
        "One-hot pour catégories"
      ],
      code: `from sklearn.preprocessing import MultiLabelBinarizer

# Encoder tags
mlb_tags = MultiLabelBinarizer()
tags_encoded = mlb_tags.fit_transform(df['tags'])

# Garder top-50
from collections import Counter
all_tags = [tag for tags in df['tags'] for tag in tags]
top_tags = [tag for tag, _ in Counter(all_tags).most_common(50)]
df['tags_filtered'] = df['tags'].apply(lambda x: [t for t in x if t in top_tags])`
    },
    {
      step: 4,
      title: "Split Temporel",
      tasks: [
        "70% train (anciennes vidéos)",
        "15% validation",
        "15% test (récentes)",
        "Pas de shuffle aléatoire"
      ],
      code: `# Split chronologique CRUCIAL
n = len(df)
train_idx = int(n * 0.7)
val_idx = int(n * 0.85)

train = df[:train_idx]
val = df[train_idx:val_idx]
test = df[val_idx:]

print(f"Train: {train['upload_date'].min()} → {train['upload_date'].max()}")
print(f"Test: {test['upload_date'].min()} → {test['upload_date'].max()}")`
    },
    {
      step: 5,
      title: "LSTM - Génération Features",
      tasks: [
        "Créer séquences de longueur 10",
        "Normaliser les inputs",
        "Architecture multi-output",
        "Entraîner sur GPU si possible"
      ],
      code: `from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Préparer séquences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Modèle LSTM
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(n_outputs)  # duration + categories + tags
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))`
    },
    {
      step: 6,
      title: "Random Forest - Prédiction Performance",
      tasks: [
        "Utiliser features LSTM générées",
        "Ajouter contexte historique",
        "Multi-output regression",
        "Optimiser hyperparamètres"
      ],
      code: `from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Features pour RF
def build_features(row, history):
    features = [
        row['duration'],
        row['day_of_week'], row['hour'], row['month'],
        row['day_sin'], row['day_cos'],
        *row['tags_encoded'],
        *row['categories_encoded'],
        history['view_count'].mean(),
        history['likes'].mean(),
        (history['timestamp'].iloc[-1] - history['timestamp'].iloc[-2]) / 86400
    ]
    return features

# Modèle
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

# Entraîner
X_train = [build_features(row, train[:i]) for i, row in train.iterrows()]
y_train = train[['view_count', 'likes', 'comment_count', 'dislikes']]
rf.fit(X_train, y_train)`
    },
    {
      step: 7,
      title: "Génération Vidéo Complète",
      tasks: [
        "LSTM prédit features créatives",
        "RF prédit performances",
        "Validation cohérence",
        "Export résultats"
      ],
      code: `def generate_next_video(history_df, lstm_model, rf_model):
    # Phase 1: LSTM génère features
    sequence = prepare_sequence(history_df[-10:])
    lstm_output = lstm_model.predict(sequence)
    
    next_date = predict_date(history_df)
    duration = lstm_output[0]
    categories = decode_categories(lstm_output[1:4])
    tags = decode_tags(lstm_output[4:])
    
    # Phase 2: RF prédit performance
    features = build_features({
        'duration': duration,
        'date': next_date,
        'categories': categories,
        'tags': tags
    }, history_df[-10:])
    
    predictions = rf_model.predict([features])[0]
    
    return {
        'upload_date': next_date,
        'duration': int(duration),
        'categories': categories,
        'tags': tags,
        'view_count': int(predictions[0]),
        'likes': int(predictions[1]),
        'comment_count': int(predictions[2]),
        'dislikes': int(predictions[3]),
        'sponsored': True,  # Pattern observé
        'age_limit': 0
    }`
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <Brain className="w-10 h-10 text-purple-400" />
            <h1 className="text-4xl font-bold">Prédicteur IA YouTube - Amixem</h1>
          </div>
          <p className="text-slate-300 text-lg">
            Architecture Hybride : LSTM (Génération) + Random Forest (Performance)
          </p>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 bg-slate-800/50 p-2 rounded-lg">
          {['overview', 'architecture', 'pipeline', 'code'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                activeTab === tab
                  ? 'bg-purple-600 text-white shadow-lg'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              {tab === 'overview' && '📊 Vue d\'ensemble'}
              {tab === 'architecture' && '🏗️ Architecture'}
              {tab === 'pipeline' && '⚙️ Pipeline'}
              {tab === 'code' && '💻 Code Complet'}
            </button>
          ))}
        </div>

        {/* Content */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Data Insights */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {dataInsights.map((insight, idx) => (
                <div key={idx} className="bg-slate-800/70 rounded-xl p-6 border border-slate-700">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-slate-400">{insight.icon}</div>
                    <span className={`text-xs px-2 py-1 rounded ${
                      insight.trend === 'up' ? 'bg-green-500/20 text-green-400' : 'bg-blue-500/20 text-blue-400'
                    }`}>
                      {insight.trend === 'up' ? '↑' : '→'}
                    </span>
                  </div>
                  <div className="text-2xl font-bold mb-1">{insight.value}</div>
                  <div className="text-sm text-slate-400">{insight.label}</div>
                </div>
              ))}
            </div>

            {/* Patterns détectés */}
            <div className="bg-slate-800/70 rounded-xl p-6 border border-slate-700">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <CheckCircle2 className="w-6 h-6 text-green-400" />
                Patterns Détectés dans vos Données
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-purple-400 rounded-full mt-2"></div>
                    <div>
                      <div className="font-semibold text-purple-300">Cadence régulière</div>
                      <div className="text-sm text-slate-400">Publication tous les ~7 jours, favorisant le samedi</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-blue-400 rounded-full mt-2"></div>
                    <div>
                      <div className="font-semibold text-blue-300">Durée croissante</div>
                      <div className="text-sm text-slate-400">Vidéos de plus en plus longues (21-66 min)</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-green-400 rounded-full mt-2"></div>
                    <div>
                      <div className="font-semibold text-green-300">Tags récurrents</div>
                      <div className="text-sm text-slate-400">['amixem', 'humour', 'delire'] + thématiques variables</div>
                    </div>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full mt-2"></div>
                    <div>
                      <div className="font-semibold text-yellow-300">Format viral</div>
                      <div className="text-sm text-slate-400">"ON [VERBE]" très performant (sosies: 7.5M vues)</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-red-400 rounded-full mt-2"></div>
                    <div>
                      <div className="font-semibold text-red-300">100% sponsorisé</div>
                      <div className="text-sm text-slate-400">Toutes les vidéos ont un sponsor (feature importante)</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-pink-400 rounded-full mt-2"></div>
                    <div>
                      <div className="font-semibold text-pink-300">Corrélation durée/vues</div>
                      <div className="text-sm text-slate-400">Vidéos longues (+3000s) = plus de vues</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Objectif */}
            <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-xl p-6 border border-purple-500/30">
              <h3 className="text-xl font-bold mb-3">🎯 Objectif du Projet</h3>
              <p className="text-slate-300 mb-4">
                Créer un système IA capable de <span className="text-purple-300 font-semibold">générer automatiquement</span> les 
                caractéristiques complètes d'une future vidéo (date, durée, tags, catégories) et de 
                <span className="text-blue-300 font-semibold"> prédire sa performance</span> (vues, likes, commentaires) sans aucun input manuel.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-purple-500/20 rounded-full text-sm">Génération autonome</span>
                <span className="px-3 py-1 bg-blue-500/20 rounded-full text-sm">Prédiction temporelle</span>
                <span className="px-3 py-1 bg-green-500/20 rounded-full text-sm">Multi-output</span>
                <span className="px-3 py-1 bg-yellow-500/20 rounded-full text-sm">Approche hybride</span>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'architecture' && (
          <div className="space-y-6">
            {/* Architecture Flow */}
            <div className="bg-slate-800/70 rounded-xl p-8 border border-slate-700">
              <h3 className="text-2xl font-bold mb-6">Architecture Hybride Complète</h3>
              <div className="space-y-6">
                {architectureSteps.map((step, idx) => (
                  <div key={idx} className="relative">
                    <div className={`bg-${step.color}-900/30 border-2 border-${step.color}-500/50 rounded-xl p-6`}>
                      <div className="flex items-center gap-4 mb-4">
                        <div className={`p-3 bg-${step.color}-500/20 rounded-lg`}>
                          {step.icon}
                        </div>
                        <div>
                          <h4 className="text-xl font-bold">{step.title}</h4>
                          <p className="text-sm text-slate-400">Modèle: {step.model}</p>
                        </div>
                      </div>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div>
                          <div className="text-sm font-semibold text-slate-400 mb-2">INPUTS</div>
                          <ul className="space-y-1">
                            {step.inputs.map((input, i) => (
                              <li key={i} className="text-sm flex items-center gap-2">
                                <ChevronRight className="w-4 h-4 text-green-400" />
                                {input}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <div className="text-sm font-semibold text-slate-400 mb-2">OUTPUTS</div>
                          <ul className="space-y-1">
                            {step.outputs.map((output, i) => (
                              <li key={i} className="text-sm flex items-center gap-2">
                                <ChevronRight className="w-4 h-4 text-purple-400" />
                                {output}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                    {idx < architectureSteps.length - 1 && (
                      <div className="flex justify-center my-4">
                        <div className="w-1 h-8 bg-gradient-to-b from-blue-500 to-green-500"></div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Avantages */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-6">
                <h4 className="font-bold text-green-300 mb-3 flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5" />
                  Avantages de l'approche
                </h4>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <span className="text-green-400">✓</span>
                    <span>Séparation logique des tâches (génération vs prédiction)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-400">✓</span>
                    <span>LSTM capte les patterns temporels complexes</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-400">✓</span>
                    <span>Random Forest robuste et interprétable</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-400">✓</span>
                    <span>Génération autonome sans input utilisateur</span>
                  </li>
                </ul>
              </div>
              <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-6">
                <h4 className="font-bold text-yellow-300 mb-3 flex items-center gap-2">
                  <AlertCircle className="w-5 h-5" />
                  Points d'attention
                </h4>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-400">⚠</span>
                    <span>LSTM nécessite beaucoup de données (min 100+ vidéos)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-400">⚠</span>
                    <span>Drift temporel : channel évolue dans le temps</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-400">⚠</span>
                    <span>Split temporel CRUCIAL (pas de shuffle)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-400">⚠</span>
                    <span>Prédire log(views) plutôt que views directement</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'pipeline' && (
          <div className="space-y-4">
            {pipeline.map((phase, idx) => (
              <div key={idx} className="bg-slate-800/70 rounded-xl border border-slate-700 overflow-hidden">
                <button
                  onClick={() => setAnalysisStep(analysisStep === idx ? -1 : idx)}
                  className="w-full p-6 flex items-center justify-between hover:bg-slate-700/50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center font-bold text-purple-300">
                      {phase.step}
                    </div>
                    <div className="text-left">
                      <h4 className="font-bold text-lg">{phase.title}</h4>
                      <p className="text-sm text-slate-400">{phase.tasks.length} tâches</p>
                    </div>
                  </div>
                  <ChevronRight className={`w-5 h-5 transition-transform ${analysisStep === idx ? 'rotate-90' : ''}`} />
                </button>
                {analysisStep === idx && (
                  <div className="p-6 pt-0 space-y-4">
                    <div className="space-y-2">
                      {phase.tasks.map((task, i) => (
                        <div key={i} className="flex items-center gap-2 text-sm">
                          <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0" />
                          <span>{task}</span>
                        </div>
                      ))}
                    </div>
                    <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
                      <pre className="text-xs text-green-400 font-mono whitespace-pre">
                        {phase.code}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {activeTab === 'code' && (
          <div className="bg-slate-800/70 rounded-xl p-6 border border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold">Script Python Complet</h3>
              <span className="text-sm text-slate-400">Prêt à l'emploi</span>
            </div>
            <div className="bg-slate-900 rounded-lg p-6 overflow-x-auto">
              <pre className="text-sm text-green-400 font-mono whitespace-pre">
{`# =============================================================================
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

# =============================================================================
# 5. RANDOM FOREST (PHASE 2)
# =============================================================================

def build_features_for_rf(row, history, tags_encoded, cat_encoded):
    """Construit le vecteur de features pour Random Forest"""
    
    features = [
        # Features LSTM générées
        row['duration'],
        row['day_of_week'],
        row['hour'],
        row['month'],
        row['day_sin'],
        row['day_cos'],
        row['hour_sin'],
        row['hour_cos'],
        
        # Contexte historique
        history['view_count'].mean(),
        history['view_count'].std(),
        history['likes'].mean(),
        history['comment_count'].mean(),
        history['days_since_last'].mean(),
        history['engagement_rate'].mean(),
        history['trend_views'].iloc[-1] if len(history) > 0 else 0,
        
        # Tags et catégories encodés
        *tags_encoded,
        *cat_encoded,
        
        # Sponsored (constant dans cet exemple)
        1  # True
    ]
    
    return features

def train_random_forest(X_train, y_train):
    """Entraîne Random Forest multi-output"""
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Multi-output
    multi_rf = MultiOutputRegressor(rf)
    multi_rf.fit(X_train, y_train)
    
    return multi_rf

# =============================================================================
# 6. GÉNÉRATION VIDÉO COMPLÈTE
# =============================================================================

class YouTubeVideoGenerator:
    """Générateur complet de vidéos"""
    
    def __init__(self, lstm_model, rf_model, scaler, mlb_tags, mlb_cat):
        self.lstm_model = lstm_model
        self.rf_model = rf_model
        self.scaler = scaler
        self.mlb_tags = mlb_tags
        self.mlb_cat = mlb_cat
    
    def generate_next_video(self, history_df, n_context=10):
        """Génère une vidéo complète"""
        
        # Phase 1: LSTM prédit features créatives
        last_videos = history_df.tail(n_context)
        sequence = self.prepare_sequence(last_videos)
        
        lstm_output = self.lstm_model.predict(sequence, verbose=0)[0]
        
        # Décoder les prédictions LSTM
        duration = int(np.clip(lstm_output[0], 600, 5000))  # 10min-83min
        day_of_week = int(np.clip(lstm_output[1], 0, 6))
        hour = int(np.clip(lstm_output[2], 0, 23))
        
        # Prédire la prochaine date
        last_date = history_df['upload_date'].iloc[-1]
        days_to_add = 7  # Pattern observé
        next_date = last_date + pd.Timedelta(days=days_to_add)
        
        # Générer tags (top-3 les plus probables)
        tags = self.generate_tags(history_df)
        categories = ['Comedy']  # Constant dans vos données
        
        # Phase 2: RF prédit performance
        tags_enc = self.mlb_tags.transform([tags])[0]
        cat_enc = self.mlb_cat.transform([categories])[0]
        
        features = build_features_for_rf(
            {
                'duration': duration,
                'day_of_week': day_of_week,
                'hour': hour,
                'month': next_date.month,
                'day_sin': np.sin(2*np.pi*day_of_week/7),
                'day_cos': np.cos(2*np.pi*day_of_week/7),
                'hour_sin': np.sin(2*np.pi*hour/24),
                'hour_cos': np.cos(2*np.pi*hour/24),
            },
            history_df.tail(10),
            tags_enc,
            cat_enc
        )
        
        predictions = self.rf_model.predict([features])[0]
        
        return {
            'upload_date': next_date.strftime('%Y-%m-%d %H:%M'),
            'duration': duration,
            'duration_minutes': f"{duration//60}:{duration%60:02d}",
            'categories': categories,
            'tags': tags,
            'view_count': int(np.expm1(predictions[0])),  # Inverse log
            'likes': int(np.expm1(predictions[1])),
            'comment_count': int(predictions[2]),
            'dislikes': int(predictions[3]),
            'sponsored': True,
            'age_limit': 0,
            'is_live': False,
            'was_live': False
        }
    
    def prepare_sequence(self, videos):
        """Prépare séquence pour LSTM"""
        feature_cols = [
            'duration', 'day_of_week', 'hour', 'month',
            'log_views', 'log_likes', 'days_since_last'
        ]
        seq = videos[feature_cols].values
        seq = self.scaler.transform(seq)
        return seq.reshape(1, len(seq), -1)
    
    def generate_tags(self, history_df):
        """Génère tags basés sur fréquence historique"""
        from collections import Counter
        
        # Tags récents
        recent_tags = [tag for tags in history_df.tail(5)['tags'] 
                      for tag in tags]
        
        # Tags de base toujours présents
        base_tags = ['amixem', 'humour', 'delire']
        
        # Ajouter 2-3 tags thématiques
        tag_counts = Counter(recent_tags)
        thematic = [tag for tag, _ in tag_counts.most_common(5) 
                   if tag not in base_tags][:3]
        
        return base_tags + thematic

# =============================================================================
# 7. PIPELINE COMPLET
# =============================================================================

def main_pipeline(csv_path):
    """Pipeline complet d'entraînement et prédiction"""
    
    print("🚀 Démarrage du pipeline...")
    
    # 1. Chargement
    print("\\n📊 Chargement des données...")
    df = load_and_clean_data(csv_path)
    print(f"   {len(df)} vidéos chargées")
    
    # 2. Feature Engineering
    print("\\n🔧 Feature Engineering...")
    df = engineer_features(df)
    tags_enc, cat_enc, mlb_tags, mlb_cat = encode_tags_categories(df)
    
    # 3. Split temporel
    print("\\n✂️ Split temporel...")
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
    print("\\n🧠 Préparation séquences LSTM...")
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
    print("\\n🔥 Entraînement LSTM...")
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
    
    # 6. Préparation Random Forest
    print("\\n🌲 Préparation Random Forest...")
    X_train_rf = []
    y_train_rf = []
    
    for i in range(10, len(train)):
        history_videos = train.iloc[i-10:i]
        current = train.iloc[i]
        
        features = build_features_for_rf(
            current,
            history_videos,
            tags_enc[i],
            cat_enc[i]
        )
        
        targets = [
            current['log_views'],
            current['log_likes'],
            current['comment_count'],
            current['dislikes']
        ]
        
        X_train_rf.append(features)
        y_train_rf.append(targets)
    
    X_train_rf = np.array(X_train_rf)
    y_train_rf = np.array(y_train_rf)
    
    # 7. Entraînement RF
    print("\\n🎯 Entraînement Random Forest...")
    rf_model = train_random_forest(X_train_rf, y_train_rf)
    
    # 8. Évaluation
    print("\\n📈 Évaluation sur test set...")
    X_test_rf = []
    y_test_rf = []
    
    for i in range(val_idx + 10, len(df)):
        history_videos = df.iloc[i-10:i]
        current = df.iloc[i]
        
        features = build_features_for_rf(
            current,
            history_videos,
            tags_enc[i],
            cat_enc[i]
        )
        
        targets = [
            current['log_views'],
            current['log_likes'],
            current['comment_count'],
            current['dislikes']
        ]
        
        X_test_rf.append(features)
        y_test_rf.append(targets)
    
    X_test_rf = np.array(X_test_rf)
    y_test_rf = np.array(y_test_rf)
    
    predictions = rf_model.predict(X_test_rf)
    
    # Métriques
    mae_views = mean_absolute_error(
        np.expm1(y_test_rf[:, 0]),
        np.expm1(predictions[:, 0])
    )
    r2_views = r2_score(y_test_rf[:, 0], predictions[:, 0])
    
    print(f"\\n   MAE Vues: {mae_views:,.0f}")
    print(f"   R² Vues: {r2_views:.3f}")
    
    # 9. Créer générateur
    print("\\n✨ Création du générateur...")
    generator = YouTubeVideoGenerator(
        lstm, rf_model, scaler, mlb_tags, mlb_cat
    )
    
    # 10. Générer prédiction
    print("\\n🎬 Génération prochaine vidéo...")
    next_video = generator.generate_next_video(df)
    
    print("\\n" + "="*60)
    print("🎥 PRÉDICTION PROCHAINE VIDÉO")
    print("="*60)
    for key, value in next_video.items():
        print(f"   {key:20s}: {value}")
    print("="*60)
    
    return generator, df

# =============================================================================
# 8. EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    generator, df = main_pipeline('youtube_data.csv')
    
    # Générer plusieurs prédictions
    print("\\n🔮 Génération de 5 vidéos futures...")
    for i in range(5):
        video = generator.generate_next_video(df)
        print(f"\\nVidéo {i+1}:")
        print(f"  Date: {video['upload_date']}")
        print(f"  Durée: {video['duration_minutes']}")
        print(f"  Vues prédites: {video['view_count']:,}")
        print(f"  Tags: {', '.join(video['tags'])}")
        
        # Ajouter à l'historique pour next iteration
        # (Dans un vrai cas, vous ajouteriez cette vidéo simulée)

print("\\n✅ Pipeline terminé !")`}
              </pre>
            </div>
            
            <div className="mt-6 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
              <h4 className="font-bold text-blue-300 mb-2">📝 Instructions d'utilisation</h4>
              <ol className="space-y-2 text-sm">
                <li className="flex gap-2">
                  <span className="font-bold">1.</span>
                  <span>Installer les dépendances: <code className="bg-slate-800 px-2 py-1 rounded text-green-400">pip install pandas numpy scikit-learn tensorflow matplotlib seaborn</code></span>
                </li>
                <li className="flex gap-2">
                  <span className="font-bold">2.</span>
                  <span>Placer votre CSV dans le même dossier que le script</span>
                </li>
                <li className="flex gap-2">
                  <span className="font-bold">3.</span>
                  <span>Lancer: <code className="bg-slate-800 px-2 py-1 rounded text-green-400">python youtube_predictor.py</code></span>
                </li>
                <li className="flex gap-2">
                  <span className="font-bold">4.</span>
                  <span>Le script entraîne automatiquement et génère les prédictions</span>
                </li>
              </ol>
            </div>
          </div>
        )}

        {/* Footer avec métriques d'évaluation */}
        <div className="mt-8 bg-gradient-to-r from-slate-800/50 to-slate-700/50 rounded-xl p-6 border border-slate-600">
          <h3 className="text-lg font-bold mb-4">📊 Métriques d'évaluation recommandées</h3>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="font-semibold text-purple-300 mb-1">MAE (Mean Absolute Error)</div>
              <div className="text-slate-400">Erreur moyenne en nombre de vues</div>
            </div>
            <div>
              <div className="font-semibold text-blue-300 mb-1">MAPE (%)</div>
              <div className="text-slate-400">Erreur en pourcentage</div>
            </div>
            <div>
              <div className="font-semibold text-green-300 mb-1">R² Score</div>
              <div className="text-slate-400">Variance expliquée (0-1)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default YouTubePredictorDashboard;