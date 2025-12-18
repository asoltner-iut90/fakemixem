import pandas as pd
import numpy as np
import os
import joblib
import ast
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class AmixemPredictorPhase2:
    def __init__(self, models_dir="../models"):
        """
        Initialise le prédicteur.
        :param models_dir: Dossier où les modèles entraînés seront stockés.
        """
        self.models_dir = models_dir
        self.model_views = None
        self.model_likes = None
        self.model_comments = None
        
        # S'assurer que le dossier existe
        os.makedirs(self.models_dir, exist_ok=True)

    def _get_duration_class(self, seconds):
        """Classification inspirée de l'IA LSTM précédente"""
        mins = seconds / 60
        if mins < 26: return 0  # Court/Standard
        elif mins < 48: return 1 # Long
        else: return 2           # Très long / Spécial
        
    def _preprocess_features(self, df, date_col, duration_col):
        """
        Enrichit le dataframe avec des features cycliques et des classes de durée.
        """
        # --- Gestion de la Date ---
        if not np.issubdtype(df[date_col].dtype, np.datetime64):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        df = df.dropna(subset=[date_col])
        
        # Extraction basique
        day_of_week = df[date_col].dt.dayofweek
        month = df[date_col].dt.month
        
        # --- Features cycliques (Inspiration IA précédente) ---
        # Permet au modèle de comprendre la continuité (ex: Dec -> Janvier)
        df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        df['day_of_year'] = df[date_col].dt.dayofyear
        
        # --- Gestion de la durée ---
        df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')
        df = df.dropna(subset=[duration_col])

        # Classification précise (<26, <48, >48)
        df['duration_class'] = df[duration_col].apply(self._get_duration_class)
        
        # Log de la durée (gestion des échelles)
        df['log_duration'] = np.log1p(df[duration_col])

        return df

    def _parse_tags(self, tag_entry):
        """Gère les tags qu'ils soient string, list ou string-de-list"""
        if pd.isna(tag_entry):
            return ""
        if isinstance(tag_entry, str):
            # Si c'est une string qui ressemble à une liste "['a', 'b']"
            if tag_entry.strip().startswith('[') and tag_entry.strip().endswith(']'):
                try:
                    # On tente de l'évaluer comme une liste Python
                    parsed = ast.literal_eval(tag_entry)
                    if isinstance(parsed, list):
                        return " ".join(parsed)
                except:
                    pass
            # Sinon c'est juste une string de tags
            return tag_entry
        if isinstance(tag_entry, list):
            return " ".join(tag_entry)
        return str(tag_entry)

    def train(self, dataset_path):
        """
        Entraîne les modèles sur le dataset complet et les sauvegarde.
        """
        print(f"Chargement des données depuis {dataset_path}...")
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"ERREUR: Fichier non trouvé à {dataset_path}")
            return

        # --- Détection automatique des noms de colonnes ---
        def find_column(options, mandatory=True):
            for opt in options:
                if opt in df.columns:
                    return opt
            if mandatory:
                raise KeyError(f"Impossible de trouver une colonne correspondant à : {options}. Vérifiez votre CSV.")
            return None

        col_date = find_column(['publishedAt', 'date', 'Date', 'publication_date', 'upload_date'])
        col_views = find_column(['viewCount', 'views', 'view_count', 'Vues', 'vues'])
        col_likes = find_column(['likeCount', 'likes', 'like_count', 'Likes'])
        col_comments = find_column(['commentCount', 'comments', 'comment_count', 'Commentaires'])
        col_duration = find_column(['duration', 'duration_seconds', 'length', 'duree'])
        col_tags = find_column(['tags', 'keywords', 'Tags', 'mots_cles'])

        print(f"Colonnes identifiées : Date={col_date}, Durée={col_duration}, Tags={col_tags}")

        # Nettoyage et parsing des tags AVANT le split
        df[col_tags] = df[col_tags].apply(self._parse_tags)

        # Feature engineering avancé
        df = self._preprocess_features(df, date_col=col_date, duration_col=col_duration)
        
        # Conversion cibles
        for col in [col_views, col_likes, col_comments]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Liste des features mises à jour avec l'inspiration IA précédente
        # Note: on utilise maintenant les features cycliques (sin/cos) au lieu de OneHot le mois brut
        features_input = [
            col_tags, 
            col_duration, 'log_duration', 'duration_class',
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 'day_of_year'
        ]
        targets = [col_views, col_likes, col_comments]
        
        df = df.dropna(subset=features_input + targets)
        
        if len(df) == 0:
            raise ValueError("Dataset vide après nettoyage.")

        X = df[features_input]
        y_views = df[col_views]
        y_likes = df[col_likes]
        y_comments = df[col_comments]

        # --- Pipeline mis à jour ---
        preprocessor = ColumnTransformer(
            transformers=[
                # Texte : TF-IDF (Plus robuste que MultiLabelBinarizer pour la régression pure)
                ('tags_tfidf', TfidfVectorizer(max_features=200), col_tags),
                
                # Numérique : Scaling standard pour durées et features cycliques
                ('num_scaled', StandardScaler(), [
                    col_duration, 'log_duration', 
                    'day_sin', 'day_cos', 'month_sin', 'month_cos', 'day_of_year'
                ]),
                
                # Catégorique : Duration Class (0, 1, 2) traitée comme catégorie
                ('cat_encoded', OneHotEncoder(handle_unknown='ignore'), ['duration_class'])
            ])

        # Random Forest robuste
        rf_params = {'n_estimators': 200, 'random_state': 42, 'min_samples_leaf': 2}

        self.model_views = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(**rf_params))])
        self.model_likes = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(**rf_params))])
        self.model_comments = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(**rf_params))])

        print("Entraînement des modèles (Views, Likes, Comments)...")
        self.model_views.fit(X, y_views)
        self.model_likes.fit(X, y_likes)
        self.model_comments.fit(X, y_comments)

        self._save_models()
        print("Entraînement terminé et modèles sauvegardés.")

    def _save_models(self):
        joblib.dump(self.model_views, os.path.join(self.models_dir, 'model_views.pkl'))
        joblib.dump(self.model_likes, os.path.join(self.models_dir, 'model_likes.pkl'))
        joblib.dump(self.model_comments, os.path.join(self.models_dir, 'model_comments.pkl'))

    def load_models(self):
        try:
            self.model_views = joblib.load(os.path.join(self.models_dir, 'model_views.pkl'))
            self.model_likes = joblib.load(os.path.join(self.models_dir, 'model_likes.pkl'))
            self.model_comments = joblib.load(os.path.join(self.models_dir, 'model_comments.pkl'))
            return True
        except FileNotFoundError:
            return False

    def predict_performance(self, predicted_date, predicted_duration, predicted_tags):
        if self.model_views is None:
            if not self.load_models():
                print("ERREUR CRITIQUE : Modèles non chargés.")
                return None

        # Préparation tags
        if isinstance(predicted_tags, list):
            predicted_tags = " ".join(predicted_tags)

        if isinstance(predicted_date, str):
            dt_obj = datetime.strptime(predicted_date, "%Y-%m-%d")
        else:
            dt_obj = predicted_date

        # Récupération dynamique des noms de colonnes
        try:
            transformers = self.model_views.named_steps['preprocessor'].transformers_
            col_tags_name = transformers[0][2] 
            col_duration_name = transformers[1][2][0] 
        except:
            col_tags_name = 'tags'
            col_duration_name = 'duration'

        input_data = pd.DataFrame({
            'publishedAt': [dt_obj], 
            col_duration_name: [predicted_duration],
            col_tags_name: [predicted_tags]
        })

        # Feature engineering (IDENTIQUE AU TRAIN)
        input_data = self._preprocess_features(input_data, date_col='publishedAt', duration_col=col_duration_name)

        pred_views = self.model_views.predict(input_data)[0]
        pred_likes = self.model_likes.predict(input_data)[0]
        pred_comments = self.model_comments.predict(input_data)[0]

        engagement_rate = ((pred_likes + pred_comments) / pred_views) * 100 if pred_views > 0 else 0

        return {
            "predicted_views": int(pred_views),
            "predicted_likes": int(pred_likes),
            "predicted_comments": int(pred_comments),
            "estimated_engagement_rate": round(engagement_rate, 2),
            "input_metadata": {
                "date": str(dt_obj.date()),
                "duration_min": round(predicted_duration/60, 2),
                "duration_class": int(input_data['duration_class'].iloc[0]),
                "tags_extract": predicted_tags[:50] + "..."
            }
        }

if __name__ == "__main__":
    predictor = AmixemPredictorPhase2()
    
    possible_paths = [
        "../datasets/amixem_20251023.csv",
        "../Datasets/amixem_20251023.csv",
        "amixem_20251023.csv",
        "datasets/amixem_20251023.csv"
    ]
    
    dataset_found = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Dataset trouvé : {path}")
            try:
                predictor.train(path)
                dataset_found = True
                break
            except Exception as e:
                print(f"Erreur avec {path}: {e}")
    
    if not dataset_found and not predictor.load_models():
        print("ERREUR : Impossible de trouver le dataset ou les modèles.")
    else:
        print("\n--- Test Comparatif Durée (Inspiration IA LSTM) ---")
        
        # Test sur les classes spécifiques : <26m vs >48m
        res_class0 = predictor.predict_performance("2025-11-15", 20*60, "vlog studio") # 20 min (Class 0)
        res_class2 = predictor.predict_performance("2025-11-15", 50*60, "voyage extreme") # 50 min (Class 2)
        
        if res_class0 and res_class2:
            print(f"Classe 0 (20min) : {res_class0['predicted_views']:,} vues")
            print(f"Classe 2 (50min) : {res_class2['predicted_views']:,} vues")