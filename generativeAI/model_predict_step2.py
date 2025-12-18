import sys
import os

# --- BLOC D'IMPORT DYNAMIQUE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
analysis_path = os.path.join(project_root, 'analysis')

if analysis_path not in sys.path:
    sys.path.append(analysis_path)

# --- IMPORT ---
try:
    from ia_step2 import AmixemPredictorPhase2
except ImportError as e:
    print(f"ERREUR CRITIQUE : Impossible d'importer 'ia_step2' depuis {analysis_path}.")
    raise e

# --- FONCTION DE GÉNÉRATION ---
def generate_performance(phase1_data):
    """
    Prend les données de la Phase 1 et génère les performances prédites.
    Format attendu phase1_data: { 'date': str, 'duration': int(sec), 'tags': str }
    """
    try:
        predictor = AmixemPredictorPhase2()
        
        # Tentative de chargement
        if not predictor.load_models():
            # Fallback chemin absolu
            models_absolute_path = os.path.join(project_root, 'models')
            predictor.models_dir = models_absolute_path
            if not predictor.load_models():
                return {"error": "Modèles non trouvés dans /models"}

        # Prédiction
        results = predictor.predict_performance(
            predicted_date=phase1_data.get('date'),
            predicted_duration=phase1_data.get('duration'),
            predicted_tags=phase1_data.get('tags')
        )
        return results

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Test direct de Step 2")
    mock_input = {"date": "2025-12-01", "duration": 1500, "tags": "test python"}
    print(generate_performance(mock_input))