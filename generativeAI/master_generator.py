import sys
import os
import ast

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Masque les INFO et WARNINGS TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force l'utilisation du CPU sans chercher de GPU

# Ajouter le dossier courant au path pour importer les autres modules du dossier
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import des étapes 1 et 2
try:
    import model_predict_step1 as step1
    import model_predict_step2 as step2
except ImportError as e:
    print("Erreur d'importation des modules step1 ou step2.")
    print("Assurez-vous que 'model_predict_step1.py' et 'model_predict_step2.py' sont dans le même dossier.")
    raise e

def generate_full_prediction(nb_videos=1):
    """
    Génère une prédiction complète en chaînant Step 1 (Contenu/Date) et Step 2 (Performance).
    
    :param nb_videos: Nombre de vidéos à générer.
    :return: Liste de dictionnaires contenant toutes les infos fusionnées et converties en string.
    """
    print(f"--- Lancement de la génération Maître pour {nb_videos} vidéo(s) ---")
    
    # 1. GÉNÉRATION STEP 1 (Date, Durée, Tags, Titre/Format)
    try:
        step1_results = step1.predict_multiple(nb_videos)
    except Exception as e:
        return [{"error": f"Erreur lors de l'étape 1 : {str(e)}"}]
    
    full_results = []
    
    # 2. BOUCLE SUR CHAQUE RÉSULTAT POUR STEP 2
    for vid in step1_results:
        # Préparation des données pour Step 2
        # Step 1 donne la durée en minutes (string), Step 2 la veut en secondes (int)
        try:
            duration_minutes = float(vid.get('duration_min', 0))
            duration_seconds = int(duration_minutes * 60)
            
            # Nettoyage des tags (Step 1 renvoie parfois une string de liste "['a', 'b']")
            tags_raw = vid.get('tags', "")
            
            step2_input = {
                "date": vid.get('date_str'),
                "duration": duration_seconds,
                "tags": tags_raw
            }
            
            # Appel Step 2
            perf_prediction = step2.generate_performance(step2_input)
            
            # Fusion des données
            merged_video = {**vid, **perf_prediction}
            
            # Petit nettoyage cosmétique : on retire les métadonnées techniques de Step 2
            if "input_metadata" in merged_video:
                del merged_video["input_metadata"]
            
            # --- CONVERSION EN STRINGS (FORMATAGE) ---
            final_video_str = {}
            for key, value in merged_video.items():
                if isinstance(value, float):
                    # Formatage des floats : 2 chiffres après la virgule (ex: 12.345 -> '12.35')
                    final_video_str[key] = f"{value:.2f}"
                elif isinstance(value, int):
                    # Conversion des int en string
                    final_video_str[key] = str(value)
                else:
                    # Conversion par défaut (str, bool, None, etc.)
                    final_video_str[key] = str(value)
            
            full_results.append(final_video_str)
            
        except Exception as e:
            print(f"Erreur sur la vidéo {vid.get('date_str')}: {e}")
            # En cas d'erreur, on renvoie quand même ce qu'on a, converti en string
            error_dict = {k: str(v) for k, v in vid.items()}
            error_dict['error_step2'] = str(e)
            full_results.append(error_dict)

    return full_results

if __name__ == "__main__":
    # Test local
    results = generate_full_prediction(nb_videos=3)
    
    print(results)
    
    print("\n" + "="*50)
    print("RÉSULTATS GLOBAUX (FORMAT STRING VÉRIFIÉ)")
    print("="*50)
    
    for i, res in enumerate(results):
        print(f"\nVIDÉO #{i+1} :")
        for k, v in res.items():
            # Affiche le type pour prouver que ce sont bien des strings
            print(f"  - {k}: '{v}'")