import os
import pandas as pd

file = os.path.join(os.path.dirname(__file__), 'amixem_20251023.csv')

df = pd.read_csv(file)


# get the n last video titles
def get_n_last_video_titles(n: int = 5) -> list[str]:
    """
    Retourne les titres des n dernières vidéos publiées (les plus récentes).
    """
    # 1. Trier le DataFrame par timestamp décroissant (du plus récent au plus ancien)
    # Si le timestamp manque, on peut utiliser 'upload_date'
    df_sorted = df.sort_values(by='timestamp', ascending=False)

    # 2. Sélectionner la colonne 'title', prendre les n premières lignes et convertir en liste
    titles = df_sorted['title'].head(n).tolist()

    return titles


# Exemple d'utilisation (optionnel)
if __name__ == "__main__":
    print(get_n_last_video_titles(5))