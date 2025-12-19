# Fakemixem

## Objectifs

### IA prédictive
Capable de prédire les informations suivantes de la prochaines video
- La date
- La durée
- Le sponsor
- Si c'est une suite ou un concept original


### IA générative
Génère le titre et la description de la prochaine video.

## Structure du projet

- `analysis/` : Contient les scripts d'analyse des données et l'entraînement des modèles IA.
- `datasets/` : Contient les jeux de données utilisés pour l'entraînement et les tests et les méthodes de collecte de données.
- `generativeAI/`: Contient les scripts pour l'IA générative (Gemini).
- `models/` : Contient les modèles IA entraînés.
- `out/` : Contient les images générées et les résultats des analyses.
- `test_and_trial/` : Contient les scripts de test et d'expérimentation.
- `youtubeTools/` : Contient les outils pour récupérer les données YouTube.
- `app.py` : Point d'entrée principal de l'application.
- `LICENSE` : Licence du projet.
- `requirements.txt` : Liste des dépendances Python nécessaires pour le projet.
- `README.md` : Documentation du projet.
- `thumbnail.png` : Image miniature du projet.

## Installation

### Dépendances

Sur linux :
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sur Windows :
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Variables d'environnement

Créer un fichier `secrets.toml` dans un dossier `.streamlit` à la racine du projet avec le contenu suivant :

```toml
GOOGLE_API_KEY=[Votre clé API Google ici]
```

## Utilisation

Lancer l'application principale avec la commande suivante :

```bash
streamlit run app.py
```
