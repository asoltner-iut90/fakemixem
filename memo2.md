# Projet — Prédiction et génération pour vidéos YouTube (ex. Amixem)

> **Objectif** : concevoir deux composantes principales : une **IA prédictive** (estimation d'attributs mesurables) et une **IA générative** (proposition de titre et description réalistes à partir des prédictions et d'exemples existants).
---
## 1) Données à prédire (IA prédictive)
### Attributs principaux à prédire automatiquement

| Nom du champ             | Type     | Description                                                                                 |
| ------------------------ | -------- | ------------------------------------------------------------------------------------------- |
| `date_de_sortie`         | datetime | Date probable de publication (prédite à partir de la fréquence historique).                 |
| `nombre_de_vues`     | int      | Nombre estimé de vues.                                                       |
| `duree_seconds`          | int      | Durée estimée de la vidéo en secondes.                                                      |
| `sponsor_present`        | bool     | Présence probable d’un sponsor.                                                             |
| `nombre_de_likes`    | int      | Likes estimés.                                                               |
| `nombre_de_dislikes` | int      | Dislikes estimés.                                                            |
| `est_suite`              | bool     | Indique si la vidéo est une suite (ex: « partie 2 », « épisode 3 ») ou un concept original. |

### Données utilisées pour la prédiction

* Historique des métadonnées des vidéos précédentes : titre, description, date, vues, likes.
* Jours entre les publications.
* Tendances temporelles (jour de la semaine, heure de publication).
* Mots-clés extraits du titre/description.
* Embeddings de texte pour similarité entre concepts.

---

## 2) Données générées (IA générative)

L’IA générative utilisera les prédictions comme base pour créer :

| Nom du champ  | Type   | Description                                                       |
| ------------- | ------ | ----------------------------------------------------------------- |
| `titre`       | string | Titre plausible, cohérent avec les vidéos précédentes similaires. |
| `description` | string | Description complète (introduction, sponsor, tags et CTA).        |

---

## 3) Workflow simplifié de génération

1. **Prédiction** : L’IA prédictive estime les valeurs de durée, vues, likes, sponsor, etc. à partir des vidéos existantes.
2. **Sélection d’exemples similaires** : le système recherche n vidéos dont la durée et les vues sont proches de la vidéo prédite.
3. **Extraction de descriptions** : récupération des descriptions des 3 vidéos les plus pertinentes via un outil externe.
4. **Génération** :

   * Utilisation des descriptions et du sponsor prédit pour générer un nouveau **titre** et une **description** cohérents.
   * Le modèle génératif (ex : API Mistral) combine ces informations pour produire du contenu adapté.

---

## 4) Exemple de schéma de données (vidéo)

```csv
video_id,title,description,timestamp,duration_seconds,views_30d,likes_30d,dislikes_30d,sponsor_present,est_suite
```

---

## 5) Étapes du pipeline complet

1. **Collecte automatique** des données via YouTube Data API.
2. **Extraction automatique** de features : durée, fréquence de publication, mots-clés, etc.
3. **Prédiction** des attributs par l’IA prédictive (modèles de régression/classification légers).
4. **Recherche de vidéos similaires** basée sur embeddings textuels et durée/vues.
5. **Récupération de descriptions** de ces vidéos via un outil externe.
6. **Génération finale** du titre et de la description par le modèle génératif.

---

## 6) Vue d’ensemble du système

```
[YouTube Data API]
      ↓
[Collecte automatique des données]
      ↓
[IA prédictive] → {durée, vues, likes, sponsor, est_suite}
      ↓
[Sélection vidéos similaires]
      ↓
[IA générative (Mistral)] → {titre, description}
```

---