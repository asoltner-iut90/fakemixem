import streamlit as st
import pandas as pd
import os
from generativeAI.gemini_tools import IA
from generativeAI.assistant import Assistant
import re

# Configuration de la page
st.set_page_config(
    page_title="Mon Application",
    page_icon="🎬",
    layout="wide"
)

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))

if "assistant" not in st.session_state and api_key:
    ia = IA(api_key)
    st.session_state.assistant = Assistant(ia)

# Initialisation de l'historique du chat pour l'affichage
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "gallery" not in st.session_state:
    st.session_state.gallery = {}

# Définition du conteneur
llm = st.container()

hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden; display: none;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem;
    }
    [data-testid="stDecoration"] {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Titre principal
<<<<<<< Updated upstream
st.title("Fakemixem")
=======
st.title("FakeMixem AI 🎬")
>>>>>>> Stashed changes

home, data, llm, tech_tab = st.tabs(["🏠 Objectif", "📊 Données", "🤖 LLM", "⚙️ Architecture IA"])

# --- Onglet Objectif ---
with home:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.header("Objectifs de l'application")
        st.markdown("""
        Cette application vise à :
        - **Prédire des informations clés** sur les prochaines vidéos (date, durée, tags).
        - **Générer du contenu** (titres, descriptions) pour les vidéos.
        - **Imaginer des concepts** de miniatures.
        """)
        st.info("L'objectif principal est d'aider à la prise de décision stratégique pour le contenu vidéo.")

# --- Onglet Données ---
with data:
    st.header("Le dataset")
    st.write("L'IA ne devine pas au hasard. Elle s'entraîne sur l'historique réel de la chaîne.")

    try:
        file_path = "datasets/amixem_20251219.csv" 
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Tri de plus récent au plus ancien
            df.sort_values(by='upload_date', ascending=False, inplace=True)
            
            # Make duration in minutes rounded to 2 decimals
            if 'duration' in df.columns:
                df['duration'] = (df['duration'] / 60).round(2)
            
            
            
            if 'upload_date' in df.columns:
                df['upload_date'] = pd.to_datetime(df['upload_date'], format='%Y%m%d', errors='coerce')
                df['year'] = df['upload_date'].dt.year
                df['day_name'] = df['upload_date'].dt.day_name()
            
            st.markdown("### Vue d'ensemble")
            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
            
            with col_kpi1:
                st.metric("Total vidéos", f"{len(df)}")
            with col_kpi2:
                # Calcul des vues totales en millions
                total_views = df['view_count'].sum()
                st.metric("Vues cumulées", f"{total_views/1e9:.2f} Md")
            with col_kpi3:
                # Moyenne des likes
                avg_likes = df['likes'].mean()
                st.metric("Moyenne Likes", f"{avg_likes/1000:.0f} k")
            with col_kpi4:
                # Année la plus ancienne
                oldest = df['upload_date'].min().year if 'upload_date' in df else "N/A"
                st.metric("Données depuis", f"{oldest}")

            st.divider()

            st.subheader("Jours de publication")
            if 'day_name' in df.columns:
                days_count = df['day_name'].value_counts()
                st.bar_chart(days_count)
                st.caption("L'IA utilise cette info pour savoir que le Dimanche est crucial.")

            st.divider()

            st.subheader("🧠 À quoi servent ces données pour l'IA ?")
            
            with st.expander("Voir le dictionnaire des variables (Feature Engineering)", expanded=True):
                st.markdown("""
                | Colonne | Rôle dans l'IA | Description |
                | :--- | :--- | :--- |
                | **title / description** | **Apprentissage sémantique** | Permet au LLM de comprendre le style, l'humour et les mots-clés qui cliquent. |
                | **tags** | **Associations** | Utilisé par le *Random Forest* pour lier des concepts (ex: "Lego" + "Construction"). |
                | **view_count** | **Target (Cible)** | C'est la note que l'IA essaie de prédire. C'est son objectif de réussite. |
                | **upload_date** | **Saisonnalité** | Permet de comprendre qu'une vidéo "Ski" marche mieux en Janvier qu'en Juillet. |
                | **duration** | **Format** | Aide l'IA à décider si le concept mérite 10min ou 40min. |
                """)

            st.subheader("Explorateur brut")
            st.dataframe(
                df[['title', 'upload_date', 'view_count', 'duration', 'tags']], 
                use_container_width=True,
                hide_index=True
            )
            
            # Bouton de téléchargement
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger le dataset complet (CSV)",
                data=csv,
                file_name='amixem_dataset_export.csv',
                mime='text/csv',
            )

        else:
            st.error(f"Le fichier de données est introuvable à l'emplacement : `{file_path}`")
            st.info("Assurez-vous que le fichier .csv est bien dans le dossier /datasets à la racine de votre projet.")

    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement des données : {e}")

# --- Onglet LLM ---
with llm:
    col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
    with col_h2:
        st.header("Studio Créatif Amixem 🎬")
        st.write("Décrivez un concept de vidéo, l'IA s'occupe du reste.")

        if not api_key:
            st.warning("⚠️ Clé API introuvable.")

    if api_key:
        if st.session_state.chat_history:
            with st.container(height=500, border=True):
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        content_text = msg["content"]

                        if isinstance(msg.get("images"), dict):
                            st.session_state.gallery.update(msg["images"])

                        # Regex qui capture la balise entière <show_image ... />
                        # (.*? est 'lazy' pour s'arrêter à la première fermeture >)
                        parts = re.split(r'(<show_image.*?>)', content_text)

                        for part in parts:
                            # Cas A : C'est une balise image détectée
                            if "<show_image" in part:
                                # Regex pour extraire l'ID, compatible avec " et '
                                match = re.search(r'id=["\'](.*?)["\']', part)

                                if match:
                                    img_id = match.group(1).strip()  # .strip() enlève les espaces parasites

                                    # Récupération dans la galerie
                                    image_data = st.session_state.gallery.get(img_id)

                                    if image_data:
                                        # Affichage centré
                                        c1, c2, c3 = st.columns([1, 4, 1])
                                        with c2:
                                            try:
                                                st.image(image_data, use_container_width=True)
                                            except:
                                                # Fallback bytes
                                                if hasattr(image_data, "image_bytes"):
                                                    st.image(image_data.image_bytes, use_container_width=True)
                                    else:
                                        st.error(f"Image introuvable : '{img_id}'")
                                else:
                                    pass  # Balise mal formée, on l'ignore

                            # Cas B : C'est du texte normal
                            elif part.strip():
                                st.markdown(part)

        col_i1, col_i2, col_i3 = st.columns([1, 2, 1])
        with col_i2:
            prompt_input = st.text_area("Votre message", height=100,
                                        placeholder="Ex: On passe 24h dans un bunker en Lego... trouve un titre et fais la miniature.",
                                        key="user_input")

            generate_btn = st.button("✨ Envoyer / Générer", type="primary")

            if generate_btn and prompt_input:
                st.session_state.chat_history.append({"role": "user", "content": prompt_input})

                with st.spinner("Le directeur artistique réfléchit..."):
                    try:
                        assistant = st.session_state.assistant
                        response = assistant.send_message(prompt_input)

                        new_images_dict = response.get("images", {})

                        if new_images_dict:
                            st.session_state.gallery.update(new_images_dict)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response["message"],
                            "images": new_images_dict
                        })

                        st.rerun()

                    except Exception as e:
                        st.error(f"Une erreur est survenue : {e}")

            # Bouton reset
            if st.session_state.chat_history:
                st.divider()
                if st.button("Effacer l'historique"):
                    st.session_state.chat_history = []
                    st.session_state.gallery = {}  # On vide aussi la galerie ? À toi de voir
                    st.rerun()

# --- Onglet architecture IA ---
with tech_tab:
    st.header("Le cerveau de l'IA")

    st.markdown("""
    Cette IA repose sur une approche **en deux temps** :  
    1. **Le Planificateur (Phase 1)** : Un réseau de neurones (LSTM) qui imagine le calendrier et le contenu.
    2. **L'Analyste (Phase 2)** : Un algorithme de Forêts Aléatoires qui estime le succès de ce contenu.
    
    Cette séparation permet d'avoir d'un côté la créativité (imaginer des vidéos) et de l'autre le réalisme (prédire les vues).
    """)
    
    st.divider()
    
    st.header("Phase 1 : Le planificateur de contenu")

    st.divider()

    st.subheader("1. La gravité du dimanche")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.graphviz_chart("""
            digraph {
                rankdir=TB;
                node [shape=box, style=filled, color="#444", fillcolor="#f7f7f9", fontname="sans-serif"];
                IA [label="Prédiction brute\n(Jour estimé)", shape=ellipse, fillcolor="#ffe5b4"];
                Adjust [label="Correction\npost-processing", shape=diamond, fillcolor="#d6eaff"];

                Sunday [label="→ Décalage vers dimanche\n(Grosse vidéo)", fillcolor="#c8ffcf"];
                Week [label="→ Maintien en semaine\nBonus/Standard", fillcolor="#fff7c2"];

                IA -> Adjust;
                Adjust -> Sunday [label="Si proche dimanche"];
                Adjust -> Week [label="Sinon"];
            }
        """)

    with col2:
        st.markdown("""
        L’IA prédit d’abord **une date brute** sans connaître les habitudes réelles du créateur.

        Une fois cette date obtenue, une étape de **post-processing** intervient pour intégrer la logique
        observée dans le comportement d’Amixem.

        ### La règle appliquée
        - Si la prédiction est **proche d’un dimanche**, la sortie est **ajustée** pour tomber exactement ce jour-là.
        - Si la date est clairement en semaine, elle est simplement confirmée.

        ### Pourquoi cette correction ?
        Le dimanche concentre généralement les **grosses vidéos** : plus longues, plus ambitieuses, plus travaillées.  
        Le post-processing agit donc comme une **force d’attraction contrôlée**, qui réaligne la prédiction brute
        sur un schéma de publication crédible.
        """)

    st.divider()

    st.subheader("2. Cohérence sémantique des tags")

    col3, col4 = st.columns(2)

    with col3:
        st.info("Problème des prédictions brutes")
        st.caption("Une IA peut mélanger des tags incompatibles.")
        st.code("Tags proposés : ['Voyage', 'Jeu', 'Horreur', 'Exploration', 'Vlog']")

    with col4:
        st.success("Solution : cohérence sémantique")
        st.caption("On ne garde que les tags qui apparaissent naturellement ensemble.")
        st.code("Tags retenus : ['Voyage', 'Exploration', 'Vlog']")

    st.markdown("""
    L’IA s’appuie sur un réseau d'affinités entre les tags (matrice de co-occurrence) :   
    certains apparaissent souvent ensemble, d'autres jamais.
    """)

    st.divider()

    st.subheader("3. Priorités d'apprentissage")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric(label="Jour de publication", value="Priorité Max")
        st.progress(1.00)
        
    with colB:
        st.metric(label="Durée / Format", value="Priorité Haute")
        st.progress(0.80)

    with colC:
        st.metric(label="Tags & catégories", value="Priorité Moyenne")
        st.progress(0.30)
        
    st.divider()

    st.header("Phase 2 : Analyse de performance")
    
    st.markdown("""
    Une fois la vidéo imaginée (titre, date, durée), nous passons le relais à une seconde IA spécialisée.
    Son but n'est pas de créer, mais de **juger**.
    """)

    col_p2_1, col_p2_2 = st.columns([1, 1])
    
    with col_p2_1:
        st.markdown("#### Le conseil des experts (Random Forest)")
        st.write("""
        Pour prédire le nombre de vues, nous utilisons un **algorithme de Forêts Aléatoires** (Random Forest).
        
        On utilise **200 arbres** :
        - L'arbre A regarde uniquement la durée de la vidéo.
        - L'arbre B regarde si c'est les vacances scolaires.
        - L'arbre C analyse les mots-clés ("Réaction" vs "Voyage").
        
        À la fin, l'IA fait la **moyenne** de ces 200 avis pour donner une estimation robuste, qui évite les erreurs grossières.
        """)
        
        st.info("""
        Contrairement à une régression linéaire simple, ce modèle comprend les règles non-linéaires 
        (ex: une vidéo très longue marche bien le dimanche, mais mal le mardi).
        """)

    with col_p2_2:
        st.graphviz_chart("""
            digraph {
                rankdir=TD;
                node [shape=box, style=filled, fillcolor="#fff", fontname="sans-serif"];
                
                Input [label="Entrée Phase 1\n(Date, Durée, Tags)", shape=note, fillcolor="#e1f5fe"];
                
                subgraph cluster_forest {
                    label = "Random Forest (200 Arbres)";
                    style=dashed;
                    color="#aaa";
                    bgcolor="#f9f9f9";
                    
                    Tree1 [label="Arbre 1\n(Analyse Durée)", fontsize=10];
                    Tree2 [label="Arbre 2\n(Analyse Saison)", fontsize=10];
                    Tree3 [label="Arbre 3\n(Analyse Mots)", fontsize=10];
                    TreeN [label="...", shape=plaintext];
                }
                
                Avg [label="Moyenne\ndes prédictions", shape=diamond, fillcolor="#d6eaff"];
                Output [label="Sortie Finale\n(Vues, Likes, Commentaires)", shape=ellipse, fillcolor="#c8ffcf", style="filled,bold"];

                Input -> Tree1;
                Input -> Tree2;
                Input -> Tree3;
                
                Tree1 -> Avg;
                Tree2 -> Avg;
                Tree3 -> Avg;
                
                Avg -> Output;
            }
        """)

    st.markdown("#### Les variables clés pour l'IA")
    
    col_var1, col_var2, col_var3 = st.columns(3)
    
    with col_var1:
        st.markdown("**1. La Temporalité**")
        st.caption("Mois, jour de la semaine, vacances...")
        st.progress(0.9)
        st.markdown("*L'IA sait que Décembre est un mois fort.*")

    with col_var2:
        st.markdown("**2. Le Contenu (Tags)**")
        st.caption("Analyse TF-IDF (Poids des mots)")
        st.progress(0.7)
        st.markdown("*L'IA sait que 'Concept' performe mieux que 'Vlog'.*")
        
    with col_var3:
        st.markdown("**3. Le Format**")
        st.caption("Durée (Courte vs Longue)")
        st.progress(0.6)
        st.markdown("*L'IA pénalise les formats courts le dimanche.*")
