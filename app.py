import streamlit as st
import pandas as pd
import os
from generativeAI.gemini_tools import IA
from generativeAI.assistant import Assistant

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

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# Initialisation de l'historique du chat pour l'affichage
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 4. INTERFACE UTILISATEUR (Structure demandée) ---

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


# --- GESTION DE L'ÉTAT (SESSION STATE) ---
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# Titre principal
st.title("Bienvenue sur mon Application Streamlit")

# --- MODIFICATION ICI : AJOUT DU 4ème ONGLET ---
home, data, llm, tech_tab = st.tabs(["🏠 Objectif", "📊 Données", "🤖 LLM", "⚙️ Architecture IA"])

# --- Onglet Accueil ---
with home:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.header("Page d'accueil")
        st.write("Ceci est une application Streamlit avec des onglets.")

        try:
            st.image("thumbnail.png", caption="Image de présentation", width=400)
        except:
            st.info("Ajoutez une image 'thumbnail.png' à la racine pour la voir ici.")

        nom = st.text_input("Entrez votre nom")
        if nom:
            st.success(f"Bonjour {nom}!")

        if st.button("Cliquez-moi"):
            st.balloons()
            st.write("Merci d'avoir cliqué!")

# --- Onglet Données ---
with data:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.header("Page des données")
        st.write("Voici la présentation des données.")

        df = pd.DataFrame({
            'Colonne A': [1, 2, 3, 4],
            'Colonne B': [10, 20, 30, 40]
        })
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger CSV", csv, "data.csv", "text/csv")

# --- Onglet LLM ---
with llm:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.header("Studio Créatif Amixem 🎬")
        st.write("Décrivez un concept de vidéo, l'IA s'occupe du reste.")

        if not api_key:
            st.warning("⚠️ Clé API introuvable.")
        else:
            # --- A. Affichage de l'historique ---
            # On affiche les anciens messages ici, avant la zone de saisie
            if st.session_state.chat_history:
                with st.container(height=400, border=True):
                    for msg in st.session_state.chat_history:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

            # --- B. Zone de saisie ---
            prompt_input = st.text_area("Votre message", height=100,
                                        placeholder="Ex: On passe 24h dans un bunker en Lego... trouve un titre et fais la miniature.",
                                        key="user_input")

            generate_btn = st.button("✨ Envoyer / Générer", type="primary")

            # 1. Logique de génération
            if generate_btn and prompt_input:
                # Ajout immédiat du message utilisateur à l'historique
                st.session_state.chat_history.append({"role": "user", "content": prompt_input})

                with st.spinner("Le Directeur Artistique réfléchit..."):
                    try:
                        # Appel via la classe Assistant stockée en session
                        assistant = st.session_state.assistant
                        response_text = assistant.send_message(prompt_input)

                        # Ajout de la réponse à l'historique
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

                        # Force le rechargement pour afficher les nouveaux messages dans l'historique ci-dessus
                        st.rerun()

                    except Exception as e:
                        st.error(f"Une erreur est survenue : {e}")

            # 2. Logique d'affichage des images (toujours en bas)
            if st.session_state.generated_images:
                st.divider()
                st.success("Image disponible !")

                # Afficher la dernière image générée
                last_img = st.session_state.generated_images[-1]
                st.image(last_img, caption="Miniature générée", use_container_width=True)

                # Afficher l'historique si plus d'une image
                if len(st.session_state.generated_images) > 1:
                    with st.expander("Voir l'historique des images"):
                        cols = st.columns(3)
                        for i, img in enumerate(st.session_state.generated_images[:-1]):
                            with cols[i % 3]:
                                st.image(img, use_container_width=True)

                if st.button("Effacer les images"):
                    st.session_state.generated_images = []
                    st.rerun()

# Onglet Architecture IA
with tech_tab:
    st.header("Le cerveau hybride de l'IA")

    st.markdown("""
    Cette IA repose sur une approche **hybride** :  
    une base de prédiction issue d’un réseau de neurones, combinée à des 
    **règles inspirées du comportement réel** d’Amixem.  
    Cette fusion permet d’obtenir des résultats réalistes, structurés et cohérents
    """)

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

        ### Effets naturels
        - **Dimanche = formats longs**  
        Les vidéos majeures ont plus de chances d'être programmées ce jour-là.

        - **Semaine = formats bonus**  
        Les contenus plus courts ou plus spontanés restent en semaine.

        Ce mécanisme garantit une dynamique temporelle fidèle à ce que l’on observe réellement.
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

    ### Comment fonctionne cette cohérence ?
    - Le tag principal (le plus pertinent) sert de **pivot**  
    - On lui associe ensuite des tags **compatibles**, basés sur l’historique réel
    - Les associations incongrues sont **éliminées** naturellement
    - Certains tags ne sont retenus que s’ils correspondent au format (court / long)
    """)

    st.divider()

    st.subheader("3. Ce que l'IA apprend en premier (priorités)")

    st.markdown("""
    Toutes les informations n'ont pas la même importance, on change donc leurs poids dans le modèle.  
    L'IA apprend à prioriser certains aspects qui ont plus d'impact que d'autres.
    """)

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric(label="Jour de publication", value="Priorité maximale")
        st.progress(1.00)
        st.caption("Les habitudes de publication sont cruciales pour un planning réaliste. Ces données sont utilisées dans les autres étapes de prédiction/de génération, il est important qu'elles soient réalistes.")

    with colB:
        st.metric(label="Durée / Format", value="Priorité élevée")
        st.progress(0.80)
        st.caption("La durée influence également le reste de la prédiction/génération. On met donc un poids important sur cette donnée car les formats courts et longs ont des caractéristiques différentes.")

    with colC:
        st.metric(label="Tags & catégories", value="Priorité flexible")
        st.progress(0.30)
        st.caption("Les tags sont principalement gérés par notre post-processing. La catégorie ne changent presque jamais dans notre dataset, elle a donc un poids faible dans la prédiction initiale.")

    st.divider()

    st.markdown("""
    Notre **phase 1** de l'IA combine un LTSM pour la **prédiction initiale** et un système de **règles** pour le post-processing.  
    Cela nous permet d'obtenir des résultats **cohérents** pour permettre un meilleur prédiction en phase 2 et une meilleure génération finale.
    """)

