import streamlit as st
import pandas as pd
import os
from generativeAI.gemini_tools import IA

# Configuration de la page
st.set_page_config(
    page_title="Mon Application",
    page_icon="🎬",
    layout="wide"
)

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

# Système d'onglets
home, data, llm = st.tabs(["🏠 Objectif", "📊 Données", "🤖 LLM"])

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
        st.header("Générateur d'images Gemini")
        st.write("Décrivez une image et l'IA la créera pour vous.")

        api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))

        if not api_key:
            st.warning("⚠️ Clé API introuvable.")
        else:
            prompt_input = st.text_area("Votre description (Prompt)", height=100,
                                        placeholder="Ex: Un chat cosmonaute...")
            generate_btn = st.button("✨ Générer l'image", type="primary")

            # 1. Logique de génération
            if generate_btn and prompt_input:
                with st.spinner("Gemini est en train de peindre..."):
                    try:
                        mon_ia = IA(gemini_api_key=api_key)
                        images = mon_ia.generate_image(prompt_input)

                        if images:
                            st.session_state.generated_images = images
                        else:
                            st.error("Aucune image n'a été retournée.")
                    except Exception as e:
                        st.error(f"Une erreur est survenue : {e}")

            # 2. Logique d'affichage
            if st.session_state.generated_images:
                st.success("Image disponible !")
                for img in st.session_state.generated_images:
                    # CORRECTION: use_container_width remplacé par width="stretch"
                    st.image(img, caption="Image générée", width="stretch")

                if st.button("Effacer l'image"):
                    st.session_state.generated_images = []
                    st.rerun()


