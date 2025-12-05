import streamlit as st
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Mon Application",
    page_icon="🎬",
    layout="wide"
)

# Titre principal
st.title("Bienvenue sur mon Application Streamlit")

# Sidebar
st.sidebar.header("Navigation")
st.sidebar.info("Application créée avec Streamlit")

# Système d'onglets
home, data, llm = st.tabs(["🏠 Objectif", "📊 Données", "🤖 LLM"])

with home:
    # Centrer le contenu avec des colonnes
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.header("Page d'accueil")
        st.write("Ceci est une application Streamlit avec des onglets.")

        st.image("thumbnail.png", caption="Image de présentation", width=400)

        # Exemple d'input utilisateur
        nom = st.text_input("Entrez votre nom")
        if nom:
            st.success(f"Bonjour {nom}!")

        # Exemple de bouton
        if st.button("Cliquez-moi"):
            st.balloons()
            st.write("Merci d'avoir cliqué!")

with data:
    # Centrer le contenu avec des colonnes
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.header("Page des données")
        st.write("Voici la présentation des données. Comment elles ont été récupérées et traitées.")

        # Exemple de dataframe
        df = pd.DataFrame({
            'Colonne A': [1, 2, 3, 4],
            'Colonne B': [10, 20, 30, 40]
        })
        st.dataframe(df)

        # Téléchargement CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger CSV", csv, "data.csv", "text/csv")

with llm:
    # Centrer le contenu avec des colonnes
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.header("Page LLM")
        st.write("Ici vous pouvez intégrer votre modèle de langage.")

        # Exemple de graphique
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['A', 'B', 'C']
        )
        st.line_chart(chart_data)
