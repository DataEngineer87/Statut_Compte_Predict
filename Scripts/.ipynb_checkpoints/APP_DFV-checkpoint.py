import os
import streamlit as st
st.cache_data.clear()
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from PIL import Image
import subprocess
import sys

# Fonction pour installer une version précise de scikit-learn
def install_sklearn_version(version="1.2.2"):
    try:
        import sklearn
        from packaging import version as pkg_version
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "packaging"])
        import sklearn
        from packaging import version as pkg_version

    if pkg_version.parse(sklearn.__version__) != pkg_version.parse(version):
        st.warning(f"Version de scikit-learn détectée : {sklearn.__version__}. Installation de la version {version}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"scikit-learn=={version}"])
        import importlib
        importlib.reload(sklearn)
        st.success(f"scikit-learn version {version} installée.")

# Configuration de la page
st.set_page_config(
    page_title="Failure Classifier",
    page_icon="images/icone.png",
)

# Titre
st.markdown("""
    <h3 style='text-align: center;'>
        🏡 Estimation du Prix d'un Bien Immobilier
    </h3>
""", unsafe_allow_html=True)
st.write("Fichiers dans le dossier images :", os.listdir("images"))

# Image
image = Image.open('images/immo.jpg')
image_resized = image.resize((700, 300))
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image(image_resized)
st.markdown("</div>", unsafe_allow_html=True)

# === Téléchargement du modèle depuis Google Drive ===

def download_model_from_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        st.warning("Téléchargement du modèle depuis Google Drive...")
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, dest_path, quiet=False)
            st.success("Modèle téléchargé avec succès.")
        except Exception as e:
            st.error("Échec du téléchargement du modèle.")
            st.text(str(e))

# === Modèle de prédiction ===

def load_model():
    modele_path = "Model_DVF.pkl"
    drive_file_id = "1Z79gZJ5R2NzWBHDiZLTxDfOsamm0nkkF"
    download_model_from_drive(drive_file_id, modele_path)

    if not os.path.exists(modele_path):
        st.error("Le fichier 'Model_DVF.pkl' est introuvable.")
        return None

    # Installer la version compatible de scikit-learn (ajuste si besoin)
    install_sklearn_version("1.2.2")

    with open(modele_path, "rb") as file:
        return pickle.load(file)

model = load_model()

if model is not None:
    st.subheader("Entrer les données du bien")
    code_postal_encoded = st.number_input("Code postal (encodé)", value=75000)
    Surface_terrain = st.number_input("Surface du terrain (m²)", 0.0, 5000.0, 100.0)
    Surface_reelle_bati = st.number_input("Surface réelle bâtie (m²)", 10.0, 1000.0, 50.0)
    Nombre_pieces_principales = st.number_input("Nombre de pièces principales", 1, 20, 3)
    annee_mutation = st.slider("Année de la mutation", 2000, 2025, 2022)
    mois_mutation = st.slider("Mois de la mutation", 1, 12, 6)
    Nature_mutation = st.selectbox("Nature de la mutation", [
        "Adjudication", "Echange", "Expropriation", "Vente", "VEFA", "Terrain à bâtir"
    ])
    Type_local = st.selectbox("Type de bien", ["Appartement", "Maison"])

    # Encodage one-hot
    Nature_mutation_Adjudication = int(Nature_mutation == "Adjudication")
    Nature_mutation_Echange = int(Nature_mutation == "Echange")
    Nature_mutation_Expropriation = int(Nature_mutation == "Expropriation")
    Nature_mutation_Vente = int(Nature_mutation == "Vente")
    Nature_mutation_Vente_etat_futur_achevement = int(Nature_mutation == "VEFA")
    Nature_mutation_Vente_terrain_a_batir = int(Nature_mutation == "Terrain à bâtir")
    Type_local_Appartement = int(Type_local == "Appartement")
    Type_local_Maison = int(Type_local == "Maison")

    # Données utilisateur
    donnees_utilisateur = pd.DataFrame([[
        Surface_terrain, Surface_reelle_bati, Nombre_pieces_principales,
        annee_mutation, mois_mutation, code_postal_encoded,
        Nature_mutation_Adjudication, Nature_mutation_Echange, Nature_mutation_Expropriation,
        Nature_mutation_Vente, Nature_mutation_Vente_etat_futur_achevement,
        Nature_mutation_Vente_terrain_a_batir, Type_local_Appartement, Type_local_Maison
    ]], columns=[
        'Surface_terrain', 'Surface_reelle_bati', 'Nombre_pieces_principales',
        'annee_mutation', 'mois_mutation', 'code_postal_encoded',
        'Nature_mutation_Adjudication', 'Nature_mutation_Echange', 'Nature_mutation_Expropriation',
        'Nature_mutation_Vente', 'Nature_mutation_Vente_etat_futur_achevement',
        'Nature_mutation_Vente_terrain_a_batir',
        'Type_local_Appartement', 'Type_local_Maison'
    ])

    st.write("Données utilisées pour la prédiction :")
    st.dataframe(donnees_utilisateur)

    try:
        prediction = model.predict(donnees_utilisateur)[0]
        st.info(f"Estimation du prix total : **{prediction * Surface_reelle_bati:.2f} €**")
    except Exception as e:
        st.error("Erreur lors de la prédiction.")
        st.text(str(e))

    # === SHAP ===
    st.subheader("Explication de la prédiction avec SHAP")
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(donnees_utilisateur)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=15)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Impossible d'afficher l'explication SHAP.")
        st.text(str(e))

else:
    st.stop()
