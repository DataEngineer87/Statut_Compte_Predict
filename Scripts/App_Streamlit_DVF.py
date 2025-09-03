import os
import streamlit as st
st.cache_data.clear()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from PIL import Image
import joblib

# === Fonction pour obtenir le chemin absolu des fichiers ===
def resource_path(*paths):
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, *paths)

# === Configuration de la page ===
st.set_page_config(
    page_title="Estimation Prix Immobilier",
    page_icon=resource_path("images", "icone.png"),
)

# Titre
st.markdown("""
    <h3 style='text-align: center;'>
        🏡 Estimation du Prix d'un Bien Immobilier
    </h3>
""", unsafe_allow_html=True)

# Image
image_path = resource_path("images", "immo.jpg")
image = Image.open(image_path)
image_resized = image.resize((700, 300))
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image(image_resized)
st.markdown("</div>", unsafe_allow_html=True)

# === Téléchargement du modèle depuis Google Drive si nécessaire ===
def download_model_from_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        st.warning("Téléchargement du modèle depuis Google Drive...")
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)
        st.success("Modèle téléchargé avec succès.")

# === Chargement du modèle ===
def load_model():
    modele_path = resource_path("model_DVF_compress.pkl")
    drive_file_id = "VOTRE_FILE_ID_ICI"  # Remplacez par l'ID réel
    download_model_from_drive(drive_file_id, modele_path)

    if not os.path.exists(modele_path):
        st.error("Le fichier 'model_DVF_compress.pkl' est introuvable.")
        return None

    return joblib.load(modele_path)

model = load_model()

# === Interface utilisateur pour la prédiction ===
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

    # Encodage one-hot avec noms exacts du modèle
    Nature_mutation_Adjudication = float(Nature_mutation == "Adjudication")
    Nature_mutation_Echange = float(Nature_mutation == "Echange")
    Nature_mutation_Expropriation = float(Nature_mutation == "Expropriation")
    Nature_mutation_Vente = float(Nature_mutation == "Vente")
    Nature_mutation_Vente_etat_futur_achevement = float(Nature_mutation == "VEFA")
    Nature_mutation_Vente_terrain_a_batir = float(Nature_mutation == "Terrain à bâtir")
    Type_local_Appartement = float(Type_local == "Appartement")
    Type_local_Maison = float(Type_local == "Maison")

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

    # Forcer les colonnes en float pour SHAP
    donnees_utilisateur = donnees_utilisateur.astype(float)

    st.write("Données utilisées pour la prédiction :")
    st.dataframe(donnees_utilisateur)

    try:
        prediction = model.predict(donnees_utilisateur)[0]
        st.info(f"Estimation du prix total : **{prediction * Surface_reelle_bati:.2f} €**")
    except Exception as e:
        st.error("Erreur lors de la prédiction.")
        st.text(str(e))

else:
    st.stop()
