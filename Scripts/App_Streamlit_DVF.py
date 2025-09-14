import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib

# === Nettoyer le cache de Streamlit ===
st.cache_data.clear()

# === Fonction pour obtenir le chemin absolu des fichiers ===
def resource_path(*paths):
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, *paths)

# === Configuration de la page ===
st.set_page_config(
    page_title="Estimation Prix Immobilier",
    page_icon=resource_path("images", "icone.png"),
)

# === Titre de l'application ===
st.markdown("""
    <h3 style='text-align: center;'>
        🏡 Estimation du Prix d'un Bien Immobilier
    </h3>
""", unsafe_allow_html=True)

# === Image d'en-tête ===
image_path = resource_path("images", "immo.jpg")
if os.path.exists(image_path):
    image = Image.open(image_path)
    image_resized = image.resize((700, 300))
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(image_resized)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("Image 'immo.jpg' introuvable.")

# === Chemin vers le modèle ===
MODELE_PATH = resource_path("models", "model_compress.pkl")

# === Chargement du modèle ===
def load_model():
    if not os.path.exists(MODELE_PATH):
        st.error(f"Le fichier du modèle est introuvable : {MODELE_PATH}")
        return None
    try:
        return joblib.load(MODELE_PATH)
    except Exception as e:
        st.error("Erreur lors du chargement du modèle.")
        st.text(str(e))
        return None

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

    # === Encodage one-hot ===
    Nature_dict = {
        "Adjudication": 0, "Echange": 0, "Expropriation": 0,
        "Vente": 0, "VEFA": 0, "Terrain à bâtir": 0
    }
    Nature_dict[Nature_mutation] = 1

    Type_dict = {"Appartement": 0, "Maison": 0}
    Type_dict[Type_local] = 1

    # === Création du DataFrame utilisateur ===
    donnees_utilisateur = pd.DataFrame([[
        Surface_terrain, Surface_reelle_bati, Nombre_pieces_principales,
        annee_mutation, mois_mutation, code_postal_encoded,
        Nature_dict["Adjudication"], Nature_dict["Echange"], Nature_dict["Expropriation"],
        Nature_dict["Vente"], Nature_dict["VEFA"], Nature_dict["Terrain à bâtir"],
        Type_dict["Appartement"], Type_dict["Maison"]
    ]], columns=[
        'Surface_terrain', 'Surface_reelle_bati', 'Nombre_pieces_principales',
        'annee_mutation', 'mois_mutation', 'code_postal_encoded',
        'Nature_mutation_Adjudication', 'Nature_mutation_Echange', 'Nature_mutation_Expropriation',
        'Nature_mutation_Vente', 'Nature_mutation_Vente_etat_futur_achevement',
        'Nature_mutation_Vente_terrain_a_batir',
        'Type_local_Appartement', 'Type_local_Maison'
    ]).astype(float)

    st.write("Données utilisées pour la prédiction :")
    st.dataframe(donnees_utilisateur)

    # === Prédiction ===
    try:
        prediction = model.predict(donnees_utilisateur)[0]
        st.info(f"Estimation du prix total : **{prediction * Surface_reelle_bati:.2f} €**")
    except Exception as e:
        st.error("Erreur lors de la prédiction.")
        st.text(str(e))
else:
    st.stop()
