import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib

# === Configuration de la page ===
st.set_page_config(
    page_title="Estimation Prix Immobilier",
    page_icon="🏠"
)

# === Clear cache si besoin ===
st.cache_data.clear()

# === Titre ===
st.markdown("""
    <h3 style='text-align: center;'>
        🏡 Estimation du Prix d'un Bien Immobilier
    </h3>
""", unsafe_allow_html=True)

# === Image ===
image_path = os.path.join("images", "immo.jpg")
if os.path.exists(image_path):
    image = Image.open(image_path)
    image_resized = image.resize((700, 300))
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(image_resized)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("Image introuvable.")

# === Chargement du modèle depuis le repo ===
def load_model():
    modele_path = os.path.join("models", "model_DVF_compress.pkl")  # chemin GitHub
    if not os.path.exists(modele_path):
        st.error(f"Le fichier modèle est introuvable à {modele_path}")
        st.stop()
    with st.spinner("Chargement du modèle, cela peut prendre quelques secondes..."):
        model = joblib.load(modele_path)
    st.success("Modèle chargé ! ✅")
    return model

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

    # Encodage one-hot
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

    donnees_utilisateur = donnees_utilisateur.astype(float)

    st.write("Données utilisées pour la prédiction :")
    st.dataframe(donnees_utilisateur)

    # Prédiction
    try:
        prediction = model.predict(donnees_utilisateur)[0]
        st.info(f"Estimation du prix total : **{prediction * Surface_reelle_bati:.2f} €**")
    except Exception as e:
        st.error("Erreur lors de la prédiction.")
        st.text(str(e))
