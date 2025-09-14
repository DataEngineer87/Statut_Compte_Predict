#!/usr/bin/env python
# coding: utf-8

# # Alseny
# ### Data Scient & Machine Learning Engineer

# # A- Packages nécessaires pour le projet

# In[1]:


from joblib import Parallel, delayed
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import sklearn
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import calendar
from sklearn.model_selection import train_test_split
import geopandas as gpd
from sklearn.preprocessing import KBinsDiscretizer  # Pour la discrétisation optimale
import matplotlib.colors as mcolors
import mapclassify as mc  # Nécessaire pour Jenks Natural Breaks
from scipy.stats import iqr
import nbformat
from nbconvert import PythonExporter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


# # B- Les fonctions utiles pour le projet

# ### B.1- Chargement dess données et exploration 

# In[2]:


def load_and_save_data(file_path, sep=None, export=False, export_path=None,
                       export_format=None, sample=None, random_state=None,
                       chunksize=None, index_col=None):

    ext = os.path.splitext(file_path)[1].lower()
    chunk_supported = ext in ['.csv', '.txt']

    def detect_separator(path):
        for s in [',', ';', '\t', '|']:
            try:
                if len(pd.read_csv(path, sep=s, nrows=5).columns) > 1:
                    return s
            except pd.errors.ParserError:
                continue
        raise ValueError("Séparateur non détecté. Spécifiez 'sep'.")

    def process_chunk(chunk):
        return chunk  # Personnalisez si besoin

    if chunksize and not chunk_supported:
        raise ValueError(f"Le format {ext} ne supporte pas 'chunksize'.")

    # ====================== Lecture par morceaux ======================
    if chunksize:
        sep = sep or detect_separator(file_path)
        reader = pd.read_csv(file_path, sep=sep, chunksize=chunksize)

        if export:
            if not export_path:
                raise ValueError("Spécifiez 'export_path' pour l'export.")
            export_ext = '.' + export_format.lower() if export_format else os.path.splitext(export_path)[1].lower()
            if export_ext not in ['.csv', '.txt']:
                raise ValueError(f"Export par chunks non supporté pour {export_ext}")
            with open(export_path, 'w', encoding='utf-8', newline='') as f:
                for i, chunk in enumerate(reader):
                    chunk = process_chunk(chunk)
                    if sample:
                        chunk = chunk.sample(n=sample if isinstance(sample, int)
                                             else None,
                                             frac=sample if isinstance(sample, float)
                                             else None,
                                             random_state=random_state)
                    chunk.to_csv(f, sep=sep, index=False, header=(i == 0), mode='a')
        else:
            for chunk in reader:
                process_chunk(chunk)  # À adapter
        return None

    # ====================== Lecture complète ======================
    if sep is None and ext in ['.csv', '.txt']:
        sep = detect_separator(file_path)

    read_funcs = {
        '.csv': lambda: pd.read_csv(file_path, sep=sep, index_col=index_col),
        '.txt': lambda: pd.read_csv(file_path, sep=sep, index_col=index_col),
        '.json': lambda: pd.read_json(file_path),
        '.xlsx': lambda: pd.read_excel(file_path),
        '.pickle': lambda: pd.read_pickle(file_path),
        '.parquet': lambda: pd.read_parquet(file_path)
    }

    if ext not in read_funcs:
        raise ValueError(f"Format non pris en charge : {ext}")

    df = read_funcs[ext]()

    if sample:
        df = df.sample(n=sample if isinstance(sample, int)
                       else None,
                       frac=sample if isinstance(sample, float)
                       else None,
                       random_state=random_state)

    # ====================== Export ======================
    if export:
        export_ext = '.' + export_format.lower() if export_format else os.path.splitext(export_path)[1].lower()
        if export_ext not in ['.csv', '.txt', '.json', '.xlsx', '.pickle', '.parquet']:
            raise ValueError(f"Export non supporté pour : {export_ext}")
        if export_ext in ['.csv', '.txt']:
            df.to_csv(export_path, sep=sep or ',', index=False)
        elif export_ext == '.json':
            df.to_json(export_path, orient='records', lines=True)
        elif export_ext == '.xlsx':
            df.to_excel(export_path, index=False)
        elif export_ext == '.pickle':
            df.to_pickle(export_path)
        elif export_ext == '.parquet':
            df.to_parquet(export_path, index=False)

    return df


# In[3]:


#Fonction d'identification de la nature des variables

def compter_variables(dataset):
    """Affiche et compte les variables quantitatives et qualitatives d'un dataset.

    Parameters:
    dataset (pandas.DataFrame): Le DataFrame contenant les variables à analyser.

    """
    # Sélection des colonnes quantitatives (numériques) et qualitatives (non numériques)
    quantitatives = dataset.select_dtypes(include=['number']).columns
    qualitatives = dataset.select_dtypes(exclude=['number']).columns

    # Affichage des colonnes quantitatives et de leur nombre
    print("Les variables quantitatives:\n", quantitatives)
    print(f"Nombre de variables quantitatives : {len(quantitatives)}\n")

    # Affichage des colonnes qualitatives et de leur nombre
    print("Les variables qualitatives:\n", qualitatives)
    print(f"Nombre de variables qualitatives : {len(qualitatives)}")


# In[4]:


# Identification des lignes dupliquées dans le DataFrame

def duplicated_values_summary(df):
    # Calcul de la somme totale des lignes dupliquées
    total_duplicated_rows = df.duplicated().sum()

    # Identification des colonnes contenant des valeurs dupliquées
    duplicated_columns = [col for col in df.columns if df[col].duplicated().any()]

    # Affichage des résultats
    print("Total de lignes dupliquées dans le DataFrame :", total_duplicated_rows)
    print("Colonnes contenant des valeurs dupliquées :", duplicated_columns)

    return {"total_duplicated_rows": total_duplicated_rows, "duplicated_columns": duplicated_columns}


# In[5]:


# Comptage du taux de données manquantes dans le DataFrame

def pie_nan(dataframe):
    """Affiche un graphique en camembert montrant le taux de données manquantes dans le DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): Le DataFrame à analyser.
    """
    # Calcul du nombre de lignes et de colonnes
    lignes = dataframe.shape[0]
    colonnes = dataframe.shape[1]

    # Nombre de données non manquantes
    nb_data = dataframe.count().sum()

    # Nombre total de données dans le jeu de données (colonnes * lignes)
    nb_total = colonnes * lignes

    # Taux de remplissage du jeu de données
    rate_data_ok = nb_data / nb_total
    print(f"Le jeu de données est rempli à {rate_data_ok:.2%}")
    print(f"Il a {1 - rate_data_ok:.2%} de données manquantes")
    print("\n")

    # Création du pie chart
    rates = [rate_data_ok, 1 - rate_data_ok]
    labels = ["Données", "NAN"]
    explode = (0, 0.1)
    colors = ['gold', 'pink']

    # Plot
    plt.figure(figsize=(8, 10))
    plt.pie(rates, explode=explode, labels=labels, colors=colors,
            autopct='%.2f%%', shadow=True, textprops={'fontsize': 26})

    # Titre du graphique
    ttl = plt.title("Taux des valeurs manquantes dans la base de données", fontsize=30)
    ttl.set_position([0.5, 0.50])

    # Assurer que le graphique est bien circulaire
    plt.axis('equal')
    plt.tight_layout()

    # Afficher le graphique
    plt.show()


# In[6]:


# Affichege du pourcentage de valeurs manquantes par colonne

def show_all_missing(data_set):
    """Calcule et retourne le pourcentage de valeurs manquantes par colonne dans un DataFrame.
    """
    return (data_set.isna().sum() / data_set.shape[0]).sort_values(ascending=True) * 100


# In[7]:


def format_commune_insee(dept_code, commune_code):
    """
    Construit le code INSEE à partir du code département et code commune.
    """
    # Gestion des départements alphanumériques (ex: '2A', '2B') ou numériques
    dept_str = str(dept_code).zfill(2) if str(dept_code).isnumeric() else str(dept_code)
    commune_str = str(commune_code).zfill(3)
    return dept_str + commune_str

# Création de nouvelle colonne 'commune_insee' à partir de dataset valeur foncière
def create_commune_insee(df, dept_col, commune_col):

    # Application de la fonction à chaque ligne du DataFrame
    df['commune_insee'] = df.apply(lambda row: format_commune_insee(row[dept_col], row[commune_col]), axis=1)

    # Affichage des premières lignes pour validation
    print("Exemple de colonne 'commune_insee' :")
    print(df[[dept_col, commune_col, 'commune_insee']].head())

    return df


# In[8]:


# Ajout de 0 devant les code postaux à 4 chiffres et suppression de .0 dans le dataset valeur foncière
def process_code_postal(df):

    # Convertir la colonne en chaîne de caractères
    df['Code postal'] = df['Code postal'].astype(str)

    # Supprimer le .0 des codes postaux
    df['Code postal'] = df['Code postal'].str.replace(r'\.0$', '', regex=True)

    # Ajouter un 0 devant les codes à 4 chiffres
    df['Code postal'] = df['Code postal'].apply(lambda x: x.zfill(5) if len(x) == 4 else x)

    return df


# In[9]:


# Ajout d’un 0 devant les codes postaux à 4 chiffres dans la variable code_postal du dataset region.
def process_code_postal_region(df):
    df['code_postal'] = df['code_postal'].astype(str).apply(lambda x: '0' + x if len(x) == 4 else x)
    return df


# In[10]:


# Conversion de type des variables
def transform_data_types(df, col_types):
    for col, dtype in col_types.items():
        if dtype == 'float64':
            # If the target type is float64, replace commas with dots before conversion
            # Check if the column contains strings before applying str.replace
            if df[col].dtype == 'object': # If column is of type object, potentially containing string values
                df[col] = pd.to_numeric(df[col].str.replace(',', '.', regex=False), errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Attempt direct numeric conversion if not an object
        else:
            df[col] = df[col].astype(dtype)
    return df


# In[11]:


# Création des colonnes annee_mutation, mois_mutation
def dateFiltered(df):
    df['Date mutation'] = pd.to_datetime(df['Date mutation'], format='%d/%m/%Y', errors='coerce') # Conversion de la colonne 'Date mutation' en type datetime 
    df["annee_mutation"] = df['Date mutation'].dt.year
    df["mois_mutation"] = df['Date mutation'].dt.month
    return df


# In[12]:


# Gestion des valeurs abérantes
def detect_outliers_iqr(df):
    """
    Supprime les valeurs aberrantes des colonnes numériques du DataSet
    en utilisant la méthode de l'écart interquartile (IQR).
    """
    # Identifier automatiquement les colonnes numériques (float ou int)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    # Supprimer les outliers pour chaque colonne numérique
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


# In[13]:


def handle_missing_values(df, method='median', fill_value=None):

    # Traitement spécifique pour la colonne 'Type de voie'
    if 'Type de voie' in df.columns:
        if not df['Type de voie'].dropna().empty:
            df['Type de voie'] = df['Type de voie'].fillna(df['Type de voie'].mode()[0])

    if method == 'drop':
        df = df.dropna()  # Supprime les lignes contenant des valeurs NaN

    elif method == 'fill':
        if fill_value is None:
            raise ValueError("Un 'fill_value' doit être spécifié pour la méthode 'fill'.")
        df = df.fillna(fill_value)  # Remplit toutes les valeurs NaN avec la valeur spécifiée

    elif method == 'median':
        # Remplir les colonnes numériques avec leur médiane
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].median())

        # Remplir les colonnes non numériques avec la valeur la plus fréquente (mode)
        for col in df.select_dtypes(exclude=['number']).columns:
            if col != 'Type de voie' and not df[col].dropna().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

    else:
        raise ValueError("Méthode non valide fournie. Utilisez 'drop', 'fill' ou 'median'.")

    return df


# In[14]:


# suppression des doublons dans le dataSet
def remove_duplicates(df, subset=None):

    return df.drop_duplicates(subset=subset)


# In[15]:


# Standardisation des variables numériques
def scale_numeric_features(df):

    # Variables numériques continues à scaler
    numeric_columns = ['Surface reelle bati', 'Nombre pieces principales',
                       'Surface terrain', 'prix_m2', 'annee_mutation', 
                       'mois_mutation']

    # Appliquer le StandardScaler uniquement à ces colonnes
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df


# In[16]:


# Nettoyage des variables
import re
import unidecode

def clean_column_names(columns):
    df = []
    for col in columns:
        # Enlever les accents
        col = unidecode.unidecode(col)
        # Remplacer les apostrophes et caractères spéciaux par rien ou un underscore
        col = re.sub(r"[^\w\s]", "", col)
        # Remplacer les espaces ou tirets par underscore
        col = re.sub(r"[\s\-]+", "_", col)
        df.append(col)
    return df


# In[17]:


# Affichage de heatmap de matrice de corrélation avec un seuil d'affichage des étiquettes de corrélation.
def plot_half_correlation_matrix(matrix_num, thres=0.1, size=(10, 8), cmap="coolwarm"):
    """
    Ce texte décrit une fonction qui affiche une heatmap représentant la moitié supérieure 
    d'une matrice de corrélation. Elle permet de visualiser uniquement les coefficients 
    de corrélation supérieurs à un certain seuil. Les paramètres incluent la matrice 
    de corrélation (matrix_num), un seuil d’affichage (thres), 
    la taille de la figure (size) et la palette de couleurs utilisée (cmap).
    """

    plt.figure(figsize=size)
    mask = np.triu(np.ones_like(matrix_num, dtype=bool))

    # Création de la heatmap avec le masque de la moitié supérieure
    ax = sns.heatmap(
        matrix_num,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matrice de Corrélation")

    # Ajuster les étiquettes annotées pour n'afficher que celles supérieures au seuil
    for t in ax.texts:
        try:
            # Conversion en float si possible
            value = float(t.get_text())
            # Afficher ou masquer en fonction du seuil
            if value >= thres:
                t.set_text(f"{value:.2f}")
            else:
                t.set_text("")
        except ValueError:
            # Si la conversion échoue, masquer l'étiquette
            t.set_text("")

    plt.show()


# In[18]:


# prompt: dans la base df_region, ajoute 0 devant tout nombre ayant 4 caractères dans la variable code_postal
def process_code_postal_region(df):
    df['code_postal'] = df['code_postal'].astype(str).apply(lambda x: '0' + x if len(x) == 4 else x)
    return df


# In[19]:


def preprocessing(df):
    # 0. Nettoyage des noms de colonnes dès le début pour homogénéiser les traitements
    df.columns = clean_column_names(df.columns)

    # 1. Suppression des doublons
    df = remove_duplicates(df).copy()

    # 2. Gestion des valeurs extrêmes pour toutes les colonnes
    df = detect_outliers_iqr(df).copy()

    # 3. Nettoyage spécifique de la variable cible 'Valeur_fonciere'
    if 'Valeur_fonciere' in df.columns:
        # a. Traitement des valeurs aberrantes (IQR)
        q1 = df['Valeur_fonciere'].quantile(0.25)
        q3 = df['Valeur_fonciere'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df['Valeur_fonciere'] >= lower_bound) & (df['Valeur_fonciere'] <= upper_bound)].copy()

        # b. Remplacement des valeurs manquantes par la médiane
        median_valeur = df['Valeur_fonciere'].median()
        df['Valeur_fonciere'] = df['Valeur_fonciere'].fillna(median_valeur)

    # 4. Gestion des valeurs manquantes (mode pour catégorielles, médiane pour numériques)
    df = handle_missing_values(df, method='median').copy()

    # 5. Remplacer les infinis par NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 6. Clipper les valeurs extrêmes pour éviter les instabilités numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e10, upper=1e10) 

    # 7. Conversion des booléens en numériques
    df = convert_bool_to_numeric(df)

    # 8. Séparation X / y
    X = df.drop(['Valeur_fonciere'], axis=1)
    y = df['Valeur_fonciere']

    return X, y


# In[20]:


# Représentation graphique de valeurs foncières par département
# Cette fonction trace une carte colorée des départements selon une variable géospatiale (par défaut, la valeur foncière), discrétisée en classes optimales.
def afficher_carte_departements(df_dep, departements, var_a_afficher="Valeur fonciere", nom_variable_commune="Code departement", cmap="RdYlGn_r", nb_classes=5):

    # 1. Fusionner les bases de données sur le code département
    base = pd.merge(departements, df_dep, left_on="code", right_on=nom_variable_commune)

    # 2. Convertir la base en GeoDataFrame si elle ne l'est pas déjà
    if not isinstance(base, gpd.GeoDataFrame):
        base = gpd.GeoDataFrame(base, geometry="geometry")  # Assurez-vous que la colonne 'geometry' existe

    # 3. Gérer les NaN dans la colonne à afficher
    base = base.dropna(subset=[var_a_afficher])

    # 4. Convertir la variable à afficher en millions d'euros
    base[var_a_afficher] = base[var_a_afficher] / 1000  # Conversion en millions d'euros

    # 5. Discrétiser la variable à afficher (par Quantiles ici)
    base['classe_discrete'] = pd.qcut(base[var_a_afficher], nb_classes, labels=False)

    # 6. Créer la carte
    fig, ax = plt.subplots(figsize=(10, 6))

    # 7. Afficher la carte avec un coloriage basé sur la variable discrétisée
    base.plot(column='classe_discrete',
              ax=ax,
              legend=False,  # Désactiver la légende des classes discrètes
              cmap=cmap)

    # 8. Définir les limites de la vue (si pertinent pour votre zone géographique)
    ax.set_xlim(-5, 10)  # Vous pouvez ajuster selon vos données géographiques
    ax.set_ylim(41, 51)

    # 9. Supprimer les axes pour une visualisation propre
    ax.set_axis_off()

    # 10. Ajouter une légende avec les valeurs réelles en millions d'euros
    min_val = base[var_a_afficher].min()
    max_val = base[var_a_afficher].max()

    # Créer une normalisation basée sur les valeurs réelles
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap_obj = plt.get_cmap(cmap)

    # Créer une barre de couleur basée sur les valeurs réelles
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array(base[var_a_afficher])

    # Ajouter la barre de couleur avec des montants en millions d'euros
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"{var_a_afficher} (en millions d'euros)")

    # 11. Afficher la carte
    plt.show()


# In[21]:


# Affiche une carte des communes colorée en fonction d'une variable discrétisée.
def afficher_carte_communes(df_commu, codes_postaux, var_a_afficher="Valeur fonciere",
                            nom_variable_commune="Code postal", cmap="RdYlGn_r",
                            nb_classes=5, methode_discretisation='quantile'):

    # Fusion des données
    base = codes_postaux.merge(df_commu, left_on="codePostal", right_on=nom_variable_commune, how="left").dropna(subset=[var_a_afficher])

    # Conversion en numérique et en millions d'euros
    base[var_a_afficher] = pd.to_numeric(base[var_a_afficher], errors='coerce') / 1_000_000

    # Choix de la méthode de discrétisation
    method_dict = {
        'quantile': mc.Quantiles,
        'uniform': mc.EqualInterval,
        'jenks': mc.NaturalBreaks
    }

    method_class = method_dict.get(methode_discretisation, mc.Quantiles)  # Par défaut : quantile

    try:
        classifier = method_class(base[var_a_afficher], k=nb_classes)
        base['classe_discrete'] = classifier.yb  # Indices des classes
        bins = classifier.bins  # Bornes des classes
    except Exception as e:
        print(f"Erreur de discrétisation : {e}")
        return

    # Création de la carte
    fig, ax = plt.subplots(figsize=(12, 8))
    base.plot(column='classe_discrete', cmap=cmap, linewidth=0.5, edgecolor='black', ax=ax, legend=True)

    # Suppression des axes pour un affichage propre
    ax.set_axis_off()
    plt.title(f"Carte des valeurs foncières ({var_a_afficher} en M€)", fontsize=12)

    # Affichage
    plt.show()


# In[24]:


from category_encoders import TargetEncoder
import numpy as np

def target_encoding(df_train_cleaned, df_test_cleaned):
    # --- Target Encoding (manuelle) ---
    for column in ['code_postal', 'Code_commune', 'Code_departement']:
        mean_value_per_category = df_train_cleaned.groupby(column)['Valeur_fonciere'].mean()
        global_mean = df_train_cleaned['Valeur_fonciere'].mean()

        df_train_cleaned[column + '_encoded'] = df_train_cleaned[column].map(mean_value_per_category)
        df_test_cleaned[column + '_encoded'] = df_test_cleaned[column].map(mean_value_per_category).fillna(global_mean)

        df_train_cleaned.drop(columns=[column], inplace=True)
        df_test_cleaned.drop(columns=[column], inplace=True)

    # --- Target Encoding (avec category_encoders) pour haute cardinalité ---
    high_card_cols = ['Voie', 'Type_de_voie']
    encoder = TargetEncoder(cols=high_card_cols)

    # Fit uniquement sur le trainFonction_utils
    encoder.fit(df_train_cleaned[high_card_cols], df_train_cleaned['Valeur_fonciere'])

    # Transform train et test
    df_train_encoded = encoder.transform(df_train_cleaned[high_card_cols])
    df_test_encoded = encoder.transform(df_test_cleaned[high_card_cols])

    # Renommer les colonnes encodées
    df_train_encoded.columns = [col + '_te' for col in high_card_cols]
    df_test_encoded.columns = [col + '_te' for col in high_card_cols]

    # Ajouter au dataframe principal
    df_train_cleaned = df_train_cleaned.drop(columns=high_card_cols)
    df_test_cleaned = df_test_cleaned.drop(columns=high_card_cols)

    df_train_cleaned = pd.concat([df_train_cleaned, df_train_encoded], axis=1)
    df_test_cleaned = pd.concat([df_test_cleaned, df_test_encoded], axis=1)

    # --- One-Hot Encoding ---
    df_train_cleaned = pd.get_dummies(df_train_cleaned, columns=['Nature_mutation', 'Type_local'], drop_first=False)
    df_test_cleaned = pd.get_dummies(df_test_cleaned, columns=['Nature_mutation', 'Type_local'], drop_first=False)

    # Harmoniser les colonnes entre les deux datasets
    df_train_cleaned, df_test_cleaned = df_train_cleaned.align(df_test_cleaned, join='left', axis=1, fill_value=0)

    return df_train_cleaned, df_test_cleaned


# In[25]:


# Convertir_colonnes_booleennes_en_entiers
def convert_bool_to_numeric(df):
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    return df


# In[ ]:





# In[ ]:





# In[ ]:




