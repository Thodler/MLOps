import os
import pandas as pd
import sqlite3
from zipfile import ZipFile
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split

def download_and_extract_data(data_url, data_zip_path, extract_dir):
    """Télécharge et extrait les données depuis l'URL."""
    # Créer le dossier `data` s'il n'existe pas
    os.makedirs(extract_dir, exist_ok=True)

    # Télécharger le fichier ZIP s'il n'existe pas
    if not os.path.exists(data_zip_path):
        print(f"Téléchargement des données depuis {data_url}...")
        urlretrieve(data_url, data_zip_path)
        print(f"Données téléchargées et sauvegardées dans {data_zip_path}")
    else:
        print(f"Le fichier ZIP existe déjà à {data_zip_path}")

    # Extraire les fichiers du ZIP
    print(f"Extraction des données dans {extract_dir}...")
    with ZipFile(data_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Données extraites avec succès.")

def split_and_load_data_to_sqlite(csv_path, db_path):
    """Divise le fichier CSV en deux ensembles et les charge dans SQLite."""
    # Vérifier si le fichier CSV existe
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas.")

    print(f"Chargement du fichier CSV depuis {csv_path}...")
    df = pd.read_csv(csv_path)

    # Diviser les données en ensembles d'entraînement et de test (80 % / 20 %)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Créer le dossier contenant la base de données si nécessaire
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Charger les données dans SQLite
    conn = sqlite3.connect(db_path)
    train_df.to_sql("train", conn, if_exists="replace", index=False)
    test_df.to_sql("test", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Données chargées dans SQLite à {db_path}")

def cleanup_temp_files(csv_path, zip_path):
    """Supprime les fichiers temporaires (.csv et .zip)."""
    if os.path.exists(csv_path):
        print(f"Suppression du fichier CSV temporaire : {csv_path}")
        os.remove(csv_path)
    else:
        print(f"Le fichier CSV {csv_path} n'existe pas.")

    if os.path.exists(zip_path):
        print(f"Suppression du fichier ZIP temporaire : {zip_path}")
        os.remove(zip_path)
    else:
        print(f"Le fichier ZIP {zip_path} n'existe pas.")

def main():
    data_url = "https://github.com/eishkina-estia/ML2023/raw/main/data/New_York_City_Taxi_Trip_Duration.zip"
    data_zip_path = "data/New_York_City_Taxi_Trip_Duration.zip"
    extract_dir = "data"
    db_path = "data/nyc_taxi_trip_duration.db"
    csv_path = os.path.join(extract_dir, "New_York_City_Taxi_Trip_Duration.csv")

    # Télécharger et extraire les données
    download_and_extract_data(data_url, data_zip_path, extract_dir)

    # Diviser les données et les charger dans SQLite
    split_and_load_data_to_sqlite(csv_path, db_path)

    # Supprimer les fichiers temporaires
    cleanup_temp_files(csv_path, data_zip_path)

if __name__ == "__main__":
    main()