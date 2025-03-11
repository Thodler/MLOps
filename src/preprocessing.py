import pandas as pd
import numpy as np
import yaml
import os

# Charger les configurations depuis config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
RAW_DATA_PATH = config['path']['raw_data_path']
PROCESSED_DATA_PATH = config['path']['processed_data_path']

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance entre deux points en kilomètres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Rayon de la Terre en km
    return km

def preprocess_data():
    """Prétraitement des données brutes."""
    # Charger les données
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Convertir les dates
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
    
    # Extraire des caractéristiques temporelles
    df['hour_of_day'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    
    # Calculer la distance
    df['distance_km'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Nettoyer les données
    # Supprimer les trajets avec une durée négative ou excessive
    df = df[(df['trip_duration'] > 10) & (df['trip_duration'] < 10000)]
    
    # Supprimer les valeurs aberrantes de distance
    df = df[df['distance_km'] < 100]
    
    # S'assurer que le dossier existe
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    # Sauvegarder les données prétraitées
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"Données prétraitées sauvegardées dans {PROCESSED_DATA_PATH}")
    return df

if __name__ == "__main__":
    preprocess_data()
