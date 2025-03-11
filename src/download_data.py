import numpy as np
import pandas as pd
import os
import yaml

# Charger les configurations depuis config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

RAW_DATA_PATH = config['path']['raw_data_path']

def generate_sample_data(n_samples=10000):
    """Génère un ensemble de données d'exemple pour les trajets de taxis NYC."""
    np.random.seed(42)
    
    data = {
        'id': [f'id_{i}' for i in range(n_samples)],
        'vendor_id': np.random.choice([1, 2], size=n_samples),
        'pickup_datetime': pd.date_range(start='2023-01-01', periods=n_samples, freq='10T'),
        'dropoff_datetime': pd.date_range(start='2023-01-01 00:10:00', periods=n_samples, freq='10T'),
        'passenger_count': np.random.randint(1, 7, size=n_samples),
        'pickup_longitude': np.random.uniform(-74.03, -73.77, size=n_samples),
        'pickup_latitude': np.random.uniform(40.63, 40.85, size=n_samples),
        'dropoff_longitude': np.random.uniform(-74.03, -73.77, size=n_samples),
        'dropoff_latitude': np.random.uniform(40.63, 40.85, size=n_samples),
        'store_and_fwd_flag': np.random.choice(['Y', 'N'], size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculer la durée du trajet en secondes
    df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()
    
    # S'assurer que le dossier existe
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    # Sauvegarder les données
    df.to_csv(RAW_DATA_PATH, index=False)
    
    print(f"Données générées et sauvegardées dans {RAW_DATA_PATH}")
    return df

if __name__ == "__main__":
    generate_sample_data()
