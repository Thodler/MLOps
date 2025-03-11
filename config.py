# Configuration du projet

# Chemins des données
RAW_DATA_PATH = "data/raw/nyc_taxi_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_taxi_data.csv"

# Chemin pour sauvegarder le modèle
MODEL_PATH = "models/taxi_duration_model.joblib"

# Paramètres pour l'entraînement
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Caractéristiques (features) à utiliser pour le modèle
FEATURES = [
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count',
    'hour_of_day',
    'day_of_week',
    'distance_km'
]

# Variable cible
TARGET = 'trip_duration'
