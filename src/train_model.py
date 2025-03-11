import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge  # Remplacement de RandomForestRegressor par Ridge
from sklearn.metrics import mean_absolute_error
import joblib
import os

def load_data_from_sqlite(db_path, table_name):
    """Charge les données depuis SQLite."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def preprocess_data(train_df):
    """Prétraite les données pour l'entraînement."""
    # Supprimer les valeurs manquantes
    train_df.dropna(inplace=True)

    # Convertir les colonnes de date/heure
    train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
    train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

    # Calculer la durée du trajet en secondes (cible)
    train_df['trip_duration'] = (train_df['dropoff_datetime'] - train_df['pickup_datetime']).dt.total_seconds()

    # Sélectionner les caractéristiques et la cible
    features = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    X = train_df[features]
    y = train_df['trip_duration']

    return X, y

def train_and_save_model(X_train, X_val, y_train, y_val, model_path, scaler_path):
    """Entraîne un modèle Ridge et sauvegarde les artefacts."""
    # Normalisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Entraîner un modèle Ridge Regression
    model = Ridge(alpha=1.0)  # alpha est le paramètre de régularisation
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de validation
    y_pred = model.predict(X_val)

    # Évaluer le modèle
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Erreur absolue moyenne (MAE) : {mae} secondes")

    # Sauvegarder le modèle et le scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modèle Ridge sauvegardé à {model_path}")
    print(f"Scaler sauvegardé à {scaler_path}")

def main():
    db_path = "data/nyc_taxi_trip_duration.db"
    model_path = "models/taxi_trip_duration_model.pkl"
    scaler_path = "models/scaler.pkl"

    # Charger les données d'entraînement depuis SQLite
    print("Chargement des données d'entraînement depuis SQLite...")
    train_df = load_data_from_sqlite(db_path, "train")

    # Prétraiter les données
    print("Prétraitement des données...")
    X, y = preprocess_data(train_df)

    # Diviser les données en ensembles d'entraînement et de validation
    print("Division des données en ensembles d'entraînement et de validation...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner et sauvegarder le modèle Ridge
    print("Entraînement du modèle Ridge...")
    train_and_save_model(X_train, X_val, y_train, y_val, model_path, scaler_path)

if __name__ == "__main__":
    main()