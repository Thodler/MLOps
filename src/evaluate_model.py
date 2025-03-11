import pandas as pd
import sqlite3
import joblib
from sklearn.metrics import mean_absolute_error

def load_data_from_sqlite(db_path, table_name):
    """Charge les données depuis SQLite."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def preprocess_data(test_df, scaler):
    """Prétraite les données pour l'évaluation."""
    # Convertir les colonnes de date/heure
    test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])
    test_df['dropoff_datetime'] = pd.to_datetime(test_df['dropoff_datetime'])

    # Calculer la durée du trajet en secondes (cible)
    test_df['trip_duration'] = (test_df['dropoff_datetime'] - test_df['pickup_datetime']).dt.total_seconds()

    # Sélectionner les caractéristiques et la cible
    features = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    X = test_df[features]
    y = test_df['trip_duration']

    # Normalisation des données avec le scaler sauvegardé
    X = scaler.transform(X)

    return X, y

def evaluate_model(model, X, y):
    """Évalue le modèle sur l'ensemble de test."""
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f"Erreur absolue moyenne (MAE) sur l'ensemble de test : {mae} secondes")

def main():
    db_path = "data/nyc_taxi_trip_duration.db"
    model_path = "models/taxi_trip_duration_model.pkl"
    scaler_path = "models/scaler.pkl"

    # Charger les données de test depuis SQLite
    print("Chargement des données de test depuis SQLite...")
    test_df = load_data_from_sqlite(db_path, "test")

    # Charger le modèle Ridge et le scaler
    print("Chargement du modèle Ridge et du scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Prétraiter les données de test
    print("Prétraitement des données de test...")
    X_test, y_test = preprocess_data(test_df, scaler)

    # Évaluer le modèle sur l'ensemble de test
    print("Évaluation du modèle Ridge sur l'ensemble de test...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()