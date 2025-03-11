import pandas as pd
import joblib

def preprocess_new_data(new_data, scaler):
    """Prétraite les nouvelles données pour la prédiction."""
    # Convertir les colonnes de date/heure
    new_data['pickup_datetime'] = pd.to_datetime(new_data['pickup_datetime'])
    new_data['dropoff_datetime'] = pd.to_datetime(new_data['dropoff_datetime'])

    # Calculer la durée du trajet en secondes (cible)
    new_data['trip_duration'] = (new_data['dropoff_datetime'] - new_data['pickup_datetime']).dt.total_seconds()

    # Sélectionner les caractéristiques
    features = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    X = new_data[features]

    # Normalisation des données avec le scaler sauvegardé
    X = scaler.transform(X)

    return X

def predict_new_data(model, scaler, new_data):
    """Fait des prédictions sur de nouvelles données."""
    # Prétraiter les nouvelles données
    X = preprocess_new_data(new_data, scaler)

    # Faire des prédictions
    predictions = model.predict(X)

    # Ajouter les prédictions aux nouvelles données
    new_data['predicted_trip_duration'] = predictions

    return new_data

def main():
    # Chemins des fichiers
    model_path = "models/taxi_trip_duration_model.pkl"
    scaler_path = "models/scaler.pkl"

    # Charger le modèle Ridge et le scaler
    print("Chargement du modèle Ridge et du scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Exemple de nouvelles données (à remplacer par vos propres données)
    new_data = pd.DataFrame({
        'id': [1, 2],
        'vendor_id': [1, 2],
        'pickup_datetime': ['2023-10-01 12:00:00', '2023-10-01 12:30:00'],
        'dropoff_datetime': ['2023-10-01 12:30:00', '2023-10-01 13:00:00'],
        'passenger_count': [1, 2],
        'pickup_longitude': [-73.9857, -73.9881],
        'pickup_latitude': [40.7484, 40.7490],
        'dropoff_longitude': [-73.9881, -73.9857],
        'dropoff_latitude': [40.7490, 40.7484],
        'store_and_fwd_flag': ['N', 'N']
    })

    # Faire des prédictions sur les nouvelles données
    print("Prédiction des durées de trajet...")
    predictions = predict_new_data(model, scaler, new_data)

    # Afficher les prédictions
    print(predictions[['id', 'predicted_trip_duration']])

if __name__ == "__main__":
    main()