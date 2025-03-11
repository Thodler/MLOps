import pandas as pd
import joblib
from config import MODEL_PATH, FEATURES

def load_model():
    """Charge le modèle entraîné."""
    return joblib.load(MODEL_PATH)

def predict(model, data):
    """Fait des prédictions sur de nouvelles données."""
    # S'assurer que les données ont les bonnes features
    X = data[FEATURES]
    # Faire des prédictions
    predictions = model.predict(X)
    return predictions

def predict_single_trip(model, trip_data):
    """Prédit la durée d'un seul trajet."""
    # Convertir en DataFrame
    df = pd.DataFrame([trip_data])
    
    # Vérifier que toutes les features nécessaires sont présentes
    for feature in FEATURES:
        if feature not in df.columns:
            raise ValueError(f"La caractéristique '{feature}' est manquante")
    
    # Faire la prédiction
    duration = predict(model, df)[0]
    return duration

if __name__ == "__main__":
    # Exemple d'utilisation
    model = load_model()
    
    # Créer une entrée d'exemple
    sample_trip = {
        'pickup_longitude': -73.9,
        'pickup_latitude': 40.7,
        'dropoff_longitude': -73.8,
        'dropoff_latitude': 40.8,
        'passenger_count': 2,
        'hour_of_day': 14,
        'day_of_week': 3,
        'distance_km': 5.2
    }
    
    predicted_duration = predict_single_trip(model, sample_trip)
    print(f"Durée prévue du trajet: {predicted_duration:.2f} secondes")
