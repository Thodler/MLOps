from src.download_data import generate_sample_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.predict import load_model, predict_single_trip

def run_pipeline():
    """Exécute le pipeline complet."""
    print("Étape 1: Téléchargement/génération des données")
    generate_sample_data()
    
    print("\nÉtape 2: Prétraitement des données")
    preprocess_data()
    
    print("\nÉtape 3: Entraînement du modèle")
    train_model()
    
    print("\nÉtape b4: Vérification de l'inférence")
    model = load_model()
    
    # Exemple de trajet pour tester l'inférence
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
    print(f"                      = {predicted_duration/60:.2f} minutes")

if __name__ == "__main__":
    run_pipeline()
