from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle et le scaler
model = joblib.load("models/taxi_trip_duration_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Définir le schéma des données d'entrée
class TripData(BaseModel):
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float

# Créer l'application FastAPI
app = FastAPI()

# Endpoint pour faire des prédictions
@app.post("/predict")
def predict(trip_data: TripData):
    try:
        # Convertir les données d'entrée en tableau numpy
        input_data = np.array([
            trip_data.passenger_count,
            trip_data.pickup_longitude,
            trip_data.pickup_latitude,
            trip_data.dropoff_longitude,
            trip_data.dropoff_latitude
        ]).reshape(1, -1)

        # Normaliser les données
        input_data = scaler.transform(input_data)

        # Faire la prédiction
        prediction = model.predict(input_data)
        return {"predicted_trip_duration": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Lancer l'application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)