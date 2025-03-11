import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge  # Importer Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from config import PROCESSED_DATA_PATH, MODEL_PATH, FEATURES, TARGET, TEST_SIZE, RANDOM_STATE

def train_model():
    """Entraîne un modèle de prédiction de durée de trajet."""
    # Charger les données prétraitées
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Préparer les données
    X = df[FEATURES]
    y = df[TARGET]
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
        
    # Initialiser le modèle
    model = Ridge(
        alpha=1.0,  # Paramètre de régularisation
        random_state=RANDOM_STATE
    )
    
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Évaluer le modèle
    y_pred = model.predict(X_test)
    
    # Calculer les métriques
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("Performance du modèle:")
    print(f"MAE: {mae:.2f} secondes")
    print(f"RMSE: {rmse:.2f} secondes")
    print(f"R²: {r2:.3f}")
    
    # S'assurer que le dossier existe
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Sauvegarder le modèle
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")
    
    return model, (X_test, y_test, y_pred)

if __name__ == "__main__":
    train_model()
