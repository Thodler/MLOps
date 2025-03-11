import os
from src.load_data import main as load_data
from src.train_model import main as train_model
from src.evaluate_model import main as evaluate_model
from src.predict import main as predict

def main():
    # Étape 1 : Télécharger et charger les données dans SQLite
    print("Étape 1 : Téléchargement et chargement des données...")
    load_data()

    # Étape 2 : Entraîner le modèle
    print("\nÉtape 2 : Entraînement du modèle...")
    train_model()

    # Étape 3 : Évaluer le modèle
    print("\nÉtape 3 : Évaluation du modèle...")
    evaluate_model()
    
    # Étape 3 : Évaluer le modèle
    print("\nÉtape 3 : Prediction...")
    predict()

    print("\nPipeline terminé avec succès !")
    

if __name__ == "__main__":
    main()