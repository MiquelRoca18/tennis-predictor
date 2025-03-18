import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

def train_model():
    # Cargar dataset
    # Nota: Deberás reemplazar esto con tu propio dataset
    data = pd.read_csv("tennis_matches.csv")
    
    # Preprocesamiento
    # Convertir categorías a valores numéricos
    if 'surface' in data.columns:
        surfaces = {'hard': 0, 'clay': 1, 'grass': 2}
        data['surface_code'] = data['surface'].map(surfaces)
    
    # Seleccionar features
    features = ["ranking_1", "ranking_2", "winrate_1", "winrate_2"]
    if 'surface_code' in data.columns:
        features.append('surface_code')
    
    X = data[features]
    y = data["winner"]  # Asumiendo que winner es 0 para player_1 y 1 para player_2
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluación
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Guardar modelo
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')
    
    return model

if __name__ == "__main__":
    train_model()