#!/usr/bin/env python3
"""
Script para entrenar el modelo de predicción de tenis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
from datetime import datetime
from utils import TennisFeatureEngineering

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml-service/logs/train_model.log'),
        logging.StreamHandler()
    ]
)

def train_model(data_path: str, model_path: str = 'ml-service/model/model.pkl', 
                test_size: float = 0.2):
    """
    Entrena el modelo de predicción de tenis.
    
    Args:
        data_path: Ruta al archivo CSV con datos de entrenamiento
        model_path: Ruta donde guardar el modelo entrenado
        test_size: Proporción de datos para prueba
        
    Returns:
        Diccionario con métricas de rendimiento
    """
    try:
        # Crear directorio de modelo si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Cargar datos
        data = pd.read_csv(data_path)
        logging.info(f"Datos cargados: {len(data)} partidos")
        
        # Preparar características
        feature_engineering = TennisFeatureEngineering()
        X = feature_engineering.extract_features(data)
        y = data['winner']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Balancear clases si es necesario
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Entrenar modelo
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train_balanced, y_train_balanced)
        logging.info("Modelo entrenado exitosamente")
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Guardar modelo
        joblib.dump(model, model_path)
        logging.info(f"Modelo guardado en {model_path}")
        
        # Guardar metadatos
        metadata = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model_type': 'RandomForestClassifier',
            'features': X.columns.tolist(),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'metrics': metrics
        }
        
        metadata_path = os.path.join(os.path.dirname(model_path), 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=4)
        
        logging.info(f"Metadatos guardados en {metadata_path}")
        logging.info("Métricas de rendimiento:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error entrenando modelo: {e}")
        return None

def main():
    """Función principal para entrenar el modelo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo de predicción de tenis')
    parser.add_argument('--data', required=True, help='Ruta al archivo CSV con datos de entrenamiento')
    parser.add_argument('--model', default='ml-service/model/model.pkl', help='Ruta donde guardar el modelo')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporción de datos para prueba')
    
    args = parser.parse_args()
    
    metrics = train_model(args.data, args.model, args.test_size)
    if metrics:
        logging.info("Entrenamiento completado exitosamente")
    else:
        logging.error("Error en el entrenamiento del modelo")

if __name__ == '__main__':
    main()