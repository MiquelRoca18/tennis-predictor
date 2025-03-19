#!/usr/bin/env python3
"""
Script para probar el sistema de predicción de tenis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
        logging.FileHandler('ml-service/logs/test_system.log'),
        logging.StreamHandler()
    ]
)

def test_system(model_path: str, test_data_path: str, output_dir: str = 'ml-service/results'):
    """
    Prueba el sistema de predicción con datos de prueba.
    
    Args:
        model_path: Ruta al modelo entrenado
        test_data_path: Ruta a los datos de prueba
        output_dir: Directorio para guardar resultados
        
    Returns:
        Diccionario con métricas de rendimiento
    """
    try:
        # Crear directorio de resultados si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar modelo y datos
        model = joblib.load(model_path)
        data = pd.read_csv(test_data_path)
        logging.info(f"Datos de prueba cargados: {len(data)} partidos")
        
        # Preparar características
        feature_engineering = TennisFeatureEngineering()
        X = feature_engineering.extract_features(data)
        y = data['winner']
        
        # Hacer predicciones
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba)
        }
        
        # Guardar resultados
        results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model_path': model_path,
            'test_data_path': test_data_path,
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
        
        results_path = os.path.join(output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            import json
            json.dump(results, f, indent=4)
        
        logging.info(f"Resultados guardados en {results_path}")
        logging.info("Métricas de rendimiento:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error probando sistema: {e}")
        return None

def main():
    """Función principal para probar el sistema."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Probar sistema de predicción de tenis')
    parser.add_argument('--model', required=True, help='Ruta al modelo entrenado')
    parser.add_argument('--test-data', required=True, help='Ruta a los datos de prueba')
    parser.add_argument('--output-dir', default='ml-service/results',
                      help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    metrics = test_system(args.model, args.test_data, args.output_dir)
    if metrics:
        logging.info("Prueba del sistema completada exitosamente")
    else:
        logging.error("Error en la prueba del sistema")

if __name__ == '__main__':
    main()