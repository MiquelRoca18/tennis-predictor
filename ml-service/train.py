#!/usr/bin/env python3
"""
Script para entrenar modelo de predicción de tenis con optimizaciones.
"""

import os
import sys
import logging
import argparse
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, f1_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from model_ensemble import TennisEnsembleModel, TennisXGBoostModel
from utils import TennisFeatureEngineering

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_detailed.log', mode='w'),
        logging.StreamHandler()
    ]
)

def train_models(data_path, model_dir='model', test_size=0.2):
    """
    Entrena múltiples modelos de predicción de tenis.
    
    Args:
        data_path: Ruta al archivo CSV con datos de entrenamiento
        model_dir: Directorio donde guardar los modelos
        test_size: Proporción de datos para prueba
    """
    start_time = time.time()
    
    try:
        # Crear directorio de modelos si no existe
        os.makedirs(model_dir, exist_ok=True)
        
        # Cargar datos
        logging.info("Iniciando carga de datos...")
        data = pd.read_csv(data_path)
        logging.info(f"Datos cargados: {len(data)} partidos")
        
        # Preparar características
        feature_engineering = TennisFeatureEngineering()
        X = feature_engineering.extract_features(data)
        y = data['winner']
        
        # Dividir datos con estratificación
        logging.info("Dividiendo datos de entrenamiento...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
        
        # Entrenar modelo ensemble
        logging.info("Entrenando modelo ensemble...")
        ensemble_model = TennisEnsembleModel()
        ensemble_model.fit(X_train, y_train)
        
        # Evaluar modelo ensemble
        y_pred_ensemble = ensemble_model.predict(X_test)
        metrics_ensemble = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble, average='weighted'),
            'recall': recall_score(y_test, y_pred_ensemble, average='weighted'),
            'f1': f1_score(y_test, y_pred_ensemble, average='weighted')
        }
        
        # Guardar modelo ensemble
        ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
        ensemble_model.save(ensemble_path)
        
        # Entrenar modelo XGBoost
        logging.info("Entrenando modelo XGBoost...")
        xgb_model = TennisXGBoostModel()
        xgb_model.fit(X_train, y_train, optimize_hyperparams=True)
        
        # Evaluar modelo XGBoost
        y_pred_xgb = xgb_model.predict(X_test)
        metrics_xgb = {
            'accuracy': accuracy_score(y_test, y_pred_xgb),
            'precision': precision_score(y_test, y_pred_xgb, average='weighted'),
            'recall': recall_score(y_test, y_pred_xgb, average='weighted'),
            'f1': f1_score(y_test, y_pred_xgb, average='weighted')
        }
        
        # Guardar modelo XGBoost
        xgb_path = os.path.join(model_dir, 'xgb_model.pkl')
        xgb_model.save(xgb_path)
        
        # Calcular y guardar importancia de características
        importance = xgb_model.feature_importance(plot=True)
        importance.to_csv(os.path.join(model_dir, 'feature_importance.csv'))
        
        # Logging de métricas
        logging.info("\nMétricas del Modelo Ensemble:")
        for metric, value in metrics_ensemble.items():
            logging.info(f"{metric}: {value:.4f}")
        
        logging.info("\nMétricas del Modelo XGBoost:")
        for metric, value in metrics_xgb.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # Informe de clasificación
        logging.info("\nInforme de Clasificación - Ensemble:")
        logging.info(classification_report(y_test, y_pred_ensemble))
        
        logging.info("\nInforme de Clasificación - XGBoost:")
        logging.info(classification_report(y_test, y_pred_xgb))
        
        # Matrices de confusión
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(confusion_matrix(y_test, y_pred_ensemble), annot=True, fmt='d')
        plt.title('Matriz de Confusión - Ensemble')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d')
        plt.title('Matriz de Confusión - XGBoost')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'confusion_matrices.png'))
        plt.close()
        
        # Tiempo total de entrenamiento
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"\nTiempo total de entrenamiento: {total_time:.2f} segundos")
        
        return {
            'ensemble_metrics': metrics_ensemble,
            'xgb_metrics': metrics_xgb
        }
        
    except Exception as e:
        logging.error(f"Error entrenando modelos: {e}", exc_info=True)
        return None

def main():
    """Función principal para entrenar los modelos."""
    parser = argparse.ArgumentParser(description='Entrenar modelos de predicción de tenis')
    parser.add_argument('--data', required=True, help='Ruta al archivo CSV con datos de entrenamiento')
    parser.add_argument('--model-dir', default='model', help='Directorio donde guardar los modelos')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporción de datos para prueba')
    
    args = parser.parse_args()
    
    metrics = train_models(args.data, args.model_dir, args.test_size)
    if metrics:
        logging.info("Entrenamiento completado exitosamente")
        print("Entrenamiento completado exitosamente")
    else:
        logging.error("Error en el entrenamiento de los modelos")
        print("Error en el entrenamiento de los modelos")

if __name__ == '__main__':
    main()