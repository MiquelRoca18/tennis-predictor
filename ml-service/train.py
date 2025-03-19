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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_detailed.log', mode='w'),
        logging.StreamHandler()
    ]
)

class TennisFeatureEngineering:
    """Clase para preparación avanzada de características."""
    
    def extract_features(self, data):
        """
        Extracción de características con ingeniería más robusta.
        """
        # Columnas numéricas base
        numeric_columns = [
            'ranking_1', 'ranking_2', 
            'winrate_1', 'winrate_2', 
            'surface_winrate_1', 'surface_winrate_2',
            'surface_winrate_diff', 'winrate_diff'
        ]
        
        # Codificar superficie de manera más segura
        surface_mapping = {
            'hard': 0, 
            'clay': 1, 
            'grass': 2,
            'carpet': 3
        }
        data['surface_encoded'] = data['surface'].map(surface_mapping).fillna(-1)
        
        # Incluir columna de superficie
        features_columns = numeric_columns + ['surface_encoded']
        
        # Extraer características
        X = data[features_columns].copy()
        
        # Manejar valores nulos de manera más robusta
        for col in X.columns:
            # Usar la mediana para reemplazar nulos
            X[col] = X[col].fillna(X[col].median())
        
        # Añadir características derivadas con verificación
        try:
            X['ranking_diff'] = data['ranking_1'] - data['ranking_2']
            X['absolute_ranking_diff'] = np.abs(data['ranking_1'] - data['ranking_2'])
        except Exception as e:
            logging.warning(f"Error al crear características derivadas: {e}")
        
        # Añadir algunas características de interacción
        try:
            X['winrate_ranking_interaction'] = X['winrate_1'] / (X['ranking_1'] + 1)
            X['winrate_ranking_interaction_2'] = X['winrate_2'] / (X['ranking_2'] + 1)
        except Exception as e:
            logging.warning(f"Error al crear características de interacción: {e}")
        
        # Eliminar columnas constantes o con varianza muy baja
        selector = VarianceThreshold(threshold=0.01)  # Umbral de varianza bajo
        X_selected = selector.fit_transform(X)
        
        # Logging de características seleccionadas
        selected_features = [X.columns[i] for i in selector.get_support(indices=True)]
        logging.info(f"Características seleccionadas: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

def train_model(data_path, model_path='model.pkl', test_size=0.2):
    """
    Entrenar modelo de predicción de tenis con optimizaciones.
    """
    start_time = time.time()
    
    try:
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
        
        # Crear pipeline con escalador robusto y clasificador
        pipeline = Pipeline([
            ('scaler', RobustScaler()),  # Más robusto a outliers
            ('classifier', RandomForestClassifier(
                n_jobs=-1,  # Usar todos los núcleos
                random_state=42
            ))
        ])
        
        # Parámetros para búsqueda de hiperparámetros
        param_grid = {
            'classifier__n_estimators': [200, 500, 1000],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        
        # Búsqueda de hiperparámetros con validación cruzada
        logging.info("Iniciando búsqueda de hiperparámetros...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5,  # 5-fold cross-validation 
            scoring='accuracy',
            n_jobs=-1  # Usar todos los núcleos
        )
        
        # Entrenar con búsqueda de hiperparámetros
        grid_search.fit(X_train, y_train)
        
        # Obtener mejor modelo
        best_model = grid_search.best_estimator_
        logging.info("Mejores hiperparámetros:")
        logging.info(grid_search.best_params_)
        
        # Predecir en conjunto de prueba
        y_pred = best_model.predict(X_test)
        
        # Métricas detalladas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Logging de métricas detalladas
        logging.info("Métricas de rendimiento:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # Informe de clasificación
        logging.info("\nInforme de Clasificación:")
        logging.info(classification_report(y_test, y_pred))
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        logging.info("\nMatriz de Confusión:")
        logging.info(conf_matrix)
        
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        
        # Guardar modelo
        joblib.dump(best_model, model_path)
        
        # Tiempo total de entrenamiento
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"\nTiempo total de entrenamiento: {total_time:.2f} segundos")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error entrenando modelo: {e}", exc_info=True)
        return None

def main():
    """Función principal para entrenar el modelo."""
    parser = argparse.ArgumentParser(description='Entrenar modelo de predicción de tenis')
    parser.add_argument('--data', required=True, help='Ruta al archivo CSV con datos de entrenamiento')
    parser.add_argument('--model', default='model/model.pkl', help='Ruta donde guardar el modelo')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporción de datos para prueba')
    
    args = parser.parse_args()
    
    metrics = train_model(args.data, args.model, args.test_size)
    if metrics:
        logging.info("Entrenamiento completado exitosamente")
        print("Entrenamiento completado exitosamente")
    else:
        logging.error("Error en el entrenamiento del modelo")
        print("Error en el entrenamiento del modelo")

if __name__ == '__main__':
    main()