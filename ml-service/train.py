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
import tensorflow as tf

from model_ensemble import TennisEnsembleModel, TennisXGBoostModel, TennisNeuralNetwork
from utils import TennisFeatureEngineering
from elo_system import TennisEloSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_detailed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_features(data):
    """
    Prepara las características para el entrenamiento.
    
    Args:
        data: DataFrame con datos de partidos
        
    Returns:
        X: Características preparadas
        y: Etiquetas (1 si gana player1, 0 si gana player2)
    """
    logging.info("Iniciando preparación de características...")
    
    # Crear copia del DataFrame para no modificar el original
    df = data.copy()
    
    # Reorganizar aleatoriamente los jugadores
    logging.info("Reorganizando aleatoriamente los jugadores...")
    mask = np.random.rand(len(df)) > 0.5
    df.loc[mask, ['player1_id', 'player2_id']] = df.loc[mask, ['player2_id', 'player1_id']].values
    
    # Crear etiquetas (1 si gana player1, 0 si gana player2)
    y = (df['winner_id'] == df['player1_id']).astype(int)
    
    # Características de ELO
    logging.info("Calculando características de ELO...")
    elo_system = TennisEloSystem()
    elo_features = elo_system.calculate_elo_features(df)
    
    # Características temporales
    logging.info("Añadiendo características temporales...")
    df['match_date'] = pd.to_datetime(df['match_date'])
    df['day_of_week'] = df['match_date'].dt.dayofweek
    df['month'] = df['match_date'].dt.month
    df['year'] = df['match_date'].dt.year
    df['season'] = df['month'].apply(lambda x: 'winter' if x in [12,1,2] else 
                                   'spring' if x in [3,4,5] else 
                                   'summer' if x in [6,7,8] else 'fall')
    
    # Características del partido
    logging.info("Añadiendo características del partido...")
    df['sets_played'] = df['sets_played'].fillna(3)  # Valor por defecto
    df['minutes'] = df['minutes'].fillna(df['minutes'].mean())
    df['round_numeric'] = df['round'].map({
        'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1
    }).fillna(1)
    
    # Crear características básicas
    features = []
    
    # Características de superficie (one-hot encoding)
    surface_dummies = pd.get_dummies(df['surface'], prefix='surface')
    features.append(surface_dummies)
    
    # Características de torneo (one-hot encoding)
    tournament_dummies = pd.get_dummies(df['tournament_category'], prefix='tournament')
    features.append(tournament_dummies)
    
    # Características temporales (one-hot encoding)
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    features.append(season_dummies)
    
    # Características numéricas
    numeric_features = pd.DataFrame({
        'sets_played': df['sets_played'],
        'minutes': df['minutes'],
        'round_numeric': df['round_numeric'],
        'day_of_week': df['day_of_week'],
        'month': df['month'],
        'year': df['year']
    })
    features.append(numeric_features)
    
    # Características de ELO
    if not elo_features.empty:
        features.append(elo_features)
    
    # Combinar todas las características
    X = pd.concat(features, axis=1)
    
    # Verificar y eliminar columnas no numéricas
    logging.info("Verificando tipos de datos de las características...")
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        logging.warning(f"Se encontraron columnas no numéricas: {non_numeric_cols}")
        logging.info("Eliminando columnas no numéricas...")
        X = X.select_dtypes(include=['number'])
    
    logging.info(f"Características preparadas: {X.shape}")
    logging.info(f"Tipos de características: {X.dtypes}")
    return X, y

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
        logger.info("Iniciando carga de datos...")
        data = pd.read_csv(data_path)
        logger.info(f"Datos cargados: {len(data)} partidos")
        
        # Preparar características
        logger.info("Preparando características...")
        X, y = prepare_features(data)
        logger.info(f"Características preparadas: {X.shape}")
        
        # Dividir datos con validación temporal
        logger.info("Dividiendo datos de entrenamiento...")
        tscv = TimeSeriesSplit(n_splits=5)
        all_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"\nEntrenando fold {fold}/5...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Crear conjunto de validación
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Entrenar modelo ensemble
            logger.info("Entrenando modelo ensemble...")
            ensemble_model = TennisEnsembleModel()
            ensemble_model.fit(X_train_final, y_train_final)
            
            # Entrenar modelo XGBoost
            logger.info("Entrenando modelo XGBoost...")
            xgb_model = TennisXGBoostModel()
            xgb_model.fit(X_train_final, y_train_final, optimize_hyperparams=True)
            
            # Entrenar red neuronal
            logger.info("Entrenando red neuronal...")
            nn_model = TennisNeuralNetwork(input_dim=X_train_final.shape[1])
            nn_model.fit(X_train_final, y_train_final, epochs=100, batch_size=32)
            
            # Evaluar modelos
            models = {
                'ensemble': ensemble_model,
                'xgb': xgb_model,
                'nn': nn_model
            }
            
            fold_metrics = {}
            for name, model in models.items():
                logger.info(f"\nEvaluando modelo {name} en fold {fold}...")
                y_pred = model.predict(X_test)
                
                fold_metrics[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                # Guardar modelo
                model_path = os.path.join(model_dir, f'{name}_model_fold_{fold}.pkl')
                model.save(model_path)
                
                # Informe de clasificación
                logger.info(f"\nInforme de Clasificación - {name} (Fold {fold}):")
                logger.info(classification_report(y_test, y_pred))
            
            all_metrics.append(fold_metrics)
        
        # Calcular métricas promedio
        avg_metrics = {}
        for model_name in models.keys():
            avg_metrics[model_name] = {
                metric: np.mean([fold[model_name][metric] for fold in all_metrics])
                for metric in ['accuracy', 'precision', 'recall', 'f1']
            }
        
        # Visualizaciones
        plt.figure(figsize=(15, 5))
        
        # Matrices de confusión promedio
        for i, (name, model) in enumerate(models.items(), 1):
            plt.subplot(1, 3, i)
            y_pred = model.predict(X_test)
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
            plt.title(f'Matriz de Confusión - {name}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'confusion_matrices.png'))
        plt.close()
        
        # Gráfico de métricas comparativas
        metrics_df = pd.DataFrame(avg_metrics).T
        metrics_df.plot(kind='bar', figsize=(10, 6))
        plt.title('Comparación de Métricas Promedio por Modelo')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'metrics_comparison.png'))
        plt.close()
        
        # Tiempo total de entrenamiento
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"\nTiempo total de entrenamiento: {total_time:.2f} segundos")
        
        return avg_metrics
        
    except Exception as e:
        logger.error(f"Error entrenando modelos: {str(e)}", exc_info=True)
        raise

def main():
    """Función principal para entrenar los modelos."""
    parser = argparse.ArgumentParser(description='Entrenar modelos de predicción de tenis')
    parser.add_argument('--data', required=True, help='Ruta al archivo CSV con datos de entrenamiento')
    parser.add_argument('--model-dir', default='model', help='Directorio donde guardar los modelos')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporción de datos para prueba')
    
    args = parser.parse_args()
    
    metrics = train_models(args.data, args.model_dir, args.test_size)
    if metrics:
        logger.info("Entrenamiento completado exitosamente")
        print("Entrenamiento completado exitosamente")
    else:
        logger.error("Error en el entrenamiento de los modelos")
        print("Error en el entrenamiento de los modelos")

if __name__ == '__main__':
    main()