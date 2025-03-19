#!/usr/bin/env python3
"""
Script para evaluar el modelo de predicción de tenis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
        logging.FileHandler('ml-service/logs/evaluation.log'),
        logging.StreamHandler()
    ]
)

class TennisModelEvaluator:
    """
    Clase para evaluar el modelo de predicción de tenis.
    """
    
    def __init__(self, model_path: str, data_path: str):
        """
        Inicializa el evaluador.
        
        Args:
            model_path: Ruta al modelo entrenado
            data_path: Ruta a los datos de evaluación
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.feature_engineering = TennisFeatureEngineering()
    
    def load_data(self) -> bool:
        """Carga el modelo y los datos."""
        try:
            # Cargar modelo
            self.model = joblib.load(self.model_path)
            logging.info(f"Modelo cargado desde {self.model_path}")
            
            # Cargar datos
            self.data = pd.read_csv(self.data_path)
            logging.info(f"Datos cargados: {len(self.data)} partidos")
            
            return True
            
        except Exception as e:
            logging.error(f"Error cargando datos: {e}")
            return False
    
    def evaluate(self, test_size: float = 0.2, n_folds: int = 5) -> dict:
        """
        Evalúa el modelo usando validación cruzada y conjunto de prueba.
        
        Args:
            test_size: Proporción de datos para prueba
            n_folds: Número de pliegues para validación cruzada
            
        Returns:
            Diccionario con métricas de evaluación
        """
        try:
            if not self.load_data():
                return None
            
            # Preparar características
            X = self.feature_engineering.extract_features(self.data)
            y = self.data['winner']
            
            # Validación cruzada
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'auc': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Entrenar modelo en este pliegue
                self.model.fit(X_train, y_train)
                
                # Predicciones
                y_pred = self.model.predict(X_val)
                y_pred_proba = self.model.predict_proba(X_val)[:, 1]
                
                # Calcular métricas
                cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                cv_scores['precision'].append(precision_score(y_val, y_pred))
                cv_scores['recall'].append(recall_score(y_val, y_pred))
                cv_scores['f1'].append(f1_score(y_val, y_pred))
                cv_scores['auc'].append(roc_auc_score(y_val, y_pred_proba))
            
            # Calcular promedios
            results = {
                'cv_mean': {k: np.mean(v) for k, v in cv_scores.items()},
                'cv_std': {k: np.std(v) for k, v in cv_scores.items()}
            }
            
            # Evaluación en conjunto de prueba
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            results['test'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Guardar resultados
            self._save_results(results)
            
            return results
            
        except Exception as e:
            logging.error(f"Error en evaluación: {e}")
            return None
    
    def _save_results(self, results: dict):
        """Guarda los resultados de la evaluación."""
        try:
            # Crear directorio de resultados si no existe
            output_dir = 'ml-service/results'
            os.makedirs(output_dir, exist_ok=True)
            
            # Preparar resultados para guardar
            output = {
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'model_path': self.model_path,
                'data_path': self.data_path,
                'results': results
            }
            
            # Guardar resultados
            output_path = os.path.join(output_dir, 'evaluation_results.json')
            with open(output_path, 'w') as f:
                import json
                json.dump(output, f, indent=4)
            
            logging.info(f"Resultados guardados en {output_path}")
            
            # Mostrar resultados
            logging.info("\nResultados de validación cruzada:")
            for metric, mean in results['cv_mean'].items():
                std = results['cv_std'][metric]
                logging.info(f"{metric}: {mean:.4f} (±{std:.4f})")
            
            logging.info("\nResultados en conjunto de prueba:")
            for metric, value in results['test'].items():
                logging.info(f"{metric}: {value:.4f}")
            
        except Exception as e:
            logging.error(f"Error guardando resultados: {e}")

def main():
    """Función principal para evaluar el modelo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluar modelo de predicción de tenis')
    parser.add_argument('--model', required=True, help='Ruta al modelo entrenado')
    parser.add_argument('--data', required=True, help='Ruta a los datos de evaluación')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proporción de datos para prueba')
    parser.add_argument('--n-folds', type=int, default=5,
                      help='Número de pliegues para validación cruzada')
    
    args = parser.parse_args()
    
    evaluator = TennisModelEvaluator(args.model, args.data)
    results = evaluator.evaluate(args.test_size, args.n_folds)
    
    if results:
        logging.info("Evaluación completada exitosamente")
    else:
        logging.error("Error en la evaluación del modelo")

if __name__ == '__main__':
    main() 