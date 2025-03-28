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
from typing import Dict, List, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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

class TennisEvaluator:
    def __init__(self):
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score
        }
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Evalúa el modelo usando métricas estándar y específicas de tenis.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            data: DataFrame con datos adicionales del partido
            
        Returns:
            Diccionario con todas las métricas
        """
        logger.info("Evaluando modelo...")
        
        # Métricas estándar
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(y_true, y_pred)
        
        # Métricas específicas de tenis si tenemos datos adicionales
        if data is not None:
            tennis_metrics = self._calculate_tennis_metrics(y_true, y_pred, data)
            results.update(tennis_metrics)
        
        logger.info("Evaluación completada")
        return results
    
    def _calculate_tennis_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                data: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula métricas específicas para tenis.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            data: DataFrame con datos del partido
            
        Returns:
            Diccionario con métricas específicas de tenis
        """
        results = {}
        
        # Evaluación por superficie
        for surface in ['hard', 'clay', 'grass', 'carpet']:
            surface_mask = data['surface'] == surface
            if surface_mask.any():
                surface_accuracy = accuracy_score(y_true[surface_mask], y_pred[surface_mask])
                results[f'{surface}_accuracy'] = surface_accuracy
        
        # Evaluación por ranking
        ranking_diffs = data['player1_rank'] - data['player2_rank']
        for diff_range in [(0, 10), (10, 50), (50, 100), (100, float('inf'))]:
            mask = (ranking_diffs >= diff_range[0]) & (ranking_diffs < diff_range[1])
            if mask.any():
                range_accuracy = accuracy_score(y_true[mask], y_pred[mask])
                results[f'rank_diff_{diff_range[0]}_{diff_range[1]}_accuracy'] = range_accuracy
        
        # Evaluación por head-to-head
        h2h_mask = data['h2h_matches'] > 0
        if h2h_mask.any():
            h2h_accuracy = accuracy_score(y_true[h2h_mask], y_pred[h2h_mask])
            results['h2h_accuracy'] = h2h_accuracy
        
        # Evaluación por forma reciente
        recent_mask = data['recent_matches'] >= 5
        if recent_mask.any():
            recent_accuracy = accuracy_score(y_true[recent_mask], y_pred[recent_mask])
            results['recent_form_accuracy'] = recent_accuracy
        
        # Evaluación por tipo de torneo
        for tournament_type in data['tournament_category'].unique():
            tournament_mask = data['tournament_category'] == tournament_type
            if tournament_mask.any():
                tournament_accuracy = accuracy_score(y_true[tournament_mask], y_pred[tournament_mask])
                results[f'{tournament_type}_accuracy'] = tournament_accuracy
        
        return results
    
    def evaluate_by_surface(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evalúa el modelo separadamente para cada superficie.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            data: DataFrame con datos del partido
            
        Returns:
            Diccionario con métricas por superficie
        """
        results = {}
        
        for surface in ['hard', 'clay', 'grass', 'carpet']:
            surface_mask = data['surface'] == surface
            if surface_mask.any():
                surface_metrics = {}
                for name, metric in self.metrics.items():
                    surface_metrics[name] = metric(y_true[surface_mask], y_pred[surface_mask])
                results[surface] = surface_metrics
        
        return results
    
    def evaluate_by_ranking_difference(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evalúa el modelo por diferencia de ranking.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            data: DataFrame con datos del partido
            
        Returns:
            Diccionario con métricas por diferencia de ranking
        """
        results = {}
        
        ranking_diffs = data['player1_rank'] - data['player2_rank']
        diff_ranges = [(0, 10), (10, 50), (50, 100), (100, float('inf'))]
        
        for diff_range in diff_ranges:
            mask = (ranking_diffs >= diff_range[0]) & (ranking_diffs < diff_range[1])
            if mask.any():
                range_metrics = {}
                for name, metric in self.metrics.items():
                    range_metrics[name] = metric(y_true[mask], y_pred[mask])
                results[f'rank_diff_{diff_range[0]}_{diff_range[1]}'] = range_metrics
        
        return results
    
    def evaluate_by_tournament_type(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evalúa el modelo por tipo de torneo.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            data: DataFrame con datos del partido
            
        Returns:
            Diccionario con métricas por tipo de torneo
        """
        results = {}
        
        for tournament_type in data['tournament_category'].unique():
            tournament_mask = data['tournament_category'] == tournament_type
            if tournament_mask.any():
                tournament_metrics = {}
                for name, metric in self.metrics.items():
                    tournament_metrics[name] = metric(y_true[tournament_mask], y_pred[tournament_mask])
                results[tournament_type] = tournament_metrics
        
        return results
    
    def evaluate_by_recent_form(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evalúa el modelo por forma reciente del jugador.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            data: DataFrame con datos del partido
            
        Returns:
            Diccionario con métricas por forma reciente
        """
        results = {}
        
        # Definir rangos de forma reciente
        form_ranges = [(0, 5), (5, 10), (10, 20), (20, float('inf'))]
        
        for form_range in form_ranges:
            mask = (data['recent_matches'] >= form_range[0]) & (data['recent_matches'] < form_range[1])
            if mask.any():
                range_metrics = {}
                for name, metric in self.metrics.items():
                    range_metrics[name] = metric(y_true[mask], y_pred[mask])
                results[f'recent_matches_{form_range[0]}_{form_range[1]}'] = range_metrics
        
        return results
    
    def evaluate_by_h2h_history(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evalúa el modelo por historial head-to-head.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            data: DataFrame con datos del partido
            
        Returns:
            Diccionario con métricas por historial h2h
        """
        results = {}
        
        # Definir rangos de partidos h2h
        h2h_ranges = [(0, 1), (1, 3), (3, 5), (5, float('inf'))]
        
        for h2h_range in h2h_ranges:
            mask = (data['h2h_matches'] >= h2h_range[0]) & (data['h2h_matches'] < h2h_range[1])
            if mask.any():
                range_metrics = {}
                for name, metric in self.metrics.items():
                    range_metrics[name] = metric(y_true[mask], y_pred[mask])
                results[f'h2h_matches_{h2h_range[0]}_{h2h_range[1]}'] = range_metrics
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, float]):
        """
        Imprime los resultados de la evaluación de forma formateada.
        
        Args:
            results: Diccionario con resultados de evaluación
        """
        logger.info("\nResultados de la evaluación:")
        logger.info("-" * 50)
        
        # Métricas estándar
        logger.info("\nMétricas estándar:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in results:
                logger.info(f"{metric:10}: {results[metric]:.4f}")
        
        # Métricas por superficie
        surfaces = [k for k in results.keys() if k.endswith('_accuracy') and k.split('_')[0] in ['hard', 'clay', 'grass', 'carpet']]
        if surfaces:
            logger.info("\nMétricas por superficie:")
            for surface in surfaces:
                logger.info(f"{surface:20}: {results[surface]:.4f}")
        
        # Métricas por ranking
        ranking_metrics = [k for k in results.keys() if k.startswith('rank_diff_')]
        if ranking_metrics:
            logger.info("\nMétricas por diferencia de ranking:")
            for metric in ranking_metrics:
                logger.info(f"{metric:30}: {results[metric]:.4f}")
        
        # Métricas por tipo de torneo
        tournament_metrics = [k for k in results.keys() if k.endswith('_accuracy') and k.split('_')[0] in ['grand_slam', 'masters', 'atp_tour']]
        if tournament_metrics:
            logger.info("\nMétricas por tipo de torneo:")
            for metric in tournament_metrics:
                logger.info(f"{metric:25}: {results[metric]:.4f}")
        
        logger.info("-" * 50)

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