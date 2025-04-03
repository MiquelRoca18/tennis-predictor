"""
evaluation.py

Módulo para evaluar la capacidad predictiva del sistema ELO de tenis.
Contiene funciones para calcular métricas de rendimiento, calibración de probabilidades,
y análisis de resultados de predicción.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import log_loss, brier_score_loss

# Configurar logging
logger = logging.getLogger(__name__)

def evaluate_predictive_power(elo_processor, test_matches_df: pd.DataFrame, 
                           probability_threshold: float = 0.5,
                           features_to_use: Optional[List[str]] = None) -> Dict:
    """
    Evalúa la capacidad predictiva del sistema ELO actual.
    
    Args:
        elo_processor: Instancia del procesador ELO con los ratings actuales
        test_matches_df: DataFrame con partidos de prueba
        probability_threshold: Umbral de probabilidad para predicciones
        features_to_use: Lista de características específicas a evaluar
        
    Returns:
        Diccionario con métricas de rendimiento
    """
    logger.info(f"Evaluando capacidad predictiva en {len(test_matches_df)} partidos...")
    
    # Verificar que tenemos las columnas necesarias
    required_columns = ['winner_id', 'loser_id', 'surface']
    
    if not all(col in test_matches_df.columns for col in required_columns):
        raise ValueError(f"DataFrame debe contener las columnas: {required_columns}")
    
    # Hacer copia para no modificar el original
    df = test_matches_df.copy()
    
    # Crear columnas para almacenar predicciones
    df['predicted_winner_id'] = None
    df['p1_win_probability'] = 0.0
    df['correct_prediction'] = False
    df['elo_diff'] = 0.0
    df['elo_surface_diff'] = 0.0
    
    # Estructuras para resultados
    results = {
        'total_matches': len(df),
        'correct_predictions': 0,
        'accuracy': 0.0,
        'accuracy_by_surface': {},
        'accuracy_by_threshold': {},
        'accuracy_by_tourney_level': {},
        'log_loss': 0.0,
        'brier_score': 0.0,
        'calibration': {},
        'confidence': {},
        'feature_importance': {}
    }
    
    # Preparar variables para métricas de calibración
    all_probs = []
    all_outcomes = []
    
    # Si hay fecha, ordenar por fecha
    if 'match_date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['match_date']):
            df['match_date'] = pd.to_datetime(df['match_date'])
        df = df.sort_values('match_date')
    
    # Contadores adicionales
    surface_correct = defaultdict(int)
    surface_total = defaultdict(int)
    
    tourney_correct = defaultdict(int)
    tourney_total = defaultdict(int)
    
    threshold_correct = defaultdict(int)
    threshold_total = defaultdict(int)
    
    confidence_bins = {
        '50-60%': {'correct': 0, 'total': 0},
        '60-70%': {'correct': 0, 'total': 0},
        '70-80%': {'correct': 0, 'total': 0},
        '80-90%': {'correct': 0, 'total': 0},
        '90-100%': {'correct': 0, 'total': 0}
    }
    
    # Variables para análisis de calibración
    bin_edges = np.linspace(0, 1, 11)  # 10 bins
    calibration_counts = np.zeros(10)
    calibration_correct = np.zeros(10)
    
    # Iterar por los partidos
    logger.info("Realizando predicciones...")
    
    for idx, match in df.iterrows():
        # Obtener IDs
        winner_id = str(match['winner_id'])
        loser_id = str(match['loser_id'])
        surface = elo_processor._normalize_surface(match['surface'])
        
        tourney_level = 'unknown'
        if 'tourney_level' in match:
            tourney_level = elo_processor._normalize_tournament_level(match['tourney_level'])
        
        # Realizar predicción con nuestro sistema actual
        prediction = elo_processor.predict_match(winner_id, loser_id, surface)
        
        # Extraer probabilidad de victoria
        p1_win_prob = prediction['prediction']['p1_win_probability']
        
        # Guardar resultados
        df.at[idx, 'p1_win_probability'] = p1_win_prob
        df.at[idx, 'elo_diff'] = prediction['player1']['elo_general'] - prediction['player2']['elo_general']
        
        if 'elo_surface' in prediction['player1'] and 'elo_surface' in prediction['player2']:
            df.at[idx, 'elo_surface_diff'] = prediction['player1']['elo_surface'] - prediction['player2']['elo_surface']
        
        # Determinar predicción
        predicted_winner = winner_id if p1_win_prob >= probability_threshold else loser_id
        df.at[idx, 'predicted_winner_id'] = predicted_winner
        
        # Verificar si la predicción fue correcta
        correct = predicted_winner == winner_id
        df.at[idx, 'correct_prediction'] = correct
        
        # Actualizar contadores generales
        if correct:
            results['correct_predictions'] += 1
        
        # Actualizar contadores por superficie
        surface_total[surface] += 1
        if correct:
            surface_correct[surface] += 1
        
        # Actualizar contadores por nivel de torneo
        tourney_total[tourney_level] += 1
        if correct:
            tourney_correct[tourney_level] += 1
        
        # Actualizar contadores por umbrales
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            pred_at_threshold = winner_id if p1_win_prob >= threshold else loser_id
            is_correct = pred_at_threshold == winner_id
            
            # Solo considerar partidos donde la probabilidad supera el umbral
            if max(p1_win_prob, 1-p1_win_prob) >= threshold:
                threshold_total[threshold] += 1
                if is_correct:
                    threshold_correct[threshold] += 1
        
        # Actualizar bins de confianza
        confidence = max(p1_win_prob, 1-p1_win_prob)
        
        if 0.5 <= confidence < 0.6:
            bin_key = '50-60%'
        elif 0.6 <= confidence < 0.7:
            bin_key = '60-70%'
        elif 0.7 <= confidence < 0.8:
            bin_key = '70-80%'
        elif 0.8 <= confidence < 0.9:
            bin_key = '80-90%'
        else:  # >= 0.9
            bin_key = '90-100%'
        
        confidence_bins[bin_key]['total'] += 1
        if correct:
            confidence_bins[bin_key]['correct'] += 1
        
        # Registrar para calibración
        all_probs.append(p1_win_prob)
        all_outcomes.append(1 if winner_id == prediction['player1']['id'] else 0)
        
        # Actualizar bins de calibración
        bin_idx = min(9, int(p1_win_prob * 10))
        calibration_counts[bin_idx] += 1
        if winner_id == prediction['player1']['id']:  # Si P1 ganó
            calibration_correct[bin_idx] += 1
    
    # Calcular resultados finales
    results['accuracy'] = results['correct_predictions'] / results['total_matches']
    
    # Accuracy por superficie
    for surface, count in surface_total.items():
        if count > 0:
            results['accuracy_by_surface'][surface] = surface_correct[surface] / count
    
    # Accuracy por nivel de torneo
    for level, count in tourney_total.items():
        if count > 0:
            results['accuracy_by_tourney_level'][level] = tourney_correct[level] / count
    
    # Accuracy por umbral
    for threshold, count in threshold_total.items():
        if count > 0:
            results['accuracy_by_threshold'][str(threshold)] = {
                'accuracy': threshold_correct[threshold] / count,
                'coverage': count / results['total_matches']
            }
    
    # Accuracy por nivel de confianza
    for bin_key, data in confidence_bins.items():
        if data['total'] > 0:
            results['confidence'][bin_key] = {
                'accuracy': data['correct'] / data['total'],
                'count': data['total'],
                'percentage': data['total'] / results['total_matches'] * 100
            }
    
    # Calibración (qué tan bien calibradas están las probabilidades)
    calibration_result = {}
    for i in range(len(bin_edges) - 1):
        bin_name = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        if calibration_counts[i] > 0:
            empirical_prob = calibration_correct[i] / calibration_counts[i]
            calibration_result[bin_name] = {
                'predicted_probability': (bin_edges[i] + bin_edges[i+1]) / 2,
                'empirical_probability': empirical_prob,
                'count': int(calibration_counts[i]),
                'error': empirical_prob - (bin_edges[i] + bin_edges[i+1]) / 2
            }
    
    results['calibration'] = calibration_result
    
    # Calcular métricas de evaluación adicionales
    try:
        results['log_loss'] = log_loss(all_outcomes, all_probs)
        results['brier_score'] = brier_score_loss(all_outcomes, all_probs)
    except Exception as e:
        logger.warning(f"Error calculando métricas avanzadas: {str(e)}")
    
    # Análisis de importancia de características si se proporcionan
    if features_to_use and all(feature in df.columns for feature in features_to_use):
        # Usar correlación con el resultado como indicador básico
        for feature in features_to_use:
            if pd.api.types.is_numeric_dtype(df[feature]):
                correlation = df[feature].corr(df['correct_prediction'])
                results['feature_importance'][feature] = abs(correlation)
    
    # Retornar resultados y DataFrame con predicciones
    return results

def calculate_calibration_curve(predictions: List[float], outcomes: List[int], 
                             n_bins: int = 10) -> Dict:
    """
    Calcula una curva de calibración para evaluar la calidad de las probabilidades.
    
    Args:
        predictions: Lista de probabilidades predichas
        outcomes: Lista de resultados (0 o 1)
        n_bins: Número de bins para la curva de calibración
        
    Returns:
        Diccionario con datos de calibración
    """
    # Validación básica
    if len(predictions) != len(outcomes):
        raise ValueError("Las listas de predicciones y resultados deben tener la misma longitud")
    
    # Crear bins uniformes
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Asegurar que los índices están en rango
    
    # Calcular frecuencia empírica para cada bin
    bin_sums = np.bincount(bin_indices, weights=outcomes, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_avg_predictions = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_avg_predictions[i] = np.mean(np.array(predictions)[mask])
        else:
            bin_avg_predictions[i] = (bin_edges[i] + bin_edges[i+1]) / 2
    
    # Calcular proporciones empíricas
    nonzero = bin_counts > 0
    bin_empirical_probs = np.zeros(n_bins)
    bin_empirical_probs[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
    
    # Construir resultado con datos estadísticos
    calibration_data = {
        'bin_edges': bin_edges.tolist(),
        'bin_counts': bin_counts.tolist(),
        'bin_avg_predictions': bin_avg_predictions.tolist(),
        'bin_empirical_probs': bin_empirical_probs.tolist(),
        'bin_errors': (bin_empirical_probs - bin_avg_predictions).tolist(),
        'mean_calibration_error': np.mean(np.abs(bin_empirical_probs - bin_avg_predictions)),
        'count': len(predictions)
    }
    
    return calibration_data

def compute_feature_importance(elo_processor, test_matches_df: pd.DataFrame, 
                           features: Optional[List[str]] = None) -> Dict:
    """
    Calcula la importancia de diferentes características para la predicción.
    
    Args:
        elo_processor: Instancia del procesador ELO
        test_matches_df: DataFrame con partidos de prueba
        features: Lista de características a analizar (opcional)
        
    Returns:
        Diccionario con medidas de importancia de características
    """
    # Preparar datos con características
    df = elo_processor.create_features_for_model(test_matches_df)
    
    # Si no se especifican características, usar todas las numéricas
    if features is None:
        features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) 
                  and col not in ['target', 'match_id']]
    
    # Calcular correlación con el resultado
    importance = {}
    for feature in features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            importance[feature] = abs(df[feature].corr(df['target']))
    
    # Ordenar por importancia
    importance = {k: v for k, v in sorted(importance.items(), 
                                        key=lambda item: item[1], reverse=True)}
    
    return {
        'feature_importance': importance,
        'top_features': list(importance.keys())[:10],  # Top 10 características
        'count': len(df)
    }

def analyze_prediction_errors(elo_processor, test_matches_df: pd.DataFrame) -> Dict:
    """
    Analiza patrones en los errores de predicción para identificar debilidades del modelo.
    
    Args:
        elo_processor: Instancia del procesador ELO
        test_matches_df: DataFrame con partidos de prueba
        
    Returns:
        Diccionario con análisis de errores
    """
    # Evaluar predicciones
    df = test_matches_df.copy()
    
    # Crear columnas para almacenar predicciones
    df['predicted_winner_id'] = None
    df['p1_win_probability'] = 0.0
    df['correct_prediction'] = False
    
    # Realizar predicciones
    for idx, match in df.iterrows():
        winner_id = str(match['winner_id'])
        loser_id = str(match['loser_id'])
        surface = elo_processor._normalize_surface(match['surface'])
        
        # Realizar predicción con nuestro sistema actual
        prediction = elo_processor.predict_match(winner_id, loser_id, surface)
        
        # Extraer probabilidad de victoria
        p1_win_prob = prediction['prediction']['p1_win_probability']
        
        # Guardar resultados
        df.at[idx, 'p1_win_probability'] = p1_win_prob
        predicted_winner = winner_id if p1_win_prob >= 0.5 else loser_id
        df.at[idx, 'predicted_winner_id'] = predicted_winner
        df.at[idx, 'correct_prediction'] = predicted_winner == winner_id
    
    # Separar predicciones correctas e incorrectas
    correct_df = df[df['correct_prediction'] == True]
    error_df = df[df['correct_prediction'] == False]
    
    # Análisis por superficie
    surface_errors = {}
    for surface in df['surface'].unique():
        surface_df = df[df['surface'] == surface]
        error_count = sum(~surface_df['correct_prediction'])
        surface_errors[surface] = {
            'total': len(surface_df),
            'errors': error_count,
            'error_rate': error_count / max(1, len(surface_df))
        }
    
    # Análisis por nivel de torneo
    tourney_errors = {}
    if 'tourney_level' in df.columns:
        for level in df['tourney_level'].unique():
            level_df = df[df['tourney_level'] == level]
            error_count = sum(~level_df['correct_prediction'])
            tourney_errors[level] = {
                'total': len(level_df),
                'errors': error_count,
                'error_rate': error_count / max(1, len(level_df))
            }
    
    # Análisis por probabilidad
    prob_bins = {
        '50-60%': {'errors': 0, 'total': 0},
        '60-70%': {'errors': 0, 'total': 0},
        '70-80%': {'errors': 0, 'total': 0},
        '80-90%': {'errors': 0, 'total': 0},
        '90-100%': {'errors': 0, 'total': 0}
    }
    
    for _, row in df.iterrows():
        prob = max(row['p1_win_probability'], 1 - row['p1_win_probability'])
        
        if 0.5 <= prob < 0.6:
            bin_key = '50-60%'
        elif 0.6 <= prob < 0.7:
            bin_key = '60-70%'
        elif 0.7 <= prob < 0.8:
            bin_key = '70-80%'
        elif 0.8 <= prob < 0.9:
            bin_key = '80-90%'
        else:  # >= 0.9
            bin_key = '90-100%'
        
        prob_bins[bin_key]['total'] += 1
        if not row['correct_prediction']:
            prob_bins[bin_key]['errors'] += 1
    
    # Calcular tasa de error por bin
    for bin_key, data in prob_bins.items():
        if data['total'] > 0:
            data['error_rate'] = data['errors'] / data['total']
    
    # Construir resultado
    return {
        'total_matches': len(df),
        'correct_predictions': len(correct_df),
        'incorrect_predictions': len(error_df),
        'accuracy': len(correct_df) / len(df),
        'surface_analysis': surface_errors,
        'tournament_analysis': tourney_errors,
        'probability_analysis': prob_bins,
        'avg_confidence_errors': error_df['p1_win_probability'].apply(lambda p: max(p, 1-p)).mean(),
        'avg_confidence_correct': correct_df['p1_win_probability'].apply(lambda p: max(p, 1-p)).mean()
    }