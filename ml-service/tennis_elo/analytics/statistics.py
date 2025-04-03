"""
statistics.py

Módulo para análisis estadísticos del sistema ELO de tenis.
Contiene funciones para analizar distribuciones de ratings, tendencias, 
patrones estadísticos y generar informes de resumen.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logger = logging.getLogger(__name__)

def analyze_ratings_distribution(elo_processor, surface: Optional[str] = None,
                             min_matches: int = 10) -> Dict:
    """
    Analiza la distribución de los ratings ELO.
    
    Args:
        elo_processor: Instancia del procesador ELO
        surface: Superficie específica (opcional)
        min_matches: Número mínimo de partidos jugados
        
    Returns:
        Diccionario con estadísticas de la distribución
    """
    if surface:
        surface = elo_processor._normalize_surface(surface)
        ratings_dict = elo_processor.player_ratings_by_surface.get(surface, {})
        
        # Filtrar jugadores con pocos partidos
        filtered_ratings = {
            player_id: rating 
            for player_id, rating in ratings_dict.items()
            if elo_processor.player_match_count_by_surface[surface].get(player_id, 0) >= min_matches
        }
    else:
        ratings_dict = elo_processor.player_ratings
        
        # Filtrar jugadores con pocos partidos
        filtered_ratings = {
            player_id: rating 
            for player_id, rating in ratings_dict.items()
            if elo_processor.player_match_count.get(player_id, 0) >= min_matches
        }
    
    ratings = list(filtered_ratings.values())
    
    if not ratings:
        return {
            'count': 0,
            'mean': elo_processor.initial_rating,
            'std': 0,
            'min': elo_processor.initial_rating,
            'max': elo_processor.initial_rating,
            'percentiles': {
                '5': elo_processor.initial_rating,
                '25': elo_processor.initial_rating,
                '50': elo_processor.initial_rating,
                '75': elo_processor.initial_rating,
                '95': elo_processor.initial_rating
            }
        }
    
    # Calcular estadísticas
    ratings_array = np.array(ratings)
    
    # Cálculos estadísticos básicos
    stats_data = {
        'count': len(ratings),
        'mean': float(np.mean(ratings_array)),
        'std': float(np.std(ratings_array)),
        'min': float(np.min(ratings_array)),
        'max': float(np.max(ratings_array)),
        'percentiles': {
            '5': float(np.percentile(ratings_array, 5)),
            '25': float(np.percentile(ratings_array, 25)),
            '50': float(np.percentile(ratings_array, 50)),  # Mediana
            '75': float(np.percentile(ratings_array, 75)),
            '95': float(np.percentile(ratings_array, 95))
        }
    }
    
    # Añadir distribución por cuartiles para visualización
    q_values = []
    bin_edges = []
    
    # Crear 20 bins para visualizar distribución
    hist, edges = np.histogram(ratings_array, bins=20)
    
    # Convertir a porcentajes
    total = hist.sum()
    percentages = (hist / total * 100).tolist()
    
    # Añadir al resultado
    stats_data['distribution'] = {
        'bin_counts': hist.tolist(),
        'bin_percentages': percentages,
        'bin_edges': edges.tolist(),
        'bin_centers': [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
    }
    
    # Verificar normalidad (para análisis)
    normality_test = stats.normaltest(ratings_array)
    stats_data['normality'] = {
        'statistic': float(normality_test.statistic),
        'p_value': float(normality_test.pvalue),
        'is_normal': float(normality_test.pvalue) > 0.05
    }
    
    return stats_data

def get_elo_statistics_summary(elo_processor, min_matches: int = 10) -> Dict:
    """
    Genera un resumen completo de estadísticas del sistema ELO.
    
    Args:
        elo_processor: Instancia del procesador ELO
        min_matches: Número mínimo de partidos para incluir a un jugador
        
    Returns:
        Diccionario con el resumen de estadísticas
    """
    logger.info("Generando resumen de estadísticas ELO...")
    
    # Estructuras para resultados
    summary = {
        'general': {},
        'by_surface': {},
        'top_players': {},
        'statistics': {},
        'processing': elo_processor.stats,
        'distribution': {}
    }
    
    # 1. Estadísticas generales
    total_players = len(elo_processor.player_ratings)
    total_matches = sum(elo_processor.player_match_count.values())
    
    # Jugadores con suficientes partidos
    qualified_players = {
        player_id: rating 
        for player_id, rating in elo_processor.player_ratings.items()
        if elo_processor.player_match_count.get(player_id, 0) >= min_matches
    }
    
    qualified_count = len(qualified_players)
    
    # Estadísticas básicas de ratings
    if qualified_count > 0:
        ratings_array = np.array(list(qualified_players.values()))
        
        summary['general'] = {
            'total_players': total_players,
            'qualified_players': qualified_count,
            'total_matches': total_matches,
            'average_matches_per_player': total_matches / max(1, total_players),
            'min_rating': float(np.min(ratings_array)),
            'max_rating': float(np.max(ratings_array)),
            'average_rating': float(np.mean(ratings_array)),
            'median_rating': float(np.median(ratings_array)),
            'std_dev': float(np.std(ratings_array))
        }
    else:
        summary['general'] = {
            'total_players': total_players,
            'qualified_players': 0,
            'total_matches': total_matches,
            'average_matches_per_player': total_matches / max(1, total_players),
            'min_rating': elo_processor.initial_rating,
            'max_rating': elo_processor.initial_rating,
            'average_rating': elo_processor.initial_rating,
            'median_rating': elo_processor.initial_rating,
            'std_dev': 0
        }
    
    # 2. Estadísticas por superficie
    surface_stats = {}
    for surface, ratings in elo_processor.player_ratings_by_surface.items():
        # Jugadores con suficientes partidos en esta superficie
        qualified_surface = {
            player_id: rating 
            for player_id, rating in ratings.items()
            if elo_processor.player_match_count_by_surface[surface].get(player_id, 0) >= min_matches
        }
        
        if qualified_surface:
            surface_ratings = np.array(list(qualified_surface.values()))
            
            surface_stats[surface] = {
                'total_players': len(ratings),
                'qualified_players': len(qualified_surface),
                'total_matches': sum(elo_processor.player_match_count_by_surface[surface].values()),
                'min_rating': float(np.min(surface_ratings)),
                'max_rating': float(np.max(surface_ratings)),
                'average_rating': float(np.mean(surface_ratings)),
                'median_rating': float(np.median(surface_ratings)),
                'std_dev': float(np.std(surface_ratings))
            }
        else:
            surface_stats[surface] = {
                'total_players': len(ratings),
                'qualified_players': 0,
                'total_matches': sum(elo_processor.player_match_count_by_surface[surface].values()),
                'min_rating': elo_processor.initial_rating,
                'max_rating': elo_processor.initial_rating,
                'average_rating': elo_processor.initial_rating,
                'median_rating': elo_processor.initial_rating,
                'std_dev': 0
            }
    
    summary['by_surface'] = surface_stats
    
    # 3. Top jugadores
    top_players_general = elo_processor.get_top_players(n=20, include_details=True).to_dict('records')
    summary['top_players']['general'] = top_players_general
    
    # Top jugadores por superficie
    surface_top = {}
    for surface in elo_processor.player_ratings_by_surface:
        top_by_surface = elo_processor.get_top_players(n=10, surface=surface, include_details=True).to_dict('records')
        if top_by_surface:
            surface_top[surface] = top_by_surface
    
    summary['top_players']['by_surface'] = surface_top
    
    # 4. Distribuciones estadísticas
    for surface in ['general'] + list(elo_processor.player_ratings_by_surface.keys()):
        # Usar función ya implementada para análisis detallado
        if surface == 'general':
            dist_data = analyze_ratings_distribution(elo_processor, min_matches=min_matches)
        else:
            dist_data = analyze_ratings_distribution(elo_processor, surface=surface, min_matches=min_matches)
        
        # Simplificar datos de distribución para el resumen
        simplified_dist = {
            'count': dist_data['count'],
            'mean': dist_data['mean'],
            'std': dist_data['std'],
            'median': dist_data['percentiles']['50'] if '50' in dist_data['percentiles'] else 
                    dist_data['percentiles'][50] if 50 in dist_data['percentiles'] else elo_processor.initial_rating,
            'percentiles': dist_data['percentiles']
        }
        
        summary['distribution'][surface] = simplified_dist
    
    # 5. Estadísticas de predicción (si están disponibles)
    if 'model_accuracy' in elo_processor.stats and elo_processor.stats['model_accuracy'] is not None:
        summary['statistics']['prediction'] = {
            'accuracy': elo_processor.stats['model_accuracy'],
            'calibration': elo_processor.stats['calibration_score'] if 'calibration_score' in elo_processor.stats else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    # 6. Estadísticas temporales
    if elo_processor.rating_history:
        # Convertir historial a DataFrame si no lo está ya
        if isinstance(elo_processor.rating_history, list):
            history_df = pd.DataFrame(elo_processor.rating_history)
        else:
            history_df = elo_processor.rating_history
        
        # Extraer fechas extremas
        if 'date' in history_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(history_df['date']):
                history_df['date'] = pd.to_datetime(history_df['date'])
            
            earliest_date = history_df['date'].min()
            latest_date = history_df['date'].max()
            date_range = (latest_date - earliest_date).days
            
            summary['statistics']['temporal'] = {
                'earliest_date': earliest_date.strftime('%Y-%m-%d'),
                'latest_date': latest_date.strftime('%Y-%m-%d'),
                'date_range_days': date_range,
                'total_historical_matches': len(history_df)
            }
    
    # 7. Estadísticas de incertidumbre
    if elo_processor.player_rating_uncertainty:
        uncertainty_values = list(elo_processor.player_rating_uncertainty.values())
        if uncertainty_values:
            summary['statistics']['uncertainty'] = {
                'average': float(np.mean(uncertainty_values)),
                'min': float(np.min(uncertainty_values)),
                'max': float(np.max(uncertainty_values)),
                'median': float(np.median(uncertainty_values))
            }
    
    # 8. Estadísticas por tipo de torneo, si están disponibles
    if elo_processor.rating_history:
        if isinstance(elo_processor.rating_history, list):
            history_df = pd.DataFrame(elo_processor.rating_history)
        else:
            history_df = elo_processor.rating_history
            
        if 'tourney_level' in history_df.columns:
            tourney_counts = history_df['tourney_level'].value_counts().to_dict()
            summary['statistics']['tournament_levels'] = tourney_counts
    
    # 9. Estadísticas de head-to-head
    h2h_counts = []
    for player_id, opponents in elo_processor.h2h_records.items():
        # Contar total de enfrentamientos
        for opponent_id, record in opponents.items():
            total_matches = record['wins'] + record['losses']
            if total_matches > 0:
                h2h_counts.append(total_matches)
    
    if h2h_counts:
        summary['statistics']['head_to_head'] = {
            'total_matchups': len(h2h_counts),
            'average_matches_per_matchup': float(np.mean(h2h_counts)),
            'max_matches_between_pair': int(np.max(h2h_counts))
        }
    
    return summary

def analyze_surface_specialists(elo_processor, min_matches: int = 10, 
                           min_rating_diff: float = 50.0) -> Dict[str, List[Dict]]:
    """
    Identifica especialistas por superficie comparando ratings entre superficies.
    
    Args:
        elo_processor: Instancia del procesador ELO
        min_matches: Mínimo de partidos en la superficie para ser considerado
        min_rating_diff: Diferencia mínima con otra superficie para ser especialista
        
    Returns:
        Diccionario de listas con especialistas por superficie
    """
    surfaces = ['hard', 'clay', 'grass', 'carpet']
    specialists = {surface: [] for surface in surfaces}
    
    # Para cada jugador activo
    for player_id, rating in elo_processor.player_ratings.items():
        player_specialists = []
        
        for surface in surfaces:
            # Verificar que tiene suficientes partidos en esta superficie
            surface_matches = elo_processor.player_match_count_by_surface[surface].get(player_id, 0)
            if surface_matches < min_matches:
                continue
                
            # Obtener rating en esta superficie
            surface_rating = elo_processor.player_ratings_by_surface[surface].get(player_id, elo_processor.initial_rating)
            
            # Comparar con otras superficies
            is_specialist = True
            comparison = []
            
            for other_surface in surfaces:
                if other_surface == surface:
                    continue
                
                other_matches = elo_processor.player_match_count_by_surface[other_surface].get(player_id, 0)
                if other_matches < min_matches:
                    continue
                
                other_rating = elo_processor.player_ratings_by_surface[other_surface].get(
                    player_id, elo_processor.initial_rating
                )
                
                diff = surface_rating - other_rating
                comparison.append({
                    'surface': other_surface,
                    'rating_diff': diff,
                    'matches': other_matches
                })
                
                if diff < min_rating_diff:
                    is_specialist = False
            
            # Si es especialista y hay al menos una comparación válida
            if is_specialist and comparison:
                player_name = elo_processor.get_player_name(player_id)
                specialists[surface].append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'rating': surface_rating,
                    'matches': surface_matches,
                    'comparisons': comparison,
                    'avg_diff': sum(c['rating_diff'] for c in comparison) / len(comparison)
                })
    
    # Ordenar especialistas por diferencia media (de mayor a menor)
    for surface in specialists:
        specialists[surface] = sorted(
            specialists[surface], 
            key=lambda x: x['avg_diff'], 
            reverse=True
        )
    
    return specialists

def analyze_rating_trends(elo_processor, period_days: int = 180, 
                       min_matches: int = 5) -> Dict[str, List[Dict]]:
    """
    Analiza tendencias de ratings para identificar jugadores en ascenso o declive.
    
    Args:
        elo_processor: Instancia del procesador ELO
        period_days: Período en días para analizar la tendencia
        min_matches: Mínimo de partidos en el período para ser considerado
        
    Returns:
        Diccionario con jugadores en tendencia ascendente y descendente
    """
    # Convertir historial a DataFrame si no lo está ya
    if isinstance(elo_processor.rating_history, list):
        history_df = pd.DataFrame(elo_processor.rating_history)
    else:
        history_df = elo_processor.rating_history
    
    if history_df.empty:
        return {'rising': [], 'falling': []}
    
    # Asegurar que la fecha es datetime
    if 'date' in history_df.columns and not pd.api.types.is_datetime64_any_dtype(history_df['date']):
        history_df['date'] = pd.to_datetime(history_df['date'])
    
    # Calcular fecha límite para el período
    now = datetime.now()
    period_start = now - timedelta(days=period_days)
    
    # Filtrar partidos en el período
    period_history = history_df[history_df['date'] >= period_start]
    
    if period_history.empty:
        return {'rising': [], 'falling': []}
    
    # Extraer los ratings iniciales y actuales para cada jugador
    player_trends = {}
    
    # Procesar ganadores
    for player_id in set(period_history['winner_id']):
        player_matches = period_history[
            (period_history['winner_id'] == player_id) | 
            (period_history['loser_id'] == player_id)
        ].sort_values('date')
        
        if len(player_matches) < min_matches:
            continue
        
        # Obtener rating inicial y final
        initial_rating = None
        final_rating = None
        
        # Primera aparición
        first_match = player_matches.iloc[0]
        if first_match['winner_id'] == player_id:
            initial_rating = first_match['winner_rating_before']
        else:
            initial_rating = first_match['loser_rating_before']
        
        # Última aparición
        last_match = player_matches.iloc[-1]
        if last_match['winner_id'] == player_id:
            final_rating = last_match['winner_rating_after']
        else:
            final_rating = last_match['loser_rating_after']
        
        # Calcular cambio en el período
        if initial_rating is not None and final_rating is not None:
            rating_change = final_rating - initial_rating
            
            # Añadir a tendencias
            player_trends[player_id] = {
                'player_id': player_id,
                'player_name': elo_processor.get_player_name(player_id),
                'initial_rating': initial_rating,
                'final_rating': final_rating,
                'rating_change': rating_change,
                'percent_change': (rating_change / initial_rating) * 100,
                'num_matches': len(player_matches),
                'period_days': period_days
            }
    
    # Procesar perdedores que no estén ya procesados
    for player_id in set(period_history['loser_id']):
        if player_id in player_trends:
            continue
            
        player_matches = period_history[
            (period_history['winner_id'] == player_id) | 
            (period_history['loser_id'] == player_id)
        ].sort_values('date')
        
        if len(player_matches) < min_matches:
            continue
        
        # Obtener rating inicial y final (similar al código anterior)
        initial_rating = None
        final_rating = None
        
        # Primera aparición
        first_match = player_matches.iloc[0]
        if first_match['winner_id'] == player_id:
            initial_rating = first_match['winner_rating_before']
        else:
            initial_rating = first_match['loser_rating_before']
        
        # Última aparición
        last_match = player_matches.iloc[-1]
        if last_match['winner_id'] == player_id:
            final_rating = last_match['winner_rating_after']
        else:
            final_rating = last_match['loser_rating_after']
        
        # Calcular cambio en el período
        if initial_rating is not None and final_rating is not None:
            rating_change = final_rating - initial_rating
            
            # Añadir a tendencias
            player_trends[player_id] = {
                'player_id': player_id,
                'player_name': elo_processor.get_player_name(player_id),
                'initial_rating': initial_rating,
                'final_rating': final_rating,
                'rating_change': rating_change,
                'percent_change': (rating_change / initial_rating) * 100,
                'num_matches': len(player_matches),
                'period_days': period_days
            }
    
    # Separar en tendencias ascendentes y descendentes
    rising = []
    falling = []
    
    for player_id, trend in player_trends.items():
        if trend['rating_change'] > 0:
            rising.append(trend)
        else:
            falling.append(trend)
    
    # Ordenar por magnitud del cambio
    rising = sorted(rising, key=lambda x: x['rating_change'], reverse=True)
    falling = sorted(falling, key=lambda x: x['rating_change'])
    
    return {
        'rising': rising,
        'falling': falling
    }

def analyze_player_consistency(elo_processor, player_id: str, 
                           min_matches: int = 10) -> Dict:
    """
    Analiza la consistencia de un jugador en función de la volatilidad de su rating.
    
    Args:
        elo_processor: Instancia del procesador ELO
        player_id: ID del jugador a analizar
        min_matches: Mínimo de partidos para considerar el análisis válido
        
    Returns:
        Diccionario con métricas de consistencia
    """
    player_id = str(player_id)
    
    # Verificar que el jugador tiene suficientes partidos
    match_count = elo_processor.player_match_count.get(player_id, 0)
    
    if match_count < min_matches:
        return {
            'player_id': player_id,
            'player_name': elo_processor.get_player_name(player_id),
            'match_count': match_count,
            'status': 'insufficient_data',
            'message': f"El jugador necesita al menos {min_matches} partidos para un análisis de consistencia válido."
        }
    
    # Obtener historial de partidos
    matches = elo_processor.player_match_history.get(player_id, [])
    
    if not matches:
        return {
            'player_id': player_id,
            'player_name': elo_processor.get_player_name(player_id),
            'match_count': 0,
            'status': 'no_match_history',
            'message': "No se encontró historial de partidos para este jugador."
        }
    
    # Extraer cambios de rating
    rating_changes = []
    for match in matches:
        if 'elo_change' in match:
            rating_changes.append(match['elo_change'])
    
    if not rating_changes:
        return {
            'player_id': player_id,
            'player_name': elo_processor.get_player_name(player_id),
            'match_count': len(matches),
            'status': 'no_rating_changes',
            'message': "No se encontraron cambios de rating en el historial."
        }
    
    # Calcular métricas de consistencia
    avg_change = np.mean(rating_changes)
    std_change = np.std(rating_changes)
    volatility = std_change / abs(avg_change) if avg_change != 0 else float('inf')
    
    # Calcular rachas (streak analysis)
    results = [1 if m['result'] == 'win' else 0 for m in matches]
    
    # Identificar rachas
    streaks = []
    current_streak = 1
    current_type = results[0]
    
    for i in range(1, len(results)):
        if results[i] == current_type:
            current_streak += 1
        else:
            streaks.append((current_type, current_streak))
            current_type = results[i]
            current_streak = 1
    
    # Añadir la última racha
    streaks.append((current_type, current_streak))
    
    # Analizar rachas
    win_streaks = [streak[1] for streak in streaks if streak[0] == 1]
    loss_streaks = [streak[1] for streak in streaks if streak[0] == 0]
    
    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_win_streak = np.mean(win_streaks) if win_streaks else 0
    avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0
    
    # Resultados por superficie
    surface_results = {}
    for surface in ['hard', 'clay', 'grass', 'carpet']:
        surface_matches = [m for m in matches if m['surface'] == surface]
        if surface_matches:
            surface_wins = len([m for m in surface_matches if m['result'] == 'win'])
            surface_results[surface] = {
                'matches': len(surface_matches),
                'wins': surface_wins,
                'losses': len(surface_matches) - surface_wins,
                'win_rate': surface_wins / len(surface_matches)
            }
    
    # Construir resultado
    return {
        'player_id': player_id,
        'player_name': elo_processor.get_player_name(player_id),
        'match_count': len(matches),
        'status': 'success',
        'consistency': {
            'avg_rating_change': float(avg_change),
            'std_rating_change': float(std_change),
            'volatility_index': float(volatility),
            'streaks': {
                'max_win_streak': int(max_win_streak),
                'max_loss_streak': int(max_loss_streak),
                'avg_win_streak': float(avg_win_streak),
                'avg_loss_streak': float(avg_loss_streak),
                'streak_counts': {
                    'total': len(streaks),
                    'win_streaks': len(win_streaks),
                    'loss_streaks': len(loss_streaks)
                }
            },
            'surface_performance': surface_results
        }
    }

def analyze_upset_patterns(elo_processor, min_matches: int = 50, 
                        min_elo_diff: float = 75.0) -> Dict:
    """
    Analiza patrones en upsets (victorias sorpresa) para detectar jugadores
    particularmente propensos a causarlos o sufrirlos.
    
    Args:
        elo_processor: Instancia del procesador ELO
        min_matches: Mínimo de partidos para ser incluido en el análisis
        min_elo_diff: Diferencia mínima de ELO para considerar un upset
        
    Returns:
        Diccionario con análisis de patrones de upsets
    """
    # Necesitamos el historial completo
    if isinstance(elo_processor.rating_history, list):
        history_df = pd.DataFrame(elo_processor.rating_history)
    else:
        history_df = elo_processor.rating_history
    
    if history_df.empty:
        return {
            'status': 'no_data',
            'message': "No hay datos de historial para analizar upsets."
        }
    
    # Asegurar que tenemos las columnas necesarias
    required_cols = ['winner_id', 'loser_id', 'winner_rating_before', 'loser_rating_before']
    if not all(col in history_df.columns for col in required_cols):
        return {
            'status': 'missing_columns',
            'message': f"Faltan columnas necesarias: {[col for col in required_cols if col not in history_df.columns]}"
        }
    
    # Calcular diferencia de ELO y marcar upsets
    history_df['elo_diff'] = history_df['loser_rating_before'] - history_df['winner_rating_before']
    history_df['is_upset'] = history_df['elo_diff'] >= min_elo_diff
    
    # Contar upsets totales
    total_upsets = history_df['is_upset'].sum()
    total_matches = len(history_df)
    upset_rate = total_upsets / total_matches
    
    # Análisis por jugador (causando upsets)
    upset_causers = {}
    for player_id in set(history_df['winner_id']):
        # Filtrar victorias de este jugador
        player_wins = history_df[history_df['winner_id'] == player_id]
        
        if len(player_wins) < min_matches / 5:  # Al menos 20% del mínimo general
            continue
        
        # Contar upsets causados
        upsets_caused = player_wins['is_upset'].sum()
        upset_rate_caused = upsets_caused / len(player_wins)
        
        if upsets_caused > 0:
            upset_causers[player_id] = {
                'player_id': player_id,
                'player_name': elo_processor.get_player_name(player_id),
                'total_matches': len(player_wins),
                'upsets_caused': int(upsets_caused),
                'upset_rate': float(upset_rate_caused),
                'relative_rate': float(upset_rate_caused / max(0.001, upset_rate))
            }
    
    # Análisis por jugador (sufriendo upsets)
    upset_victims = {}
    for player_id in set(history_df['loser_id']):
        # Filtrar derrotas de este jugador
        player_losses = history_df[history_df['loser_id'] == player_id]
        
        if len(player_losses) < min_matches / 5:
            continue
        
        # Contar upsets sufridos
        upsets_suffered = player_losses['is_upset'].sum()
        upset_rate_suffered = upsets_suffered / len(player_losses)
        
        if upsets_suffered > 0:
            upset_victims[player_id] = {
                'player_id': player_id,
                'player_name': elo_processor.get_player_name(player_id),
                'total_matches': len(player_losses),
                'upsets_suffered': int(upsets_suffered),
                'upset_rate': float(upset_rate_suffered),
                'relative_rate': float(upset_rate_suffered / max(0.001, upset_rate))
            }
    
    # Análisis por superficie
    surface_upset_rates = {}
    if 'surface' in history_df.columns:
        for surface in history_df['surface'].unique():
            surface_matches = history_df[history_df['surface'] == surface]
            if len(surface_matches) > 0:
                surface_upsets = surface_matches['is_upset'].sum()
                surface_upset_rates[surface] = {
                    'matches': len(surface_matches),
                    'upsets': int(surface_upsets),
                    'upset_rate': float(surface_upsets / len(surface_matches))
                }
    
    # Ordenar jugadores por tasa relativa de upsets
    upset_causers_list = sorted(
        upset_causers.values(), 
        key=lambda x: x['relative_rate'], 
        reverse=True
    )
    
    upset_victims_list = sorted(
        upset_victims.values(), 
        key=lambda x: x['relative_rate'], 
        reverse=True
    )
    
    return {
        'status': 'success',
        'total_matches': total_matches,
        'total_upsets': int(total_upsets),
        'overall_upset_rate': float(upset_rate),
        'upset_causers': upset_causers_list[:20],  # Top 20 causantes
        'upset_victims': upset_victims_list[:20],  # Top 20 víctimas
        'surface_analysis': surface_upset_rates
    }

def analyze_h2h_dominance(elo_processor, min_matches: int = 5) -> List[Dict]:
    """
    Identifica relaciones de dominancia en head-to-head entre jugadores.
    
    Args:
        elo_processor: Instancia del procesador ELO
        min_matches: Mínimo de enfrentamientos para considerar una relación
        
    Returns:
        Lista de diccionarios con relaciones de dominancia
    """
    dominance_relationships = []
    
    # Iterar por todos los registros head-to-head
    for player1_id, opponents in elo_processor.h2h_records.items():
        for player2_id, record in opponents.items():
            # Calcular total de encuentros
            total_matches = record['wins'] + record['losses']
            
            # Solo considerar si hay suficientes enfrentamientos
            if total_matches >= min_matches:
                win_rate = record['wins'] / total_matches
                
                # Solo considerar relaciones de dominancia clara (>70% de victorias)
                if win_rate >= 0.7:
                    # Obtener nombres de jugadores
                    player1_name = elo_processor.get_player_name(player1_id)
                    player2_name = elo_processor.get_player_name(player2_id)
                    
                    # Añadir a la lista de relaciones
                    dominance_relationships.append({
                        'dominant_player_id': player1_id,
                        'dominant_player_name': player1_name,
                        'dominated_player_id': player2_id,
                        'dominated_player_name': player2_name,
                        'total_matches': total_matches,
                        'wins': record['wins'],
                        'losses': record['losses'],
                        'win_rate': win_rate,
                        'dominance_score': win_rate * (1 + min(0.5, (total_matches - min_matches) / 10))
                        # El score incluye un bonus por más enfrentamientos
                    })
    
    # Ordenar por puntuación de dominancia
    dominance_relationships = sorted(
        dominance_relationships,
        key=lambda x: x['dominance_score'],
        reverse=True
    )
    
    return dominance_relationships