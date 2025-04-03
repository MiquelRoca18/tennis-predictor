"""
feature_engineering.py

Módulo para la creación y preparación de características para modelos de machine learning
basados en el sistema ELO de tenis. Contiene funciones para extraer, transformar y
generar conjuntos de datos estructurados para entrenamiento y evaluación.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict

# Configurar logging
logger = logging.getLogger(__name__)

def create_features_for_model(elo_processor, matches_df: pd.DataFrame, 
                          include_context: bool = True,
                          include_temporal: bool = True,
                          include_h2h: bool = True,
                          include_surface: bool = True) -> pd.DataFrame:
    """
    Crea características basadas en ELO para entrenar modelos de ML.
    
    Args:
        elo_processor: Instancia del procesador ELO
        matches_df: DataFrame con partidos a procesar
        include_context: Si debe incluir características contextuales
        include_temporal: Si debe incluir características temporales
        include_h2h: Si debe incluir características head-to-head
        include_surface: Si debe incluir características específicas por superficie
        
    Returns:
        DataFrame con características para aprendizaje automático
    """
    logger.info("Creando características para modelo de machine learning...")
    
    # Verificar que tenemos el mínimo de columnas necesarias
    required_columns = ['winner_id', 'loser_id', 'match_date', 'surface']
    
    if not all(col in matches_df.columns for col in required_columns):
        raise ValueError(f"DataFrame debe contener las columnas: {required_columns}")
    
    # Hacer copia para no modificar el original
    df = matches_df.copy()
    
    # Asegurar que match_date es datetime
    if not pd.api.types.is_datetime64_any_dtype(df['match_date']):
        df['match_date'] = pd.to_datetime(df['match_date'])
    
    # Características base: ELO general de ambos jugadores
    logger.info("Añadiendo características de ELO general...")
    
    # Creamos un diccionario para hacer seguimiento a los ratings para cada fecha
    temporal_ratings = {}
    temporal_ratings_surface = {}
    
    # Ordenamos los partidos por fecha
    df = df.sort_values('match_date')
    
    # Características básicas de ELO
    df['p1_elo'] = 0.0
    df['p2_elo'] = 0.0
    df['elo_diff'] = 0.0
    df['p1_win_probability'] = 0.0
    
    # Características específicas por superficie
    if include_surface:
        logger.info("Añadiendo características de ELO por superficie...")
        df['p1_elo_surface'] = 0.0
        df['p2_elo_surface'] = 0.0
        df['elo_surface_diff'] = 0.0
        df['p1_win_probability_surface'] = 0.0
    
    # Características contextuales
    if include_context:
        logger.info("Añadiendo características contextuales...")
        df['p1_uncertainty'] = 0.0
        df['p2_uncertainty'] = 0.0
        df['p1_form'] = 0.0
        df['p2_form'] = 0.0
        df['match_importance'] = 0.0
        
        # Codificar variables categóricas
        if 'tourney_level' in df.columns:
            # One-hot encoding para nivel de torneo
            df['tourney_level_normalized'] = df['tourney_level'].apply(
                elo_processor._normalize_tournament_level
            )
            tourney_dummies = pd.get_dummies(df['tourney_level_normalized'], prefix='tourney')
            df = pd.concat([df, tourney_dummies], axis=1)
        
        if 'round' in df.columns:
            # Mapa ordinal para rondas
            round_rank = {
                'F': 7,       # Final
                'SF': 6,      # Semifinal
                'QF': 5,      # Cuartos de final
                'R16': 4,     # Octavos
                'R32': 3,     # 1/16
                'R64': 2,     # 1/32
                'R128': 1,    # 1/64
                'RR': 4       # Round Robin (similar a octavos)
            }
            df['round_rank'] = df['round'].map(lambda x: round_rank.get(str(x), 0))
    
    # Características de head-to-head
    if include_h2h:
        logger.info("Añadiendo características head-to-head...")
        df['p1_h2h_wins'] = 0
        df['p2_h2h_wins'] = 0
        df['p1_h2h_ratio'] = 0.5
        df['h2h_factor'] = 1.0
        
        if include_surface:
            df['p1_h2h_surface_wins'] = 0
            df['p2_h2h_surface_wins'] = 0
            df['p1_h2h_surface_ratio'] = 0.5
    
    # Características temporales
    if include_temporal:
        logger.info("Añadiendo características temporales...")
        df['p1_days_since_last_match'] = 0
        df['p2_days_since_last_match'] = 0
        df['p1_matches_last_90days'] = 0
        df['p2_matches_last_90days'] = 0
        df['p1_win_ratio_last_90days'] = 0.5
        df['p2_win_ratio_last_90days'] = 0.5
        
        # Diccionarios para seguimiento temporal
        last_match_date = {}
        match_history_90days = defaultdict(list)
    
    # Iterar por los partidos cronológicamente
    logger.info(f"Procesando {len(df)} partidos para crear características...")
    
    for idx, match in df.iterrows():
        match_date = match['match_date']
        p1_id = str(match['winner_id'])
        p2_id = str(match['loser_id'])
        surface = elo_processor._normalize_surface(match['surface'])
        
        # Obtener ratings desde diccionario temporal o usar el inicial
        p1_elo = temporal_ratings.get(p1_id, elo_processor.initial_rating)
        p2_elo = temporal_ratings.get(p2_id, elo_processor.initial_rating)
        
        # Guardar features de ELO general
        df.at[idx, 'p1_elo'] = p1_elo
        df.at[idx, 'p2_elo'] = p2_elo
        df.at[idx, 'elo_diff'] = p1_elo - p2_elo
        
        # Calcular probabilidad usando la fórmula ELO
        p1_win_prob = 1.0 / (1.0 + 10.0 ** ((p2_elo - p1_elo) / 400.0))
        df.at[idx, 'p1_win_probability'] = p1_win_prob
        
        # Características específicas por superficie
        if include_surface:
            # Inicializar ratings por superficie si no existen
            if p1_id not in temporal_ratings_surface:
                temporal_ratings_surface[p1_id] = {}
            if p2_id not in temporal_ratings_surface:
                temporal_ratings_surface[p2_id] = {}
            
            # Ratings por superficie, usar el general si no hay específico
            p1_elo_surface = temporal_ratings_surface[p1_id].get(surface, p1_elo)
            p2_elo_surface = temporal_ratings_surface[p2_id].get(surface, p2_elo)
            
            df.at[idx, 'p1_elo_surface'] = p1_elo_surface
            df.at[idx, 'p2_elo_surface'] = p2_elo_surface
            df.at[idx, 'elo_surface_diff'] = p1_elo_surface - p2_elo_surface
            
            # Probabilidad específica por superficie
            p1_win_prob_surface = 1.0 / (1.0 + 10.0 ** ((p2_elo_surface - p1_elo_surface) / 400.0))
            df.at[idx, 'p1_win_probability_surface'] = p1_win_prob_surface
        
        # Características contextuales adicionales
        if include_context:
            # Características de incertidumbre
            p1_matches = sum(1 for h in elo_processor.player_match_history.get(p1_id, []) 
                        if h['date'] < match_date)
            p2_matches = sum(1 for h in elo_processor.player_match_history.get(p2_id, []) 
                        if h['date'] < match_date)
            
            p1_uncertainty = 350 / (p1_matches + 5)
            p2_uncertainty = 350 / (p2_matches + 5)
            
            df.at[idx, 'p1_uncertainty'] = p1_uncertainty
            df.at[idx, 'p2_uncertainty'] = p2_uncertainty
            
            # Factores de forma
            # Calcular forma basada en partidos previos
            p1_recent_matches = [m for m in elo_processor.player_match_history.get(p1_id, []) 
                            if m['date'] < match_date]
            p2_recent_matches = [m for m in elo_processor.player_match_history.get(p2_id, []) 
                            if m['date'] < match_date]
            
            # Tomar los últimos N partidos
            window = elo_processor.form_window_matches
            p1_recent_form = [1 if m['result'] == 'win' else 0 
                        for m in sorted(p1_recent_matches, key=lambda x: x['date'], reverse=True)[:window]]
            p2_recent_form = [1 if m['result'] == 'win' else 0 
                        for m in sorted(p2_recent_matches, key=lambda x: x['date'], reverse=True)[:window]]
            
            # Calcular forma ponderada
            p1_form = 1.0  # Valor neutral
            p2_form = 1.0
            
            if p1_recent_form:
                # Dar más peso a partidos recientes
                weights = [1.5**i for i in range(len(p1_recent_form))]
                p1_form = sum(f*w for f, w in zip(p1_recent_form, weights)) / sum(weights)
                p1_form = 0.8 + (p1_form * 0.4)  # Mapear a rango 0.8-1.2
            
            if p2_recent_form:
                weights = [1.5**i for i in range(len(p2_recent_form))]
                p2_form = sum(f*w for f, w in zip(p2_recent_form, weights)) / sum(weights)
                p2_form = 0.8 + (p2_form * 0.4)  # Mapear a rango 0.8-1.2
            
            df.at[idx, 'p1_form'] = p1_form
            df.at[idx, 'p2_form'] = p2_form
            
            # Importancia del partido
            if 'tourney_level' in df.columns and 'round' in df.columns:
                tourney_level = match['tourney_level']
                round_name = match['round']
                
                # Usar las funciones existentes para calcular importancia
                match_importance = elo_processor._get_match_importance_factor(
                    tourney_level, round_name, p1_id, p2_id
                )
                df.at[idx, 'match_importance'] = match_importance
        
        # Características head-to-head
        if include_h2h:
            # Obtener historial previo a este partido
            p1_h2h_wins = sum(1 for m in elo_processor.player_match_history.get(p1_id, [])
                        if m['date'] < match_date and m['opponent_id'] == p2_id and m['result'] == 'win')
            
            p2_h2h_wins = sum(1 for m in elo_processor.player_match_history.get(p2_id, [])
                        if m['date'] < match_date and m['opponent_id'] == p1_id and m['result'] == 'win')
            
            df.at[idx, 'p1_h2h_wins'] = p1_h2h_wins
            df.at[idx, 'p2_h2h_wins'] = p2_h2h_wins
            
            total_h2h = p1_h2h_wins + p2_h2h_wins
            h2h_ratio = 0.5  # Valor neutral
            
            if total_h2h > 0:
                h2h_ratio = p1_h2h_wins / total_h2h
            
            df.at[idx, 'p1_h2h_ratio'] = h2h_ratio
            
            # Factor de ventaja H2H
            max_factor = min(0.1, 0.05 + (total_h2h * 0.005))
            h2h_factor = 1.0 + ((h2h_ratio - 0.5) * 2 * max_factor)
            df.at[idx, 'h2h_factor'] = h2h_factor
            
            # H2H por superficie
            if include_surface:
                p1_h2h_surface_wins = sum(1 for m in elo_processor.player_match_history.get(p1_id, [])
                                    if m['date'] < match_date and m['opponent_id'] == p2_id 
                                    and m['result'] == 'win' and m['surface'] == surface)
                
                p2_h2h_surface_wins = sum(1 for m in elo_processor.player_match_history.get(p2_id, [])
                                    if m['date'] < match_date and m['opponent_id'] == p1_id 
                                    and m['result'] == 'win' and m['surface'] == surface)
                
                df.at[idx, 'p1_h2h_surface_wins'] = p1_h2h_surface_wins
                df.at[idx, 'p2_h2h_surface_wins'] = p2_h2h_surface_wins
                
                total_h2h_surface = p1_h2h_surface_wins + p2_h2h_surface_wins
                h2h_surface_ratio = 0.5
                
                if total_h2h_surface > 0:
                    h2h_surface_ratio = p1_h2h_surface_wins / total_h2h_surface
                
                df.at[idx, 'p1_h2h_surface_ratio'] = h2h_surface_ratio
        
        # Características temporales
        if include_temporal:
            # Días desde último partido
            p1_last_date = last_match_date.get(p1_id)
            p2_last_date = last_match_date.get(p2_id)
            
            p1_days_since = 30  # Valor por defecto si no hay partidos previos
            p2_days_since = 30
            
            if p1_last_date:
                p1_days_since = (match_date - p1_last_date).days
            
            if p2_last_date:
                p2_days_since = (match_date - p2_last_date).days
            
            df.at[idx, 'p1_days_since_last_match'] = p1_days_since
            df.at[idx, 'p2_days_since_last_match'] = p2_days_since
            
            # Partidos en los últimos 90 días
            date_90days_ago = match_date - pd.Timedelta(days=90)
            
            # Limpiar historial antiguo
            for player_id in list(match_history_90days.keys()):
                match_history_90days[player_id] = [
                    m for m in match_history_90days[player_id] 
                    if m['date'] >= date_90days_ago
                ]
            
            # Contar partidos en los últimos 90 días
            p1_matches_90d = len(match_history_90days.get(p1_id, []))
            p2_matches_90d = len(match_history_90days.get(p2_id, []))
            
            df.at[idx, 'p1_matches_last_90days'] = p1_matches_90d
            df.at[idx, 'p2_matches_last_90days'] = p2_matches_90d
            
            # Calcular ratio de victorias en los últimos 90 días
            p1_wins_90d = sum(1 for m in match_history_90days.get(p1_id, []) if m['result'] == 'win')
            p2_wins_90d = sum(1 for m in match_history_90days.get(p2_id, []) if m['result'] == 'win')
            
            p1_win_ratio_90d = p1_wins_90d / max(1, p1_matches_90d)
            p2_win_ratio_90d = p2_wins_90d / max(1, p2_matches_90d)
            
            df.at[idx, 'p1_win_ratio_last_90days'] = p1_win_ratio_90d
            df.at[idx, 'p2_win_ratio_last_90days'] = p2_win_ratio_90d
        
        # Actualizar ratings para el próximo partido
        # Simulamos el cambio de ELO igual que en update_ratings pero de manera simplificada
        k_factor = elo_processor.k_factor_base
        
        # Probabilidad esperada (ya calculada antes)
        expected_prob = p1_win_prob
        
        # Resultado actual (sabemos que p1 ganó)
        actual_result = 1.0
        
        # Calcular cambios de ELO básicos
        p1_elo_change = k_factor * (actual_result - expected_prob)
        p2_elo_change = -k_factor * expected_prob
        
        # Actualizar ratings temporales
        temporal_ratings[p1_id] = p1_elo + p1_elo_change
        temporal_ratings[p2_id] = p2_elo + p2_elo_change
        
        # Actualizar ratings por superficie
        if include_surface:
            # Actualizar con mayor especificidad para superficie
            surface_mult = elo_processor.surface_specificity.get(surface, 1.0)
            
            if surface not in temporal_ratings_surface[p1_id]:
                temporal_ratings_surface[p1_id][surface] = elo_processor.initial_rating
            if surface not in temporal_ratings_surface[p2_id]:
                temporal_ratings_surface[p2_id][surface] = elo_processor.initial_rating
            
            p1_surface_rating = temporal_ratings_surface[p1_id][surface]
            p2_surface_rating = temporal_ratings_surface[p2_id][surface]
            
            # Usar probabilidad específica por superficie
            expected_prob_surface = p1_win_prob_surface
            
            p1_surface_change = k_factor * surface_mult * (actual_result - expected_prob_surface)
            p2_surface_change = -k_factor * surface_mult * expected_prob_surface
            
            temporal_ratings_surface[p1_id][surface] = p1_surface_rating + p1_surface_change
            temporal_ratings_surface[p2_id][surface] = p2_surface_rating + p2_surface_change
        
        # Actualizar tracking temporal
        if include_temporal:
            last_match_date[p1_id] = match_date
            last_match_date[p2_id] = match_date
            
            # Añadir al historial de 90 días
            match_history_90days[p1_id].append({
                'date': match_date,
                'result': 'win',
                'opponent_id': p2_id
            })
            
            match_history_90days[p2_id].append({
                'date': match_date,
                'result': 'loss',
                'opponent_id': p1_id
            })
    
    # Añadir variable objetivo para modelos de ML
    df['target'] = 1  # El ganador siempre está en la columna p1
    
    # Procesar variables categóricas restantes
    if 'surface' in df.columns:
        # One-hot encoding para superficie
        df['surface_normalized'] = df['surface'].apply(elo_processor._normalize_surface)
        surface_dummies = pd.get_dummies(df['surface_normalized'], prefix='surface')
        df = pd.concat([df, surface_dummies], axis=1)
    
    logger.info(f"Generadas {len(df.columns)} características para modelo de ML")
    
    return df

def create_balanced_dataset(matches_df: pd.DataFrame, 
                         features: List[str], 
                         target_col: str = 'target',
                         test_size: float = 0.2,
                         split_by_date: bool = True,
                         balance_method: str = 'undersample') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Crea un conjunto de datos balanceado para entrenamiento de modelos.
    
    Args:
        matches_df: DataFrame con partidos y características
        features: Lista de características a incluir
        target_col: Nombre de la columna objetivo
        test_size: Fracción de datos para test
        split_by_date: Si es True, divide cronológicamente
        balance_method: Método de balanceo ('undersample', 'oversample', 'none')
        
    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    df = matches_df.copy()
    
    # Asegurar que tenemos las columnas necesarias
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Faltan las siguientes características: {missing_features}")
    
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo {target_col} no encontrada")
    
    # Preparar datos
    X = df[features]
    y = df[target_col]
    
    # División de datos
    if split_by_date and 'match_date' in df.columns:
        # Ordenar por fecha
        if not pd.api.types.is_datetime64_any_dtype(df['match_date']):
            df['match_date'] = pd.to_datetime(df['match_date'])
        df = df.sort_values('match_date')
        
        # Dividir cronológicamente
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]
        y_test = test_df[target_col]
    else:
        # División aleatoria
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    # Balanceo de clases
    if balance_method != 'none':
        if balance_method == 'undersample':
            # Submuestra la clase mayoritaria
            from sklearn.utils import resample
            
            # Encontrar la clase minoritaria
            class_counts = y_train.value_counts()
            min_class = class_counts.idxmin()
            min_count = class_counts.min()
            
            # Separar por clase
            train_data = pd.concat([X_train, y_train], axis=1)
            train_by_class = {cls: train_data[train_data[target_col] == cls] for cls in y_train.unique()}
            
            # Submuestrear clases mayoritarias
            balanced_dfs = []
            for cls, cls_df in train_by_class.items():
                if len(cls_df) > min_count:
                    balanced_df = resample(cls_df, replace=False, n_samples=min_count, random_state=42)
                else:
                    balanced_df = cls_df
                balanced_dfs.append(balanced_df)
            
            # Combinar datos balanceados
            balanced_train = pd.concat(balanced_dfs)
            
            # Actualizar conjuntos de entrenamiento
            X_train = balanced_train[features]
            y_train = balanced_train[target_col]
            
        elif balance_method == 'oversample':
            # Sobremuestrear clase minoritaria
            from sklearn.utils import resample
            
            # Encontrar la clase mayoritaria
            class_counts = y_train.value_counts()
            max_class = class_counts.idxmax()
            max_count = class_counts.max()
            
            # Separar por clase
            train_data = pd.concat([X_train, y_train], axis=1)
            train_by_class = {cls: train_data[train_data[target_col] == cls] for cls in y_train.unique()}
            
            # Sobremuestrear clases minoritarias
            balanced_dfs = []
            for cls, cls_df in train_by_class.items():
                if len(cls_df) < max_count:
                    balanced_df = resample(cls_df, replace=True, n_samples=max_count, random_state=42)
                else:
                    balanced_df = cls_df
                balanced_dfs.append(balanced_df)
            
            # Combinar datos balanceados
            balanced_train = pd.concat(balanced_dfs)
            
            # Actualizar conjuntos de entrenamiento
            X_train = balanced_train[features]
            y_train = balanced_train[target_col]
    
    return X_train, X_test, y_train, y_test

def create_upsets_dataset(elo_processor, matches_df: pd.DataFrame, 
                       min_elo_diff: float = 100.0) -> pd.DataFrame:
    """
    Crea un conjunto de datos especializado para el análisis de upsets (victorias sorpresa).
    
    Args:
        elo_processor: Instancia del procesador ELO
        matches_df: DataFrame con partidos
        min_elo_diff: Diferencia mínima de ELO para considerar un upset
        
    Returns:
        DataFrame con datos de upsets procesados
    """
    df = matches_df.copy()
    
    # Asegurar que tenemos las columnas necesarias
    required_columns = ['winner_id', 'loser_id', 'surface']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame debe contener las columnas: {required_columns}")
    
    # Crear columnas de análisis
    df['winner_elo'] = 0.0
    df['loser_elo'] = 0.0
    df['elo_diff'] = 0.0
    df['expected_win_prob'] = 0.0
    df['is_upset'] = False
    
    # Iterar por partidos
    for idx, match in df.iterrows():
        winner_id = str(match['winner_id'])
        loser_id = str(match['loser_id'])
        surface = elo_processor._normalize_surface(match['surface'])
        
        # Obtener ratings actuales
        winner_elo = elo_processor.get_player_rating(winner_id)
        loser_elo = elo_processor.get_player_rating(loser_id)
        
        # Guardar datos
        df.at[idx, 'winner_elo'] = winner_elo
        df.at[idx, 'loser_elo'] = loser_elo
        df.at[idx, 'elo_diff'] = loser_elo - winner_elo  # Positivo si el perdedor tenía mayor rating
        
        # Calcular probabilidad esperada
        expected_prob = 1.0 / (1.0 + 10.0 ** ((loser_elo - winner_elo) / 400.0))
        df.at[idx, 'expected_win_prob'] = expected_prob
        
        # Determinar si es upset
        df.at[idx, 'is_upset'] = (loser_elo - winner_elo) >= min_elo_diff
    
    # Filtrar solo upsets si se requiere
    upsets_df = df[df['is_upset']].copy()
    
    # Añadir características contextuales
    if 'tourney_level' in df.columns:
        upsets_df['tourney_level_normalized'] = upsets_df['tourney_level'].apply(
            elo_processor._normalize_tournament_level
        )
    
    if 'round' in df.columns:
        # Mapeo de rondas a valores numéricos para análisis
        round_importance = {
            'F': 7,       # Final
            'SF': 6,      # Semifinal
            'QF': 5,      # Cuartos
            'R16': 4,     # Octavos
            'R32': 3,     # 1/16
            'R64': 2,     # 1/32
            'R128': 1,    # 1/64
            'RR': 4       # Round Robin
        }
        upsets_df['round_importance'] = upsets_df['round'].map(
            lambda x: round_importance.get(str(x), 0)
        )
    
    # Añadir h2h previo si está disponible
    if hasattr(elo_processor, 'h2h_records'):
        upsets_df['h2h_prior_wins'] = 0
        upsets_df['h2h_prior_losses'] = 0
        
        # Calcular historial h2h previo
        for idx, match in upsets_df.iterrows():
            winner_id = str(match['winner_id'])
            loser_id = str(match['loser_id'])
            
            if winner_id in elo_processor.h2h_records and loser_id in elo_processor.h2h_records[winner_id]:
                h2h = elo_processor.h2h_records[winner_id][loser_id]
                upsets_df.at[idx, 'h2h_prior_wins'] = h2h.get('wins', 0)
            
            if loser_id in elo_processor.h2h_records and winner_id in elo_processor.h2h_records[loser_id]:
                h2h = elo_processor.h2h_records[loser_id][winner_id]
                upsets_df.at[idx, 'h2h_prior_losses'] = h2h.get('wins', 0)
    
    # Añadir información de superficie si está disponible
    if 'surface' in upsets_df.columns:
        # One-hot encoding para superficie
        upsets_df['surface_normalized'] = upsets_df['surface'].apply(elo_processor._normalize_surface)
        surface_dummies = pd.get_dummies(upsets_df['surface_normalized'], prefix='surface')
        upsets_df = pd.concat([upsets_df, surface_dummies], axis=1)
    
    return upsets_df

def feature_selection(X: pd.DataFrame, y: pd.Series, 
                   method: str = 'importance', 
                   n_features: int = 10) -> List[str]:
    """
    Selecciona las características más importantes usando diferentes métodos.
    
    Args:
        X: DataFrame con características
        y: Series con variable objetivo
        method: Método de selección ('importance', 'correlation', 'mutual_info')
        n_features: Número de características a seleccionar
        
    Returns:
        Lista con los nombres de las características seleccionadas
    """
    if method == 'importance':
        # Usando Random Forest para importancia
        from sklearn.ensemble import RandomForestClassifier
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Obtener importancias
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Seleccionar las n_features más importantes
        selected_features = [X.columns[i] for i in indices[:n_features]]
        
        return selected_features
    
    elif method == 'correlation':
        # Usando correlación con la variable objetivo
        # Solo funciona para características numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Calcular correlaciones
        correlations = {}
        for col in numeric_cols:
            correlations[col] = abs(X[col].corr(y))
        
        # Ordenar por correlación
        sorted_corrs = {k: v for k, v in sorted(correlations.items(), 
                                               key=lambda item: item[1], 
                                               reverse=True)}
        
        # Seleccionar las n_features más correlacionadas
        selected_features = list(sorted_corrs.keys())[:min(n_features, len(sorted_corrs))]
        
        return selected_features
    
    elif method == 'mutual_info':
        # Usando información mutua
        from sklearn.feature_selection import mutual_info_classif
        
        # Calcular información mutua
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_indices = np.argsort(mi_scores)[::-1]
        
        # Seleccionar las n_features con mayor información mutua
        selected_features = [X.columns[i] for i in mi_indices[:n_features]]
        
        return selected_features
    
    else:
        raise ValueError(f"Método de selección '{method}' no reconocido")

def prepare_data_for_sequential_prediction(matches_df: pd.DataFrame, 
                                        features: List[str],
                                        sequence_length: int = 5,
                                        group_by: str = 'player_id',
                                        target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara datos en formato secuencial para modelos recurrentes (RNN, LSTM, etc.)
    
    Args:
        matches_df: DataFrame con partidos y características
        features: Lista de características a incluir
        sequence_length: Longitud de la secuencia para cada muestra
        group_by: Columna para agrupar secuencias (generalmente 'player_id')
        target_col: Columna objetivo
        
    Returns:
        Tupla (X_sequences, y_sequences) con datos procesados
    """
    df = matches_df.copy()
    
    # Asegurar que tenemos las columnas necesarias
    if group_by not in df.columns:
        raise ValueError(f"Columna de agrupación '{group_by}' no encontrada")
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Faltan las siguientes características: {missing_features}")
    
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
    
    # Asegurar ordenamiento cronológico
    if 'match_date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['match_date']):
            df['match_date'] = pd.to_datetime(df['match_date'])
        df = df.sort_values(['match_date', group_by])
    
    # Crear estructura para almacenar secuencias
    X_sequences = []
    y_sequences = []
    
    # Agrupar por jugador o el criterio especificado
    grouped = df.groupby(group_by)
    
    for _, group in grouped:
        # Solo considerar grupos con suficientes partidos
        if len(group) >= sequence_length + 1:  # +1 para tener al menos un objetivo
            for i in range(len(group) - sequence_length):
                # Extraer secuencia
                X_seq = group.iloc[i:i+sequence_length][features].values
                y_seq = group.iloc[i+sequence_length][target_col]
                
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
    
    # Convertir a arrays numpy
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    return X_sequences, y_sequences

def generate_player_features(elo_processor, player_id: str, 
                          match_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Genera un conjunto completo de características para un jugador en un momento dado.
    Útil para API de predicción y análisis de jugadores.
    
    Args:
        elo_processor: Instancia del procesador ELO
        player_id: ID del jugador
        match_date: Fecha para la que generar características (None = actual)
        
    Returns:
        Diccionario con características del jugador
    """
    player_id = str(player_id)
    
    # Si no se especifica fecha, usar la actual
    if match_date is None:
        match_date = datetime.now()
    
    # Características base: ELO y estadísticas generales
    features = {
        'player_id': player_id,
        'player_name': elo_processor.get_player_name(player_id),
        'elo_rating': elo_processor.get_player_rating(player_id),
        'uncertainty': elo_processor.get_player_uncertainty(player_id),
        'match_count': elo_processor.player_match_count.get(player_id, 0),
        'form': elo_processor.get_player_form(player_id)
    }
    
    # Ratings por superficie
    for surface in ['hard', 'clay', 'grass', 'carpet']:
        features[f'elo_{surface}'] = elo_processor.get_player_rating(player_id, surface)
        features[f'matches_{surface}'] = elo_processor.player_match_count_by_surface[surface].get(player_id, 0)
        features[f'form_{surface}'] = elo_processor.get_player_form(player_id, surface)
    
    # Historial de partidos reciente
    matches = elo_processor.player_match_history.get(player_id, [])
    
    # Filtrar por fecha si se especifica
    if match_date:
        matches = [m for m in matches if m['date'] <= match_date]
    
    # Ordenar por fecha (más recientes primero)
    matches = sorted(matches, key=lambda x: x['date'], reverse=True)
    
    # Extraer estadísticas recientes
    if matches:
        # Últimos 10 partidos
        last_10 = matches[:10]
        wins_last_10 = sum(1 for m in last_10 if m['result'] == 'win')
        
        features['wins_last_10'] = wins_last_10
        features['losses_last_10'] = len(last_10) - wins_last_10
        features['win_rate_last_10'] = wins_last_10 / len(last_10)
        
        # Por superficie en últimos 20 partidos
        last_20 = matches[:20]
        for surface in ['hard', 'clay', 'grass', 'carpet']:
            surface_matches = [m for m in last_20 if m['surface'] == surface]
            if surface_matches:
                surface_wins = sum(1 for m in surface_matches if m['result'] == 'win')
                features[f'win_rate_{surface}_recent'] = surface_wins / len(surface_matches)
            else:
                features[f'win_rate_{surface}_recent'] = 0.5  # Valor neutral
        
        # Por nivel de torneo
        for level in ['G', 'M', 'A', 'D', 'F', 'C', 'S', 'O']:
            level_matches = [m for m in matches if m.get('tourney_level') == level]
            if level_matches:
                level_wins = sum(1 for m in level_matches if m['result'] == 'win')
                features[f'win_rate_tourney_{level}'] = level_wins / len(level_matches)
                features[f'matches_tourney_{level}'] = len(level_matches)
            else:
                features[f'win_rate_tourney_{level}'] = 0.5  # Valor neutral
                features[f'matches_tourney_{level}'] = 0
    else:
        # Valores por defecto si no hay partidos
        features['wins_last_10'] = 0
        features['losses_last_10'] = 0
        features['win_rate_last_10'] = 0.5
    
    # Características de tendencia
    if len(matches) >= 2:
        # Calcular tendencia de ELO en los últimos partidos
        recent_elo_changes = []
        
        for match in matches[:10]:  # últimos 10 partidos
            if 'elo_change' in match:
                recent_elo_changes.append(match['elo_change'])
        
        if recent_elo_changes:
            features['elo_trend'] = sum(recent_elo_changes) / len(recent_elo_changes)
        else:
            features['elo_trend'] = 0.0
    else:
        features['elo_trend'] = 0.0
    
    return features