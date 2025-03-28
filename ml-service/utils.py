#!/usr/bin/env python3
"""
Módulo de utilidades para el sistema de predicción de tenis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
from datetime import datetime, timedelta
import logging
import os
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import json
import psycopg2
import math

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Definir la ruta del directorio de logs
log_dir = os.path.join(current_dir, 'logs')

# Crear el directorio si no existe
os.makedirs(log_dir, exist_ok=True)

# Configurar logging con ruta absoluta
logging.basicConfig(
    filename=os.path.join(log_dir, 'tennis_ml.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from sklearn.preprocessing import StandardScaler
import math

class TennisFeatureEngineering:
    """
    Clase para generar características avanzadas para el modelo de predicción de tenis
    """
    
    def __init__(self, db_connection_string):
        """
        Inicializa el generador de características
        
        Args:
            db_connection_string: String de conexión a PostgreSQL
        """
        self.conn_string = db_connection_string
    
    def _get_connection(self):
        """Establece conexión con la base de datos"""
        return psycopg2.connect(self.conn_string)
    
    def extract_h2h_features(self, player1_id, player2_id, match_date=None, surface=None):
        """
        Extrae características de historial head-to-head
        
        Args:
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            match_date: Fecha hasta la cual considerar partidos (None = todos)
            surface: Superficie específica (opcional)
            
        Returns:
            Dict con características H2H
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Construir condición de fecha
            date_condition = ""
            params = [player1_id, player2_id, player2_id, player1_id]
            
            if match_date:
                date_condition = "AND date < %s"
                params.extend([match_date, match_date])
            
            # Construir condición de superficie
            surface_condition = ""
            if surface:
                surface_condition = "AND surface = %s"
                params.extend([surface, surface])
            
            # Obtener historial H2H
            cursor.execute(f"""
            WITH h2h_matches AS (
                -- Partidos donde player1 ganó a player2
                SELECT 
                    1 as p1_won,
                    winner_id as winner,
                    loser_id as loser,
                    date,
                    surface,
                    score,
                    w_ace as winner_aces,
                    l_ace as loser_aces,
                    w_df as winner_df,
                    l_df as loser_df,
                    w_bpSaved as winner_bp_saved,
                    w_bpFaced as winner_bp_faced,
                    l_bpSaved as loser_bp_saved,
                    l_bpFaced as loser_bp_faced
                FROM matches
                WHERE winner_id = %s AND loser_id = %s
                {date_condition}
                {surface_condition}
                
                UNION ALL
                
                -- Partidos donde player2 ganó a player1
                SELECT 
                    0 as p1_won,
                    winner_id as winner,
                    loser_id as loser,
                    date,
                    surface,
                    score,
                    w_ace as winner_aces,
                    l_ace as loser_aces,
                    w_df as winner_df,
                    l_df as loser_df,
                    w_bpSaved as winner_bp_saved,
                    w_bpFaced as winner_bp_faced,
                    l_bpSaved as loser_bp_saved,
                    l_bpFaced as loser_bp_faced
                FROM matches
                WHERE winner_id = %s AND loser_id = %s
                {date_condition}
                {surface_condition}
            )
            SELECT 
                COUNT(*) as total_matches,
                SUM(p1_won) as p1_wins,
                SUM(1 - p1_won) as p2_wins,
                -- Últimos 3 partidos (1 para p1_victoria, 0 para p2_victoria)
                ARRAY(SELECT p1_won FROM h2h_matches ORDER BY date DESC LIMIT 3) as last_3_results,
                -- Promedio de estadísticas cuando p1 gana
                AVG(CASE WHEN p1_won = 1 THEN 
                    CASE WHEN winner = %s THEN winner_aces ELSE loser_aces END 
                END) as p1_win_avg_aces,
                AVG(CASE WHEN p1_won = 1 THEN 
                    CASE WHEN winner = %s THEN winner_df ELSE loser_df END 
                END) as p1_win_avg_df,
                AVG(CASE WHEN p1_won = 1 THEN 
                    CASE WHEN winner = %s THEN CAST(winner_bp_saved AS FLOAT) / NULLIF(winner_bp_faced, 0) ELSE CAST(loser_bp_saved AS FLOAT) / NULLIF(loser_bp_faced, 0) END 
                END) as p1_win_avg_bp_saved_pct,
                -- Promedio de estadísticas cuando p2 gana
                AVG(CASE WHEN p1_won = 0 THEN 
                    CASE WHEN winner = %s THEN winner_aces ELSE loser_aces END 
                END) as p2_win_avg_aces,
                AVG(CASE WHEN p1_won = 0 THEN 
                    CASE WHEN winner = %s THEN winner_df ELSE loser_df END 
                END) as p2_win_avg_df,
                AVG(CASE WHEN p1_won = 0 THEN 
                    CASE WHEN winner = %s THEN CAST(winner_bp_saved AS FLOAT) / NULLIF(winner_bp_faced, 0) ELSE CAST(loser_bp_saved AS FLOAT) / NULLIF(loser_bp_faced, 0) END 
                END) as p2_win_avg_bp_saved_pct
            FROM h2h_matches
            """, params + [player1_id, player1_id, player1_id, player2_id, player2_id, player2_id])
            
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                total_matches, p1_wins, p2_wins, last_3_results = result[0:4]
                
                # Calcular porcentaje H2H
                h2h_ratio = p1_wins / total_matches if total_matches > 0 else 0.5
                
                # Calcular tendencia reciente (últimos 3 partidos, ponderados)
                recent_trend = 0
                weights = [0.6, 0.3, 0.1]  # Mayor peso a partidos más recientes
                
                for i, res in enumerate(last_3_results[:min(3, len(last_3_results))]):
                    if i < len(weights):
                        recent_trend += res * weights[i]
                
                # Normalizar si hay menos de 3 partidos
                if len(last_3_results) > 0:
                    recent_trend /= sum(weights[:min(3, len(last_3_results))])
                else:
                    recent_trend = 0.5  # Valor neutral
                
                # Extraer estadísticas promedio
                p1_win_stats = {
                    'aces': result[4] if result[4] is not None else 0,
                    'df': result[5] if result[5] is not None else 0,
                    'bp_saved_pct': result[6] if result[6] is not None else 0
                }
                
                p2_win_stats = {
                    'aces': result[7] if result[7] is not None else 0,
                    'df': result[8] if result[8] is not None else 0,
                    'bp_saved_pct': result[9] if result[9] is not None else 0
                }
                
                return {
                    'h2h_total_matches': total_matches,
                    'h2h_p1_win_ratio': h2h_ratio,
                    'h2h_recent_trend': recent_trend,
                    'h2h_p1_win_stats': p1_win_stats,
                    'h2h_p2_win_stats': p2_win_stats,
                    'h2h_win_diff': p1_wins - p2_wins
                }
            else:
                # No hay historial H2H
                return {
                    'h2h_total_matches': 0,
                    'h2h_p1_win_ratio': 0.5,  # Valor neutral
                    'h2h_recent_trend': 0.5,  # Valor neutral
                    'h2h_p1_win_stats': {'aces': 0, 'df': 0, 'bp_saved_pct': 0},
                    'h2h_p2_win_stats': {'aces': 0, 'df': 0, 'bp_saved_pct': 0},
                    'h2h_win_diff': 0
                }
                
        except Exception as e:
            print(f"Error en extracción de características H2H: {e}")
            # Devolver valores por defecto
            return {
                'h2h_total_matches': 0,
                'h2h_p1_win_ratio': 0.5,
                'h2h_recent_trend': 0.5,
                'h2h_p1_win_stats': {'aces': 0, 'df': 0, 'bp_saved_pct': 0},
                'h2h_p2_win_stats': {'aces': 0, 'df': 0, 'bp_saved_pct': 0},
                'h2h_win_diff': 0
            }
        finally:
            cursor.close()
            conn.close()
    
    def extract_player_style_features(self, player1_id, player2_id):
        """
        Extrae características de estilo de juego y físicas de los jugadores
        
        Args:
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            
        Returns:
            Dict con características de estilo
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Obtener datos físicos y de estilo
            cursor.execute("""
            SELECT 
                id,
                height,
                weight,
                play_style,
                hand,
                backhand,
                turned_pro,
                birth_date
            FROM players
            WHERE id IN (%s, %s)
            """, (player1_id, player2_id))
            
            players_data = {}
            for row in cursor.fetchall():
                player_id, height, weight, play_style, hand, backhand, turned_pro, birth_date = row
                
                players_data[player_id] = {
                    'height': height if height else 183,  # Valor promedio si falta
                    'weight': weight if weight else 78,   # Valor promedio si falta
                    'play_style': play_style if play_style else 'unknown',
                    'hand': hand if hand else 'R',        # Diestro por defecto
                    'backhand': backhand if backhand else 'two-handed',
                    'turned_pro': turned_pro if turned_pro else datetime.now().year - 5,
                    'birth_date': birth_date
                }
            
            # Si falta algún jugador, usar valores por defecto
            for player_id in [player1_id, player2_id]:
                if player_id not in players_data:
                    players_data[player_id] = {
                        'height': 183,
                        'weight': 78,
                        'play_style': 'unknown',
                        'hand': 'R',
                        'backhand': 'two-handed',
                        'turned_pro': datetime.now().year - 5,
                        'birth_date': datetime.now() - timedelta(days=365*25)
                    }
            
            # Calcular diferencias y características derivadas
            height_diff = players_data[player1_id]['height'] - players_data[player2_id]['height']
            weight_diff = players_data[player1_id]['weight'] - players_data[player2_id]['weight']
            
            # Calcular edad actual aproximada
            p1_age = None
            p2_age = None
            
            if players_data[player1_id]['birth_date']:
                p1_age = (datetime.now() - players_data[player1_id]['birth_date']).days / 365.25
            
            if players_data[player2_id]['birth_date']:
                p2_age = (datetime.now() - players_data[player2_id]['birth_date']).days / 365.25
            
            # Valores por defecto si faltan
            p1_age = p1_age if p1_age else 27
            p2_age = p2_age if p2_age else 27
            
            age_diff = p1_age - p2_age
            
            # Calcular años de experiencia
            p1_experience = datetime.now().year - players_data[player1_id]['turned_pro']
            p2_experience = datetime.now().year - players_data[player2_id]['turned_pro']
            experience_diff = p1_experience - p2_experience
            
            # Codificación de mano dominante (1 si igual, 0 si diferente)
            same_handed = int(players_data[player1_id]['hand'] == players_data[player2_id]['hand'])
            
            # Codificación de revés (1 si igual, 0 si diferente)
            same_backhand = int(players_data[player1_id]['backhand'] == players_data[player2_id]['backhand'])
            
            # Obtener estadísticas de servicio/resto
            cursor.execute("""
            WITH player_stats AS (
                -- Estadísticas como ganador
                SELECT
                    winner_id as player_id,
                    AVG(w_ace) as avg_aces,
                    AVG(w_df) as avg_df,
                    AVG(w_svpt) as avg_serve_points,
                    AVG(w_1stIn) as avg_first_serves_in,
                    AVG(w_1stWon) as avg_first_serves_won,
                    AVG(w_2ndWon) as avg_second_serves_won,
                    AVG(w_SvGms) as avg_service_games,
                    AVG(w_bpSaved) as avg_bp_saved,
                    AVG(w_bpFaced) as avg_bp_faced
                FROM matches
                WHERE winner_id IN (%s, %s)
                GROUP BY winner_id
                
                UNION ALL
                
                -- Estadísticas como perdedor
                SELECT
                    loser_id as player_id,
                    AVG(l_ace) as avg_aces,
                    AVG(l_df) as avg_df,
                    AVG(l_svpt) as avg_serve_points,
                    AVG(l_1stIn) as avg_first_serves_in,
                    AVG(l_1stWon) as avg_first_serves_won,
                    AVG(l_2ndWon) as avg_second_serves_won,
                    AVG(l_SvGms) as avg_service_games,
                    AVG(l_bpSaved) as avg_bp_saved,
                    AVG(l_bpFaced) as avg_bp_faced
                FROM matches
                WHERE loser_id IN (%s, %s)
                GROUP BY loser_id
            )
            SELECT
                player_id,
                AVG(avg_aces) as avg_aces,
                AVG(avg_df) as avg_df,
                AVG(CAST(avg_first_serves_in AS FLOAT) / NULLIF(avg_serve_points, 0)) as first_serve_pct,
                AVG(CAST(avg_first_serves_won AS FLOAT) / NULLIF(avg_first_serves_in, 0)) as first_serve_win_pct,
                AVG(CAST(avg_second_serves_won AS FLOAT) / NULLIF(avg_serve_points - avg_first_serves_in, 0)) as second_serve_win_pct,
                AVG(CAST(avg_bp_saved AS FLOAT) / NULLIF(avg_bp_faced, 0)) as bp_saved_pct
            FROM player_stats
            GROUP BY player_id
            """, (player1_id, player2_id, player1_id, player2_id))
            
            # Inicializar estadísticas por defecto
            players_stats = {
                player1_id: {
                    'avg_aces': 0, 'avg_df': 0, 'first_serve_pct': 0,
                    'first_serve_win_pct': 0, 'second_serve_win_pct': 0, 'bp_saved_pct': 0
                },
                player2_id: {
                    'avg_aces': 0, 'avg_df': 0, 'first_serve_pct': 0,
                    'first_serve_win_pct': 0, 'second_serve_win_pct': 0, 'bp_saved_pct': 0
                }
            }
            
            # Actualizar con datos reales si están disponibles
            for row in cursor.fetchall():
                player_id, avg_aces, avg_df, first_serve_pct, first_serve_win_pct, second_serve_win_pct, bp_saved_pct = row
                
                # Corregir valores None
                players_stats[player_id] = {
                    'avg_aces': avg_aces if avg_aces is not None else 0,
                    'avg_df': avg_df if avg_df is not None else 0,
                    'first_serve_pct': first_serve_pct if first_serve_pct is not None else 0,
                    'first_serve_win_pct': first_serve_win_pct if first_serve_win_pct is not None else 0,
                    'second_serve_win_pct': second_serve_win_pct if second_serve_win_pct is not None else 0,
                    'bp_saved_pct': bp_saved_pct if bp_saved_pct is not None else 0
                }
            
            # Calcular diferencias en estadísticas de servicio
            serve_stats_diffs = {
                'aces_diff': players_stats[player1_id]['avg_aces'] - players_stats[player2_id]['avg_aces'],
                'df_diff': players_stats[player1_id]['avg_df'] - players_stats[player2_id]['avg_df'],
                'first_serve_pct_diff': players_stats[player1_id]['first_serve_pct'] - players_stats[player2_id]['first_serve_pct'],
                'first_serve_win_pct_diff': players_stats[player1_id]['first_serve_win_pct'] - players_stats[player2_id]['first_serve_win_pct'],
                'second_serve_win_pct_diff': players_stats[player1_id]['second_serve_win_pct'] - players_stats[player2_id]['second_serve_win_pct'],
                'bp_saved_pct_diff': players_stats[player1_id]['bp_saved_pct'] - players_stats[player2_id]['bp_saved_pct']
            }
            
            # Calcular índice de ventaja de servicio
            p1_serve_index = (players_stats[player1_id]['avg_aces'] * 0.3 + 
                            players_stats[player1_id]['first_serve_pct'] * 0.3 + 
                            players_stats[player1_id]['first_serve_win_pct'] * 0.2 + 
                            players_stats[player1_id]['second_serve_win_pct'] * 0.2 -
                            players_stats[player1_id]['avg_df'] * 0.1)
            
            p2_serve_index = (players_stats[player2_id]['avg_aces'] * 0.3 + 
                            players_stats[player2_id]['first_serve_pct'] * 0.3 + 
                            players_stats[player2_id]['first_serve_win_pct'] * 0.2 + 
                            players_stats[player2_id]['second_serve_win_pct'] * 0.2 -
                            players_stats[player2_id]['avg_df'] * 0.1)
            
            serve_advantage = p1_serve_index - p2_serve_index
            
            # Compilar todas las características de estilo
            style_features = {
                'height_diff': height_diff,
                'weight_diff': weight_diff,
                'age_diff': age_diff,
                'experience_diff': experience_diff,
                'same_handed': same_handed,
                'same_backhand': same_backhand,
                'p1_height': players_data[player1_id]['height'],
                'p2_height': players_data[player2_id]['height'],
                'p1_left_handed': 1 if players_data[player1_id]['hand'] == 'L' else 0,
                'p2_left_handed': 1 if players_data[player2_id]['hand'] == 'L' else 0,
                'serve_advantage': serve_advantage,
                'p1_serve_index': p1_serve_index,
                'p2_serve_index': p2_serve_index
            }
            
            # Añadir estadísticas individuales
            for stat_name, stat_diff in serve_stats_diffs.items():
                style_features[stat_name] = stat_diff
            
            for player_id, stats in players_stats.items():
                prefix = 'p1_' if player_id == player1_id else 'p2_'
                for stat_name, stat_value in stats.items():
                    style_features[f"{prefix}{stat_name}"] = stat_value
            
            return style_features
            
        except Exception as e:
            print(f"Error en extracción de características de estilo: {e}")
            # Devolver valores por defecto
            return {
                'height_diff': 0,
                'weight_diff': 0,
                'age_diff': 0,
                'experience_diff': 0,
                'same_handed': 1,
                'same_backhand': 1,
                'p1_height': 183,
                'p2_height': 183,
                'p1_left_handed': 0,
                'p2_left_handed': 0,
                'serve_advantage': 0,
                'p1_serve_index': 0,
                'p2_serve_index': 0,
                'aces_diff': 0,
                'df_diff': 0,
                'first_serve_pct_diff': 0,
                'first_serve_win_pct_diff': 0,
                'second_serve_win_pct_diff': 0,
                'bp_saved_pct_diff': 0
            }
        finally:
            cursor.close()
            conn.close()
    
    def extract_temporal_features(self, player1_id, player2_id, match_date, surface=None):
        """
        Extrae características temporales (forma reciente, fatiga, momentum)
        
        Args:
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            match_date: Fecha del partido
            surface: Superficie específica (opcional)
            
        Returns:
            Dict con características temporales
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Definir períodos de tiempo para análisis
            date_1m_ago = match_date - timedelta(days=30)
            date_3m_ago = match_date - timedelta(days=90)
            date_6m_ago = match_date - timedelta(days=180)
            date_12m_ago = match_date - timedelta(days=365)
            
            # Condición de superficie
            surface_condition = ""
            if surface:
                surface_condition = f"AND surface = '{surface}'"
            
            # Función auxiliar para calcular características temporales de un jugador
            def get_player_temporal_features(player_id):
                # Forma reciente: últimos 1, 3, 6, 12 meses
                cursor.execute(f"""
                WITH player_matches AS (
                    -- Partidos ganados
                    SELECT 
                        1 as won,
                        date,
                        surface,
                        round
                    FROM matches
                    WHERE winner_id = %s
                    AND date < %s
                    {surface_condition}
                    
                    UNION ALL
                    
                    -- Partidos perdidos
                    SELECT 
                        0 as won,
                        date,
                        surface,
                        round
                    FROM matches
                    WHERE loser_id = %s
                    AND date < %s
                    {surface_condition}
                )
                SELECT
                    -- Forma reciente (últimos 1, 3, 6, 12 meses)
                    SUM(CASE WHEN date >= %s THEN won ELSE 0 END) as wins_1m,
                    COUNT(CASE WHEN date >= %s THEN 1 ELSE NULL END) as matches_1m,
                    
                    SUM(CASE WHEN date >= %s THEN won ELSE 0 END) as wins_3m,
                    COUNT(CASE WHEN date >= %s THEN 1 ELSE NULL END) as matches_3m,
                    
                    SUM(CASE WHEN date >= %s THEN won ELSE 0 END) as wins_6m,
                    COUNT(CASE WHEN date >= %s THEN 1 ELSE NULL END) as matches_6m,
                    
                    SUM(CASE WHEN date >= %s THEN won ELSE 0 END) as wins_12m,
                    COUNT(CASE WHEN date >= %s THEN 1 ELSE NULL END) as matches_12m,
                    
                    -- Últimos 10 partidos (1: victoria, 0: derrota)
                    ARRAY(
                        SELECT won
                        FROM player_matches
                        ORDER BY date DESC
                        LIMIT 10
                    ) as last_10_results
                FROM player_matches
                """, (player_id, match_date, player_id, match_date, 
                      date_1m_ago, date_1m_ago, 
                      date_3m_ago, date_3m_ago, 
                      date_6m_ago, date_6m_ago, 
                      date_12m_ago, date_12m_ago))
                
                result = cursor.fetchone()
                
                if result:
                    wins_1m, matches_1m, wins_3m, matches_3m, wins_6m, matches_6m, wins_12m, matches_12m, last_10_results = result
                    
                    # Calcular win rates (evitar división por cero)
                    win_rate_1m = wins_1m / matches_1m if matches_1m and matches_1m > 0 else 0.5
                    win_rate_3m = wins_3m / matches_3m if matches_3m and matches_3m > 0 else 0.5
                    win_rate_6m = wins_6m / matches_6m if matches_6m and matches_6m > 0 else 0.5
                    win_rate_12m = wins_12m / matches_12m if matches_12m and matches_12m > 0 else 0.5
                    
                    # Partidos jugados recientemente (indicador de actividad)
                    activity_level = matches_1m * 3 + matches_3m
                    
                    # Calcular índice de momentum (ponderación decreciente de resultados recientes)
                    momentum = 0
                    weights = [0.20, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05]  # Suma = 1
                    
                    for i, res in enumerate(last_10_results[:min(10, len(last_10_results))]):
                        if i < len(weights):
                            momentum += res * weights[i]
                    
                    # Normalizar si hay menos de 10 partidos
                    if len(last_10_results) > 0:
                        momentum /= sum(weights[:min(10, len(last_10_results))])
                    else:
                        momentum = 0.5  # Valor neutral
                    
                    # Obtener último partido
                    cursor.execute(f"""
                    WITH player_last_match AS (
                        -- Último partido como ganador
                        SELECT 
                            date,
                            minutes
                        FROM matches
                        WHERE winner_id = %s
                        AND date < %s
                        ORDER BY date DESC
                        LIMIT 1
                        
                        UNION ALL
                        
                        -- Último partido como perdedor
                        SELECT 
                            date,
                            minutes
                        FROM matches
                        WHERE loser_id = %s
                        AND date < %s
                        ORDER BY date DESC
                        LIMIT 1
                    )
                    SELECT 
                        date,
                        minutes
                    FROM player_last_match
                    ORDER BY date DESC
                    LIMIT 1
                    """, (player_id, match_date, player_id, match_date))
                    
                    last_match = cursor.fetchone()
                    
                    # Calcular días de descanso
                    days_since_last_match = 30  # Valor por defecto
                    last_match_duration = 90    # Valor por defecto
                    
                    if last_match and last_match[0]:
                        days_since_last_match = (match_date - last_match[0]).days
                        last_match_duration = last_match[1] if last_match[1] else 90
                    
                    # Índice de fatiga (inverso a días de descanso, ponderado por duración del último partido)
                    fatigue_index = 0
                    if days_since_last_match <= 14:  # Solo considerar fatiga hasta 2 semanas
                        fatigue_base = max(0, (14 - days_since_last_match) / 14)
                        duration_factor = min(1.5, last_match_duration / 90)  # Normalizado respecto a 90 min
                        fatigue_index = fatigue_base * duration_factor
                    
                    return {
                        'win_rate_1m': win_rate_1m,
                        'win_rate_3m': win_rate_3m,
                        'win_rate_6m': win_rate_6m,
                        'win_rate_12m': win_rate_12m,
                        'matches_1m': matches_1m,
                        'matches_3m': matches_3m,
                        'momentum': momentum,
                        'days_since_last_match': days_since_last_match,
                        'fatigue_index': fatigue_index,
                        'activity_level': activity_level
                    }
                else:
                    # No hay datos disponibles
                    return {
                        'win_rate_1m': 0.5,
                        'win_rate_3m': 0.5,
                        'win_rate_6m': 0.5,
                        'win_rate_12m': 0.5,
                        'matches_1m': 0,
                        'matches_3m': 0,
                        'momentum': 0.5,
                        'days_since_last_match': 30,
                        'fatigue_index': 0,
                        'activity_level': 0
                    }
            
            # Obtener características para ambos jugadores
            p1_features = get_player_temporal_features(player1_id)
            p2_features = get_player_temporal_features(player2_id)
            
            # Calcular diferencias
            temporal_features = {}
            
            for key in p1_features.keys():
                p1_value = p1_features[key]
                p2_value = p2_features[key]
                
                # Añadir valores individuales
                temporal_features[f'p1_{key}'] = p1_value
                temporal_features[f'p2_{key}'] = p2_value
                
                # Añadir diferencia
                temporal_features[f'{key}_diff'] = p1_value - p2_value
            
            # Calcular índice de forma combinado (ponderando diferentes períodos)
            temporal_features['p1_form_index'] = (
                p1_features['win_rate_1m'] * 0.50 +
                p1_features['win_rate_3m'] * 0.30 +
                p1_features['win_rate_6m'] * 0.15 +
                p1_features['win_rate_12m'] * 0.05
            )
            
            temporal_features['p2_form_index'] = (
                p2_features['win_rate_1m'] * 0.50 +
                p2_features['win_rate_3m'] * 0.30 +
                p2_features['win_rate_6m'] * 0.15 +
                p2_features['win_rate_12m'] * 0.05
            )
            
            temporal_features['form_index_diff'] = temporal_features['p1_form_index'] - temporal_features['p2_form_index']
            
            return temporal_features
            
        except Exception as e:
            print(f"Error en extracción de características temporales: {e}")
            # Devolver valores por defecto
            return {
                'p1_win_rate_1m': 0.5, 'p2_win_rate_1m': 0.5, 'win_rate_1m_diff': 0,
                'p1_win_rate_3m': 0.5, 'p2_win_rate_3m': 0.5, 'win_rate_3m_diff': 0,
                'p1_win_rate_6m': 0.5, 'p2_win_rate_6m': 0.5, 'win_rate_6m_diff': 0,
                'p1_win_rate_12m': 0.5, 'p2_win_rate_12m': 0.5, 'win_rate_12m_diff': 0,
                'p1_momentum': 0.5, 'p2_momentum': 0.5, 'momentum_diff': 0,
                'p1_form_index': 0.5, 'p2_form_index': 0.5, 'form_index_diff': 0,
                'p1_fatigue_index': 0, 'p2_fatigue_index': 0, 'fatigue_index_diff': 0,
                'p1_days_since_last_match': 30, 'p2_days_since_last_match': 30, 'days_since_last_match_diff': 0
            }
        finally:
            cursor.close()
            conn.close()
    
    def extract_tournament_features(self, tournament_id, player1_id, player2_id, match_date):
        """
        Extrae características específicas del torneo
        
        Args:
            tournament_id: ID del torneo
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            match_date: Fecha del partido
            
        Returns:
            Dict con características del torneo
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Obtener información del torneo
            cursor.execute("""
            SELECT 
                name,
                surface,
                draw_size,
                tourney_level,
                EXTRACT(YEAR FROM date) as year
            FROM tournaments
            WHERE id = %s
            """, (tournament_id,))
            
            tourney_info = cursor.fetchone()
            
            if not tourney_info:
                # Torneo desconocido, usar valores por defecto
                return {
                    'tournament_level_G': 0,
                    'tournament_level_M': 0,
                    'tournament_level_A': 0,
                    'tournament_level_D': 0,
                    'tournament_level_F': 0,
                    'draw_size': 32,
                    'p1_tournament_history': 0.5,
                    'p2_tournament_history': 0.5,
                    'tournament_history_diff': 0
                }
            
            name, surface, draw_size, tourney_level, year = tourney_info
            
            # Codificar nivel del torneo
            tournament_level_features = {
                'tournament_level_G': 1 if tourney_level == 'G' else 0,  # Grand Slam
                'tournament_level_M': 1 if tourney_level == 'M' else 0,  # Masters 1000
                'tournament_level_A': 1 if tourney_level == 'A' else 0,  # ATP 500
                'tournament_level_D': 1 if tourney_level == 'D' else 0,  # ATP 250
                'tournament_level_F': 1 if tourney_level == 'F' else 0,  # Finals
                'draw_size': draw_size if draw_size else 32
            }
            
            # Historial de los jugadores en este torneo
            def get_player_tournament_history(player_id):
                # Buscar partidos anteriores en el mismo torneo
                cursor.execute("""
                WITH tournament_matches AS (
                    -- Partidos ganados en este torneo
                    SELECT 
                        1 as won,
                        date,
                        round
                    FROM matches
                    JOIN tournaments ON matches.tournament_id = tournaments.id
                    WHERE winner_id = %s
                    AND tournaments.name = %s
                    AND date < %s
                    
                    UNION ALL
                    
                    -- Partidos perdidos en este torneo
                    SELECT 
                        0 as won,
                        date,
                        round
                    FROM matches
                    JOIN tournaments ON matches.tournament_id = tournaments.id
                    WHERE loser_id = %s
                    AND tournaments.name = %s
                    AND date < %s
                )
                SELECT
                    COUNT(*) as total_matches,
                    SUM(won) as wins,
                    -- Partidos más recientes tienen más peso
                    SUM(won * (1 / (EXTRACT(YEAR FROM %s) - EXTRACT(YEAR FROM date) + 1))) / 
                    NULLIF(SUM(1 / (EXTRACT(YEAR FROM %s) - EXTRACT(YEAR FROM date) + 1)), 0) as recent_weighted_win_rate
                FROM tournament_matches
                """, (player_id, name, match_date, player_id, name, match_date, match_date, match_date))
                
                result = cursor.fetchone()
                
                if result and result[0] > 0:
                    total_matches, wins, recent_weighted_win_rate = result
                    win_rate = wins / total_matches if total_matches > 0 else 0.5
                    
                    return {
                        'matches_played': total_matches,
                        'win_rate': win_rate,
                        'recent_weighted_win_rate': recent_weighted_win_rate if recent_weighted_win_rate is not None else 0.5
                    }
                else:
                    # Sin historial en este torneo
                    return {
                        'matches_played': 0,
                        'win_rate': 0.5,
                        'recent_weighted_win_rate': 0.5
                    }
            
            # Obtener historial para ambos jugadores
            p1_history = get_player_tournament_history(player1_id)
            p2_history = get_player_tournament_history(player2_id)
            
            # Factores adicionales
            tournament_features = tournament_level_features.copy()
            
            # Añadir historial de torneo
            tournament_features['p1_tournament_matches'] = p1_history['matches_played']
            tournament_features['p2_tournament_matches'] = p2_history['matches_played']
            tournament_features['p1_tournament_win_rate'] = p1_history['win_rate']
            tournament_features['p2_tournament_win_rate'] = p2_history['win_rate']
            tournament_features['p1_tournament_recent_win_rate'] = p1_history['recent_weighted_win_rate']
            tournament_features['p2_tournament_recent_win_rate'] = p2_history['recent_weighted_win_rate']
            
            # Diferencias
            tournament_features['tournament_win_rate_diff'] = p1_history['win_rate'] - p2_history['win_rate']
            tournament_features['tournament_recent_win_rate_diff'] = p1_history['recent_weighted_win_rate'] - p2_history['recent_weighted_win_rate']
            tournament_features['tournament_experience_diff'] = p1_history['matches_played'] - p2_history['matches_played']
            
            return tournament_features
            
        except Exception as e:
            print(f"Error en extracción de características del torneo: {e}")
            # Devolver valores por defecto
            return {
                'tournament_level_G': 0,
                'tournament_level_M': 0,
                'tournament_level_A': 0,
                'tournament_level_D': 0,
                'tournament_level_F': 0,
                'draw_size': 32,
                'p1_tournament_win_rate': 0.5,
                'p2_tournament_win_rate': 0.5,
                'tournament_win_rate_diff': 0
            }
        finally:
            cursor.close()
            conn.close()
    
    def generate_complete_feature_set(self, match_data):
        """
        Genera conjunto completo de características para un partido
        
        Args:
            match_data: Dict con datos del partido (player1_id, player2_id, date, tournament_id, surface)
            
        Returns:
            Dict con todas las características
        """
        # Extraer datos principales
        player1_id = match_data['player1_id']
        player2_id = match_data['player2_id']
        match_date = match_data['date']
        tournament_id = match_data.get('tournament_id')
        surface = match_data.get('surface')
        
        # Recopilar todas las características
        features = {}
        
        # Características de H2H
        h2h_features = self.extract_h2h_features(player1_id, player2_id, match_date, surface)
        features.update(h2h_features)
        
        # Características de estilo
        style_features = self.extract_player_style_features(player1_id, player2_id)
        features.update(style_features)
        
        # Características temporales
        temporal_features = self.extract_temporal_features(player1_id, player2_id, match_date, surface)
        features.update(temporal_features)
        
        # Características del torneo
        if tournament_id:
            tournament_features = self.extract_tournament_features(tournament_id, player1_id, player2_id, match_date)
            features.update(tournament_features)
        
        # Añadir datos específicos del partido
        features['surface_hard'] = 1 if surface == 'hard' else 0
        features['surface_clay'] = 1 if surface == 'clay' else 0
        features['surface_grass'] = 1 if surface == 'grass' else 0
        features['surface_carpet'] = 1 if surface == 'carpet' else 0
        
        # Información temporal
        features['month'] = match_date.month
        features['year'] = match_date.year
        features['day_of_week'] = match_date.weekday()
        
        return features
    
    def prepare_features_batch(self, matches_df):
        """
        Prepara características para un lote de partidos
        
        Args:
            matches_df: DataFrame con partidos (debe tener player1_id, player2_id, date, etc.)
            
        Returns:
            DataFrame con características completas
        """
        all_features = []
        
        # Procesar cada partido
        for idx, row in matches_df.iterrows():
            # Convertir fila a diccionario
            match_data = row.to_dict()
            
            # Asegurar que los campos necesarios estén presentes
            if 'winner_id' in match_data and 'loser_id' in match_data:
                match_data['player1_id'] = match_data['winner_id']
                match_data['player2_id'] = match_data['loser_id']
            
            # Obtener características
            features = self.generate_complete_feature_set(match_data)
            
            # Añadir ID del partido si existe
            if 'match_id' in match_data:
                features['match_id'] = match_data['match_id']
            
            # Añadir resultado si existe
            if 'winner_id' in match_data and 'player1_id' in match_data:
                features['target'] = 1 if match_data['winner_id'] == match_data['player1_id'] else 0
            
            all_features.append(features)
        
        # Convertir a DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Llenar valores faltantes
        features_df = features_df.fillna(0)
        
        return features_df
    
    def normalize_features(self, features_df, scaler=None, fit=True):
        """
        Normaliza características numéricas
        
        Args:
            features_df: DataFrame con características
            scaler: Scaler preentrenado (opcional)
            fit: Indica si entrenar el scaler (True) o usar uno existente (False)
            
        Returns:
            Tupla (DataFrame normalizado, scaler)
        """
        # Hacer copia para no modificar el original
        df = features_df.copy()
        
        # Separar características categóricas
        categorical_cols = [col for col in df.columns if 
                           col.startswith('tournament_level_') or 
                           col.startswith('surface_') or
                           col in ['same_handed', 'same_backhand', 'p1_left_handed', 'p2_left_handed']]
        
        # Separar columnas especiales
        special_cols = ['match_id', 'target']
        special_data = df[special_cols].copy() if all(col in df.columns for col in special_cols) else None
        
        # Obtener columnas numéricas
        numeric_cols = [col for col in df.columns if col not in categorical_cols + special_cols]
        
        # Crear o usar scaler
        if scaler is None and fit:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif scaler is not None:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        
        # Restaurar columnas especiales si existen
        if special_data is not None:
            for col in special_cols:
                if col in special_data.columns:
                    df[col] = special_data[col]
        
        return df, scaler
    
    def select_important_features(self, features_df, importance_threshold=0.005):
        """
        Selecciona características importantes según su importancia
        
        Args:
            features_df: DataFrame con características
            importance_threshold: Umbral mínimo de importancia
            
        Returns:
            DataFrame con características seleccionadas
        """
        # Nota: Esta función requiere un modelo entrenado para determinar importancias
        # Aquí se podría implementar una selección basada en correlación, varianza, etc.
        
        # Características que sabemos son importantes basadas en análisis previo
        important_features = [
            # ELO (más importantes)
            'player1_surface_elo', 'player2_surface_elo', 'surface_elo_difference',
            'player1_elo', 'player2_elo', 'elo_difference',
            
            # Rendimiento reciente
            'p1_form_index', 'p2_form_index', 'form_index_diff',
            'p1_momentum', 'p2_momentum', 'momentum_diff',
            
            # Físicas y estilo
            'height_diff', 'serve_advantage', 'p1_left_handed', 'p2_left_handed',
            
            # H2H
            'h2h_p1_win_ratio', 'h2h_recent_trend',
            
            # Torneo
            'tournament_level_G', 'tournament_level_M', 'tournament_win_rate_diff',
            
            # Superficie
            'surface_hard', 'surface_clay', 'surface_grass', 'surface_carpet',
            
            # Temporales
            'month', 'year', 'day_of_week'
        ]
        
        # Asegurar que solo usamos columnas que existen
        available_features = [col for col in important_features if col in features_df.columns]
        
        # Añadir columnas especiales
        if 'match_id' in features_df.columns:
            available_features.append('match_id')
        if 'target' in features_df.columns:
            available_features.append('target')
        
        return features_df[available_features]
# Funciones mejoradas para compatibilidad con el código antiguo
def load_model():
    """Carga el modelo entrenado desde el archivo"""
    model_paths = [
        'ml-service/model/model.pkl',
        'model/model.pkl',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model.pkl')
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            logging.info(f"Modelo encontrado en {path}")
            return joblib.load(path)
    
    raise FileNotFoundError(f"Modelo no encontrado. Rutas probadas: {model_paths}")

def preprocess_match_data(match_data):
    """
    Función de compatibilidad con el código antiguo.
    Ahora utiliza la nueva clase TennisFeatureEngineering.
    """
    # Instanciar el motor de características
    fe = TennisFeatureEngineering()
    
    # Asegurarse de que tenemos estadísticas calculadas
    if not fe.players_stats:
        try:
            fe.build_player_statistics()
            fe.build_head_to_head_statistics()
        except Exception as e:
            logging.warning(f"No se pudieron calcular estadísticas: {e}")
    
    # Extraer características avanzadas
    try:
        df = fe.transform_match_data(match_data)
        # Convertir a diccionario para compatibilidad
        features_dict = df.iloc[0].to_dict()
        return features_dict
    except Exception as e:
        logging.warning(f"Error con características avanzadas, usando método básico: {e}")
        
        # Si falla, intentamos con un método muy básico pero que siempre funcione
        features = {}
        
        # Ranking
        if 'ranking_1' in match_data:
            features['ranking_1'] = match_data['ranking_1']
        else:
            # Buscar ranking en bases de datos externas
            player = match_data.get('player_1', match_data.get('player1'))
            if player:
                # Inicializar motor para buscar datos
                fe = TennisFeatureEngineering()
                ranking = fe._get_external_ranking(player)
                features['ranking_1'] = ranking if ranking is not None else 100
            else:
                features['ranking_1'] = 100
        
        if 'ranking_2' in match_data:
            features['ranking_2'] = match_data['ranking_2']
        else:
            # Buscar ranking en bases de datos externas
            player = match_data.get('player_2', match_data.get('player2'))
            if player:
                # Inicializar motor para buscar datos
                fe = TennisFeatureEngineering()
                ranking = fe._get_external_ranking(player)
                features['ranking_2'] = ranking if ranking is not None else 100
            else:
                features['ranking_2'] = 100
        
        # Winrate - intentar obtener de datos externos
        if 'winrate_1' in match_data:
            features['winrate_1'] = match_data['winrate_1']
        else:
            # Usamos 45 como valor neutro, ligeramente por debajo de la media
            features['winrate_1'] = 45
        
        if 'winrate_2' in match_data:
            features['winrate_2'] = match_data['winrate_2']
        else:
            features['winrate_2'] = 45
        
        # Superficie
        if 'surface' in match_data:
            surfaces = {'hard': 0, 'clay': 1, 'grass': 2, 'carpet': 3}
            features['surface_code'] = surfaces.get(match_data['surface'].lower(), 0)
        else:
            features['surface_code'] = 0  # Hard como superficie más común
            
        return features

class PlayerStatsManager:
    """
    Clase para gestionar estadísticas de jugadores basadas en datos históricos.
    """
    
    def __init__(self, data_path='data/tennis_matches.csv'):
        """
        Inicializa el gestor de estadísticas de jugadores.
        
        Args:
            data_path: Ruta al archivo CSV con datos históricos
        """
        self.data_path = data_path
        self.player_stats = {}
        self.head_to_head = {}
        self.surface_stats = {}
        self._load_data()
    
    def _load_data(self):
        """Carga los datos históricos y calcula estadísticas."""
        try:
            if not os.path.exists(self.data_path):
                logging.warning(f"No se encontró el archivo de datos: {self.data_path}")
                return
                
            logging.info(f"Cargando datos históricos desde {self.data_path}")
            data = pd.read_csv(self.data_path)
            logging.info(f"Datos cargados: {len(data)} partidos")
            
            # Calcular estadísticas generales por jugador
            self._calculate_player_stats(data)
            
            # Calcular estadísticas head-to-head
            self._calculate_head_to_head(data)
            
            # Calcular estadísticas por superficie
            self._calculate_surface_stats(data)
            
            logging.info(f"Estadísticas calculadas para {len(self.player_stats)} jugadores")
            
        except Exception as e:
            logging.error(f"Error cargando datos históricos: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_player_stats(self, data):
        """Calcula estadísticas generales por jugador."""
        players = set(data['player_1'].tolist() + data['player_2'].tolist())
        
        for player in players:
            # Partidos como player_1
            p1_matches = data[data['player_1'] == player]
            p1_wins = len(p1_matches[p1_matches['winner'] == 0])
            
            # Partidos como player_2
            p2_matches = data[data['player_2'] == player]
            p2_wins = len(p2_matches[p2_matches['winner'] == 1])
            
            # Total de partidos y victorias
            total_matches = len(p1_matches) + len(p2_matches)
            total_wins = p1_wins + p2_wins
            
            # Calcular tasa de victoria
            winrate = (total_wins / total_matches * 100) if total_matches > 0 else 50.0
            
            # Guardar estadísticas
            self.player_stats[player] = {
                'total_matches': total_matches,
                'total_wins': total_wins,
                'winrate': winrate
            }
            
            # Obtener ranking promedio (si está disponible)
            if 'ranking_1' in data.columns and 'ranking_2' in data.columns:
                rankings = []
                for _, row in p1_matches.iterrows():
                    if pd.notna(row['ranking_1']):
                        rankings.append(row['ranking_1'])
                for _, row in p2_matches.iterrows():
                    if pd.notna(row['ranking_2']):
                        rankings.append(row['ranking_2'])
                
                avg_ranking = sum(rankings) / len(rankings) if rankings else 100.0
                self.player_stats[player]['avg_ranking'] = avg_ranking
            else:
                self.player_stats[player]['avg_ranking'] = 100.0
    
    def _calculate_head_to_head(self, data):
        """Calcula estadísticas head-to-head entre jugadores."""
        # Crear un diccionario para almacenar partidos entre pares de jugadores
        player_pairs = {}
        
        for _, row in data.iterrows():
            player1 = row['player_1']
            player2 = row['player_2']
            winner = row['winner']
            
            # Clave normalizada (orden alfabético)
            key = tuple(sorted([player1, player2]))
            
            if key not in player_pairs:
                player_pairs[key] = {'total': 0, 'player1_wins': 0, 'player2_wins': 0}
            
            player_pairs[key]['total'] += 1
            
            if winner == 0:  # player_1 ganó
                if player1 == key[0]:
                    player_pairs[key]['player1_wins'] += 1
                else:
                    player_pairs[key]['player2_wins'] += 1
            else:  # player_2 ganó
                if player2 == key[0]:
                    player_pairs[key]['player1_wins'] += 1
                else:
                    player_pairs[key]['player2_wins'] += 1
        
        # Guardar estadísticas head-to-head
        self.head_to_head = player_pairs
    
    def _calculate_surface_stats(self, data):
        """Calcula estadísticas por superficie para cada jugador."""
        players = set(data['player_1'].tolist() + data['player_2'].tolist())
        surfaces = data['surface'].unique()
        
        # Verificar si el dataset tiene columnas de winrate por superficie precalculadas
        has_precalculated = 'surface_winrate_1' in data.columns and 'surface_winrate_2' in data.columns
        
        for player in players:
            self.surface_stats[player] = {}
            
            for surface in surfaces:
                # Partidos en esta superficie como player_1
                p1_surface_matches = data[(data['player_1'] == player) & (data['surface'] == surface)]
                p1_surface_wins = len(p1_surface_matches[p1_surface_matches['winner'] == 0])
                
                # Partidos en esta superficie como player_2
                p2_surface_matches = data[(data['player_2'] == player) & (data['surface'] == surface)]
                p2_surface_wins = len(p2_surface_matches[p2_surface_matches['winner'] == 1])
                
                # Total de partidos y victorias en esta superficie
                total_surface_matches = len(p1_surface_matches) + len(p2_surface_matches)
                total_surface_wins = p1_surface_wins + p2_surface_wins
                
                # Calcular tasa de victoria en esta superficie
                if total_surface_matches > 0:
                    surface_winrate = (total_surface_wins / total_surface_matches * 100)
                elif has_precalculated:
                    # Si hay datos precalculados en el CSV y no tenemos partidos,
                    # intentamos usar esos valores precalculados
                    surface_winrates = []
                    
                    for _, row in p1_surface_matches.iterrows():
                        if 'surface_winrate_1' in row and pd.notna(row['surface_winrate_1']):
                            surface_winrates.append(row['surface_winrate_1'])
                    
                    for _, row in p2_surface_matches.iterrows():
                        if 'surface_winrate_2' in row and pd.notna(row['surface_winrate_2']):
                            surface_winrates.append(row['surface_winrate_2'])
                    
                    if surface_winrates:
                        surface_winrate = sum(surface_winrates) / len(surface_winrates)
                        # Estimar el número de partidos basado en el winrate
                        # Asumimos al menos 5 partidos si tenemos una tasa de victoria
                        total_surface_matches = 5
                        total_surface_wins = round((surface_winrate / 100) * total_surface_matches)
                    else:
                        surface_winrate = 50.0
                else:
                    surface_winrate = 50.0
                
                # Guardar estadísticas
                self.surface_stats[player][surface] = {
                    'matches': total_surface_matches,
                    'wins': total_surface_wins,
                    'winrate': surface_winrate
                }
    
    def get_player_stats(self, player_name):
        """
        Obtiene estadísticas de un jugador.
        
        Args:
            player_name: Nombre del jugador
            
        Returns:
            Diccionario con estadísticas del jugador o valores predeterminados
        """
        if player_name in self.player_stats:
            return self.player_stats[player_name]
        else:
            # Valores predeterminados para jugador desconocido
            return {
                'total_matches': 0,
                'total_wins': 0,
                'winrate': 50.0,
                'avg_ranking': 100.0
            }
    
    def get_surface_stats(self, player_name, surface):
        """
        Obtiene estadísticas de un jugador en una superficie específica.
        
        Args:
            player_name: Nombre del jugador
            surface: Superficie (clay, hard, grass, carpet)
            
        Returns:
            Diccionario con estadísticas o valores predeterminados
        """
        try:
            if os.path.exists(self.data_path):
                data = pd.read_csv(self.data_path)
                
                # Normalizar el nombre de la superficie
                surface = surface.capitalize()
                
                # Filtrar partidos de este jugador en esta superficie
                p1_matches = data[(data['player_1'] == player_name) & (data['surface'] == surface)]
                p2_matches = data[(data['player_2'] == player_name) & (data['surface'] == surface)]
                
                # Contar victorias y partidos reales
                p1_wins = len(p1_matches[p1_matches['winner'] == 0])
                p2_wins = len(p2_matches[p2_matches['winner'] == 1])
                total_matches = len(p1_matches) + len(p2_matches)
                total_wins = p1_wins + p2_wins
                
                # Obtener winrate precalculado del CSV
                surface_winrates = []
                
                # Buscar winrate en partidos como player_1
                for _, row in p1_matches.iterrows():
                    if pd.notna(row['surface_winrate_1']):
                        surface_winrates.append(row['surface_winrate_1'])
                
                # Buscar winrate en partidos como player_2
                for _, row in p2_matches.iterrows():
                    if pd.notna(row['surface_winrate_2']):
                        surface_winrates.append(row['surface_winrate_2'])
                
                # Si encontramos winrates precalculados, usar el promedio
                if surface_winrates:
                    winrate = sum(surface_winrates) / len(surface_winrates)
                else:
                    # Si no hay winrate precalculado, usar el winrate general del jugador
                    general_stats = self.get_player_stats(player_name)
                    winrate = general_stats['winrate']
                
                return {
                    'matches': total_matches,  # Número real de partidos
                    'wins': total_wins,        # Número real de victorias
                    'winrate': winrate
                }
                
        except Exception as e:
            logging.warning(f"Error obteniendo estadísticas de superficie desde CSV: {e}")
            import traceback
            traceback.print_exc()
        
        # Si hay algún error o no encontramos datos, devolver 0 en lugar de valores estimados
        return {
            'matches': 0,  # Si no hay datos, mostrar 0 partidos
            'wins': 0,     # Si no hay datos, mostrar 0 victorias
            'winrate': 50.0  # Valor neutro para winrate
        }
    
    def get_head_to_head(self, player1, player2):
        """
        Obtiene estadísticas head-to-head entre dos jugadores.
        
        Args:
            player1: Nombre del primer jugador
            player2: Nombre del segundo jugador
            
        Returns:
            Diccionario con estadísticas head-to-head
        """
        key = tuple(sorted([player1, player2]))
        
        if key in self.head_to_head:
            h2h = self.head_to_head[key]
            
            # Determinar qué jugador es cuál en las estadísticas
            if player1 == key[0]:
                p1_wins = h2h['player1_wins']
                p2_wins = h2h['player2_wins']
            else:
                p1_wins = h2h['player2_wins']
                p2_wins = h2h['player1_wins']
            
            total = h2h['total']
            
            return {
                'total_matches': total,
                'player1_wins': p1_wins,
                'player2_wins': p2_wins,
                'player1_win_pct': (p1_wins / total * 100) if total > 0 else 50.0,
                'player2_win_pct': (p2_wins / total * 100) if total > 0 else 50.0
            }
        else:
            # Valores predeterminados si no hay enfrentamientos previos
            return {
                'total_matches': 0,
                'player1_wins': 0,
                'player2_wins': 0,
                'player1_win_pct': 50.0,
                'player2_win_pct': 50.0
            }
    
    def prepare_prediction_features(self, player1, player2, surface):
        """
        Prepara características para predicción basadas en estadísticas históricas.
        
        Args:
            player1: Nombre del primer jugador
            player2: Nombre del segundo jugador
            surface: Superficie del partido
            
        Returns:
            DataFrame con características para predicción
        """
        # Obtener estadísticas
        p1_stats = self.get_player_stats(player1)
        p2_stats = self.get_player_stats(player2)
        p1_surface_stats = self.get_surface_stats(player1, surface)
        p2_surface_stats = self.get_surface_stats(player2, surface)
        
        # Características básicas
        ranking_1 = p1_stats['avg_ranking']
        ranking_2 = p2_stats['avg_ranking']
        winrate_1 = p1_stats['winrate']
        winrate_2 = p2_stats['winrate']
        
        # Características derivadas
        ranking_diff = ranking_1 - ranking_2
        absolute_ranking_diff = abs(ranking_diff)
        winrate_diff = winrate_1 - winrate_2
        
        # Características específicas de superficie
        surface_winrate_1 = p1_surface_stats['winrate']
        surface_winrate_2 = p2_surface_stats['winrate']
        surface_winrate_diff = surface_winrate_1 - surface_winrate_2
        
        # Características de interacción
        winrate_ranking_interaction = winrate_diff * ranking_diff
        winrate_ranking_interaction_2 = winrate_1/(ranking_1+1) - winrate_2/(ranking_2+1)
        
        # Crear DataFrame con todas las características
        features = {
            'ranking_1': ranking_1,
            'ranking_2': ranking_2,
            'winrate_1': winrate_1,
            'winrate_2': winrate_2,
            'ranking_diff': ranking_diff,
            'absolute_ranking_diff': absolute_ranking_diff,
            'winrate_diff': winrate_diff,
            'surface_winrate_1': surface_winrate_1,
            'surface_winrate_2': surface_winrate_2,
            'surface_winrate_diff': surface_winrate_diff,
            'winrate_ranking_interaction': winrate_ranking_interaction,
            'winrate_ranking_interaction_2': winrate_ranking_interaction_2
        }
        
        return pd.DataFrame([features])