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


class TennisFeatureEngineering:
    """
    Clase para la ingeniería de características en datos de tenis.
    """
    
    def __init__(self, data_path=None):
        """Inicializa la clase con configuraciones por defecto."""
        self.scaler = StandardScaler()
        self.feature_weights = {
            'ranking': 0.15,
            'h2h': 0.15,
            'recent_form': 0.15,
            'surface': 0.15,
            'fatigue': 0.10,
            'physical': 0.10,
            'temporal': 0.10,
            'tournament': 0.10
        }
        self.data_path = data_path
        self.players_stats = {}
        self.head_to_head_stats = {}
        self.tournament_stats = {}
        self.temporal_stats = {}
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae características de los datos de partidos de tenis.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            DataFrame con características calculadas
            
        Raises:
            ValueError: Si el DataFrame está vacío o no tiene el formato esperado
            TypeError: Si el input no es un DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Los datos de entrada deben ser un pandas DataFrame")
            
        if data.empty:
            raise ValueError("El DataFrame está vacío")
            
        try:
            features = pd.DataFrame(index=data.index)
            
            # Lista de características a procesar con sus columnas requeridas
            feature_configs = {
                'ranking': {
                    'columns': ['ranking_1', 'ranking_2'],
                    'features': {
                        'ranking_diff': lambda x: x['ranking_1'] - x['ranking_2'],
                        'ranking_ratio': lambda x: x['ranking_1'] / np.where(x['ranking_2'] == 0, 1, x['ranking_2'])
                    }
                },
                'elo': {
                    'columns': ['elo_winner', 'elo_loser'],
                    'features': {
                        'elo_diff': lambda x: x['elo_winner'] - x['elo_loser'],
                        'elo_ratio': lambda x: x['elo_winner'] / np.where(x['elo_loser'] == 0, 1, x['elo_loser'])
                    }
                },
                'elo_surface': {
                    'columns': ['elo_winner_surface', 'elo_loser_surface'],
                    'features': {
                        'elo_surface_diff': lambda x: x['elo_winner_surface'] - x['elo_loser_surface'],
                        'elo_surface_ratio': lambda x: x['elo_winner_surface'] / np.where(x['elo_loser_surface'] == 0, 1, x['elo_loser_surface'])
                    }
                }
            }
            
            # Procesar cada grupo de características
            for feature_group, config in feature_configs.items():
                if all(col in data.columns for col in config['columns']):
                    for feature_name, feature_func in config['features'].items():
                        try:
                            features[feature_name] = feature_func(data)
                            logging.debug(f"Característica {feature_name} calculada exitosamente")
                        except Exception as e:
                            logging.warning(f"Error calculando {feature_name}: {str(e)}")
                            features[feature_name] = 0
                else:
                    missing_cols = [col for col in config['columns'] if col not in data.columns]
                    logging.warning(f"Columnas faltantes para {feature_group}: {missing_cols}")
            
            # Características simples (una sola columna)
            simple_features = [
                'surface_winrate_diff', 'winrate_diff', 'fatigue_diff',
                'physical_advantage', 'temporal_advantage', 'tournament_experience_diff',
                'h2h_advantage', 'momentum_diff', 'rest_advantage', 'time_of_day_advantage'
            ]
            
            for feature in simple_features:
                if feature in data.columns:
                    features[feature] = data[feature]
                    logging.debug(f"Característica {feature} copiada exitosamente")
            
            # Validar que tengamos al menos una característica
            if features.empty:
                logging.warning("No se encontraron características disponibles. Creando columna dummy.")
                features['dummy'] = 0
            
            # Rellenar valores nulos con la mediana de cada columna
            for col in features.columns:
                if features[col].isnull().any():
                    median_value = features[col].median()
                    features[col] = features[col].fillna(median_value)
                    logging.info(f"Valores nulos en {col} rellenados con la mediana: {median_value}")
            
            # Detectar y manejar valores extremos
            for col in features.columns:
                if col != 'dummy':
                    Q1 = features[col].quantile(0.25)
                    Q3 = features[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((features[col] < lower_bound) | (features[col] > upper_bound))
                    if outliers.any():
                        n_outliers = outliers.sum()
                        features.loc[outliers, col] = features[col].clip(lower_bound, upper_bound)
                        logging.info(f"Se detectaron y corrigieron {n_outliers} valores extremos en {col}")
            
            # Normalizar características
            features = self._normalize_features(features)
            
            # Validación final
            if features.isnull().any().any():
                raise ValueError("Hay valores nulos en las características después del procesamiento")
                
            return features
            
        except Exception as e:
            logging.error(f"Error en la extracción de características: {str(e)}")
            raise
    
    def _calculate_surface_advantage(self, data: pd.DataFrame) -> pd.Series:
        """Calcula la ventaja por superficie."""
        surface_advantage = pd.Series(0, index=data.index)
        for surface in data['surface'].unique():
            mask = data['surface'] == surface
            surface_advantage[mask] = (
                data.loc[mask, 'winner_surface_win_rate'] - 
                data.loc[mask, 'loser_surface_win_rate']
            )
        return surface_advantage
    
    def _calculate_recent_form(self, data: pd.DataFrame) -> pd.Series:
        """Calcula la forma reciente de los jugadores."""
        return (
            data['winner_recent_wins'] / (data['winner_recent_matches'] + 1) -
            data['loser_recent_wins'] / (data['loser_recent_matches'] + 1)
        )
    
    def _calculate_fatigue(self, match_data: dict) -> float:
        """
        Calcula el nivel de fatiga de los jugadores.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            Diferencia de fatiga entre jugadores
        """
        try:
            player1 = match_data.get('player_1', match_data.get('player1', ''))
            player2 = match_data.get('player_2', match_data.get('player2', ''))
            match_date = match_data.get('date')
            
            if not match_date or not player1 or not player2:
                return 0.0
            
            # Obtener estadísticas de partidos recientes
            p1_stats = self.players_stats.get(player1, {})
            p2_stats = self.players_stats.get(player2, {})
            
            # Calcular días de descanso
            p1_last_match = p1_stats.get('last_match_date')
            p2_last_match = p2_stats.get('last_match_date')
            
            if not p1_last_match or not p2_last_match:
                return 0.0
            
            p1_rest = (match_date - p1_last_match).days
            p2_rest = (match_date - p2_last_match).days
            
            # Calcular partidos jugados en los últimos 30 días
            p1_recent_matches = p1_stats.get('recent_matches', [])
            p2_recent_matches = p2_stats.get('recent_matches', [])
            
            p1_matches = len([m for m in p1_recent_matches if (match_date - m['date']).days <= 30])
            p2_matches = len([m for m in p2_recent_matches if (match_date - m['date']).days <= 30])
            
            # Calcular fatiga
            p1_fatigue = p1_matches / (p1_rest + 1)
            p2_fatigue = p2_matches / (p2_rest + 1)
            
            return p1_fatigue - p2_fatigue
            
        except Exception as e:
            logging.error(f"Error calculando fatiga: {str(e)}")
            return 0.0
    
    def _calculate_h2h_advantage(self, data: pd.DataFrame) -> pd.Series:
        """Calcula la ventaja head-to-head."""
        return (
            data['winner_h2h_wins'] / (data['winner_h2h_wins'] + data['loser_h2h_wins'] + 1) -
            data['loser_h2h_wins'] / (data['winner_h2h_wins'] + data['loser_h2h_wins'] + 1)
        )
    
    def _calculate_physical_advantage(self, match_data: dict) -> float:
        """
        Calcula ventajas físicas entre jugadores.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            Ventaja física entre jugadores
        """
        try:
            player1 = match_data.get('player_1', match_data.get('player1', ''))
            player2 = match_data.get('player_2', match_data.get('player2', ''))
            surface = match_data.get('surface', 'hard').lower()
            
            if not player1 or not player2:
                return 0.0
            
            # Obtener estadísticas de jugadores
            p1_stats = self.players_stats.get(player1, {})
            p2_stats = self.players_stats.get(player2, {})
            
            # Ventaja de altura (más importante en superficies rápidas)
            p1_height = p1_stats.get('height', 180)
            p2_height = p2_stats.get('height', 180)
            height_diff = p1_height - p2_height
            
            # Factor de superficie
            surface_speed = {
                'hard': 1.0,
                'clay': 0.5,
                'grass': 1.2,
                'carpet': 0.8
            }.get(surface, 1.0)
            
            # Ventaja de peso (más importante en superficies lentas)
            p1_weight = p1_stats.get('weight', 75)
            p2_weight = p2_stats.get('weight', 75)
            weight_diff = p1_weight - p2_weight
            
            # Calcular ventaja física total
            physical_advantage = (
                height_diff * surface_speed +
                weight_diff * (1 - surface_speed)
            ) / 100  # Normalizar a un rango razonable
            
            return physical_advantage
            
        except Exception as e:
            logging.error(f"Error calculando ventaja física: {str(e)}")
            return 0.0
    
    def _calculate_tournament_experience(self, data: pd.DataFrame) -> pd.Series:
        """Calcula la experiencia en el torneo actual."""
        tournament_experience = pd.Series(0, index=data.index)
        
        for idx, row in data.iterrows():
            tournament = row['tournament']
            winner = row['player_1']
            loser = row['player_2']
            
            # Obtener experiencia en el torneo
            winner_exp = self.tournament_stats.get((tournament, winner), {}).get('matches_played', 0)
            loser_exp = self.tournament_stats.get((tournament, loser), {}).get('matches_played', 0)
            
            # Calcular ventaja de experiencia
            tournament_experience[idx] = winner_exp - loser_exp
            
        return tournament_experience

    def _calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calcula el momentum de los jugadores basado en resultados recientes."""
        momentum = pd.Series(0, index=data.index)
        
        for idx, row in data.iterrows():
            winner = row['player_1']
            loser = row['player_2']
            
            # Obtener resultados recientes
            winner_recent = self.players_stats.get(winner, {}).get('recent_results', [])
            loser_recent = self.players_stats.get(loser, {}).get('recent_results', [])
            
            # Calcular momentum (ponderado por recencia)
            winner_momentum = sum(
                (1 - i/len(winner_recent)) * result 
                for i, result in enumerate(winner_recent)
            ) if winner_recent else 0
            
            loser_momentum = sum(
                (1 - i/len(loser_recent)) * result 
                for i, result in enumerate(loser_recent)
            ) if loser_recent else 0
            
            momentum[idx] = winner_momentum - loser_momentum
            
        return momentum

    def _calculate_seasonal_advantage(self, data: pd.DataFrame) -> pd.Series:
        """Calcula ventajas estacionales basadas en el momento del año."""
        seasonal_advantage = pd.Series(0, index=data.index)
        
        for idx, row in data.iterrows():
            match_date = pd.to_datetime(row['match_date'])
            month = match_date.month
            
            # Obtener preferencias estacionales de los jugadores
            winner_seasonal = self.players_stats.get(row['player_1'], {}).get('seasonal_preferences', {})
            loser_seasonal = self.players_stats.get(row['player_2'], {}).get('seasonal_preferences', {})
            
            # Calcular ventaja estacional
            winner_month_perf = winner_seasonal.get(month, 0.5)
            loser_month_perf = loser_seasonal.get(month, 0.5)
            
            seasonal_advantage[idx] = winner_month_perf - loser_month_perf
            
        return seasonal_advantage

    def _calculate_rest_advantage(self, match_data: dict) -> float:
        """
        Calcula la ventaja de descanso entre jugadores.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            Ventaja de descanso entre jugadores
        """
        try:
            player1 = match_data.get('player_1', match_data.get('player1', ''))
            player2 = match_data.get('player_2', match_data.get('player2', ''))
            match_date = match_data.get('date')
            
            if not match_date or not player1 or not player2:
                return 0.0
            
            # Obtener estadísticas de jugadores
            p1_stats = self.players_stats.get(player1, {})
            p2_stats = self.players_stats.get(player2, {})
            
            # Calcular días de descanso
            p1_last_match = p1_stats.get('last_match_date')
            p2_last_match = p2_stats.get('last_match_date')
            
            if not p1_last_match or not p2_last_match:
                return 0.0
            
            p1_rest = (match_date - p1_last_match).days
            p2_rest = (match_date - p2_last_match).days
            
            # Calcular ventaja de descanso
            rest_advantage = p1_rest - p2_rest
            
            # Normalizar a un rango razonable
            return rest_advantage / 7  # Dividir por 7 para normalizar a semanas
            
        except Exception as e:
            logging.error(f"Error calculando ventaja de descanso: {str(e)}")
            return 0.0
    
    def _calculate_time_of_day_advantage(self, match_data: dict) -> float:
        """
        Calcula la ventaja por hora del día.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            Ventaja por hora del día
        """
        try:
            match_time = match_data.get('time')
            if not match_time:
                return 0.0
            
            # Convertir hora a formato 24h
            hour = int(match_time.split(':')[0])
            
            # Ventajas por hora del día
            # Mañana (8-12): Ventaja para jugadores que prefieren jugar temprano
            # Tarde (12-18): Hora neutral
            # Noche (18-22): Ventaja para jugadores que prefieren jugar tarde
            if 8 <= hour < 12:
                return 0.3  # Ventaja para jugadores de mañana
            elif 12 <= hour < 18:
                return 0.0  # Hora neutral
            else:
                return -0.3  # Ventaja para jugadores de noche
            
        except Exception as e:
            logging.error(f"Error calculando ventaja por hora del día: {str(e)}")
            return 0.0
    
    def _calculate_seasonal_advantage(self, match_data: dict) -> float:
        """
        Calcula la ventaja estacional.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            Ventaja estacional
        """
        try:
            match_date = match_data.get('date')
            if not match_date:
                return 0.0
            
            # Obtener mes del partido
            month = match_date.month
            
            # Ventajas por temporada
            # Temporada de hierba (junio-julio): Ventaja para jugadores de hierba
            # Temporada de tierra batida (abril-mayo): Ventaja para jugadores de tierra
            # Temporada de pista dura (enero-marzo, agosto-diciembre): Ventaja para jugadores de pista dura
            if 6 <= month <= 7:  # Temporada de hierba
                return 0.3  # Ventaja para jugadores de hierba
            elif 4 <= month <= 5:  # Temporada de tierra
                return -0.3  # Ventaja para jugadores de tierra
            else:  # Temporada de pista dura
                return 0.0  # Hora neutral
            
        except Exception as e:
            logging.error(f"Error calculando ventaja estacional: {str(e)}")
            return 0.0
    
    def _calculate_tournament_experience(self, match_data: dict) -> float:
        """
        Calcula la diferencia de experiencia en el torneo.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            Diferencia de experiencia en el torneo
        """
        try:
            player1 = match_data.get('player_1', match_data.get('player1', ''))
            player2 = match_data.get('player_2', match_data.get('player2', ''))
            tournament = match_data.get('tournament', '')
            
            if not player1 or not player2 or not tournament:
                return 0.0
            
            # Obtener estadísticas de jugadores
            p1_stats = self.players_stats.get(player1, {})
            p2_stats = self.players_stats.get(player2, {})
            
            # Obtener experiencia en el torneo
            p1_tournament_matches = p1_stats.get('tournament_stats', {}).get(tournament, {}).get('total_matches', 0)
            p2_tournament_matches = p2_stats.get('tournament_stats', {}).get(tournament, {}).get('total_matches', 0)
            
            # Calcular diferencia de experiencia
            experience_diff = p1_tournament_matches - p2_tournament_matches
            
            # Normalizar a un rango razonable
            return experience_diff / 10  # Dividir por 10 para normalizar
            
        except Exception as e:
            logging.error(f"Error calculando experiencia en torneo: {str(e)}")
            return 0.0
    
    def _calculate_momentum(self, match_data: dict) -> float:
        """
        Calcula la diferencia de momentum entre jugadores.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            Diferencia de momentum entre jugadores
        """
        try:
            player1 = match_data.get('player_1', match_data.get('player1', ''))
            player2 = match_data.get('player_2', match_data.get('player2', ''))
            match_date = match_data.get('date')
            
            if not match_date or not player1 or not player2:
                return 0.0
            
            # Obtener estadísticas de jugadores
            p1_stats = self.players_stats.get(player1, {})
            p2_stats = self.players_stats.get(player2, {})
            
            # Obtener partidos recientes
            p1_recent_matches = p1_stats.get('recent_matches', [])
            p2_recent_matches = p2_stats.get('recent_matches', [])
            
            # Calcular racha de victorias en los últimos 5 partidos
            p1_wins = sum(1 for m in p1_recent_matches[:5] if m['winner'] == player1)
            p2_wins = sum(1 for m in p2_recent_matches[:5] if m['winner'] == player2)
            
            # Calcular diferencia de momentum
            momentum_diff = p1_wins - p2_wins
            
            # Normalizar a un rango razonable
            return momentum_diff / 5  # Dividir por 5 para normalizar
            
        except Exception as e:
            logging.error(f"Error calculando momentum: {str(e)}")
            return 0.0
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normaliza las características usando StandardScaler."""
        return pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )



    def build_player_statistics(self):
        """Construye estadísticas de jugadores desde los datos."""
        if not self.data_path:
            logging.warning("No hay ruta de datos definida para construir estadísticas")
            return
        
        try:
            data = pd.read_csv(self.data_path)
            data['match_date'] = pd.to_datetime(data['match_date'])
            players = set(data['player_1'].tolist() + data['player_2'].tolist())
            
            for player in players:
                # Partidos como jugador 1
                p1_matches = data[data['player_1'] == player]
                p1_wins = len(p1_matches[p1_matches['winner'] == 0])
                
                # Partidos como jugador 2
                p2_matches = data[data['player_2'] == player]
                p2_wins = len(p2_matches[p2_matches['winner'] == 1])
                
                # Total partidos y victorias
                total_matches = len(p1_matches) + len(p2_matches)
                total_wins = p1_wins + p2_wins
                
                # Estadísticas por superficie
                surface_stats = {}
                for surface in data['surface'].unique():
                    # Superficie como jugador 1
                    p1_surface = p1_matches[p1_matches['surface'] == surface]
                    p1_surface_wins = len(p1_surface[p1_surface['winner'] == 0])
                    
                    # Superficie como jugador 2
                    p2_surface = p2_matches[p2_matches['surface'] == surface]
                    p2_surface_wins = len(p2_surface[p2_surface['winner'] == 1])
                    
                    # Total partidos y victorias en superficie
                    surface_matches = len(p1_surface) + len(p2_surface)
                    surface_wins = p1_surface_wins + p2_surface_wins
                    
                    if surface_matches > 0:
                        surface_win_rate = (surface_wins / surface_matches) * 100
                    else:
                        surface_win_rate = 0
                    
                    surface_stats[surface] = {
                        'matches': surface_matches,
                        'wins': surface_wins,
                        'win_rate': surface_win_rate
                    }
                
                # Obtener ranking promedio
                if 'ranking_1' in data.columns:
                    p1_rankings = p1_matches['ranking_1'].tolist()
                    p2_rankings = p2_matches['ranking_2'].tolist()
                    avg_ranking = np.mean(p1_rankings + p2_rankings) if (p1_rankings + p2_rankings) else 100
                else:
                    avg_ranking = 100
                
                # Calcular tasa de victoria
                win_rate = (total_wins / total_matches) * 100 if total_matches > 0 else 50
                
                # Nuevas estadísticas temporales
                recent_results = []
                seasonal_preferences = {}
                time_preferences = {}
                
                # Ordenar partidos por fecha
                all_matches = pd.concat([p1_matches, p2_matches]).sort_values('match_date')
                
                # Calcular resultados recientes (últimos 10 partidos)
                for _, match in all_matches.tail(10).iterrows():
                    if match['player_1'] == player:
                        recent_results.append(1 if match['winner'] == 0 else 0)
                    else:
                        recent_results.append(1 if match['winner'] == 1 else 0)
                
                # Calcular preferencias estacionales
                for month in range(1, 13):
                    month_matches = all_matches[all_matches['match_date'].dt.month == month]
                    if not month_matches.empty:
                        month_wins = sum(
                            1 if (row['player_1'] == player and row['winner'] == 0) or
                                (row['player_2'] == player and row['winner'] == 1)
                            else 0
                            for _, row in month_matches.iterrows()
                        )
                        seasonal_preferences[month] = month_wins / len(month_matches)
                
                # Calcular preferencias por hora del día
                for hour in range(24):
                    hour_matches = all_matches[all_matches['match_date'].dt.hour == hour]
                    if not hour_matches.empty:
                        hour_wins = sum(
                            1 if (row['player_1'] == player and row['winner'] == 0) or
                                (row['player_2'] == player and row['winner'] == 1)
                            else 0
                            for _, row in hour_matches.iterrows()
                        )
                        time_preferences[hour] = hour_wins / len(hour_matches)
                
                # Calcular días de descanso promedio
                if len(all_matches) > 1:
                    rest_days = []
                    for i in range(1, len(all_matches)):
                        days = (all_matches.iloc[i]['match_date'] - all_matches.iloc[i-1]['match_date']).days
                        if days > 0:
                            rest_days.append(days)
                    avg_rest_days = np.mean(rest_days) if rest_days else 0
                else:
                    avg_rest_days = 0
                
                # Guardar estadísticas del jugador
                self.players_stats[player] = {
                    'total_matches': total_matches,
                    'total_wins': total_wins,
                    'win_rate': win_rate,
                    'avg_ranking': avg_ranking,
                    'surface_stats': surface_stats,
                    'recent_results': recent_results,
                    'seasonal_preferences': seasonal_preferences,
                    'time_preferences': time_preferences,
                    'days_rest': avg_rest_days
                }
            
            logging.info(f"Estadísticas calculadas para {len(self.players_stats)} jugadores")
            
        except Exception as e:
            logging.error(f"Error construyendo estadísticas de jugadores: {e}")
    
    def build_head_to_head_statistics(self):
        """Construye estadísticas head-to-head entre jugadores."""
        if not self.data_path:
            logging.warning("No hay ruta de datos definida para construir estadísticas")
            return
        
        try:
            data = pd.read_csv(self.data_path)
            data['match_date'] = pd.to_datetime(data['match_date'])
            
            for _, row in data.iterrows():
                player1 = row['player_1']
                player2 = row['player_2']
                winner = row['winner']
                
                # Usar tupla ordenada como clave para asegurar consistencia
                player_pair = tuple(sorted([player1, player2]))
                
                # Inicializar si no existe
                if player_pair not in self.head_to_head_stats:
                    self.head_to_head_stats[player_pair] = {
                        'total_matches': 0,
                        'player1_wins': 0,
                        'player2_wins': 0,
                        'player1_win_pct': 0,
                        'player2_win_pct': 0,
                        'surface_stats': {},
                        'recent_matches': [],
                        'avg_duration': 0,
                        'total_duration': 0,
                        'last_match_date': None,
                        'first_match_date': None,
                        'win_streak': 0,
                        'current_streak_holder': None
                    }
                
                # Actualizar estadísticas básicas
                self.head_to_head_stats[player_pair]['total_matches'] += 1
                
                if winner == 0:  # Jugador 1 ganó
                    if player1 == player_pair[0]:
                        self.head_to_head_stats[player_pair]['player1_wins'] += 1
                    else:
                        self.head_to_head_stats[player_pair]['player2_wins'] += 1
                else:  # Jugador 2 ganó
                    if player2 == player_pair[0]:
                        self.head_to_head_stats[player_pair]['player1_wins'] += 1
                    else:
                        self.head_to_head_stats[player_pair]['player2_wins'] += 1
                
                # Actualizar estadísticas por superficie
                surface = row['surface']
                if surface not in self.head_to_head_stats[player_pair]['surface_stats']:
                    self.head_to_head_stats[player_pair]['surface_stats'][surface] = {
                        'total_matches': 0,
                        'player1_wins': 0,
                        'player2_wins': 0,
                        'player1_win_pct': 0,
                        'player2_win_pct': 0
                    }
                
                surface_stats = self.head_to_head_stats[player_pair]['surface_stats'][surface]
                surface_stats['total_matches'] += 1
                
                if winner == 0:  # Jugador 1 ganó
                    if player1 == player_pair[0]:
                        surface_stats['player1_wins'] += 1
                    else:
                        surface_stats['player2_wins'] += 1
                else:  # Jugador 2 ganó
                    if player2 == player_pair[0]:
                        surface_stats['player1_wins'] += 1
                    else:
                        surface_stats['player2_wins'] += 1
                
                # Actualizar partidos recientes
                match_info = {
                    'date': row['match_date'],
                    'winner': player1 if winner == 0 else player2,
                    'surface': surface,
                    'tournament': row['tournament']
                }
                self.head_to_head_stats[player_pair]['recent_matches'].append(match_info)
                
                # Mantener solo los últimos 10 partidos
                self.head_to_head_stats[player_pair]['recent_matches'] = sorted(
                    self.head_to_head_stats[player_pair]['recent_matches'],
                    key=lambda x: x['date']
                )[-10:]
                
                # Actualizar fechas de primer y último partido
                match_date = row['match_date']
                if (self.head_to_head_stats[player_pair]['first_match_date'] is None or 
                    match_date < self.head_to_head_stats[player_pair]['first_match_date']):
                    self.head_to_head_stats[player_pair]['first_match_date'] = match_date
                if (self.head_to_head_stats[player_pair]['last_match_date'] is None or 
                    match_date > self.head_to_head_stats[player_pair]['last_match_date']):
                    self.head_to_head_stats[player_pair]['last_match_date'] = match_date
                
                # Actualizar duración promedio
                if 'duration' in row:
                    self.head_to_head_stats[player_pair]['total_duration'] += row['duration']
                    self.head_to_head_stats[player_pair]['avg_duration'] = (
                        self.head_to_head_stats[player_pair]['total_duration'] / 
                        self.head_to_head_stats[player_pair]['total_matches']
                    )
                
                # Calcular racha actual
                recent_matches = sorted(
                    self.head_to_head_stats[player_pair]['recent_matches'],
                    key=lambda x: x['date']
                )
                current_streak = 0
                current_holder = None
                
                for match in reversed(recent_matches):
                    if current_holder is None:
                        current_holder = match['winner']
                        current_streak = 1
                    elif match['winner'] == current_holder:
                        current_streak += 1
                    else:
                        break
                
                self.head_to_head_stats[player_pair]['win_streak'] = current_streak
                self.head_to_head_stats[player_pair]['current_streak_holder'] = current_holder
            
            # Calcular porcentajes finales
            for pair_stats in self.head_to_head_stats.values():
                total_matches = pair_stats['total_matches']
                if total_matches > 0:
                    pair_stats['player1_win_pct'] = pair_stats['player1_wins'] / total_matches
                    pair_stats['player2_win_pct'] = pair_stats['player2_wins'] / total_matches
                
                # Calcular porcentajes por superficie
                for surface_stats in pair_stats['surface_stats'].values():
                    surface_total = surface_stats['total_matches']
                    if surface_total > 0:
                        surface_stats['player1_win_pct'] = surface_stats['player1_wins'] / surface_total
                        surface_stats['player2_win_pct'] = surface_stats['player2_wins'] / surface_total
            
            logging.info(f"Estadísticas head-to-head calculadas para {len(self.head_to_head_stats)} pares de jugadores")
            
        except Exception as e:
            logging.error(f"Error construyendo estadísticas head-to-head: {e}")
    
    def transform_match_data(self, match_data):
        """
        Transforma los datos de un partido para predecir el resultado.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            DataFrame con características para el modelo
        """
        try:
            # Extraer datos básicos
            player1 = match_data.get('player_1', match_data.get('player1', ''))
            player2 = match_data.get('player_2', match_data.get('player2', ''))
            surface = match_data.get('surface', 'hard').lower()
            match_date = match_data.get('date')
            
            # Preparar características
            features = {}
            
            # Ranking y ELO
            ranking1 = match_data.get('ranking_1')
            if ranking1 is None and player1 in self.players_stats:
                ranking1 = self.players_stats[player1].get('avg_ranking', 100)
            elif ranking1 is None:
                ranking1 = 100
            
            ranking2 = match_data.get('ranking_2')
            if ranking2 is None and player2 in self.players_stats:
                ranking2 = self.players_stats[player2].get('avg_ranking', 100)
            elif ranking2 is None:
                ranking2 = 100
            
            features['ranking_1'] = ranking1
            features['ranking_2'] = ranking2
            
            # ELO ratings
            elo1 = match_data.get('elo_1')
            if elo1 is None and player1 in self.players_stats:
                elo1 = self.players_stats[player1].get('elo_rating', 1500)
            elif elo1 is None:
                elo1 = 1500
            
            elo2 = match_data.get('elo_2')
            if elo2 is None and player2 in self.players_stats:
                elo2 = self.players_stats[player2].get('elo_rating', 1500)
            elif elo2 is None:
                elo2 = 1500
            
            features['elo_winner'] = elo1
            features['elo_loser'] = elo2
            
            # ELO por superficie
            elo_surface1 = match_data.get('elo_surface_1')
            if elo_surface1 is None and player1 in self.players_stats:
                elo_surface1 = self.players_stats[player1].get('surface_elo', {}).get(surface, elo1)
            elif elo_surface1 is None:
                elo_surface1 = elo1
            
            elo_surface2 = match_data.get('elo_surface_2')
            if elo_surface2 is None and player2 in self.players_stats:
                elo_surface2 = self.players_stats[player2].get('surface_elo', {}).get(surface, elo2)
            elif elo_surface2 is None:
                elo_surface2 = elo2
            
            features['elo_winner_surface'] = elo_surface1
            features['elo_loser_surface'] = elo_surface2
            
            # Tasas de victoria
            winrate1 = match_data.get('winrate_1')
            if winrate1 is None and player1 in self.players_stats:
                winrate1 = self.players_stats[player1].get('win_rate', 50)
            elif winrate1 is None:
                winrate1 = 50
            
            winrate2 = match_data.get('winrate_2')
            if winrate2 is None and player2 in self.players_stats:
                winrate2 = self.players_stats[player2].get('win_rate', 50)
            elif winrate2 is None:
                winrate2 = 50
            
            features['winrate_1'] = winrate1
            features['winrate_2'] = winrate2
            features['winrate_diff'] = winrate1 - winrate2
            
            # Superficie
            surfaces = {'hard': 0, 'clay': 1, 'grass': 2, 'carpet': 3}
            features['surface_code'] = surfaces.get(surface, 0)
            
            # Estadísticas por superficie
            if player1 in self.players_stats and player2 in self.players_stats:
                p1_surface_stats = self.players_stats[player1].get('surface_stats', {}).get(surface, {})
                p2_surface_stats = self.players_stats[player2].get('surface_stats', {}).get(surface, {})
                
                features['p1_surface_winrate'] = p1_surface_stats.get('win_rate', 50)
                features['p2_surface_winrate'] = p2_surface_stats.get('win_rate', 50)
                features['surface_winrate_diff'] = features['p1_surface_winrate'] - features['p2_surface_winrate']
            else:
                features['p1_surface_winrate'] = 50
                features['p2_surface_winrate'] = 50
                features['surface_winrate_diff'] = 0
            
            # Estadísticas head-to-head
            player_pair = tuple(sorted([player1, player2]))
            if player_pair in self.head_to_head_stats:
                h2h = self.head_to_head_stats[player_pair]
                features['h2h_advantage'] = h2h.get('player1_win_pct', 0.5) - h2h.get('player2_win_pct', 0.5)
                features['h2h_matches'] = h2h.get('total_matches', 0)
                
                # Head-to-head por superficie
                surface_h2h = h2h.get('surface_stats', {}).get(surface, {})
                features['h2h_surface_advantage'] = surface_h2h.get('player1_win_pct', 0.5) - surface_h2h.get('player2_win_pct', 0.5)
            else:
                features['h2h_advantage'] = 0
                features['h2h_matches'] = 0
                features['h2h_surface_advantage'] = 0
            
            # Fatiga y descanso
            if match_date:
                features['fatigue_diff'] = self._calculate_fatigue(match_data)
                features['rest_advantage'] = self._calculate_rest_advantage(match_data)
            
            # Características físicas
            features['physical_advantage'] = self._calculate_physical_advantage(match_data)
            
            # Características temporales
            if match_date:
                features['temporal_advantage'] = self._calculate_time_of_day_advantage(match_data)
                features['seasonal_advantage'] = self._calculate_seasonal_advantage(match_data)
            
            # Experiencia en torneo
            features['tournament_experience_diff'] = self._calculate_tournament_experience(match_data)
            
            # Momentum
            features['momentum_diff'] = self._calculate_momentum(match_data)
            
            # Convertir a DataFrame
            df_features = pd.DataFrame([features])
            
            return df_features
            
        except Exception as e:
            logging.error(f"Error transformando datos del partido: {str(e)}")
            raise
    
    def _get_external_ranking(self, player):
        """Método para obtener ranking desde fuentes externas."""
        # Esto sería para implementar integración con APIs externas
        # Por ahora devolvemos None y usamos valores predeterminados
        return None
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