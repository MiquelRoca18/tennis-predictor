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
            'ranking': 0.3,
            'h2h': 0.2,
            'recent_form': 0.2,
            'surface': 0.15,
            'fatigue': 0.15
        }
        self.data_path = data_path
        self.players_stats = {}
        self.head_to_head_stats = {}

        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae características de los datos de partidos de tenis.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            DataFrame con características calculadas
        """
        try:
            features = pd.DataFrame()
            
            # Características básicas
            features['ranking_diff'] = data['winner_rank'] - data['loser_rank']
            features['age_diff'] = data['winner_age'] - data['loser_age']
            features['height_diff'] = data['winner_ht'] - data['loser_ht']
            
            # Características de superficie
            features['surface_advantage'] = self._calculate_surface_advantage(data)
            
            # Características de forma reciente
            features['recent_form'] = self._calculate_recent_form(data)
            
            # Características de fatiga
            features['fatigue'] = self._calculate_fatigue(data)
            
            # Características head-to-head
            features['h2h_advantage'] = self._calculate_h2h_advantage(data)
            
            # Normalizar características
            features = self._normalize_features(features)
            
            return features
            
        except Exception as e:
            logging.error(f"Error extrayendo características: {e}")
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
    
    def _calculate_fatigue(self, data: pd.DataFrame) -> pd.Series:
        """Calcula el nivel de fatiga de los jugadores."""
        return (
            data['winner_matches_played'] / (data['winner_days_rest'] + 1) -
            data['loser_matches_played'] / (data['loser_days_rest'] + 1)
        )
    
    def _calculate_h2h_advantage(self, data: pd.DataFrame) -> pd.Series:
        """Calcula la ventaja head-to-head."""
        return (
            data['winner_h2h_wins'] / (data['winner_h2h_wins'] + data['loser_h2h_wins'] + 1) -
            data['loser_h2h_wins'] / (data['winner_h2h_wins'] + data['loser_h2h_wins'] + 1)
        )
    
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
                
                # Guardar estadísticas del jugador
                self.players_stats[player] = {
                    'total_matches': total_matches,
                    'total_wins': total_wins,
                    'win_rate': win_rate,
                    'avg_ranking': avg_ranking,
                    'surface_stats': surface_stats
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
                        'player2_win_pct': 0
                    }
                
                # Actualizar estadísticas
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
                
                # Calcular porcentajes
                total = self.head_to_head_stats[player_pair]['total_matches']
                p1_wins = self.head_to_head_stats[player_pair]['player1_wins']
                p2_wins = self.head_to_head_stats[player_pair]['player2_wins']
                
                self.head_to_head_stats[player_pair]['player1_win_pct'] = (p1_wins / total) * 100 if total > 0 else 50
                self.head_to_head_stats[player_pair]['player2_win_pct'] = (p2_wins / total) * 100 if total > 0 else 50
            
            logging.info(f"Estadísticas H2H calculadas para {len(self.head_to_head_stats)} pares de jugadores")
            
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
            # Extraer datos
            player1 = match_data.get('player_1', match_data.get('player1', ''))
            player2 = match_data.get('player_2', match_data.get('player2', ''))
            surface = match_data.get('surface', 'hard').lower()
            
            # Preparar características
            features = {}
            
            # Ranking
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
            features['ranking_diff'] = ranking1 - ranking2
            
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
                
                # Determinar quién es quién en las estadísticas H2H
                if player1 == player_pair[0]:
                    features['p1_h2h_wins'] = h2h['player1_wins']
                    features['p2_h2h_wins'] = h2h['player2_wins']
                    features['p1_h2h_winrate'] = h2h['player1_win_pct']
                    features['p2_h2h_winrate'] = h2h['player2_win_pct']
                else:
                    features['p1_h2h_wins'] = h2h['player2_wins']
                    features['p2_h2h_wins'] = h2h['player1_wins']
                    features['p1_h2h_winrate'] = h2h['player2_win_pct']
                    features['p2_h2h_winrate'] = h2h['player1_win_pct']
                
                features['h2h_diff'] = features['p1_h2h_winrate'] - features['p2_h2h_winrate']
            else:
                features['p1_h2h_wins'] = 0
                features['p2_h2h_wins'] = 0
                features['p1_h2h_winrate'] = 50
                features['p2_h2h_winrate'] = 50
                features['h2h_diff'] = 0
            
            # Convertir a DataFrame
            return pd.DataFrame([features])
            
        except Exception as e:
            logging.error(f"Error transformando datos del partido: {e}")
            # Datos mínimos si hay error
            return pd.DataFrame([{
                'ranking_1': match_data.get('ranking_1', 100),
                'ranking_2': match_data.get('ranking_2', 100),
                'winrate_1': match_data.get('winrate_1', 50),
                'winrate_2': match_data.get('winrate_2', 50),
                'surface_code': {'hard': 0, 'clay': 1, 'grass': 2}.get(match_data.get('surface', 'hard').lower(), 0)
            }])
    
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