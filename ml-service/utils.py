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
    
    def __init__(self):
        """Inicializa la clase con configuraciones por defecto."""
        self.scaler = StandardScaler()
        self.feature_weights = {
            'ranking': 0.3,
            'h2h': 0.2,
            'recent_form': 0.2,
            'surface': 0.15,
            'fatigue': 0.15
        }
        
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