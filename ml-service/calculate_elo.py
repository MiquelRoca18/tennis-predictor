#!/usr/bin/env python3
"""
calculate_elo.py

Script mejorado para calcular las puntuaciones ELO para todos los jugadores de tenis.
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EloSystem:
    def __init__(self, k_factor=32, initial_rating=1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.player_ratings = {}
        self.player_ratings_by_surface = {
            'Hard': {},
            'Clay': {},
            'Grass': {},
            'Carpet': {}
        }
    
    def get_player_rating(self, player_id: str, surface: str = None) -> float:
        """Obtener el rating ELO de un jugador."""
        if surface:
            return self.player_ratings_by_surface[surface].get(player_id, self.initial_rating)
        return self.player_ratings.get(player_id, self.initial_rating)
    
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calcular la probabilidad esperada de victoria."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, winner_id: str, loser_id: str, surface: str = None, k_factor: float = None):
        """Actualizar los ratings ELO después de un partido."""
        if k_factor is None:
            k_factor = self.k_factor
            
        # Ajustar k_factor según la superficie
        if surface:
            if surface == 'Grass':
                k_factor *= 1.2  # Mayor volatilidad en hierba
            elif surface == 'Clay':
                k_factor *= 0.8  # Menor volatilidad en tierra
        
        # Obtener ratings actuales
        if surface:
            winner_rating = self.get_player_rating(winner_id, surface)
            loser_rating = self.get_player_rating(loser_id, surface)
        else:
            winner_rating = self.get_player_rating(winner_id)
            loser_rating = self.get_player_rating(loser_id)
        
        # Calcular probabilidades esperadas
        winner_expected = self.calculate_expected_score(winner_rating, loser_rating)
        loser_expected = 1 - winner_expected
        
        # Actualizar ratings
        winner_new_rating = winner_rating + k_factor * (1 - winner_expected)
        loser_new_rating = loser_rating + k_factor * (0 - loser_expected)
        
        if surface:
            self.player_ratings_by_surface[surface][winner_id] = winner_new_rating
            self.player_ratings_by_surface[surface][loser_id] = loser_new_rating
        else:
            self.player_ratings[winner_id] = winner_new_rating
            self.player_ratings[loser_id] = loser_new_rating

def process_matches(df: pd.DataFrame) -> Tuple[pd.DataFrame, EloSystem]:
    """Procesar todos los partidos y calcular ratings ELO."""
    elo_system = EloSystem()
    
    # Ordenar partidos por fecha
    df = df.sort_values('match_date')
    
    # Columnas para almacenar ratings
    df['elo_winner'] = 0.0
    df['elo_loser'] = 0.0
    df['elo_winner_surface'] = 0.0
    df['elo_loser_surface'] = 0.0
    
    for idx, match in df.iterrows():
        # Obtener IDs de jugadores
        winner_id = str(match['winner'])
        loser_id = str(match['player_1'] if match['winner'] == match['player_2'] else match['player_2'])
        
        # Guardar ratings actuales
        df.at[idx, 'elo_winner'] = elo_system.get_player_rating(winner_id)
        df.at[idx, 'elo_loser'] = elo_system.get_player_rating(loser_id)
        
        # Solo procesar ratings por superficie si la superficie es válida
        surface = match['surface']
        if pd.notna(surface) and surface in elo_system.player_ratings_by_surface:
            df.at[idx, 'elo_winner_surface'] = elo_system.get_player_rating(winner_id, surface)
            df.at[idx, 'elo_loser_surface'] = elo_system.get_player_rating(loser_id, surface)
            elo_system.update_ratings(winner_id, loser_id, surface)
        
        # Actualizar ratings generales
        elo_system.update_ratings(winner_id, loser_id)
    
    return df, elo_system

def main():
    parser = argparse.ArgumentParser(description='Calcular ratings ELO para jugadores de tenis')
    parser.add_argument('--input_file', required=True, help='Archivo CSV con los partidos')
    parser.add_argument('--output_file', required=True, help='Archivo CSV donde guardar los resultados')
    args = parser.parse_args()
    
    try:
        # Cargar datos
        logger.info(f"Cargando datos desde {args.input_file}")
        df = pd.read_csv(args.input_file)
        
        # Procesar partidos
        logger.info("Calculando ratings ELO")
        df_with_elo, elo_system = process_matches(df)
        
        # Guardar resultados
        logger.info(f"Guardando resultados en {args.output_file}")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        df_with_elo.to_csv(args.output_file, index=False)
        
        # Guardar ratings finales
        ratings_file = os.path.join(os.path.dirname(args.output_file), 'final_elo_ratings.json')
        with open(ratings_file, 'w') as f:
            json.dump({
                'general': elo_system.player_ratings,
                'by_surface': elo_system.player_ratings_by_surface
            }, f, indent=4)
        
        logger.info("Proceso completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el proceso: {str(e)}")
        raise

if __name__ == "__main__":
    main()