#!/usr/bin/env python3
"""
elo_system.py

Sistema ELO mejorado para tenis.
Implementa un sistema de rating ELO con características específicas para tenis:
- Factores K ajustados por superficie
- ELO específico por superficie
- Actualización automática con nuevos resultados
- Historial de ELO para análisis de tendencias
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs/elo_system.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TennisEloSystem:
    def __init__(self, k_factor: float = 32.0):
        self.k_factor = k_factor
        self.surface_k_factors = {
            'Hard': 32.0,
            'Clay': 32.0,
            'Grass': 32.0,
            'Carpet': 32.0
        }
        self.ratings = {}
        self.surface_ratings = {}
        self.logger = logging.getLogger(__name__)
        
    def calculate_expected_score(self, player_rating: float, opponent_rating: float) -> float:
        """Calcula la probabilidad esperada de victoria."""
        return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))
    
    def update_rating(self, player_rating: float, opponent_rating: float, 
                     result: float, surface: str = None) -> float:
        """Actualiza el rating ELO de un jugador."""
        k = self.surface_k_factors.get(surface, self.k_factor)
        expected = self.calculate_expected_score(player_rating, opponent_rating)
        return player_rating + k * (result - expected)
    
    def initialize_ratings(self, initial_rating: float = 1500.0):
        """Inicializa los ratings para todos los jugadores."""
        self.ratings = {}
        self.surface_ratings = {}
        
    def process_match(self, winner_id: str, loser_id: str, surface: str = None):
        """Procesa un partido y actualiza los ratings de ambos jugadores."""
        # Obtener ratings actuales o inicializarlos si no existen
        winner_rating = self.ratings.get(winner_id, 1500.0)
        loser_rating = self.ratings.get(loser_id, 1500.0)
        
        # Actualizar ratings generales
        new_winner_rating = self.update_rating(winner_rating, loser_rating, 1.0, surface)
        new_loser_rating = self.update_rating(loser_rating, winner_rating, 0.0, surface)
        
        self.ratings[winner_id] = new_winner_rating
        self.ratings[loser_id] = new_loser_rating
        
        # Actualizar ratings por superficie si se especifica
        if surface:
            winner_surface_rating = self.surface_ratings.get((winner_id, surface), 1500.0)
            loser_surface_rating = self.surface_ratings.get((loser_id, surface), 1500.0)
            
            new_winner_surface_rating = self.update_rating(
                winner_surface_rating, loser_surface_rating, 1.0, surface)
            new_loser_surface_rating = self.update_rating(
                loser_surface_rating, winner_surface_rating, 0.0, surface)
            
            self.surface_ratings[(winner_id, surface)] = new_winner_surface_rating
            self.surface_ratings[(loser_id, surface)] = new_loser_surface_rating
    
    def calculate_historical_ratings(self, matches_df: pd.DataFrame):
        """Calcula ratings históricos para todos los partidos."""
        self.logger.info("Calculando ratings históricos...")
        
        # Ordenar partidos por fecha
        matches_df = matches_df.sort_values('match_date')
        
        # Inicializar ratings
        self.initialize_ratings()
        
        # Procesar cada partido
        for _, match in matches_df.iterrows():
            winner_id = str(match['winner_id'])
            loser_id = str(match['player2_id'] if match['winner_id'] == match['player1_id'] else match['player1_id'])
            
            self.process_match(
                winner_id,
                loser_id,
                match['surface']
            )
            
        return self.ratings
    
    def get_player_rating(self, player_id: str, surface: str = None) -> float:
        """Obtiene el rating de un jugador (general o por superficie)."""
        if surface:
            return self.surface_ratings.get((player_id, surface), 1500.0)
        return self.ratings.get(player_id, 1500.0)
    
    def get_rating_difference(self, player1_id: str, player2_id: str, 
                            surface: str = None) -> float:
        """Calcula la diferencia de rating entre dos jugadores."""
        rating1 = self.get_player_rating(player1_id, surface)
        rating2 = self.get_player_rating(player2_id, surface)
        return rating1 - rating2
    
    def get_win_probability(self, player1_id: str, player2_id: str, 
                          surface: str = None) -> float:
        """Calcula la probabilidad de victoria basada en ratings ELO."""
        rating1 = self.get_player_rating(player1_id, surface)
        rating2 = self.get_player_rating(player2_id, surface)
        return self.calculate_expected_score(rating1, rating2)
    
    def calculate_elo_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula las características ELO para todos los partidos.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            DataFrame con características ELO añadidas
        """
        self.logger.info("Calculando características ELO...")
        
        # Calcular ratings históricos
        self.calculate_historical_ratings(data)
        
        # Crear copia del DataFrame para no modificar el original
        data_with_elo = data.copy()
        
        # Añadir ratings ELO
        data_with_elo['player1_elo'] = data_with_elo.apply(
            lambda row: self.get_player_rating(str(row['player1_id'])), axis=1)
        data_with_elo['player2_elo'] = data_with_elo.apply(
            lambda row: self.get_player_rating(str(row['player2_id'])), axis=1)
        
        # Añadir ratings por superficie
        data_with_elo['player1_surface_elo'] = data_with_elo.apply(
            lambda row: self.get_player_rating(str(row['player1_id']), row['surface']), axis=1)
        data_with_elo['player2_surface_elo'] = data_with_elo.apply(
            lambda row: self.get_player_rating(str(row['player2_id']), row['surface']), axis=1)
        
        # Calcular diferencias de rating
        data_with_elo['elo_difference'] = data_with_elo['player1_elo'] - data_with_elo['player2_elo']
        data_with_elo['surface_elo_difference'] = data_with_elo['player1_surface_elo'] - data_with_elo['player2_surface_elo']
        
        # Seleccionar solo las columnas numéricas de ELO para devolver
        elo_features = data_with_elo[['player1_elo', 'player2_elo', 
                                     'player1_surface_elo', 'player2_surface_elo',
                                     'elo_difference', 'surface_elo_difference']]
        
        self.logger.info("Características ELO calculadas correctamente")
        return elo_features
    
    def recalculate_all_elo(self):
        """Recalcula todos los ratings ELO desde cero."""
        # TO DO: Implementar lógica para recalcular todos los ratings
        pass
    
    def get_elo_ranking(self, surface: str = None, limit: int = 20) -> pd.DataFrame:
        """Obtiene el ranking ELO de jugadores."""
        # TO DO: Implementar lógica para obtener el ranking ELO
        pass
    
    def plot_elo_history(self, player_names: List[str], output_path: str, surface: str = None, show_plot: bool = True):
        """Grafica el historial ELO de jugadores."""
        # TO DO: Implementar lógica para graficar el historial ELO
        pass
    
    def export_elo_data(self, output_path: str) -> bool:
        """Exporta los datos ELO a un archivo."""
        # TO DO: Implementar lógica para exportar los datos ELO
        pass

def setup_db_schema(connection):
    """
    Configura el esquema de base de datos para el sistema ELO.
    Crea las tablas necesarias si no existen.
    
    Args:
        connection: Conexión a PostgreSQL
    """
    try:
        cursor = connection.cursor()
        
        # Crear tabla de jugadores si no existe
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                country VARCHAR(3),
                birth_date DATE,
                dominant_hand CHAR(1),
                height INT,
                current_ranking INT,
                best_ranking INT,
                elo_rating FLOAT DEFAULT 1500,
                elo_hard FLOAT DEFAULT 1500,
                elo_clay FLOAT DEFAULT 1500,
                elo_grass FLOAT DEFAULT 1500,
                elo_carpet FLOAT DEFAULT 1500,
                elo_last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                matches_played INT DEFAULT 0,
                CONSTRAINT unique_player_name UNIQUE (name)
            )
        """)
        
        # Crear tabla de historial ELO si no existe
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_elo_history (
                id SERIAL PRIMARY KEY,
                player_id INT REFERENCES players(id),
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                elo_rating FLOAT,
                elo_hard FLOAT,
                elo_clay FLOAT,
                elo_grass FLOAT,
                elo_carpet FLOAT,
                match_id INT,
                notes TEXT
            )
        """)
        
        # Crear índice para búsquedas rápidas
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_elo_history_player
            ON player_elo_history (player_id)
        """)
        
        connection.commit()
        cursor.close()
        logging.info("Esquema de base de datos para ELO configurado correctamente")
        
    except Exception as e:
        logging.error(f"Error configurando esquema de base de datos: {e}")
        connection.rollback()

def main():
    """Función principal para probar el sistema ELO."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de puntuación ELO para tenis')
    parser.add_argument('--data', type=str, help='Ruta al archivo CSV con datos históricos')
    parser.add_argument('--recalculate', action='store_true', help='Recalcular ELO desde cero')
    parser.add_argument('--export', type=str, help='Exportar datos a archivo JSON')
    parser.add_argument('--db-connection', action='store_true', help='Usar conexión a base de datos')
    parser.add_argument('--top', type=int, default=20, help='Mostrar los N mejores jugadores')
    parser.add_argument('--plot', nargs='+', help='Nombres de jugadores para graficar su historial ELO')
    parser.add_argument('--plot-output', type=str, help='Ruta para guardar el gráfico')
    parser.add_argument('--surface', type=str, choices=['hard', 'clay', 'grass', 'carpet'],
                      help='Superficie específica para rankings o gráficos')
    
    # Argumentos para conexión a base de datos
    parser.add_argument('--db-host', default='localhost', help='Host de PostgreSQL')
    parser.add_argument('--db-port', type=int, default=5432, help='Puerto de PostgreSQL')
    parser.add_argument('--db-name', help='Nombre de la base de datos')
    parser.add_argument('--db-user', help='Usuario de PostgreSQL')
    parser.add_argument('--db-password', help='Contraseña de PostgreSQL')
    
    args = parser.parse_args()
    
    # Inicializar sistema ELO
    db_connection = None
    
    if args.db_connection and args.db_name and args.db_user:
        try:
            logging.info(f"Conectando a base de datos {args.db_name}@{args.db_host}")
            db_connection = psycopg2.connect(
                host=args.db_host,
                port=args.db_port,
                dbname=args.db_name,
                user=args.db_user,
                password=args.db_password
            )
            
            # Configurar esquema si es necesario
            setup_db_schema(db_connection)
            
        except Exception as e:
            logging.error(f"Error conectando a base de datos: {e}")
            db_connection = None
    
    # Crear sistema ELO
    elo_system = TennisEloSystem()
    
    # Recalcular ELO si se solicita
    if args.recalculate:
        logging.info("Iniciando recálculo de ELO...")
        if elo_system.recalculate_all_elo():
            logging.info("Recálculo de ELO completado exitosamente")
        else:
            logging.error("Error en recálculo de ELO")
    
    # Mostrar top jugadores
    top_players = elo_system.get_elo_ranking(surface=args.surface, limit=args.top)
    if not top_players.empty:
        if args.surface:
            print(f"\nTop {args.top} jugadores por ELO en superficie {args.surface}:")
        else:
            print(f"\nTop {args.top} jugadores por ELO general:")
        
        # Dar formato a la salida
        pd.set_option('display.max_rows', args.top)
        pd.set_option('display.width', 120)
        print(top_players)
    
    # Graficar historial si se solicita
    if args.plot:
        output_path = args.plot_output or 'elo_history.png'
        elo_system.plot_elo_history(
            player_names=args.plot,
            output_path=output_path,
            surface=args.surface,
            show_plot=True
        )
    
    # Exportar datos si se solicita
    if args.export:
        if elo_system.export_elo_data(args.export):
            logging.info(f"Datos exportados a {args.export}")
        else:
            logging.error(f"Error exportando datos a {args.export}")
    
    # Cerrar conexión a base de datos
    if db_connection:
        db_connection.close()
        logging.info("Conexión a base de datos cerrada")

if __name__ == "__main__":
    main()