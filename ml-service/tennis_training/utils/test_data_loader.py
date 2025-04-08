"""
Pruebas unitarias para el módulo data_loader.

Este archivo contiene pruebas para verificar la funcionalidad
del cargador de datos para el sistema de predicción de tenis.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Asegurar que el módulo está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar el módulo a probar
from utils.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Clase de pruebas para DataLoader."""
    
    def setUp(self):
        """Configuración previa a cada prueba."""
        # Crear directorio temporal para las pruebas
        self.test_dir = tempfile.mkdtemp()
        
        # Crear estructura de directorios simulada
        self.atp_dir = os.path.join(self.test_dir, 'atp')
        self.wta_dir = os.path.join(self.test_dir, 'wta')
        self.elo_dir = os.path.join(self.test_dir, 'elo')
        self.elo_ratings_dir = os.path.join(self.elo_dir, 'ratings')
        
        os.makedirs(self.atp_dir, exist_ok=True)
        os.makedirs(self.wta_dir, exist_ok=True)
        os.makedirs(self.elo_ratings_dir, exist_ok=True)
        
        # Crear archivos de prueba
        self._create_test_files()
        
        # Inicializar el DataLoader
        self.data_loader = DataLoader(base_dir=self.test_dir)
    
    def tearDown(self):
        """Limpieza posterior a cada prueba."""
        # Eliminar directorio temporal
        shutil.rmtree(self.test_dir)
    
    def _create_test_files(self):
        """Crea archivos de prueba para las pruebas."""
        # Crear archivo de partidos ATP
        atp_matches = pd.DataFrame({
            'match_id': range(1, 11),
            'player1_id': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'player2_id': [200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
            'winner_id': [100, 201, 102, 203, 104, 205, 106, 207, 108, 209],
            'date': pd.date_range(start='2022-01-01', periods=10),
            'surface': ['hard', 'clay', 'grass', 'hard', 'clay', 'grass', 'hard', 'clay', 'grass', 'hard'],
            'tournament_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
        })
        atp_matches.to_csv(os.path.join(self.atp_dir, 'atp_matches_main_2000_2024.csv'), index=False)
        
        # Crear archivo de jugadores ATP
        atp_players = pd.DataFrame({
            'player_id': np.concatenate([np.arange(100, 110), np.arange(200, 210)]),
            'name': [f"Player {i}" for i in np.concatenate([np.arange(100, 110), np.arange(200, 210)])],
            'hand': ['R', 'L', 'R', 'R', 'L', 'R', 'L', 'R', 'R', 'L'] * 2,
            'height': [185, 190, 178, 182, 195, 180, 188, 183, 187, 181] * 2,
            'country': ['ESP', 'USA', 'FRA', 'GER', 'SUI', 'ARG', 'ITA', 'BRA', 'JPN', 'CAN'] * 2
        })
        atp_players.to_csv(os.path.join(self.atp_dir, 'atp_players.csv'), index=False)
        
        # Crear archivo de rankings ATP
        atp_rankings = pd.DataFrame({
            'ranking_date': np.repeat(pd.date_range(start='2022-01-01', periods=5), 4),
            'player_id': np.tile(np.concatenate([np.arange(100, 104), np.arange(200, 204)]), 5),
            'ranking': np.tile([1, 2, 3, 4, 5, 6, 7, 8], 5),
            'ranking_points': np.tile([10000, 8000, 7000, 6000, 5000, 4000, 3000, 2000], 5)
        })
        atp_rankings.to_csv(os.path.join(self.atp_dir, 'atp_rankings_2000_2024.csv'), index=False)
        
        # Crear archivo ELO de ejemplo
        elo_general = {
            "100": 2100,
            "101": 2050,
            "102": 2000,
            "103": 1950,
            "104": 1900,
            "200": 2080,
            "201": 2030,
            "202": 1980,
            "203": 1930,
            "204": 1880
        }
        
        elo_by_surface = {
            "hard": {
                "100": 2120,
                "101": 2040,
                "200": 2090,
                "201": 2010
            },
            "clay": {
                "100": 2080,
                "101": 2070,
                "200": 2060,
                "201": 2050
            },
            "grass": {
                "100": 2050,
                "101": 2030,
                "200": 2040,
                "201": 2020
            }
        }
        
        # Guardar como JSON
        import json
        with open(os.path.join(self.elo_ratings_dir, 'elo_ratings_general_20250406_081013.json'), 'w') as f:
            json.dump(elo_general, f)
        
        with open(os.path.join(self.elo_ratings_dir, 'elo_ratings_by_surface_20250406_081013.json'), 'w') as f:
            json.dump(elo_by_surface, f)
    
    def test_load_matches(self):
        """Prueba la carga de partidos."""
        # Probar carga de partidos
        matches = self.data_loader.load_matches(tour='atp', level='main')
        
        # Verificar que el DataFrame no está vacío
        self.assertFalse(matches.empty)
        
        # Verificar número de filas
        self.assertEqual(len(matches), 10)
        
        # Verificar columnas esenciales
        expected_columns = ['match_id', 'player1_id', 'player2_id', 'winner_id', 'date', 'surface']
        for col in expected_columns:
            self.assertIn(col, matches.columns)
        
        # Verificar conversión de fecha
        self.assertIsInstance(matches['date'].iloc[0], pd.Timestamp)
    
    def test_load_players(self):
        """Prueba la carga de información de jugadores."""
        players = self.data_loader.load_players(tour='atp')
        
        # Verificar que el DataFrame no está vacío
        self.assertFalse(players.empty)
        
        # Verificar número de filas
        self.assertEqual(len(players), 20)
        
        # Verificar columnas esenciales
        expected_columns = ['player_id', 'name', 'hand', 'height', 'country']
        for col in expected_columns:
            self.assertIn(col, players.columns)
    
    def test_load_rankings(self):
        """Prueba la carga de rankings."""
        rankings = self.data_loader.load_rankings(tour='atp')
        
        # Verificar que el DataFrame no está vacío
        self.assertFalse(rankings.empty)
        
        # Verificar columnas esenciales
        expected_columns = ['ranking_date', 'player_id', 'ranking', 'ranking_points']
        for col in expected_columns:
            self.assertIn(col, rankings.columns)
        
        # Verificar conversión de fecha
        self.assertIsInstance(rankings['ranking_date'].iloc[0], pd.Timestamp)
    
    def test_load_elo_ratings(self):
        """Prueba la carga de ratings ELO."""
        # Probar carga de ELO general
        elo_general = self.data_loader.load_elo_ratings(rating_type='general', level='main')
        
        # Verificar que no está vacío
        self.assertTrue(elo_general)
        
        # Verificar algunos valores
        self.assertEqual(elo_general.get('100'), 2100)
        self.assertEqual(elo_general.get('101'), 2050)
        
        # Probar carga de ELO por superficie
        elo_hard = self.data_loader.load_elo_ratings(rating_type='hard', level='main')
        
        # Verificar valores
        self.assertEqual(elo_hard.get('100'), 2120)
        self.assertEqual(elo_hard.get('101'), 2040)
    
    def test_merge_player_data(self):
        """Prueba la unión de datos de partidos con información de jugadores."""
        matches = self.data_loader.load_matches(tour='atp')
        players = self.data_loader.load_players(tour='atp')
        
        merged = self.data_loader.merge_player_data(matches, players)
        
        # Verificar que el DataFrame no está vacío
        self.assertFalse(merged.empty)
        
        # Verificar que tiene información de jugadores
        self.assertIn('name_p1', merged.columns)
        self.assertIn('name_p2', merged.columns)
    
    def test_add_rankings_to_matches(self):
        """Prueba la adición de rankings a los partidos."""
        matches = self.data_loader.load_matches(tour='atp')
        rankings = self.data_loader.load_rankings(tour='atp')
        
        result = self.data_loader.add_rankings_to_matches(matches, rankings)
        
        # Verificar columnas de ranking añadidas
        self.assertIn('player1_rank', result.columns)
        self.assertIn('player2_rank', result.columns)
    
    def test_prepare_training_data(self):
        """Prueba la preparación de datos para entrenamiento."""
        # Preparar datos básicos
        data = self.data_loader.prepare_training_data(
            tour='atp',
            level='main',
            include_elo=True
        )
        
        # Verificar que no está vacío
        self.assertFalse(data.empty)
        
        # Verificar columna objetivo
        self.assertIn('player1_won', data.columns)
        
        # Verificar que tiene ELO
        self.assertIn('player1_elo', data.columns)
        self.assertIn('player2_elo', data.columns)
        
        # Probar filtrado por superficie
        data_hard = self.data_loader.prepare_training_data(
            tour='atp',
            level='main',
            surfaces=['hard']
        )
        self.assertLess(len(data_hard), len(data))
        self.assertTrue(all(s == 'hard' for s in data_hard['surface']))
    
    def test_split_train_test(self):
        """Prueba la división en datos de entrenamiento y prueba."""
        data = self.data_loader.prepare_training_data(tour='atp', level='main')
        
        # División temporal
        train, test = self.data_loader.split_train_test(data, test_size=0.3, temporal=True)
        
        # Verificar tamaños
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        
        # Verificar que la división es realmente temporal
        if 'date' in data.columns:
            max_train_date = train['date'].max()
            min_test_date = test['date'].min()
            self.assertLessEqual(max_train_date, min_test_date)
    
    def test_get_features_and_target(self):
        """Prueba la extracción de características y objetivo."""
        data = self.data_loader.prepare_training_data(tour='atp', level='main')
        
        # Extracción con columnas específicas
        feature_cols = ['player1_elo', 'player2_elo', 'player1_rank', 'player2_rank']
        X, y = self.data_loader.get_features_and_target(data, feature_cols=feature_cols)
        
        # Verificar columnas
        self.assertEqual(set(X.columns), set(feature_cols))
        
        # Verificar que y es la columna objetivo
        self.assertEqual(len(y), len(data))


if __name__ == '__main__':
    unittest.main()