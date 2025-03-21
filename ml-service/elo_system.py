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
        logging.FileHandler('ml-service/logs/elo_system.log'),
        logging.StreamHandler()
    ]
)

class TennisELOSystem:
    """
    Sistema avanzado de ELO para tenis con múltiples mejoras:
    - ELO específico por superficie
    - K-factor personalizado por superficie e importancia del torneo
    - Actualización más precisa basada en diferencia de rendimiento
    - Ajustes por ranking oficial y edad del jugador
    """
    
    def __init__(self, db_connection=None, data_path=None):
        """
        Inicializa el sistema ELO con base de datos o archivo CSV.
        
        Args:
            db_connection: Conexión a PostgreSQL (opcional)
            data_path: Ruta al archivo CSV con datos históricos (opcional)
        """
        self.db_connection = db_connection
        self.data_path = data_path
        self.csv_mode = data_path is not None and os.path.exists(data_path)
        
        # ELO inicial y parámetros de configuración
        self.initial_elo = 1500
        self.new_player_penalty = 100  # Penalización para jugadores nuevos (1400 inicial)
        
        # Configurar factores K por superficie (el factor K determina cuánto cambia el ELO después de cada partido)
        self.k_factors = {
            'hard': 32,     # Superficie común, factor K estándar
            'clay': 24,     # Superficie más predecible, factor K más bajo
            'grass': 40,    # Superficie más variable, factor K más alto
            'carpet': 36,   # Superficie indoor, factor K alto-medio
            'default': 32   # Valor por defecto para otras superficies
        }
        
        # Multiplicadores de K-factor por tipo de torneo
        self.tournament_multipliers = {
            'grand_slam': 1.5,      # Grand Slams tienen mayor importancia
            'masters': 1.25,        # Masters 1000/500
            'atp_tour': 1.0,        # ATP 250 y otros torneos regulares
            'challenger': 0.8,      # Challenger
            'futures': 0.6,         # ITF Futures
            'default': 1.0          # Valor por defecto
        }
        
        # Datos en memoria (modo CSV)
        self.players_elo = {}
        self.matches_history = []
        self.elo_history = {}
        
        # Cargar datos en memoria si estamos en modo CSV
        if self.csv_mode:
            self._load_csv_data()
        
        logging.info("Sistema ELO inicializado")
    
    def _load_csv_data(self):
        """Carga datos iniciales desde CSV para uso sin base de datos."""
        try:
            if not os.path.exists(self.data_path):
                logging.warning(f"El archivo {self.data_path} no existe")
                return
            
            # Cargar datos de partidos
            matches_data = pd.read_csv(self.data_path)
            logging.info(f"Datos cargados: {len(matches_data)} partidos")
            
            # Inicializar ELO para todos los jugadores
            all_players = set(matches_data['player_1'].tolist() + matches_data['player_2'].tolist())
            for player in all_players:
                self.players_elo[player] = {
                    'elo': self.initial_elo,
                    'elo_hard': self.initial_elo,
                    'elo_clay': self.initial_elo,
                    'elo_grass': self.initial_elo,
                    'elo_carpet': self.initial_elo,
                    'last_update': None,
                    'matches': 0
                }
            
            logging.info(f"Inicializados ratings ELO para {len(self.players_elo)} jugadores")
            
        except Exception as e:
            logging.error(f"Error cargando datos CSV: {e}")
            traceback.print_exc()
    
    def _get_k_factor(self, surface: str, tournament_type: str = 'default', 
                     player_matches: int = 0) -> float:
        """
        Determina el factor K apropiado para un partido.
        
        Args:
            surface: Superficie del partido (hard, clay, grass, carpet)
            tournament_type: Tipo de torneo (grand_slam, masters, etc.)
            player_matches: Número de partidos del jugador
            
        Returns:
            Factor K ajustado para el partido
        """
        # Obtener factor K base para la superficie
        base_k = self.k_factors.get(surface.lower(), self.k_factors['default'])
        
        # Aplicar multiplicador por tipo de torneo
        tournament_mult = self.tournament_multipliers.get(tournament_type, 
                                                        self.tournament_multipliers['default'])
        
        # Ajustar K para jugadores con pocos partidos (más sensible al principio)
        experience_factor = 1.0
        if player_matches < 30:
            experience_factor = 1.5 - (player_matches / 60)  # 1.5 a 1.0 gradualmente
        
        # Calcular K final
        k_factor = base_k * tournament_mult * experience_factor
        
        return k_factor
    
    def _calculate_expected_score(self, player_elo: float, opponent_elo: float) -> float:
        """
        Calcula la puntuación esperada para un jugador.
        
        Args:
            player_elo: Puntuación ELO del jugador
            opponent_elo: Puntuación ELO del oponente
            
        Returns:
            Puntuación esperada (entre 0 y 1)
        """
        return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
    
    def calculate_elo_change(self, player_elo: float, opponent_elo: float, 
                           result: int, k_factor: float, score_diff: float = None) -> float:
        """
        Calcula el cambio en la puntuación ELO de un jugador después de un partido.
        
        Args:
            player_elo: Puntuación ELO del jugador
            opponent_elo: Puntuación ELO del oponente
            result: Resultado (1 para victoria, 0 para derrota)
            k_factor: Factor K para este partido
            score_diff: Diferencia de puntuación (opcional, para ajustar por margen de victoria)
            
        Returns:
            Cambio en la puntuación ELO
        """
        expected = self._calculate_expected_score(player_elo, opponent_elo)
        
        # Cálculo básico
        elo_change = k_factor * (result - expected)
        
        # Ajuste opcional por margen de victoria
        if score_diff is not None:
            # Ajustar puntuación - un margen mayor causa un cambio mayor (con límite)
            margin_multiplier = min(1.5, 1.0 + score_diff / 10.0)
            elo_change *= margin_multiplier
        
        return elo_change
    
    def update_player_elo(self, player1: str, player2: str, winner: int, 
                        surface: str, tournament_type: str = 'default',
                        score: str = None) -> Dict[str, Dict[str, float]]:
        """
        Actualiza las puntuaciones ELO de dos jugadores después de un partido.
        
        Args:
            player1: Nombre del primer jugador
            player2: Nombre del segundo jugador
            winner: Ganador (0 si player1 ganó, 1 si player2 ganó)
            surface: Superficie del partido
            tournament_type: Tipo de torneo (grand_slam, masters, etc.)
            score: Puntuación del partido (opcional)
            
        Returns:
            Diccionario con nuevos ELO de ambos jugadores
        """
        # Asegurarnos de que la superficie es válida
        surface = surface.lower()
        if surface not in ['hard', 'clay', 'grass', 'carpet']:
            surface = 'hard'
        
        # Calcular diferencia de puntuación si tenemos información del score
        score_diff = None
        if score:
            score_diff = self._calculate_score_difference(score)
        
        # Inicializar jugadores si no existen en nuestro sistema
        if self.csv_mode:
            # Modo CSV - usar diccionario en memoria
            for player in [player1, player2]:
                if player not in self.players_elo:
                    self.players_elo[player] = {
                        'elo': self.initial_elo - self.new_player_penalty,
                        f'elo_{surface}': self.initial_elo - self.new_player_penalty,
                        'elo_hard': self.initial_elo - self.new_player_penalty,
                        'elo_clay': self.initial_elo - self.new_player_penalty,
                        'elo_grass': self.initial_elo - self.new_player_penalty,
                        'elo_carpet': self.initial_elo - self.new_player_penalty,
                        'last_update': datetime.now().strftime('%Y-%m-%d'),
                        'matches': 0
                    }
        else:
            # Modo DB - verificar y crear jugadores en la base de datos
            self._init_players_in_db([player1, player2])
        
        # Obtener ELOs actuales
        p1_elo, p1_surface_elo, p1_matches = self._get_player_current_elo(player1, surface)
        p2_elo, p2_surface_elo, p2_matches = self._get_player_current_elo(player2, surface)
        
        # Determinar resultados (1 = victoria, 0 = derrota)
        p1_result = 1.0 if winner == 0 else 0.0
        p2_result = 1.0 if winner == 1 else 0.0
        
        # Calcular factores K adecuados
        p1_k = self._get_k_factor(surface, tournament_type, p1_matches)
        p2_k = self._get_k_factor(surface, tournament_type, p2_matches)
        
        # Calcular cambios de ELO (general)
        p1_elo_change = self.calculate_elo_change(p1_elo, p2_elo, p1_result, p1_k, score_diff)
        p2_elo_change = self.calculate_elo_change(p2_elo, p1_elo, p2_result, p2_k, score_diff)
        
        # Calcular cambios de ELO (específico por superficie)
        p1_surface_elo_change = self.calculate_elo_change(
            p1_surface_elo, p2_surface_elo, p1_result, p1_k * 1.2, score_diff)
        p2_surface_elo_change = self.calculate_elo_change(
            p2_surface_elo, p1_surface_elo, p2_result, p2_k * 1.2, score_diff)
        
        # Calcular nuevos ELOs
        new_p1_elo = p1_elo + p1_elo_change
        new_p2_elo = p2_elo + p2_elo_change
        new_p1_surface_elo = p1_surface_elo + p1_surface_elo_change
        new_p2_surface_elo = p2_surface_elo + p2_surface_elo_change
        
        # Actualizar ELOs en almacenamiento
        if self.csv_mode:
            # Actualizar en diccionario de memoria
            self._update_player_elo_in_memory(
                player1, new_p1_elo, surface, new_p1_surface_elo)
            self._update_player_elo_in_memory(
                player2, new_p2_elo, surface, new_p2_surface_elo)
        else:
            # Actualizar en base de datos
            self._update_player_elo_in_db(
                player1, new_p1_elo, surface, new_p1_surface_elo)
            self._update_player_elo_in_db(
                player2, new_p2_elo, surface, new_p2_surface_elo)
        
        # Registro para debugging y análisis
        logging.debug(f"Partido: {player1} vs {player2} en {surface} - Ganador: {'Jugador 1' if winner == 0 else 'Jugador 2'}")
        logging.debug(f"ELO {player1}: {p1_elo:.1f} -> {new_p1_elo:.1f} (Δ{p1_elo_change:.1f})")
        logging.debug(f"ELO {player2}: {p2_elo:.1f} -> {new_p2_elo:.1f} (Δ{p2_elo_change:.1f})")
        
        # Devolver nuevos valores
        return {
            player1: {
                'elo': new_p1_elo,
                f'elo_{surface}': new_p1_surface_elo,
                'change': p1_elo_change,
                'surface_change': p1_surface_elo_change
            },
            player2: {
                'elo': new_p2_elo,
                f'elo_{surface}': new_p2_surface_elo,
                'change': p2_elo_change,
                'surface_change': p2_surface_elo_change
            }
        }
    
    def _calculate_score_difference(self, score: str) -> float:
        """
        Calcula la diferencia de puntuación a partir del score.
        
        Args:
            score: String con el resultado (ej: "6-4 7-5")
            
        Returns:
            Diferencia relativa de juegos ganados (0-10 aproximadamente)
        """
        try:
            # Normalizar formato de score
            score = score.replace('(', ' ').replace(')', ' ').strip()
            
            # Procesar cada set
            sets = score.split()
            sets_diff = 0
            games_diff = 0
            sets_p1 = 0
            sets_p2 = 0
            
            for set_score in sets:
                # Omitir texto no numérico
                if '-' not in set_score:
                    continue
                
                parts = set_score.split('-')
                if len(parts) != 2:
                    continue
                
                try:
                    # Manejar tiebreaks
                    if '(' in parts[1]:
                        parts[1] = parts[1].split('(')[0]
                    
                    # Convertir a números
                    p1_games = int(parts[0])
                    p2_games = int(parts[1])
                    
                    # Contar sets ganados
                    if p1_games > p2_games:
                        sets_p1 += 1
                    else:
                        sets_p2 += 1
                    
                    # Acumular diferencia de juegos
                    games_diff += (p1_games - p2_games)
                except:
                    continue
            
            # Calcular diferencia de sets
            sets_diff = sets_p1 - sets_p2
            
            # Puntaje combinado basado en sets y juegos
            # Nos da un valor aproximadamente entre 0 y 10
            combined_diff = abs(sets_diff) * 2 + abs(games_diff) / 2
            
            # Normalizar y devolver con signo
            return combined_diff * (1 if sets_diff > 0 else -1)
            
        except Exception as e:
            logging.warning(f"Error calculando diferencia de score '{score}': {e}")
            return None
    
    def _get_player_current_elo(self, player: str, surface: str) -> Tuple[float, float, int]:
        """
        Obtiene los ratings ELO actuales de un jugador.
        
        Args:
            player: Nombre del jugador
            surface: Superficie
            
        Returns:
            Tupla (elo_general, elo_superficie, num_partidos)
        """
        if self.csv_mode:
            # Modo CSV - obtener de diccionario en memoria
            if player in self.players_elo:
                return (
                    self.players_elo[player]['elo'],
                    self.players_elo[player].get(f'elo_{surface}', self.initial_elo),
                    self.players_elo[player].get('matches', 0)
                )
            else:
                return (self.initial_elo - self.new_player_penalty, 
                        self.initial_elo - self.new_player_penalty, 0)
        else:
            # Modo DB - obtener de base de datos
            return self._get_player_elo_from_db(player, surface)
    
    def _update_player_elo_in_memory(self, player: str, new_elo: float, 
                                   surface: str, new_surface_elo: float) -> None:
        """
        Actualiza el ELO de un jugador en el diccionario en memoria.
        
        Args:
            player: Nombre del jugador
            new_elo: Nuevo ELO general
            surface: Superficie
            new_surface_elo: Nuevo ELO específico para la superficie
        """
        if player not in self.players_elo:
            self.players_elo[player] = {
                'elo': new_elo,
                f'elo_{surface}': new_surface_elo,
                'elo_hard': self.initial_elo,
                'elo_clay': self.initial_elo,
                'elo_grass': self.initial_elo,
                'elo_carpet': self.initial_elo,
                'last_update': datetime.now().strftime('%Y-%m-%d'),
                'matches': 1
            }
        else:
            # Actualizar ELO general
            self.players_elo[player]['elo'] = new_elo
            
            # Actualizar ELO específico por superficie
            self.players_elo[player][f'elo_{surface}'] = new_surface_elo
            
            # Actualizar fecha y contador de partidos
            self.players_elo[player]['last_update'] = datetime.now().strftime('%Y-%m-%d')
            self.players_elo[player]['matches'] += 1
        
        # Registrar en historial
        if player not in self.elo_history:
            self.elo_history[player] = []
        
        self.elo_history[player].append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'elo': new_elo,
            f'elo_{surface}': new_surface_elo
        })
    
    def _get_player_elo_from_db(self, player_id: Union[int, str], surface: str = None) -> Tuple[float, float, int]:
        """
        Obtiene el ELO de un jugador desde la base de datos.
        
        Args:
            player_id: ID o nombre del jugador
            surface: Superficie (opcional)
            
        Returns:
            Tupla (elo_general, elo_superficie, num_partidos)
        """
        try:
            if not self.db_connection:
                return (self.initial_elo, self.initial_elo, 0)
            
            cursor = self.db_connection.cursor()
            
            # Determinar si player_id es un ID numérico o un nombre
            if isinstance(player_id, int) or player_id.isdigit():
                where_clause = "id = %s"
            else:
                where_clause = "name = %s"
            
            # Obtener información del jugador
            query = f"""
                SELECT 
                    elo_rating, 
                    elo_hard, 
                    elo_clay, 
                    elo_grass, 
                    elo_carpet, 
                    matches_played
                FROM 
                    players
                WHERE 
                    {where_clause}
            """
            
            cursor.execute(query, (player_id,))
            result = cursor.fetchone()
            
            if result:
                elo_general, elo_hard, elo_clay, elo_grass, elo_carpet, matches = result
                
                # Determinar ELO específico por superficie
                if surface:
                    surface_column = f'elo_{surface.lower()}'
                    elo_superficie = locals().get(surface_column, elo_general)
                else:
                    elo_superficie = elo_general
                
                cursor.close()
                return (elo_general, elo_superficie, matches)
            else:
                # Jugador no encontrado
                cursor.close()
                return (self.initial_elo - self.new_player_penalty, 
                        self.initial_elo - self.new_player_penalty, 0)
                
        except Exception as e:
            logging.error(f"Error obteniendo ELO desde DB: {e}")
            return (self.initial_elo, self.initial_elo, 0)
    
    def _update_player_elo_in_db(self, player_id: Union[int, str], new_elo: float,
                               surface: str, new_surface_elo: float) -> bool:
        """
        Actualiza el ELO de un jugador en la base de datos.
        
        Args:
            player_id: ID o nombre del jugador
            new_elo: Nuevo ELO general
            surface: Superficie
            new_surface_elo: Nuevo ELO específico por superficie
            
        Returns:
            True si la actualización fue exitosa, False en caso contrario
        """
        try:
            if not self.db_connection:
                return False
            
            cursor = self.db_connection.cursor()
            
            # Determinar si player_id es un ID numérico o un nombre
            if isinstance(player_id, int) or player_id.isdigit():
                where_clause = "id = %s"
            else:
                where_clause = "name = %s"
            
            # Actualizar ELO general
            query = f"""
                UPDATE players 
                SET 
                    elo_rating = %s, 
                    elo_last_update = CURRENT_TIMESTAMP,
                    matches_played = matches_played + 1
                WHERE 
                    {where_clause}
            """
            
            cursor.execute(query, (new_elo, player_id))
            
            # Actualizar ELO específico por superficie
            surface_column = f'elo_{surface.lower()}'
            
            query = f"""
                UPDATE players 
                SET 
                    {surface_column} = %s
                WHERE 
                    {where_clause}
            """
            
            cursor.execute(query, (new_surface_elo, player_id))
            
            # Registrar en historial
            query = """
                INSERT INTO player_elo_history 
                (player_id, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet, date, match_id) 
                VALUES (
                    (SELECT id FROM players WHERE {where_clause}),
                    %s,
                    (SELECT elo_hard FROM players WHERE {where_clause}),
                    (SELECT elo_clay FROM players WHERE {where_clause}),
                    (SELECT elo_grass FROM players WHERE {where_clause}),
                    (SELECT elo_carpet FROM players WHERE {where_clause}),
                    CURRENT_TIMESTAMP,
                    NULL
                )
            """.format(where_clause=where_clause)
            
            cursor.execute(query, (player_id, new_elo, player_id, player_id, player_id, player_id))
            
            self.db_connection.commit()
            cursor.close()
            return True
            
        except Exception as e:
            logging.error(f"Error actualizando ELO en DB: {e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False
    
    def _init_players_in_db(self, players: List[str]) -> None:
        """
        Inicializa jugadores en la base de datos si no existen.
        
        Args:
            players: Lista de nombres de jugadores a inicializar
        """
        if not self.db_connection:
            return
        
        try:
            cursor = self.db_connection.cursor()
            
            for player in players:
                # Verificar si el jugador ya existe
                cursor.execute("SELECT id FROM players WHERE name = %s", (player,))
                if cursor.fetchone() is None:
                    # Crear nuevo jugador
                    query = """
                        INSERT INTO players 
                        (name, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet, 
                         elo_last_update, matches_played) 
                        VALUES 
                        (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, 0)
                    """
                    
                    initial_elo = self.initial_elo - self.new_player_penalty
                    cursor.execute(query, (
                        player, initial_elo, initial_elo, initial_elo, 
                        initial_elo, initial_elo
                    ))
            
            self.db_connection.commit()
            cursor.close()
            
        except Exception as e:
            logging.error(f"Error inicializando jugadores en DB: {e}")
            if self.db_connection:
                self.db_connection.rollback()
    
    def recalculate_all_elo(self, matches_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Recalcula todos los ratings ELO desde cero utilizando datos históricos.
        
        Args:
            matches_data: DataFrame con datos de partidos o None para usar DB/CSV configurado
            
        Returns:
            True si el recálculo fue exitoso, False en caso contrario
        """
        try:
            logging.info("Iniciando recálculo completo de ELO")
            start_time = datetime.now()
            
            # Restablecer todos los ELO a valores iniciales
            self._initialize_all_elo()
            
            # Obtener datos de partidos
            if matches_data is None:
                if self.csv_mode and self.data_path:
                    logging.info(f"Cargando partidos desde {self.data_path}")
                    matches_data = pd.read_csv(self.data_path)
                elif self.db_connection:
                    logging.info("Cargando partidos desde base de datos")
                    matches_data = self._get_matches_from_db()
                else:
                    logging.error("No hay fuente de datos para recalcular ELO")
                    return False
            
            if matches_data is None or len(matches_data) == 0:
                logging.warning("No hay datos de partidos para recalcular ELO")
                return False
            
            # Ordenar partidos por fecha
            if 'match_date' in matches_data.columns:
                matches_data = matches_data.sort_values('match_date')
            
            # Procesar cada partido cronológicamente
            total_matches = len(matches_data)
            logging.info(f"Procesando {total_matches} partidos cronológicamente")
            
            for i, match in enumerate(matches_data.iterrows()):
                if i % 1000 == 0 and i > 0:
                    logging.info(f"Procesados {i}/{total_matches} partidos ({i/total_matches*100:.1f}%)")
                
                match = match[1]  # Obtener la serie (2da parte de la tupla)
                
                # Verificar que tenemos los datos necesarios
                required_cols = ['player_1', 'player_2', 'winner', 'surface']
                if not all(col in match for col in required_cols):
                    continue
                
                player1 = match['player_1']
                player2 = match['player_2']
                winner = match['winner']
                surface = match['surface']
                
                # Determinar tipo de torneo si está disponible
                tournament_type = 'default'
                if 'tournament_type' in match:
                    tournament_type = match['tournament_type']
                elif 'tournament' in match:
                    # Intentar inferir tipo basado en nombre del torneo
                    tournament_name = str(match['tournament']).lower()
                    if any(gs in tournament_name for gs in ['grand slam', 'open', 'wimbledon']):
                        tournament_type = 'grand_slam'
                    elif any(m in tournament_name for m in ['masters', '1000']):
                        tournament_type = 'masters'
                    elif any(c in tournament_name for c in ['challenger', 'futures']):
                        tournament_type = 'challenger'
                
                # Obtener score si está disponible
                score = match.get('score', None)
                
                # Actualizar ELO
                self.update_player_elo(player1, player2, winner, surface, tournament_type, score)
            
            # Guardar resultados finales si estamos en modo CSV
            if self.csv_mode:
                self._save_elo_to_csv()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"Recálculo de ELO completado en {duration:.2f} segundos")
            
            return True
            
        except Exception as e:
            logging.error(f"Error recalculando ELO: {e}")
            traceback.print_exc()
            return False
    
    def _initialize_all_elo(self) -> bool:
        """
        Inicializa los ratings ELO para todos los jugadores.
        
        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        """
        try:
            if self.csv_mode:
                # Modo CSV - inicializar en memoria
                for player in self.players_elo:
                    self.players_elo[player] = {
                        'elo': self.initial_elo,
                        'elo_hard': self.initial_elo,
                        'elo_clay': self.initial_elo,
                        'elo_grass': self.initial_elo,
                        'elo_carpet': self.initial_elo,
                        'last_update': datetime.now().strftime('%Y-%m-%d'),
                        'matches': 0
                    }
                
                # Limpiar historial
                self.elo_history = {}
                
                logging.info(f"Inicializados {len(self.players_elo)} jugadores en memoria")
                
            elif self.db_connection:
                # Modo DB - inicializar en base de datos
                cursor = self.db_connection.cursor()
                
                # Inicializar ELO para todos los jugadores
                cursor.execute(
                    "UPDATE players SET elo_rating = %s, elo_hard = %s, elo_clay = %s, "
                    "elo_grass = %s, elo_carpet = %s, elo_last_update = CURRENT_TIMESTAMP, "
                    "matches_played = 0",
                    (self.initial_elo, self.initial_elo, self.initial_elo, 
                     self.initial_elo, self.initial_elo)
                )
                
                # Limpiar historial ELO
                cursor.execute("DELETE FROM player_elo_history")
                
                self.db_connection.commit()
                cursor.close()
                
                logging.info("Inicializados todos los jugadores en base de datos")
            
            return True
            
        except Exception as e:
            logging.error(f"Error inicializando ELO: {e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False
    
    def _get_matches_from_db(self) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de partidos desde la base de datos.
        
        Returns:
            DataFrame con datos de partidos o None si hay error
        """
        try:
            if not self.db_connection:
                return None
            
            query = """
                SELECT 
                    m.id, 
                    p1.name as player_1, 
                    p2.name as player_2, 
                    CASE WHEN m.winner_id = m.player1_id THEN 0 ELSE 1 END as winner, 
                    m.surface, 
                    m.match_date,
                    t.name as tournament,
                    t.category as tournament_type,
                    m.score
                FROM 
                    matches m
                JOIN 
                    players p1 ON m.player1_id = p1.id
                JOIN 
                    players p2 ON m.player2_id = p2.id
                LEFT JOIN
                    tournaments t ON m.tournament_id = t.id
                WHERE 
                    m.player1_id IS NOT NULL AND 
                    m.player2_id IS NOT NULL AND 
                    m.winner_id IS NOT NULL AND 
                    m.surface IS NOT NULL
                ORDER BY 
                    m.match_date
            """
            
            matches_data = pd.read_sql_query(query, self.db_connection)
            logging.info(f"Cargados {len(matches_data)} partidos desde la base de datos")
            
            return matches_data
            
        except Exception as e:
            logging.error(f"Error obteniendo partidos desde DB: {e}")
            return None
    
    def _save_elo_to_csv(self, output_path: str = None) -> bool:
        """
        Guarda los ratings ELO actuales a un archivo CSV.
        
        Args:
            output_path: Ruta de salida (por defecto: elo_rankings.csv)
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            if not output_path:
                output_path = 'data/elo_rankings.csv'
            
            # Crear DataFrame con datos de ELO
            data = []
            for player, elo_data in self.players_elo.items():
                row = {'player': player}
                row.update(elo_data)
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Ordenar por ELO descendente
            df = df.sort_values(by='elo', ascending=False)
            
            # Guardar a CSV
            df.to_csv(output_path, index=False)
            logging.info(f"Ratings ELO guardados en {output_path}")
            
            # Guardar también historial
            history_path = output_path.replace('.csv', '_history.csv')
            history_data = []
            
            for player, history in self.elo_history.items():
                for entry in history:
                    row = {'player': player}
                    row.update(entry)
                    history_data.append(row)
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                history_df.to_csv(history_path, index=False)
                logging.info(f"Historial ELO guardado en {history_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error guardando ELO a CSV: {e}")
            return False
    
    def get_elo_ranking(self, surface: Optional[str] = None, 
                      min_matches: int = 10, limit: int = 100) -> pd.DataFrame:
        """
        Obtiene el ranking de jugadores basado en ELO.
        
        Args:
            surface: Superficie específica o None para ranking general
            min_matches: Mínimo de partidos jugados para aparecer en el ranking
            limit: Límite de jugadores a devolver
            
        Returns:
            DataFrame con ranking ELO
        """
        try:
            if self.csv_mode:
                # Modo CSV - generar ranking desde memoria
                data = []
                for player, elo_data in self.players_elo.items():
                    matches = elo_data.get('matches', 0)
                    if matches < min_matches:
                        continue
                    
                    if surface:
                        elo_value = elo_data.get(f'elo_{surface.lower()}', elo_data['elo'])
                    else:
                        elo_value = elo_data['elo']
                    
                    data.append({
                        'player': player,
                        'elo': elo_value,
                        'matches': matches
                    })
                
                # Crear DataFrame y ordenar
                ranking_df = pd.DataFrame(data).sort_values(by='elo', ascending=False)
                
                # Limitar resultados
                if len(ranking_df) > limit:
                    ranking_df = ranking_df.head(limit)
                
                return ranking_df
                
            elif self.db_connection:
                # Modo DB - obtener ranking desde base de datos
                cursor = self.db_connection.cursor()
                
                # Determinar qué columna ELO usar
                elo_column = f"elo_{surface.lower()}" if surface else "elo_rating"
                
                query = f"""
                    SELECT 
                        id, name, {elo_column} AS elo, current_ranking AS atp_ranking,
                        country, dominant_hand, matches_played
                    FROM 
                        players
                    WHERE 
                        {elo_column} IS NOT NULL AND
                        matches_played >= %s
                    ORDER BY 
                        {elo_column} DESC
                    LIMIT 
                        %s
                """
                
                cursor.execute(query, (min_matches, limit))
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                cursor.close()
                
                # Convertir a DataFrame
                ranking_df = pd.DataFrame(results, columns=columns)
                
                return ranking_df
            
            # Si no hay modo válido, devolver DataFrame vacío
            return pd.DataFrame()
            
        except Exception as e:
            logging.error(f"Error obteniendo ranking ELO: {e}")
            return pd.DataFrame()
    
    def get_player_elo_history(self, player_id: Union[int, str]) -> pd.DataFrame:
        """
        Obtiene el historial de ELO de un jugador.
        
        Args:
            player_id: ID o nombre del jugador
            
        Returns:
            DataFrame con historial ELO
        """
        try:
            if self.csv_mode:
                # Modo CSV - obtener desde memoria
                if isinstance(player_id, int):
                    # No podemos buscar por ID en modo CSV
                    return pd.DataFrame()
                
                # Buscar por nombre
                player_name = player_id
                if player_name in self.elo_history:
                    return pd.DataFrame(self.elo_history[player_name])
                else:
                    return pd.DataFrame()
                
            elif self.db_connection:
                # Modo DB - obtener desde base de datos
                cursor = self.db_connection.cursor()
                
                # Determinar si player_id es un ID numérico o un nombre
                if isinstance(player_id, int) or (isinstance(player_id, str) and player_id.isdigit()):
                    where_clause = "player_id = %s"
                else:
                    where_clause = "player_id = (SELECT id FROM players WHERE name = %s)"
                
                query = f"""
                    SELECT 
                        player_id, 
                        date, 
                        elo_rating, 
                        elo_hard, 
                        elo_clay, 
                        elo_grass, 
                        elo_carpet,
                        match_id, 
                        notes
                    FROM 
                        player_elo_history
                    WHERE 
                        {where_clause}
                    ORDER BY 
                        date
                """
                
                cursor.execute(query, (player_id,))
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                cursor.close()
                
                # Convertir a DataFrame
                history_df = pd.DataFrame(results, columns=columns)
                
                return history_df
            
            # Si no hay modo válido, devolver DataFrame vacío
            return pd.DataFrame()
            
        except Exception as e:
            logging.error(f"Error obteniendo historial ELO: {e}")
            return pd.DataFrame()
    
    def predict_match_outcome(self, player1: str, player2: str, 
                            surface: str = 'hard') -> Dict[str, float]:
        """
        Predice el resultado de un partido basado en ratings ELO.
        
        Args:
            player1: Nombre del primer jugador
            player2: Nombre del segundo jugador
            surface: Superficie del partido
            
        Returns:
            Diccionario con probabilidades
        """
        # Obtener ELOs actuales
        p1_elo, p1_surface_elo, _ = self._get_player_current_elo(player1, surface)
        p2_elo, p2_surface_elo, _ = self._get_player_current_elo(player2, surface)
        
        # Calcular ELO efectivo (combinación de general y superficie)
        p1_effective_elo = 0.6 * p1_elo + 0.4 * p1_surface_elo
        p2_effective_elo = 0.6 * p2_elo + 0.4 * p2_surface_elo
        
        # Calcular probabilidad usando fórmula ELO
        p1_win_prob = self._calculate_expected_score(p1_effective_elo, p2_effective_elo)
        p2_win_prob = 1 - p1_win_prob
        
        logging.info(
            f"Predicción: {player1} ({p1_effective_elo:.1f}) vs {player2} ({p2_effective_elo:.1f}) "
            f"en {surface} - P({player1}): {p1_win_prob:.1%}, P({player2}): {p2_win_prob:.1%}"
        )
        
        return {
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'p1_probability': p1_win_prob,
            'p2_probability': p2_win_prob,
            'p1_elo': p1_elo,
            'p2_elo': p2_elo,
            'p1_surface_elo': p1_surface_elo,
            'p2_surface_elo': p2_surface_elo,
            'p1_effective_elo': p1_effective_elo,
            'p2_effective_elo': p2_effective_elo
        }
    
    def plot_elo_history(self, player_names: List[str], output_path: str = None,
                        start_date: str = None, end_date: str = None,
                        surface: str = None, show_plot: bool = False) -> None:
        """
        Crea un gráfico con el historial de ELO de uno o más jugadores.
        
        Args:
            player_names: Lista de nombres de jugadores
            output_path: Ruta donde guardar el gráfico (opcional)
            start_date: Fecha de inicio (formato: 'YYYY-MM-DD', opcional)
            end_date: Fecha final (formato: 'YYYY-MM-DD', opcional)
            surface: Superficie específica para mostrar ELO (opcional)
            show_plot: Si es True, muestra el gráfico (en entornos interactivos)
        """
        plt.figure(figsize=(12, 8))
        
        for player_name in player_names:
            # Obtener historial
            history_df = self.get_player_elo_history(player_name)
            
            if history_df.empty:
                logging.warning(f"No hay historial ELO para {player_name}")
                continue
            
            # Aplicar filtros de fecha
            if start_date:
                history_df = history_df[history_df['date'] >= start_date]
            
            if end_date:
                history_df = history_df[history_df['date'] <= end_date]
            
            if history_df.empty:
                logging.warning(f"No hay datos para {player_name} en el rango de fechas especificado")
                continue
            
            # Determinar qué columna usar
            if surface:
                elo_column = f'elo_{surface.lower()}'
                if elo_column not in history_df.columns:
                    logging.warning(f"No hay datos de ELO para superficie {surface}")
                    continue
                label = f"{player_name} ({surface})"
            else:
                elo_column = 'elo_rating'
                label = player_name
            
            # Graficar
            plt.plot(history_df['date'], history_df[elo_column], label=label, linewidth=2)
        
        # Configurar gráfico
        plt.title('Evolución del Rating ELO', fontsize=16)
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Rating ELO', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Añadir líneas de referencia para niveles de ELO
        plt.axhline(y=2400, color='r', linestyle='--', alpha=0.5, label='Elite (2400+)')
        plt.axhline(y=2000, color='g', linestyle='--', alpha=0.5, label='Profesional (2000+)')
        plt.axhline(y=1800, color='y', linestyle='--', alpha=0.5, label='Avanzado (1800+)')
        
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logging.info(f"Gráfico guardado en {output_path}")
        
        # Mostrar si se solicita
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def export_elo_data(self, output_path: str = 'data/elo_data.json') -> bool:
        """
        Exporta todos los datos ELO a un archivo JSON.
        
        Args:
            output_path: Ruta de salida del archivo JSON
            
        Returns:
            True si la exportación fue exitosa, False en caso contrario
        """
        try:
            if self.csv_mode:
                # Exportar desde memoria
                data = {
                    'players': self.players_elo,
                    'history': self.elo_history,
                    'metadata': {
                        'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'initial_elo': self.initial_elo,
                        'k_factors': self.k_factors
                    }
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                logging.info(f"Datos ELO exportados a {output_path}")
                return True
                
            elif self.db_connection:
                # Exportar desde base de datos
                cursor = self.db_connection.cursor()
                
                # Obtener datos de jugadores
                cursor.execute("""
                    SELECT 
                        id, name, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet,
                        elo_last_update, matches_played
                    FROM 
                        players
                    WHERE 
                        elo_rating IS NOT NULL
                """)
                
                columns = [desc[0] for desc in cursor.description]
                players_data = []
                
                for row in cursor.fetchall():
                    player_dict = dict(zip(columns, row))
                    
                    # Convertir datetime a string
                    if 'elo_last_update' in player_dict and player_dict['elo_last_update']:
                        player_dict['elo_last_update'] = player_dict['elo_last_update'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    players_data.append(player_dict)
                
                # Obtener historial ELO (limitado para no sobrecargar)
                cursor.execute("""
                    SELECT 
                        player_id, 
                        (SELECT name FROM players WHERE id = player_id) as player_name,
                        date, 
                        elo_rating, 
                        elo_hard, 
                        elo_clay, 
                        elo_grass, 
                        elo_carpet
                    FROM 
                        player_elo_history
                    ORDER BY 
                        player_id, date
                    LIMIT 10000
                """)
                
                history_columns = [desc[0] for desc in cursor.description]
                history_data = []
                
                for row in cursor.fetchall():
                    history_dict = dict(zip(history_columns, row))
                    
                    # Convertir datetime a string
                    if 'date' in history_dict and history_dict['date']:
                        history_dict['date'] = history_dict['date'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    history_data.append(history_dict)
                
                cursor.close()
                
                # Crear estructura final
                data = {
                    'players': players_data,
                    'history': history_data,
                    'metadata': {
                        'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'initial_elo': self.initial_elo,
                        'k_factors': self.k_factors,
                        'tournament_multipliers': self.tournament_multipliers
                    }
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                logging.info(f"Datos ELO exportados a {output_path}")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error exportando datos ELO: {e}")
            return False

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
    elo_system = TennisELOSystem(db_connection=db_connection, data_path=args.data)
    
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