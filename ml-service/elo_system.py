#!/usr/bin/env python3
"""
Módulo que implementa el sistema ELO para tenis.
El sistema ELO proporciona una mejor medida de la habilidad de los jugadores
comparado con los rankings oficiales.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import psycopg2
from psycopg2 import sql
from typing import Dict, Tuple, Optional, List, Union

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

class TennisELOSystem:
    """
    Clase que implementa el sistema ELO para tenis.
    """
    
    def __init__(self, db_connection=None):
        """
        Inicializa el sistema ELO.
        
        Args:
            db_connection: Conexión a la base de datos PostgreSQL
        """
        self.db_connection = db_connection
        
        # Factores K por superficie (el factor K determina cuánto cambia el ELO después de cada partido)
        self.k_factors = {
            'hard': 32,     # Superficie común, factor K estándar
            'clay': 24,     # Superficie más predecible, factor K más bajo
            'grass': 40,    # Superficie más variable, factor K más alto
            'carpet': 36,   # Superficie indoor, factor K alto-medio
            'default': 32   # Valor por defecto para otras superficies
        }
        
        # ELO inicial para jugadores nuevos
        self.initial_elo = 1500
    
    def _get_k_factor(self, surface: str) -> int:
        """
        Determina el factor K apropiado para una superficie dada.
        
        Args:
            surface: Superficie del partido (hard, clay, grass, carpet)
            
        Returns:
            Factor K para la superficie
        """
        return self.k_factors.get(surface.lower(), self.k_factors['default'])
    
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
                           result: int, surface: str) -> float:
        """
        Calcula el cambio en la puntuación ELO de un jugador después de un partido.
        
        Args:
            player_elo: Puntuación ELO del jugador
            opponent_elo: Puntuación ELO del oponente
            result: Resultado (1 para victoria, 0 para derrota)
            surface: Superficie del partido
            
        Returns:
            Cambio en la puntuación ELO
        """
        expected = self._calculate_expected_score(player_elo, opponent_elo)
        k = self._get_k_factor(surface)
        return k * (result - expected)
    
    def update_player_elo(self, player_id: int, opponent_id: int, 
                        result: int, surface: str) -> Tuple[float, float]:
        """
        Actualiza las puntuaciones ELO de dos jugadores después de un partido.
        
        Args:
            player_id: ID del jugador
            opponent_id: ID del oponente
            result: Resultado (1 si player_id ganó, 0 si perdió)
            surface: Superficie del partido
            
        Returns:
            Nuevas puntuaciones ELO (player_elo, opponent_elo)
        """
        # Obtener puntuaciones ELO actuales
        player_elo, player_surface_elo = self.get_player_elo(player_id, surface)
        opponent_elo, opponent_surface_elo = self.get_player_elo(opponent_id, surface)
        
        # Calcular cambios ELO (global)
        player_elo_change = self.calculate_elo_change(player_elo, opponent_elo, result, surface)
        opponent_elo_change = self.calculate_elo_change(opponent_elo, player_elo, 1 - result, surface)
        
        # Calcular cambios ELO (específico por superficie)
        player_surface_elo_change = self.calculate_elo_change(
            player_surface_elo, opponent_surface_elo, result, surface)
        opponent_surface_elo_change = self.calculate_elo_change(
            opponent_surface_elo, player_surface_elo, 1 - result, surface)
        
        # Calcular nuevos ELO
        new_player_elo = player_elo + player_elo_change
        new_opponent_elo = opponent_elo + opponent_elo_change
        new_player_surface_elo = player_surface_elo + player_surface_elo_change
        new_opponent_surface_elo = opponent_surface_elo + opponent_surface_elo_change
        
        # Actualizar en base de datos
        if self.db_connection:
            self._update_player_elo_in_db(
                player_id, new_player_elo, surface, new_player_surface_elo)
            self._update_player_elo_in_db(
                opponent_id, new_opponent_elo, surface, new_opponent_surface_elo)
        
        return (new_player_elo, new_opponent_elo)
    
    def get_player_elo(self, player_id: int, surface: Optional[str] = None) -> Tuple[float, float]:
        """
        Obtiene la puntuación ELO actual de un jugador.
        
        Args:
            player_id: ID del jugador
            surface: Superficie (opcional)
            
        Returns:
            Tupla (elo_general, elo_superficie)
        """
        if self.db_connection:
            return self._get_player_elo_from_db(player_id, surface)
        else:
            # Valores por defecto si no hay conexión a DB
            return (self.initial_elo, self.initial_elo)
    
    def _get_player_elo_from_db(self, player_id: int, surface: Optional[str] = None) -> Tuple[float, float]:
        """
        Obtiene la puntuación ELO de un jugador desde la base de datos.
        
        Args:
            player_id: ID del jugador
            surface: Superficie (opcional)
            
        Returns:
            Tupla (elo_general, elo_superficie)
        """
        try:
            cursor = self.db_connection.cursor()
            
            # Obtener ELO general
            cursor.execute(
                "SELECT elo_rating FROM players WHERE id = %s", 
                (player_id,)
            )
            result = cursor.fetchone()
            elo_general = result[0] if result else self.initial_elo
            
            # Obtener ELO específico por superficie
            if surface:
                surface_column = f"elo_{surface.lower()}"
                cursor.execute(
                    sql.SQL("SELECT {} FROM players WHERE id = %s").format(
                        sql.Identifier(surface_column)
                    ), 
                    (player_id,)
                )
                result = cursor.fetchone()
                elo_superficie = result[0] if result else self.initial_elo
            else:
                elo_superficie = self.initial_elo
                
            cursor.close()
            return (elo_general, elo_superficie)
            
        except Exception as e:
            logging.error(f"Error obteniendo ELO desde DB: {e}")
            return (self.initial_elo, self.initial_elo)
    
    def _update_player_elo_in_db(self, player_id: int, new_elo: float, 
                               surface: str, new_surface_elo: float) -> bool:
        """
        Actualiza la puntuación ELO de un jugador en la base de datos.
        
        Args:
            player_id: ID del jugador
            new_elo: Nueva puntuación ELO general
            surface: Superficie
            new_surface_elo: Nueva puntuación ELO específica por superficie
            
        Returns:
            True si la actualización fue exitosa, False en caso contrario
        """
        try:
            cursor = self.db_connection.cursor()
            
            # Actualizar ELO general
            cursor.execute(
                "UPDATE players SET elo_rating = %s, elo_last_update = CURRENT_TIMESTAMP WHERE id = %s",
                (new_elo, player_id)
            )
            
            # Actualizar ELO específico por superficie
            surface_column = f"elo_{surface.lower()}"
            cursor.execute(
                sql.SQL("UPDATE players SET {} = %s WHERE id = %s").format(
                    sql.Identifier(surface_column)
                ),
                (new_surface_elo, player_id)
            )
            
            # Registrar cambio en historial
            cursor.execute(
                "INSERT INTO player_elo_history (player_id, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet) "
                "VALUES (%s, %s, "
                "(SELECT elo_hard FROM players WHERE id = %s), "
                "(SELECT elo_clay FROM players WHERE id = %s), "
                "(SELECT elo_grass FROM players WHERE id = %s), "
                "(SELECT elo_carpet FROM players WHERE id = %s))",
                (player_id, new_elo, player_id, player_id, player_id, player_id)
            )
            
            self.db_connection.commit()
            cursor.close()
            return True
            
        except Exception as e:
            logging.error(f"Error actualizando ELO en DB: {e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False
    
    def recalculate_all_elo(self, matches_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Recalcula todas las puntuaciones ELO desde cero utilizando datos históricos.
        
        Args:
            matches_data: DataFrame con datos de partidos o None para usar DB
            
        Returns:
            True si el recálculo fue exitoso, False en caso contrario
        """
        try:
            # Inicializar ELO para todos los jugadores
            self._initialize_all_player_elo()
            
            # Obtener datos de partidos ordenados por fecha
            if matches_data is None and self.db_connection:
                matches_data = self._get_matches_from_db()
            
            if matches_data is None or len(matches_data) == 0:
                logging.warning("No hay datos de partidos para recalcular ELO")
                return False
            
            # Ordenar partidos por fecha
            matches_data = matches_data.sort_values('match_date')
            
            # Procesar cada partido
            for _, match in matches_data.iterrows():
                player1_id = match['player1_id']
                player2_id = match['player2_id']
                winner_id = match['winner_id']
                surface = match['surface']
                
                # Determinar resultado (1 = victoria, 0 = derrota)
                result_player1 = 1 if winner_id == player1_id else 0
                
                # Actualizar ELO
                self.update_player_elo(player1_id, player2_id, result_player1, surface)
            
            logging.info("Recálculo de ELO completado exitosamente")
            return True
            
        except Exception as e:
            logging.error(f"Error recalculando ELO: {e}")
            return False
    
    def _initialize_all_player_elo(self) -> bool:
        """
        Inicializa las puntuaciones ELO para todos los jugadores.
        
        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        """
        if not self.db_connection:
            logging.warning("No hay conexión a DB para inicializar ELO")
            return False
        
        try:
            cursor = self.db_connection.cursor()
            
            # Inicializar ELO para todos los jugadores
            cursor.execute(
                "UPDATE players SET elo_rating = %s, elo_hard = %s, elo_clay = %s, "
                "elo_grass = %s, elo_carpet = %s, elo_last_update = CURRENT_TIMESTAMP",
                (self.initial_elo, self.initial_elo, self.initial_elo, self.initial_elo, self.initial_elo)
            )
            
            # Limpiar historial ELO
            cursor.execute("DELETE FROM player_elo_history")
            
            self.db_connection.commit()
            cursor.close()
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
            query = """
                SELECT 
                    id, player1_id, player2_id, winner_id, surface, match_date
                FROM 
                    matches 
                WHERE 
                    player1_id IS NOT NULL AND 
                    player2_id IS NOT NULL AND 
                    winner_id IS NOT NULL AND 
                    surface IS NOT NULL
                ORDER BY 
                    match_date
            """
            
            matches_data = pd.read_sql_query(query, self.db_connection)
            return matches_data
            
        except Exception as e:
            logging.error(f"Error obteniendo partidos desde DB: {e}")
            return None
    
    def get_elo_ranking(self, surface: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """
        Obtiene el ranking de jugadores basado en ELO.
        
        Args:
            surface: Superficie específica o None para ranking general
            limit: Límite de jugadores a devolver
            
        Returns:
            DataFrame con ranking ELO
        """
        if not self.db_connection:
            logging.warning("No hay conexión a DB para obtener ranking ELO")
            return pd.DataFrame()
        
        try:
            # Determinar qué columna ELO usar
            elo_column = f"elo_{surface.lower()}" if surface else "elo_rating"
            
            query = f"""
                SELECT 
                    id, name, {elo_column} AS elo, current_ranking AS atp_ranking,
                    country, dominant_hand
                FROM 
                    players
                WHERE 
                    {elo_column} IS NOT NULL
                ORDER BY 
                    {elo_column} DESC
                LIMIT 
                    {limit}
            """
            
            ranking_data = pd.read_sql_query(query, self.db_connection)
            return ranking_data
            
        except Exception as e:
            logging.error(f"Error obteniendo ranking ELO: {e}")
            return pd.DataFrame()
    
    def get_player_elo_history(self, player_id: int) -> pd.DataFrame:
        """
        Obtiene el historial de ELO de un jugador.
        
        Args:
            player_id: ID del jugador
            
        Returns:
            DataFrame con historial ELO
        """
        if not self.db_connection:
            logging.warning("No hay conexión a DB para obtener historial ELO")
            return pd.DataFrame()
        
        try:
            query = """
                SELECT 
                    player_id, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet,
                    date, match_id, notes
                FROM 
                    player_elo_history
                WHERE 
                    player_id = %s
                ORDER BY 
                    date
            """
            
            history_data = pd.read_sql_query(query, self.db_connection, params=(player_id,))
            return history_data
            
        except Exception as e:
            logging.error(f"Error obteniendo historial ELO: {e}")
            return pd.DataFrame()