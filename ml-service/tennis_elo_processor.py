"""
tennis_elo_processor.py

Script optimizado para calcular ratings ELO a partir de los datos recopilados por
CompleteTennisDataCollector. Incluye:
- Cálculo de ELO general y específico por superficie
- Factores K dinámicos basados en importancia del torneo/ronda
- Ponderación por margen de victoria
- Decaimiento temporal para jugadores inactivos
- Generación de características ELO para modelos predictivos
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import sys

# Crear directorio de logs si no existe
os.makedirs('logs', exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/elo_processing.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)

class TennisEloProcessor:
    """
    Procesador avanzado de ELO para tenis que implementa múltiples mejoras
    sobre el sistema tradicional.
    """
    
    def __init__(self, 
                initial_rating: float = 1500,
                k_factor_base: float = 32, 
                decay_rate: float = 0.995,
                surface_spec_weight: float = 0.7):
        """
        Inicializa el procesador de ELO.
        
        Args:
            initial_rating: Rating ELO inicial para nuevos jugadores
            k_factor_base: Factor K base para ajustes de rating
            decay_rate: Tasa de decaimiento mensual para jugadores inactivos
            surface_spec_weight: Peso para ELO específico por superficie vs ELO general
        """
        self.initial_rating = initial_rating
        self.k_factor_base = k_factor_base
        self.decay_rate = decay_rate
        self.surface_spec_weight = surface_spec_weight
        
        # Almacenamiento de ratings
        self.player_ratings = {}            # Ratings generales
        self.player_ratings_by_surface = {  # Ratings específicos por superficie
            'hard': {},
            'clay': {},
            'grass': {},
            'carpet': {}
        }
        self.player_last_match = {}         # Fecha del último partido para cada jugador
        self.player_match_count = {}        # Contador de partidos para cada jugador
        
        # Historial de ratings para análisis
        self.rating_history = []
        
        # Parámetros de ajuste del sistema
        self.tournament_k_factors = {
            'G': 2.0,      # Grand Slam (multiplicador alto)
            'M': 1.5,      # Masters 1000
            'A': 1.2,      # ATP 500
            'D': 1.0,      # ATP 250
            'F': 1.8,      # Tour Finals
            'C': 0.8,      # Challenger
            'S': 0.5,      # Satellite/ITF
            'O': 0.7       # Other
        }
        
        # Mapeo alternativo para columnas con distintos nombres
        self.tourney_level_mapping = {
            'G': 'G',          # Grand Slam
            'M': 'M',          # Masters 1000
            'A': 'A',          # ATP 500
            'D': 'D',          # ATP 250
            'F': 'F',          # Tour Finals
            'C': 'C',          # Challenger
            'S': 'S',          # Satellite/ITF
            'O': 'O',          # Other
            'Grand Slam': 'G',
            'Masters 1000': 'M',
            'ATP500': 'A',
            'ATP250': 'D',
            'Tour Finals': 'F',
            'Challenger': 'C',
            'Satellite': 'S',
            'Futures': 'S',
            'ITF': 'S'
        }
        
        # Multiplicadores por ronda (aumenta en rondas finales)
        self.round_multipliers = {
            'F': 1.5,     # Final
            'SF': 1.3,    # Semifinal
            'QF': 1.2,    # Cuartos de final
            'R16': 1.1,   # Octavos
            'R32': 1.0,   # 1/16
            'R64': 0.9,   # 1/32
            'R128': 0.8,  # 1/64
            'RR': 1.0     # Round Robin
        }
        
        # Multiplicadores por superficie (para ajustar especificidad)
        self.surface_specificity = {
            'hard': 1.0,
            'clay': 1.1,    # Mayor especificidad en tierra
            'grass': 1.2,   # Mayor especificidad en hierba
            'carpet': 1.0
        }
        
        # Mapeo de superficies para normalización
        self.surface_mapping = {
            'hard': 'hard',
            'clay': 'clay', 
            'grass': 'grass',
            'carpet': 'carpet',
            'h': 'hard',
            'c': 'clay',
            'g': 'grass',
            'cr': 'carpet'
        }
        
        # Estadísticas del procesamiento
        self.stats = {
            'total_matches_processed': 0,
            'invalid_matches_skipped': 0,
            'players_with_ratings': 0,
            'dates_range': (None, None)
        }
        
        logger.info("Procesador de ELO inicializado")
    
    def _normalize_surface(self, surface: str) -> str:
        """
        Normaliza el nombre de la superficie.
        
        Args:
            surface: Nombre de la superficie (puede ser cualquier formato)
            
        Returns:
            Nombre normalizado de la superficie
        """
        if pd.isna(surface) or not surface or surface == 'unknown':
            return 'hard'  # Usar hard como valor predeterminado
        
        surface_lower = str(surface).lower().strip()
        return self.surface_mapping.get(surface_lower, 'hard')
    
    def _normalize_tournament_level(self, level: str) -> str:
        """
        Normaliza el nivel del torneo.
        
        Args:
            level: Nivel del torneo (puede ser cualquier formato)
            
        Returns:
            Código normalizado del nivel del torneo
        """
        if pd.isna(level) or not level:
            return 'O'  # Otros como valor predeterminado
        
        return self.tourney_level_mapping.get(level, 'O')
    
    def get_player_rating(self, player_id: str, surface: Optional[str] = None) -> float:
        """
        Obtiene el rating ELO actual de un jugador.
        
        Args:
            player_id: ID del jugador
            surface: Superficie específica (opcional)
            
        Returns:
            Rating ELO (general o específico por superficie)
        """
        if surface:
            surface = self._normalize_surface(surface)
            return self.player_ratings_by_surface[surface].get(str(player_id), self.initial_rating)
        return self.player_ratings.get(str(player_id), self.initial_rating)
    
    def calculate_expected_win_probability(self, rating_a: float, rating_b: float) -> float:
        """
        Calcula la probabilidad esperada de victoria.
        
        Args:
            rating_a: Rating ELO del jugador A
            rating_b: Rating ELO del jugador B
            
        Returns:
            Probabilidad de que el jugador A gane (0-1)
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    def _get_dynamic_k_factor(self, player_id: str, tourney_level: str, round_name: str, surface: str) -> float:
        """
        Calcula un factor K dinámico basado en múltiples factores.
        
        Args:
            player_id: ID del jugador
            tourney_level: Nivel del torneo (G, M, A, D, etc.)
            round_name: Ronda del torneo (F, SF, QF, etc.)
            surface: Superficie de juego
            
        Returns:
            Factor K ajustado
        """
        # Factor base
        k_factor = self.k_factor_base
        
        # Normalizar valores
        tourney_level = self._normalize_tournament_level(tourney_level)
        
        # Ajuste por tipo de torneo
        tourney_multiplier = self.tournament_k_factors.get(tourney_level, 1.0)
        k_factor *= tourney_multiplier
        
        # Ajuste por ronda
        round_multiplier = self.round_multipliers.get(round_name, 1.0)
        k_factor *= round_multiplier
        
        # Ajuste por experiencia (menor K para jugadores con más experiencia)
        match_count = self.player_match_count.get(str(player_id), 0)
        experience_multiplier = max(0.7, 1.0 - (match_count / 500))
        k_factor *= experience_multiplier
        
        return k_factor
    
    def _parse_score(self, score: str) -> Tuple[int, int, int]:
        """
        Analiza el score de un partido para determinar sets y games ganados.
        
        Args:
            score: String con el resultado (e.g. '6-4 7-5')
            
        Returns:
            Tupla de (sets_ganador, sets_perdedor, diferencia_games)
        """
        if pd.isna(score) or not score:
            return 2, 0, 4  # Valores predeterminados para scores desconocidos
        
        try:
            # Limpiar y dividir el score
            score = str(score).strip()
            
            # Manejar casos especiales: abandono, walkover, etc.
            if any(term in score.lower() for term in ['ret', 'reti', 'def', 'w/o', 'walkover', 'default']):
                # Para abandonos, usar valores estándar
                return 2, 0, 4
            
            # Dividir el score por sets
            sets = score.split()
            
            sets_won_winner = 0
            sets_won_loser = 0
            games_diff = 0
            
            for set_score in sets:
                # Limpiar para manejar tiebreaks, etc.
                if '(' in set_score:
                    set_score = set_score.split('(')[0]  # Remover información de tiebreak
                
                if '-' in set_score:
                    games = set_score.split('-')
                    
                    if len(games) == 2:
                        try:
                            winner_games = int(games[0])
                            loser_games = int(games[1])
                            
                            games_diff += winner_games - loser_games
                            
                            if winner_games > loser_games:
                                sets_won_winner += 1
                            elif loser_games > winner_games:
                                sets_won_loser += 1
                        except (ValueError, TypeError):
                            # Si no se puede convertir a entero, ignorar este set
                            continue
            
            # En caso de no poder analizar correctamente, usar valores predeterminados
            if sets_won_winner == 0 and sets_won_loser == 0:
                return 2, 0, 4
                
            return sets_won_winner, sets_won_loser, games_diff
            
        except Exception as e:
            logger.debug(f"Error analizando score '{score}': {str(e)}")
            return 2, 0, 4  # Valores predeterminados en caso de error
    
    def _get_margin_multiplier(self, score: str, retirement: bool = False) -> float:
        """
        Calcula multiplicador basado en el margen de victoria.
        
        Args:
            score: String con el resultado (e.g. '6-4 7-5')
            retirement: Indica si hubo abandono
            
        Returns:
            Multiplicador para el cambio de ELO (0.8-1.2)
        """
        if retirement:
            return 0.8  # Menor impacto para partidos con abandono
        
        try:
            # Analizar el score
            sets_won_winner, sets_won_loser, games_diff = self._parse_score(score)
            
            # Calcular dominancia
            sets_diff = sets_won_winner - sets_won_loser
            sets_ratio = sets_won_winner / max(1, sets_won_winner + sets_won_loser)
            
            # Normalizar diferencia de games
            normalized_games_diff = games_diff / max(1, sets_won_winner + sets_won_loser) / 6.0
            
            # Combinar factores de dominancia (más peso a sets que a games)
            dominance = 0.7 * sets_ratio + 0.3 * normalized_games_diff
            
            # Valor entre 0.8 y 1.2 según dominancia
            return 0.8 + min(0.4, dominance * 0.4)
            
        except Exception as e:
            logger.debug(f"Error calculando margen: {str(e)}")
            return 1.0  # Valor predeterminado en caso de error
    
    def _apply_temporal_decay(self, player_id: str, current_date: datetime) -> None:
        """
        Aplica decaimiento temporal al rating de un jugador inactivo.
        
        Args:
            player_id: ID del jugador
            current_date: Fecha actual
        """
        player_id = str(player_id)
        last_match_date = self.player_last_match.get(player_id)
        
        if last_match_date and current_date > last_match_date:
            # Calcular meses de inactividad
            months_inactive = (current_date - last_match_date).days / 30.0
            
            if months_inactive > 1:
                # Aplicar decaimiento exponencial
                decay_factor = self.decay_rate ** months_inactive
                
                # Aplicar a rating general
                if player_id in self.player_ratings:
                    current_rating = self.player_ratings[player_id]
                    self.player_ratings[player_id] = current_rating * decay_factor
                
                # Aplicar a ratings por superficie
                for surface in self.player_ratings_by_surface:
                    if player_id in self.player_ratings_by_surface[surface]:
                        current_surface_rating = self.player_ratings_by_surface[surface][player_id]
                        self.player_ratings_by_surface[surface][player_id] = current_surface_rating * decay_factor
    
    def update_ratings(self, 
                      winner_id: str, 
                      loser_id: str, 
                      match_date: datetime, 
                      surface: str,
                      tourney_level: str,
                      round_name: str,
                      score: str,
                      retirement: bool = False) -> Tuple[float, float]:
        """
        Actualiza los ratings ELO después de un partido.
        
        Args:
            winner_id: ID del jugador ganador
            loser_id: ID del jugador perdedor
            match_date: Fecha del partido
            surface: Superficie de juego
            tourney_level: Nivel del torneo
            round_name: Ronda del torneo
            score: Resultado del partido
            retirement: Si hubo abandono
            
        Returns:
            Tuple (cambio_elo_ganador, cambio_elo_perdedor)
        """
        # Convertir IDs a string
        winner_id = str(winner_id)
        loser_id = str(loser_id)
        
        # Aplicar decaimiento temporal a ambos jugadores
        self._apply_temporal_decay(winner_id, match_date)
        self._apply_temporal_decay(loser_id, match_date)
        
        # Normalizar superficie
        surface = self._normalize_surface(surface)
        
        # Obtener ratings actuales
        w_general_elo = self.get_player_rating(winner_id)
        l_general_elo = self.get_player_rating(loser_id)
        w_surface_elo = self.get_player_rating(winner_id, surface)
        l_surface_elo = self.get_player_rating(loser_id, surface)
        
        # Calcular probabilidades esperadas
        w_win_prob_general = self.calculate_expected_win_probability(w_general_elo, l_general_elo)
        w_win_prob_surface = self.calculate_expected_win_probability(w_surface_elo, l_surface_elo)
        
        # Media ponderada: configurada por parámetro surface_spec_weight
        w_win_prob = (1 - self.surface_spec_weight) * w_win_prob_general + self.surface_spec_weight * w_win_prob_surface
        
        # Calcular factores K dinámicos
        k_winner = self._get_dynamic_k_factor(winner_id, tourney_level, round_name, surface)
        k_loser = self._get_dynamic_k_factor(loser_id, tourney_level, round_name, surface)
        
        # Ajustar por margen de victoria
        margin_multiplier = self._get_margin_multiplier(score, retirement)
        
        # Calcular cambios de ELO
        elo_change_winner = k_winner * margin_multiplier * (1 - w_win_prob)
        elo_change_loser = k_loser * margin_multiplier * (0 - (1 - w_win_prob))
        
        # Actualizar ratings generales
        if winner_id not in self.player_ratings:
            self.player_ratings[winner_id] = self.initial_rating
        if loser_id not in self.player_ratings:
            self.player_ratings[loser_id] = self.initial_rating
            
        self.player_ratings[winner_id] += elo_change_winner
        self.player_ratings[loser_id] += elo_change_loser
        
        # Actualizar ratings por superficie
        if winner_id not in self.player_ratings_by_surface[surface]:
            self.player_ratings_by_surface[surface][winner_id] = self.initial_rating
        if loser_id not in self.player_ratings_by_surface[surface]:
            self.player_ratings_by_surface[surface][loser_id] = self.initial_rating
            
        # Actualizar con mayor especificidad para superficie
        surface_mult = self.surface_specificity.get(surface, 1.0)
        self.player_ratings_by_surface[surface][winner_id] += elo_change_winner * surface_mult
        self.player_ratings_by_surface[surface][loser_id] += elo_change_loser * surface_mult
        
        # Actualizar fecha del último partido
        self.player_last_match[winner_id] = match_date
        self.player_last_match[loser_id] = match_date
        
        # Actualizar contador de partidos
        self.player_match_count[winner_id] = self.player_match_count.get(winner_id, 0) + 1
        self.player_match_count[loser_id] = self.player_match_count.get(loser_id, 0) + 1
        
        # Registrar historial de ratings para análisis
        self.rating_history.append({
            'date': match_date,
            'winner_id': winner_id,
            'loser_id': loser_id,
            'winner_rating_before': w_general_elo,
            'loser_rating_before': l_general_elo,
            'winner_rating_after': self.player_ratings[winner_id],
            'loser_rating_after': self.player_ratings[loser_id],
            'winner_surface_rating_before': w_surface_elo,
            'loser_surface_rating_before': l_surface_elo,
            'winner_surface_rating_after': self.player_ratings_by_surface[surface][winner_id],
            'loser_surface_rating_after': self.player_ratings_by_surface[surface][loser_id],
            'elo_change_winner': elo_change_winner,
            'elo_change_loser': elo_change_loser,
            'surface': surface,
            'tourney_level': tourney_level,
            'round': round_name,
            'score': score,
            'k_factor_winner': k_winner,
            'k_factor_loser': k_loser,
            'margin_multiplier': margin_multiplier,
            'expected_win_prob': w_win_prob
        })
        
        return elo_change_winner, elo_change_loser
    
    def process_matches_dataframe(self, matches_df: pd.DataFrame, chronological: bool = True) -> pd.DataFrame:
        """
        Procesa todos los partidos de un DataFrame calculando ELO progresivamente.
        
        Args:
            matches_df: DataFrame con datos de partidos
            chronological: Si es True, ordena los partidos cronológicamente primero
            
        Returns:
            DataFrame con columnas de ELO añadidas
        """
        if matches_df.empty:
            logger.warning("DataFrame de partidos vacío, no hay nada que procesar")
            return matches_df
        
        # Hacer copia para no modificar el original
        df = matches_df.copy()
        
                # Mapear nombres de columnas si son diferentes
        column_map = {
            'match_date': ['date', 'tourney_date'],
            'winner_id': ['player1_id', 'p1_id'],
            'loser_id': ['player2_id', 'p2_id'],
            'surface': ['surface_normalized'],
            'tourney_level': ['tournament_level', 'tournament_category', 'tourney_type'],
            'round': ['round_name'],
            'score': ['match_score'],
            'retirement': ['is_retirement', 'ret']
        }
        
        # Verificar y renombrar columnas si es necesario
        for target, alternatives in column_map.items():
            if target not in df.columns:
                for alt in alternatives:
                    if alt in df.columns:
                        df[target] = df[alt]
                        break
        
        # Asegurar que tenemos las columnas necesarias
        required_columns = [
            'winner_id', 'loser_id', 'match_date', 'surface', 
            'tourney_level', 'round', 'score'
        ]
        
        # Verificar si tenemos las columnas necesarias
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Columnas faltantes en el DataFrame: {missing_columns}")
            raise ValueError(f"DataFrame no tiene las columnas requeridas: {missing_columns}")
        
        # Asegurar que match_date es datetime
        if not pd.api.types.is_datetime64_any_dtype(df['match_date']):
            df['match_date'] = pd.to_datetime(df['match_date'])
        
        # Añadir columna para indicar abandono (si existe)
        if 'retirement' not in df.columns:
            if 'score' in df.columns:
                # Intentar detectar abandonos en el score
                df['retirement'] = df['score'].apply(
                    lambda x: any(term in str(x).lower() for term in ['ret', 'reti', 'def', 'w/o', 'walkover', 'default'])
                    if not pd.isna(x) else False
                )
            else:
                df['retirement'] = False
        
        # Ordenar cronológicamente si se especifica
        if chronological:
            df = df.sort_values('match_date')
        
        # Añadir columnas para ratings ELO
        df['winner_elo_before'] = 0.0
        df['loser_elo_before'] = 0.0
        df['winner_surface_elo_before'] = 0.0
        df['loser_surface_elo_before'] = 0.0
        df['winner_elo_after'] = 0.0
        df['loser_elo_after'] = 0.0
        df['winner_surface_elo_after'] = 0.0
        df['loser_surface_elo_after'] = 0.0
        df['elo_change_winner'] = 0.0
        df['elo_change_loser'] = 0.0
        df['expected_win_prob'] = 0.0
        
        # Procesar cada partido secuencialmente
        total_matches = len(df)
        logger.info(f"Procesando {total_matches} partidos para cálculo de ELO...")
        
        # Actualizar estadísticas
        self.stats['total_matches_processed'] = 0
        self.stats['invalid_matches_skipped'] = 0
        
        # Registrar rango de fechas
        if not df.empty:
            self.stats['dates_range'] = (df['match_date'].min(), df['match_date'].max())
        
        # Usar tqdm para mostrar barra de progreso
        for idx in tqdm(df.index, total=total_matches, desc="Calculando ELO"):
            try:
                match = df.loc[idx]
                
                # Obtener datos del partido
                winner_id = str(match['winner_id'])
                loser_id = str(match['loser_id'])
                match_date = match['match_date']
                surface = str(match['surface']).lower()
                tourney_level = str(match['tourney_level'])
                round_name = str(match['round'])
                score = str(match['score'])
                retirement = bool(match['retirement'])
                
                # Validar datos básicos
                if winner_id == loser_id or pd.isna(winner_id) or pd.isna(loser_id):
                    logger.warning(f"Partido inválido (ID ganador = ID perdedor o valores nulos): {winner_id} vs {loser_id}")
                    self.stats['invalid_matches_skipped'] += 1
                    continue
                
                # Guardar ratings actuales antes de la actualización
                df.at[idx, 'winner_elo_before'] = self.get_player_rating(winner_id)
                df.at[idx, 'loser_elo_before'] = self.get_player_rating(loser_id)
                df.at[idx, 'winner_surface_elo_before'] = self.get_player_rating(winner_id, surface)
                df.at[idx, 'loser_surface_elo_before'] = self.get_player_rating(loser_id, surface)
                
                # Calcular probabilidad esperada
                w_general_elo = self.get_player_rating(winner_id)
                l_general_elo = self.get_player_rating(loser_id)
                w_surface_elo = self.get_player_rating(winner_id, surface)
                l_surface_elo = self.get_player_rating(loser_id, surface)
                
                # Media ponderada: configurada por parámetro surface_spec_weight
                w_win_prob_general = self.calculate_expected_win_probability(w_general_elo, l_general_elo)
                w_win_prob_surface = self.calculate_expected_win_probability(w_surface_elo, l_surface_elo)
                w_win_prob = (1 - self.surface_spec_weight) * w_win_prob_general + self.surface_spec_weight * w_win_prob_surface
                df.at[idx, 'expected_win_prob'] = w_win_prob
                
                # Actualizar ratings
                elo_change_winner, elo_change_loser = self.update_ratings(
                    winner_id, loser_id, match_date, surface, tourney_level, round_name, score, retirement
                )
                
                # Guardar resultados de la actualización
                df.at[idx, 'elo_change_winner'] = elo_change_winner
                df.at[idx, 'elo_change_loser'] = elo_change_loser
                df.at[idx, 'winner_elo_after'] = self.get_player_rating(winner_id)
                df.at[idx, 'loser_elo_after'] = self.get_player_rating(loser_id)
                df.at[idx, 'winner_surface_elo_after'] = self.get_player_rating(winner_id, surface)
                df.at[idx, 'loser_surface_elo_after'] = self.get_player_rating(loser_id, surface)
                
                # Actualizar contador
                self.stats['total_matches_processed'] += 1
                
            except Exception as e:
                logger.warning(f"Error procesando partido {idx}: {str(e)}")
                self.stats['invalid_matches_skipped'] += 1
                continue
        
        # Actualizar estadística de jugadores con ratings
        self.stats['players_with_ratings'] = len(self.player_ratings)
        
        logger.info(f"Procesamiento completado: {self.stats['total_matches_processed']} partidos procesados, "
                   f"{self.stats['invalid_matches_skipped']} partidos omitidos")
        
        return df
    
    def save_ratings(self, output_dir: str = 'data/processed/elo') -> None:
        """
        Guarda los ratings ELO actuales en archivos JSON.
        
        Args:
            output_dir: Directorio donde guardar los archivos
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar ratings generales
        general_path = output_path / 'elo_ratings_general.json'
        with open(general_path, 'w') as f:
            json.dump(self.player_ratings, f, indent=2)
        
        # Guardar ratings por superficie
        surface_path = output_path / 'elo_ratings_by_surface.json'
        with open(surface_path, 'w') as f:
            json.dump(self.player_ratings_by_surface, f, indent=2)
        
        # Guardar contadores de partidos
        counts_path = output_path / 'player_match_counts.json'
        with open(counts_path, 'w') as f:
            json.dump(self.player_match_count, f, indent=2)
        
        # Guardar historial de ratings como CSV
        history_df = pd.DataFrame(self.rating_history)
        history_path = output_path / 'elo_rating_history.csv'
        history_df.to_csv(history_path, index=False)
        
        # Guardar estadísticas del procesamiento
        stats_path = output_path / 'elo_processing_stats.json'
        
        # Convertir fechas a strings para JSON
        stats_json = self.stats.copy()
        if 'dates_range' in stats_json and stats_json['dates_range'][0] is not None:
            stats_json['dates_range'] = (
                stats_json['dates_range'][0].strftime('%Y-%m-%d'),
                stats_json['dates_range'][1].strftime('%Y-%m-%d')
            )
        
        with open(stats_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        logger.info(f"Ratings ELO guardados en {output_dir}")
    
    def load_ratings(self, input_dir: str = 'data/processed/elo') -> None:
        """
        Carga ratings ELO previamente guardados.
        
        Args:
            input_dir: Directorio donde están los archivos
        """
        input_path = Path(input_dir)
        
        # Cargar ratings generales
        general_path = input_path / 'elo_ratings_general.json'
        if general_path.exists():
            with open(general_path, 'r') as f:
                self.player_ratings = json.load(f)
            logger.info(f"Ratings generales cargados: {len(self.player_ratings)} jugadores")
        
        # Cargar ratings por superficie
        surface_path = input_path / 'elo_ratings_by_surface.json'
        if surface_path.exists():
            with open(surface_path, 'r') as f:
                self.player_ratings_by_surface = json.load(f)
            logger.info(f"Ratings por superficie cargados")
        
        # Cargar contadores de partidos
        counts_path = input_path / 'player_match_counts.json'
        if counts_path.exists():
            with open(counts_path, 'r') as f:
                self.player_match_count = json.load(f)
            logger.info(f"Contadores de partidos cargados: {len(self.player_match_count)} jugadores")
        
        # Cargar historial si existe
        history_path = input_path / 'elo_rating_history.csv'
        if history_path.exists():
            try:
                history_df = pd.read_csv(history_path)
                if 'date' in history_df.columns:
                    history_df['date'] = pd.to_datetime(history_df['date'])
                self.rating_history = history_df.to_dict('records')
                logger.info(f"Historial de ratings cargado: {len(self.rating_history)} registros")
            except Exception as e:
                logger.warning(f"Error cargando historial de ratings: {str(e)}")
        
        # Cargar estadísticas si existen
        stats_path = input_path / 'elo_processing_stats.json'
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
                logger.info(f"Estadísticas de procesamiento cargadas")
            except Exception as e:
                logger.warning(f"Error cargando estadísticas: {str(e)}")
        
        logger.info(f"Ratings ELO cargados desde {input_dir}")
    
    def get_top_players(self, n: int = 20, surface: Optional[str] = None, 
                       min_matches: int = 10, date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Obtiene los mejores jugadores según su rating ELO.
        
        Args:
            n: Número de jugadores a mostrar
            surface: Superficie específica (opcional)
            min_matches: Número mínimo de partidos jugados
            date: Fecha para la cual obtener los ratings (opcional, usa los actuales si no se especifica)
            
        Returns:
            DataFrame con los mejores jugadores
        """
        # Si se especifica fecha, usar el historial
        if date is not None and self.rating_history:
            # Convertir historial a DataFrame si no lo está ya
            if isinstance(self.rating_history, list):
                history_df = pd.DataFrame(self.rating_history)
            else:
                history_df = self.rating_history
            
            # Asegurar que la fecha es datetime
            if 'date' in history_df.columns and not pd.api.types.is_datetime64_any_dtype(history_df['date']):
                history_df['date'] = pd.to_datetime(history_df['date'])
            
            # Filtrar registros anteriores a la fecha
            filtered_history = history_df[history_df['date'] <= date]
            
            if filtered_history.empty:
                logger.warning(f"No hay datos de ratings para la fecha {date}")
                return pd.DataFrame(columns=['player_id', 'elo_rating', 'matches_played'])
            
            # Obtener el último registro para cada jugador
            if surface:
                # Unir ratings de ganadores y perdedores
                winners = filtered_history[['winner_id', 'winner_surface_rating_after', 'date']].rename(
                    columns={'winner_id': 'player_id', 'winner_surface_rating_after': 'elo_rating'}
                )
                losers = filtered_history[['loser_id', 'loser_surface_rating_after', 'date']].rename(
                    columns={'loser_id': 'player_id', 'loser_surface_rating_after': 'elo_rating'}
                )
            else:
                # Unir ratings generales
                winners = filtered_history[['winner_id', 'winner_rating_after', 'date']].rename(
                    columns={'winner_id': 'player_id', 'winner_rating_after': 'elo_rating'}
                )
                losers = filtered_history[['loser_id', 'loser_rating_after', 'date']].rename(
                    columns={'loser_id': 'player_id', 'loser_rating_after': 'elo_rating'}
                )
            
            all_ratings = pd.concat([winners, losers], ignore_index=True)
            
            # Obtener el registro más reciente para cada jugador
            latest_ratings = all_ratings.sort_values('date').groupby('player_id').last().reset_index()
            
            # Añadir contador de partidos
            latest_ratings['matches_played'] = latest_ratings['player_id'].apply(
                lambda x: self.player_match_count.get(str(x), 0)
            )
            
            # Filtrar jugadores con pocos partidos
            filtered_ratings = latest_ratings[latest_ratings['matches_played'] >= min_matches]
            
            # Ordenar y devolver los N mejores
            return filtered_ratings.sort_values('elo_rating', ascending=False).head(n)
        
        # Usar ratings actuales
        if surface:
            ratings = self.player_ratings_by_surface.get(self._normalize_surface(surface), {})
        else:
            ratings = self.player_ratings
        
        # Convertir a DataFrame
        df = pd.DataFrame([
            {'player_id': player_id, 'elo_rating': rating}
            for player_id, rating in ratings.items()
        ])
        
        if df.empty:
            return pd.DataFrame(columns=['player_id', 'elo_rating', 'matches_played'])
        
        # Añadir contador de partidos
        df['matches_played'] = df['player_id'].map(
            lambda x: self.player_match_count.get(x, 0)
        )
        
        # Filtrar jugadores con pocos partidos
        df = df[df['matches_played'] >= min_matches]
        
        # Ordenar y devolver los N mejores
        return df.sort_values('elo_rating', ascending=False).head(n)
    
    def analyze_ratings_distribution(self, surface: Optional[str] = None,
                                    min_matches: int = 10) -> Dict:
        """
        Analiza la distribución de los ratings ELO.
        
        Args:
            surface: Superficie específica (opcional)
            min_matches: Número mínimo de partidos jugados
            
        Returns:
            Diccionario con estadísticas de la distribución
        """
        if surface:
            surface = self._normalize_surface(surface)
            ratings_dict = self.player_ratings_by_surface.get(surface, {})
        else:
            ratings_dict = self.player_ratings
        
        # Filtrar jugadores con pocos partidos
        filtered_ratings = {
            player_id: rating 
            for player_id, rating in ratings_dict.items()
            if self.player_match_count.get(player_id, 0) >= min_matches
        }
        
        ratings = list(filtered_ratings.values())
        
        if not ratings:
            return {
                'count': 0,
                'mean': self.initial_rating,
                'std': 0,
                'min': self.initial_rating,
                'max': self.initial_rating,
                'percentiles': {
                    '5': self.initial_rating,
                    '25': self.initial_rating,
                    '50': self.initial_rating,
                    '75': self.initial_rating,
                    '95': self.initial_rating
                }
            }
        
        # Calcular estadísticas
        ratings_array = np.array(ratings)
        
        return {
            'count': len(ratings),
            'mean': float(np.mean(ratings_array)),
            'std': float(np.std(ratings_array)),
            'min': float(np.min(ratings_array)),
            'max': float(np.max(ratings_array)),
            'percentiles': {
                '5': float(np.percentile(ratings_array, 5)),
                '25': float(np.percentile(ratings_array, 25)),
                '50': float(np.percentile(ratings_array, 50)),
                '75': float(np.percentile(ratings_array, 75)),
                '95': float(np.percentile(ratings_array, 95))
            }
        }
    
    def plot_top_players_history(self, player_ids: List[str], start_date: Optional[datetime] = None, 
                                end_date: Optional[datetime] = None, surface: Optional[str] = None,
                                save_path: Optional[str] = None, player_names: Optional[Dict[str, str]] = None) -> None:
        """
        Genera un gráfico con la evolución de ratings de los jugadores seleccionados.
        
        Args:
            player_ids: Lista de IDs de jugadores a incluir
            start_date: Fecha de inicio para el gráfico
            end_date: Fecha final para el gráfico
            surface: Superficie específica (opcional)
            save_path: Ruta para guardar el gráfico (opcional)
            player_names: Diccionario de {player_id: nombre} para etiquetas (opcional)
        """
        # Convertir historial a DataFrame si no lo está ya
        if isinstance(self.rating_history, list):
            history_df = pd.DataFrame(self.rating_history)
        else:
            history_df = self.rating_history
            
        if history_df.empty:
            logger.warning("No hay datos de historial ELO disponibles para graficar")
            return
        
        # Asegurar que la fecha es datetime
        if 'date' in history_df.columns and not pd.api.types.is_datetime64_any_dtype(history_df['date']):
            history_df['date'] = pd.to_datetime(history_df['date'])
        
        # Columnas a usar según superficie
        if surface:
            surface = self._normalize_surface(surface)
            winner_rating_col = 'winner_surface_rating_after'
            loser_rating_col = 'loser_surface_rating_after'
            
            # Filtrar por superficie
            history_df = history_df[history_df['surface'] == surface]
        else:
            winner_rating_col = 'winner_rating_after'
            loser_rating_col = 'loser_rating_after'
        
        # Preparar DataFrame para el gráfico
        plot_data = []
        
        for player_id in player_ids:
            player_id = str(player_id)
            
            # Datos cuando el jugador ganó
            winner_data = history_df[history_df['winner_id'] == player_id].copy()
            if not winner_data.empty:
                winner_data['rating'] = winner_data[winner_rating_col]
                winner_data['player_id'] = player_id
                plot_data.append(winner_data[['date', 'player_id', 'rating']])
            
            # Datos cuando el jugador perdió
            loser_data = history_df[history_df['loser_id'] == player_id].copy()
            if not loser_data.empty:
                loser_data['rating'] = loser_data[loser_rating_col]
                loser_data['player_id'] = player_id
                plot_data.append(loser_data[['date', 'player_id', 'rating']])
        
        if not plot_data:
            logger.warning(f"No hay datos disponibles para los jugadores {player_ids}")
            return
        
        # Concatenar todos los datos
        all_data = pd.concat(plot_data, ignore_index=True)
        
        # Filtrar por fechas si se especifican
        if start_date:
            all_data = all_data[all_data['date'] >= start_date]
        if end_date:
            all_data = all_data[all_data['date'] <= end_date]
        
        if all_data.empty:
            logger.warning("No hay datos para el rango de fechas especificado")
            return
            
        # Ordenar por fecha
        all_data = all_data.sort_values('date')
        
        # Crear gráfico
        plt.figure(figsize=(14, 8))
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # Colores para cada jugador
        colors = plt.cm.tab10.colors
        
        # Graficar una línea por jugador
        for i, player_id in enumerate(player_ids):
            player_id = str(player_id)
            player_data = all_data[all_data['player_id'] == player_id]
            
            if not player_data.empty:
                # Nombre para la leyenda
                if player_names and player_id in player_names:
                    label = player_names[player_id]
                else:
                    label = f'Jugador {player_id}'
                
                plt.plot(player_data['date'], player_data['rating'], '-o', 
                         label=label, color=colors[i % len(colors)], markersize=4)
        
        # Añadir título y etiquetas
        if surface:
            plt.title(f'Evolución del rating ELO en superficie {surface.upper()}', fontsize=16)
        else:
            plt.title('Evolución del rating ELO general', fontsize=16)
        
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Rating ELO', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Ajustar eje Y para mostrar un rango razonable
        ratings = all_data['rating']
        min_rating = max(ratings.min() - 50, 1000)
        max_rating = min(ratings.max() + 50, 2500)
        plt.ylim(min_rating, max_rating)
        
        # Rotar etiquetas de fecha
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Guardar o mostrar
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en {save_path}")
        else:
            plt.show()
    
    def create_features_for_model(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características basadas en ELO para un modelo de ML.
        
        Args:
            matches_df: DataFrame con datos de partidos
            
        Returns:
            DataFrame con características ELO añadidas
        """
        df = matches_df.copy()
        
        # Mapear nombres de columnas si son diferentes
        column_map = {
            'player1_id': ['winner_id', 'p1_id'],
            'player2_id': ['loser_id', 'p2_id'],
            'surface': ['surface_normalized'],
        }
        
        # Verificar y renombrar columnas si es necesario
        for target, alternatives in column_map.items():
            if target not in df.columns:
                for alt in alternatives:
                    if alt in df.columns:
                        df[target] = df[alt]
                        break
        
        # Añadir ratings ELO generales
        df['player1_elo'] = df.apply(lambda row: self.get_player_rating(str(row['player1_id'])), axis=1)
        df['player2_elo'] = df.apply(lambda row: self.get_player_rating(str(row['player2_id'])), axis=1)
        
        # Añadir ratings ELO por superficie
        if 'surface' in df.columns:
            df['player1_surface_elo'] = df.apply(
                lambda row: self.get_player_rating(str(row['player1_id']), str(row['surface'])), 
                axis=1
            )
            df['player2_surface_elo'] = df.apply(
                lambda row: self.get_player_rating(str(row['player2_id']), str(row['surface'])),
                axis=1
            )
        
        # Calcular diferencias de ELO (características muy predictivas)
        df['elo_difference'] = df['player1_elo'] - df['player2_elo']
        df['elo_difference_abs'] = df['elo_difference'].abs()
        
        if 'surface' in df.columns:
            df['surface_elo_difference'] = df['player1_surface_elo'] - df['player2_surface_elo']
            df['surface_elo_difference_abs'] = df['surface_elo_difference'].abs()
        
        # Añadir contadores de partidos
        df['player1_matches'] = df['player1_id'].apply(lambda x: self.player_match_count.get(str(x), 0))
        df['player2_matches'] = df['player2_id'].apply(lambda x: self.player_match_count.get(str(x), 0))
        df['experience_difference'] = df['player1_matches'] - df['player2_matches']
        
        # Calcular probabilidad de victoria basada en ELO
        df['p1_win_prob_general'] = df.apply(
            lambda row: self.calculate_expected_win_probability(row['player1_elo'], row['player2_elo']),
            axis=1
        )
        
        if 'surface' in df.columns:
            df['p1_win_prob_surface'] = df.apply(
                lambda row: self.calculate_expected_win_probability(row['player1_surface_elo'], row['player2_surface_elo']),
                axis=1
            )
            # Probabilidad combinada
            df['p1_win_prob'] = (1 - self.surface_spec_weight) * df['p1_win_prob_general'] + self.surface_spec_weight * df['p1_win_prob_surface']
        else:
            df['p1_win_prob'] = df['p1_win_prob_general']
        
        return df
    
    def evaluate_predictive_power(self, test_df: pd.DataFrame) -> Dict:
        """
        Evalúa el poder predictivo del sistema ELO.
        
        Args:
            test_df: DataFrame con partidos de prueba
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        if test_df.empty:
            return {
                'accuracy': 0.0,
                'accuracy_by_surface': {},
                'accuracy_by_tournament': {},
                'calibration_error': 1.0,
                'count': 0
            }
        
        # Hacer copia para no modificar el original
        df = test_df.copy()
        
        # Añadir predicciones a cada partido
        df = self.create_features_for_model(df)
        
        # La columna target debe ser 1 si p1/ganador gana, 0 si p2/perdedor gana
        if 'target' not in df.columns:
            if 'winner_id' in df.columns and 'player1_id' in df.columns:
                df['target'] = (df['winner_id'] == df['player1_id']).astype(int)
            else:
                logger.warning("No se encontró columna target o winner_id para evaluar predicciones")
                df['target'] = 1  # Asumir que player1 es el ganador como caso por defecto
        
        # Predicción final (1 si p1 gana, 0 si p2 gana)
        df['predicted_winner'] = (df['p1_win_prob'] > 0.5).astype(int)
        
        # Calcular métricas
        correct = (df['predicted_winner'] == df['target']).sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calcular precisión por superficie
        accuracy_by_surface = {}
        if 'surface' in df.columns:
            for surface in df['surface'].unique():
                if pd.notna(surface) and surface != '':
                    surface_df = df[df['surface'] == surface]
                    if len(surface_df) > 0:
                        surface_correct = (surface_df['predicted_winner'] == surface_df['target']).sum()
                        accuracy_by_surface[surface] = surface_correct / len(surface_df)
        
        # Calcular precisión por tipo de torneo
        accuracy_by_tournament = {}
        for col in ['tourney_level', 'tournament_level', 'tournament_category']:
            if col in df.columns:
                for level in df[col].unique():
                    if pd.notna(level) and level != '':
                        level_df = df[df[col] == level]
                        if len(level_df) > 0:
                            level_correct = (level_df['predicted_winner'] == level_df['target']).sum()
                            accuracy_by_tournament[level] = level_correct / len(level_df)
                break
        
        # Calcular error de calibración (diferencia entre prob. predicha y resultado)
        calibration_error = ((df['p1_win_prob'] - df['target']).abs()).mean()
        
        # Calcular precisión por diferencia de ELO
        df['elo_diff_abs'] = df['elo_difference'].abs()
        df['elo_diff_bin'] = pd.cut(df['elo_diff_abs'], 
                                    bins=[0, 50, 100, 150, 200, float('inf')],
                                    labels=['0-50', '50-100', '100-150', '150-200', '200+'])
        
        accuracy_by_elo_diff = {}
        for bin_name in df['elo_diff_bin'].dropna().unique():
            bin_df = df[df['elo_diff_bin'] == bin_name]
            if len(bin_df) > 0:
                bin_correct = (bin_df['predicted_winner'] == bin_df['target']).sum()
                accuracy_by_elo_diff[str(bin_name)] = bin_correct / len(bin_df)
        
        return {
            'accuracy': accuracy,
            'accuracy_by_surface': accuracy_by_surface,
            'accuracy_by_tournament': accuracy_by_tournament,
            'accuracy_by_elo_diff': accuracy_by_elo_diff,
            'calibration_error': calibration_error,
            'count': total
        }
    
    def get_elo_statistics_summary(self) -> Dict:
        """
        Genera un resumen completo de las estadísticas de ELO.
        
        Returns:
            Diccionario con estadísticas completas
        """
        summary = {
            'general_stats': self.analyze_ratings_distribution(),
            'surface_stats': {},
            'player_counts': {
                'total': len(self.player_ratings),
                'by_surface': {
                    surface: len(ratings) 
                    for surface, ratings in self.player_ratings_by_surface.items()
                }
            },
            'top_players': {
                'general': self.get_top_players(10).to_dict('records'),
                'by_surface': {}
            },
            'processing_stats': self.stats
        }
        
        # Añadir estadísticas por superficie
        for surface in self.player_ratings_by_surface:
            summary['surface_stats'][surface] = self.analyze_ratings_distribution(surface)
            summary['top_players']['by_surface'][surface] = self.get_top_players(10, surface).to_dict('records')
        
        return summary

def main():
    """Función principal para ejecutar el procesador de ELO."""
    parser = argparse.ArgumentParser(description='Procesador de ELO para datos de tenis')
    parser.add_argument('--input', type=str, required=True, help='Archivo CSV con datos de partidos')
    parser.add_argument('--output', type=str, default='data/processed/elo/matches_with_elo.csv', 
                       help='Archivo de salida con ratings ELO')
    parser.add_argument('--initial-rating', type=float, default=1500, help='Rating ELO inicial')
    parser.add_argument('--k-factor', type=float, default=32, help='Factor K base')
    parser.add_argument('--decay-rate', type=float, default=0.995, help='Tasa de decaimiento mensual')
    parser.add_argument('--surface-weight', type=float, default=0.7, 
                       help='Peso para ELO específico por superficie vs ELO general')
    parser.add_argument('--load-ratings', type=str, help='Cargar ratings desde directorio')
    parser.add_argument('--save-ratings', action='store_true', help='Guardar ratings finales')
    parser.add_argument('--output-dir', type=str, default='data/processed/elo', help='Directorio para archivos de salida')
    parser.add_argument('--start-year', type=int, help='Filtrar datos desde este año')
    parser.add_argument('--end-year', type=int, help='Filtrar datos hasta este año')
    parser.add_argument('--top-players', type=int, default=20, help='Mostrar N mejores jugadores al final')
    parser.add_argument('--plot-top', type=int, default=10, help='Generar gráfico con N mejores jugadores')
    parser.add_argument('--test-set', type=str, help='Archivo CSV con datos de prueba para evaluar precisión')
    parser.add_argument('--features-export', type=str, help='Exportar características ELO para modelo ML')
    
    args = parser.parse_args()
    
    try:
        # Crear directorios de salida
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Inicializar procesador de ELO
        elo_processor = TennisEloProcessor(
            initial_rating=args.initial_rating,
            k_factor_base=args.k_factor,
            decay_rate=args.decay_rate,
            surface_spec_weight=args.surface_weight
        )
        
        # Cargar ratings previos si se especifica
        if args.load_ratings:
            elo_processor.load_ratings(args.load_ratings)
        
        # Cargar datos
        logger.info(f"Cargando datos desde {args.input}")
        df = pd.read_csv(args.input)
        
        # Filtrar por año si se especifica
        if 'match_date' not in df.columns and 'date' in df.columns:
            df['match_date'] = df['date']
            
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'])
            
            if args.start_year:
                df = df[df['match_date'].dt.year >= args.start_year]
                
            if args.end_year:
                df = df[df['match_date'].dt.year <= args.end_year]
        
        # Procesar partidos
        logger.info(f"Procesando {len(df)} partidos para cálculo de ELO")
        processed_df = elo_processor.process_matches_dataframe(df)
        
        # Guardar resultados
        processed_df.to_csv(args.output, index=False)
        logger.info(f"Datos con ELO guardados en {args.output}")
        
        # Opcionalmente guardar ratings
        if args.save_ratings:
            elo_processor.save_ratings(args.output_dir)
        
        # Exportar características para modelo ML si se solicita
        if args.features_export:
            if 'test_set' in args and args.test_set:
                # Usar conjunto de prueba para crear características
                test_df = pd.read_csv(args.test_set)
                features_df = elo_processor.create_features_for_model(test_df)
            else:
                # Usar los mismos datos procesados para crear características
                features_df = elo_processor.create_features_for_model(processed_df)
            
            # Guardar características
            features_df.to_csv(args.features_export, index=False)
            logger.info(f"Características para modelo ML guardadas en {args.features_export}")
        
        # Evaluar poder predictivo si se especifica conjunto de prueba
        if args.test_set:
            logger.info(f"Evaluando poder predictivo en {args.test_set}")
            test_df = pd.read_csv(args.test_set)
            performance = elo_processor.evaluate_predictive_power(test_df)
            
            logger.info(f"Precisión general: {performance['accuracy']:.4f}")
            for surface, acc in performance['accuracy_by_surface'].items():
                logger.info(f"Precisión en {surface}: {acc:.4f}")
            
            # Guardar resultados de evaluación
            with open(os.path.join(args.output_dir, 'predictive_performance.json'), 'w') as f:
                json.dump(performance, f, indent=2)
        
        # Mostrar top jugadores si se solicita
        if args.top_players > 0:
            top_general = elo_processor.get_top_players(args.top_players)
            print("\nMejores jugadores por ELO general:")
            print(top_general[['player_id', 'elo_rating', 'matches_played']])
            
            # Mostrar por superficie si hay datos suficientes
            for surface in elo_processor.player_ratings_by_surface:
                if elo_processor.player_ratings_by_surface[surface]:
                    top_surface = elo_processor.get_top_players(args.top_players, surface)
                    if not top_surface.empty:
                        print(f"\nMejores jugadores en {surface}:")
                        print(top_surface[['player_id', 'elo_rating', 'matches_played']])
        
        # Generar gráfico de mejores jugadores si se solicita
        if args.plot_top > 0:
            # Obtener IDs de mejores jugadores
            top_players = elo_processor.get_top_players(args.plot_top)
            if not top_players.empty:
                player_ids = top_players['player_id'].tolist()
                
                # Generar gráfico
                plot_path = os.path.join(args.output_dir, 'top_players_elo_history.png')
                elo_processor.plot_top_players_history(
                    player_ids=player_ids,
                    save_path=plot_path
                )
        
        # Mostrar estadísticas finales
        elo_stats = elo_processor.analyze_ratings_distribution()
        print("\nEstadísticas de la distribución de ELO:")
        print(f"Jugadores: {elo_stats['count']}")
        print(f"Media: {elo_stats['mean']:.2f}")
        print(f"Desviación estándar: {elo_stats['std']:.2f}")
        print(f"Mínimo: {elo_stats['min']:.2f}")
        print(f"Máximo: {elo_stats['max']:.2f}")
        print(f"Mediana: {elo_stats['percentiles']['50']:.2f}")
        
        # Crear reporte completo
        summary = elo_processor.get_elo_statistics_summary()
        with open(os.path.join(args.output_dir, 'elo_statistics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Proceso completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el proceso: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()