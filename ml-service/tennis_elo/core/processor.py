"""
processor.py

Clase principal del procesador ELO para tenis que coordina las diferentes
componentes del sistema: cálculo de ratings, análisis de partidos y
estadísticas de jugadores.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import traceback
from tqdm import tqdm

from .ratings import RatingCalculator
from .match_analysis import MatchAnalyzer
from .player_stats import PlayerStatsManager
from ..utils.normalizers import normalize_surface, normalize_tournament_level

# Configurar logging
logger = logging.getLogger(__name__)

class EnhancedTennisEloProcessor:
    """
    Procesador avanzado de ELO para tenis que integra análisis de partidos,
    cálculo de ratings y gestión de estadísticas de jugadores.
    """
    
    def __init__(self, 
                initial_rating: float = 1500,
                k_factor_base: float = 32, 
                decay_rate: float = 0.995,
                surface_transfer_matrix: Optional[Dict[str, Dict[str, float]]] = None,
                data_dir: str = 'data/processed',
                min_matches_full_rating: int = 10,
                form_window_matches: int = 10,
                form_weight: float = 0.2,
                use_match_stats: bool = True,
                use_point_by_point: bool = True,
                use_bayesian_approach: bool = True,
                initialization_method: str = 'surface_adaptive'):
        """
        Inicializa el procesador de ELO avanzado.
        
        Args:
            initial_rating: Rating ELO inicial para nuevos jugadores
            k_factor_base: Factor K base para ajustes de rating
            decay_rate: Tasa de decaimiento mensual para jugadores inactivos
            surface_transfer_matrix: Matriz de transferencia entre superficies
            data_dir: Directorio con los datos procesados
            min_matches_full_rating: Mínimo de partidos para rating completo
            form_window_matches: Ventana de partidos para calcular forma
            form_weight: Peso de la forma reciente en el cálculo
            use_match_stats: Usar estadísticas detalladas de partidos si disponibles
            use_point_by_point: Usar datos punto por punto si disponibles
            use_bayesian_approach: Usar enfoque bayesiano para ratings
            initialization_method: Método para inicializar ratings ('standard', 'surface_adaptive')
        """
        self.initial_rating = initial_rating
        self.k_factor_base = k_factor_base
        self.decay_rate = decay_rate
        self.data_dir = Path(data_dir)
        self.min_matches_full_rating = min_matches_full_rating
        self.form_window_matches = form_window_matches
        self.form_weight = form_weight
        self.use_match_stats = use_match_stats
        self.use_point_by_point = use_point_by_point
        self.use_bayesian_approach = use_bayesian_approach
        self.initialization_method = initialization_method
        
        # Matriz de transferencia entre superficies (cuánto influye una superficie en otra)
        if surface_transfer_matrix is None:
            self.surface_transfer_matrix = {
                'hard': {'hard': 1.0, 'clay': 0.7, 'grass': 0.6, 'carpet': 0.8},
                'clay': {'hard': 0.7, 'clay': 1.0, 'grass': 0.5, 'carpet': 0.6},
                'grass': {'hard': 0.6, 'clay': 0.5, 'grass': 1.0, 'carpet': 0.7},
                'carpet': {'hard': 0.8, 'clay': 0.6, 'grass': 0.7, 'carpet': 1.0}
            }
        else:
            self.surface_transfer_matrix = surface_transfer_matrix
            
        # Pesos ajustados por tipo de torneo para superficies diferentes
        self.tourney_surface_weights = {
            'G': {'hard': 1.0, 'clay': 1.0, 'grass': 1.0, 'carpet': 1.0},  # Grand Slams - máximo peso en todas
            'M': {'hard': 0.9, 'clay': 0.95, 'grass': 0.9, 'carpet': 0.9},  # Masters - más peso en tierra
            'A': {'hard': 0.85, 'clay': 0.9, 'grass': 0.85, 'carpet': 0.85},  # ATP 500
            'D': {'hard': 0.8, 'clay': 0.85, 'grass': 0.8, 'carpet': 0.8},   # ATP 250
            'F': {'hard': 0.95, 'clay': 0.95, 'grass': 0.95, 'carpet': 0.95},  # Tour Finals
            'C': {'hard': 0.75, 'clay': 0.8, 'grass': 0.75, 'carpet': 0.75},  # Challenger - más peso en tierra
            'S': {'hard': 0.7, 'clay': 0.75, 'grass': 0.7, 'carpet': 0.7},   # Satellite/ITF - más peso en tierra
            'O': {'hard': 0.7, 'clay': 0.7, 'grass': 0.7, 'carpet': 0.7}     # Other
        }
        
        # Inicializar componentes modulares
        self.match_analyzer = MatchAnalyzer(
            initial_rating=initial_rating
        )
        
        self.rating_calculator = RatingCalculator(
            initial_rating=initial_rating,
            k_factor_base=k_factor_base,
            decay_rate=decay_rate,
            surface_transfer_matrix=self.surface_transfer_matrix,
            min_matches_full_rating=min_matches_full_rating,
            use_bayesian=use_bayesian_approach
        )
        
        self.player_stats = PlayerStatsManager(
            initial_rating=initial_rating,
            form_window_matches=form_window_matches
        )
        
        # Inicialización de estructuras de datos principales
        # Ratings generales y por superficie
        self.player_ratings = {}
        self.player_ratings_by_surface = {
            'hard': {},
            'clay': {},
            'grass': {},
            'carpet': {}
        }
        
        # Datos de seguimiento y estadísticas avanzadas
        self.player_last_match = {}
        self.player_match_count = {}
        self.player_match_count_by_surface = {
            'hard': {},
            'clay': {},
            'grass': {},
            'carpet': {}
        }
        self.rating_history = []      # Historial de ratings para análisis
        
        # Variables bayesianas - Para enfoque regularizado
        self.player_rating_uncertainty = {}  # Incertidumbre sobre el rating (stdev)
        self.player_rating_posterior_n = {}  # Número efectivo de partidos (enfoque bayesiano)
        
        # Caches para datos relacionados con partidos
        self.stats_cache = {}  # Caché para estadísticas de partidos
        self.pbp_cache = {}    # Caché para datos punto por punto
        
        # Estadísticas del procesamiento
        self.stats = {
            'total_matches_processed': 0,
            'invalid_matches_skipped': 0,
            'players_with_ratings': 0,
            'dates_range': (None, None),
            'surface_distribution': {'hard': 0, 'clay': 0, 'grass': 0, 'carpet': 0},
            'processing_time': 0,
            'model_accuracy': None,
            'calibration_score': None
        }
        
        # Cargar datos adicionales si están disponibles
        self._load_additional_data()
        
        logger.info("Procesador ELO avanzado inicializado")
    
    def _load_additional_data(self) -> None:
        """
        Carga datos adicionales útiles para el procesamiento:
        - Estadísticas de partidos
        - Datos punto por punto
        - Match Charting Project
        - Información de jugadores
        """
        # Intentar cargar datos de jugadores
        try:
            # Intentar cargar jugadores ATP
            atp_players_path = self.data_dir / 'atp' / 'atp_players.csv'
            if atp_players_path.exists():
                self.atp_players_df = pd.read_csv(atp_players_path, low_memory=False)
                logger.info(f"Datos de jugadores ATP cargados: {len(self.atp_players_df)} jugadores")
            else:
                self.atp_players_df = pd.DataFrame()
                
            # Intentar cargar jugadores WTA
            wta_players_path = self.data_dir / 'wta' / 'wta_players.csv'
            if wta_players_path.exists():
                self.wta_players_df = pd.read_csv(wta_players_path, low_memory=False)
                logger.info(f"Datos de jugadores WTA cargados: {len(self.wta_players_df)} jugadores")
            else:
                self.wta_players_df = pd.DataFrame()
                
            # Combinar en un solo DataFrame si hay datos de ambos
            if not self.atp_players_df.empty and not self.wta_players_df.empty:
                # Asegurar que los DataFrames tengan las mismas columnas
                common_cols = list(set(self.atp_players_df.columns) & set(self.wta_players_df.columns))
                self.players_df = pd.concat([
                    self.atp_players_df[common_cols],
                    self.wta_players_df[common_cols]
                ])
                logger.info(f"DataFrame combinado: {len(self.players_df)} jugadores totales")
            elif not self.atp_players_df.empty:
                self.players_df = self.atp_players_df
            elif not self.wta_players_df.empty:
                self.players_df = self.wta_players_df
            else:
                self.players_df = pd.DataFrame()
                logger.warning("No se pudieron cargar datos de jugadores")
        except Exception as e:
            logger.warning(f"Error cargando datos de jugadores: {str(e)}")
            self.players_df = pd.DataFrame()
        
        # Diccionario para mapear ID de jugador a su nombre
        self.player_names = {}
        if not self.players_df.empty and 'player_id' in self.players_df.columns:
            if 'name_first' in self.players_df.columns and 'name_last' in self.players_df.columns:
                for _, row in self.players_df.iterrows():
                    player_id = str(row['player_id'])
                    self.player_names[player_id] = f"{row['name_first']} {row['name_last']}"
            elif 'name' in self.players_df.columns:
                for _, row in self.players_df.iterrows():
                    player_id = str(row['player_id'])
                    self.player_names[player_id] = row['name']
        
        # Compartir nombres con el gestor de estadísticas de jugadores
        self.player_stats.player_names = self.player_names
        
        # Cargar datos estadísticos si se requiere
        if self.use_match_stats:
            try:
                # Verificar si existen archivos de estadísticas de partidos
                # Podría estar en /matches_stats.csv o /match_stats/*.csv
                stats_paths = [
                    self.data_dir / 'atp' / 'match_stats.csv',
                    self.data_dir / 'wta' / 'match_stats.csv',
                    self.data_dir / 'match_stats.csv',
                    self.data_dir / 'match_stats' / 'atp_match_stats.csv',
                    self.data_dir / 'match_stats' / 'wta_match_stats.csv'
                ]
                
                for stats_path in stats_paths:
                    if stats_path.exists():
                        # Cargar las estadísticas
                        try:
                            stats_df = pd.read_csv(stats_path)
                            if not hasattr(self, 'match_stats_df'):
                                self.match_stats_df = stats_df
                            else:
                                # Asegurar que tienen las mismas columnas
                                common_cols = list(set(self.match_stats_df.columns) & set(stats_df.columns))
                                if common_cols:
                                    self.match_stats_df = pd.concat([
                                        self.match_stats_df[common_cols],
                                        stats_df[common_cols]
                                    ])
                            logger.info(f"Estadísticas de partidos cargadas desde {stats_path}")
                        except Exception as e:
                            logger.warning(f"Error al cargar estadísticas desde {stats_path}: {str(e)}")
                
                # Si no encontramos archivos específicos, intentar con los archivos de partidos principales
                if not hasattr(self, 'match_stats_df'):
                    # Intentar con partidos ATP
                    atp_matches_path = self.data_dir / 'atp' / 'atp_matches_main_2000_2024.csv'
                    if atp_matches_path.exists():
                        atp_matches = pd.read_csv(atp_matches_path)
                        # Verificar si tiene columnas de estadísticas
                        stat_cols = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
                                   'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon']
                        if any(col in atp_matches.columns for col in stat_cols):
                            self.match_stats_df = atp_matches
                            logger.info(f"Estadísticas extraídas de partidos ATP: {len(atp_matches)} partidos")
                    
                    # Intentar con partidos WTA
                    wta_matches_path = self.data_dir / 'wta' / 'wta_matches_main_2000_2024.csv'
                    if wta_matches_path.exists():
                        wta_matches = pd.read_csv(wta_matches_path, low_memory=False)
                        # Verificar si tiene columnas de estadísticas
                        if any(col in wta_matches.columns for col in stat_cols):
                            if hasattr(self, 'match_stats_df'):
                                # Asegurar que tienen las mismas columnas
                                common_cols = list(set(self.match_stats_df.columns) & set(wta_matches.columns))
                                if common_cols:
                                    self.match_stats_df = pd.concat([
                                        self.match_stats_df[common_cols],
                                        wta_matches[common_cols]
                                    ])
                            else:
                                self.match_stats_df = wta_matches
                            logger.info(f"Estadísticas extraídas de partidos WTA: {len(wta_matches)} partidos")
                
                # Verificar si tenemos estadísticas Match Charting
                mcp_path = self.data_dir / 'match_charting' / 'all_matches_combined.csv'
                if mcp_path.exists():
                    try:
                        mcp_df = pd.read_csv(mcp_path)
                        self.mcp_matches_df = mcp_df
                        logger.info(f"Datos de Match Charting Project cargados: {len(mcp_df)} partidos")
                    except Exception as e:
                        logger.warning(f"Error cargando datos de Match Charting: {str(e)}")
                        self.mcp_matches_df = pd.DataFrame()
                else:
                    self.mcp_matches_df = pd.DataFrame()
            except Exception as e:
                logger.warning(f"Error cargando estadísticas de partidos: {str(e)}")
                self.match_stats_df = pd.DataFrame()
        else:
            self.match_stats_df = pd.DataFrame()
            
    def get_player_rating(self, player_id: str, surface: Optional[str] = None) -> float:
        """
        Obtiene el rating ELO actual de un jugador.
        
        Args:
            player_id: ID del jugador
            surface: Superficie específica (opcional)
            
        Returns:
            Rating ELO (general o específico por superficie)
        """
        return self.rating_calculator.get_player_rating(
            player_id, 
            surface,
            self.player_ratings,
            self.player_ratings_by_surface,
            self.player_match_count,
            self.player_match_count_by_surface,
            self.player_rating_uncertainty
        )
    
    def get_combined_surface_rating(self, player_id: str, surface: str) -> float:
        """
        Obtiene un rating combinado por superficie que integra información de
        todas las superficies ponderadas según la matriz de transferencia.
        
        Args:
            player_id: ID del jugador
            surface: Superficie para la que calcular el rating
            
        Returns:
            Rating ELO combinado para la superficie específica
        """
        return self.rating_calculator.get_combined_surface_rating(
            player_id,
            surface,
            self.player_ratings,
            self.player_ratings_by_surface,
            self.player_match_count,
            self.player_match_count_by_surface
        )

    def update_ratings(self, 
                  winner_id: str, 
                  loser_id: str, 
                  match_date: datetime, 
                  surface: str,
                  tourney_level: str,
                  round_name: str,
                  score: str,
                  retirement: bool = False,
                  stats: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Actualiza los ratings ELO después de un partido usando el sistema avanzado.
        
        Args:
            winner_id: ID del jugador ganador
            loser_id: ID del jugador perdedor
            match_date: Fecha del partido
            surface: Superficie de juego
            tourney_level: Nivel del torneo
            round_name: Ronda del torneo
            score: Resultado del partido
            retirement: Si hubo abandono
            stats: Estadísticas adicionales del partido (opcional)
            
        Returns:
            Tuple (cambio_elo_ganador, cambio_elo_perdedor)
        """
        try:
            # Convertir IDs a string con validación
            winner_id = str(winner_id) if not pd.isna(winner_id) else ''
            loser_id = str(loser_id) if not pd.isna(loser_id) else ''
            
            # Verificar IDs válidos
            if not winner_id or not loser_id or winner_id == loser_id:
                logger.warning(f"IDs inválidos para actualización de ratings: winner={winner_id}, loser={loser_id}")
                return 0.0, 0.0  # No hacer cambios si los IDs no son válidos
            
            # Validar match_date
            if not isinstance(match_date, datetime):
                try:
                    match_date = pd.to_datetime(match_date)
                except:
                    match_date = datetime.now()  # Valor por defecto
            
            # Aplicar decaimiento temporal a ambos jugadores
            self.rating_calculator.apply_temporal_decay(winner_id, match_date, 
                                              self.player_ratings, 
                                              self.player_ratings_by_surface,
                                              self.player_last_match,
                                              self.player_rating_uncertainty)
            
            self.rating_calculator.apply_temporal_decay(loser_id, match_date, 
                                              self.player_ratings, 
                                              self.player_ratings_by_surface,
                                              self.player_last_match,
                                              self.player_rating_uncertainty)
            
            # Normalizar superficie con validación
            if not isinstance(surface, str):
                surface = str(surface) if not pd.isna(surface) else 'hard'
            surface = normalize_surface(surface)
            
            # Validar tourney_level y round_name
            if not isinstance(tourney_level, str):
                tourney_level = str(tourney_level) if not pd.isna(tourney_level) else 'O'
            tourney_level = normalize_tournament_level(tourney_level)
            
            if not isinstance(round_name, str):
                round_name = str(round_name) if not pd.isna(round_name) else 'R32'
            
            # Validar score y retirement
            if not isinstance(score, str):
                score = str(score) if not pd.isna(score) else ''
            
            if not isinstance(retirement, bool):
                retirement = bool(retirement) if not pd.isna(retirement) else False
            
            # Validar stats
            if stats is not None and not isinstance(stats, dict):
                stats = None
            
            # Obtener ratings actuales con enfoque bayesiano
            w_general_elo = self.get_player_rating(winner_id)
            l_general_elo = self.get_player_rating(loser_id)
            
            # Ratings específicos por superficie (con transferencia entre superficies)
            w_surface_elo = self.get_combined_surface_rating(winner_id, surface)
            l_surface_elo = self.get_combined_surface_rating(loser_id, surface)
            
            # Obtener incertidumbres
            w_uncertainty = self.rating_calculator.get_player_uncertainty(winner_id, surface, 
                                                           self.player_match_count, 
                                                           self.player_match_count_by_surface,
                                                           self.player_rating_uncertainty)
            
            l_uncertainty = self.rating_calculator.get_player_uncertainty(loser_id, surface,
                                                           self.player_match_count,
                                                           self.player_match_count_by_surface,
                                                           self.player_rating_uncertainty)
            
            # Calcular probabilidades esperadas
            # 1. Probabilidad basada en ELO general
            w_win_prob_general = self.rating_calculator.calculate_expected_win_probability(
                w_general_elo, l_general_elo, 
                w_uncertainty, l_uncertainty
            )
            
            # 2. Probabilidad basada en ELO específico por superficie
            w_win_prob_surface = self.rating_calculator.calculate_expected_win_probability(
                w_surface_elo, l_surface_elo, 
                w_uncertainty, l_uncertainty
            )
            
            # 3. Obtener pesos específicos para superficie y tipo de torneo
            try:
                surface_weight = self.tourney_surface_weights.get(
                    normalize_tournament_level(tourney_level), 
                    {}
                ).get(surface, 0.7)
                
                # Asegurar que es un número
                if not isinstance(surface_weight, (int, float)):
                    surface_weight = 0.7
            except:
                surface_weight = 0.7  # Valor por defecto
            
            # 4. Media ponderada con peso específico
            w_win_prob = (1.0 - surface_weight) * w_win_prob_general + surface_weight * w_win_prob_surface
            
            # 5. Considerar ventaja head-to-head
            h2h_factor = self.player_stats.get_h2h_advantage(winner_id, loser_id)
            
            # 6. Considerar forma reciente
            w_form = self.player_stats.get_player_form(winner_id, surface)
            l_form = self.player_stats.get_player_form(loser_id, surface)
            form_ratio = w_form / l_form if l_form > 0 else 1.0
            
            # 7. Ajustar probabilidad final con h2h y forma
            # Limitar el impacto para mantener estabilidad
            adjustment = ((h2h_factor - 1.0) * 0.5) + ((form_ratio - 1.0) * 0.3)
            w_win_prob = min(0.95, max(0.05, w_win_prob + adjustment * 0.1))
            
            # 8. Calcular importancia del partido
            match_importance = self.match_analyzer.get_match_importance_factor(
                tourney_level, round_name, winner_id, loser_id, w_general_elo, l_general_elo
            )
            
            # 9. Analizar margen de victoria
            margin_multiplier = self.match_analyzer.get_margin_multiplier(score, retirement)
            
            # 10. Calcular factores de impacto para victorias importantes/inesperadas
            victory_impact = self.match_analyzer.calculate_victory_impact_factor(
                w_general_elo, l_general_elo, w_win_prob, margin_multiplier, match_importance
            )
            
            # 11. Extraer factor adicional de estadísticas si están disponibles
            if stats and isinstance(stats, dict):
                stats_factor = self.match_analyzer.get_match_stats_factor(
                    None, winner_id, loser_id, stats.get('winner'), stats.get('loser')
                )
            else:
                stats_factor = self.match_analyzer.get_match_stats_factor(
                    None, winner_id, loser_id, None, None, self.match_stats_df
                )
            
            # 12. Calcular factores K dinámicos
            k_winner = self.match_analyzer.get_dynamic_k_factor(
                winner_id, tourney_level, round_name, surface, match_importance,
                self.player_match_count.get(winner_id, 0), w_uncertainty
            )
            
            k_loser = self.match_analyzer.get_dynamic_k_factor(
                loser_id, tourney_level, round_name, surface, match_importance,
                self.player_match_count.get(loser_id, 0), l_uncertainty
            )
            
            # 13. Calcular cambios de ELO con todos los factores
            # Para el ganador: K * margen * impacto * stats * (actual - esperado)
            elo_change_winner = k_winner * margin_multiplier * victory_impact * stats_factor * (1.0 - w_win_prob)
            
            # Para el perdedor: efecto proporcional pero en dirección opuesta
            # Asegura que sea negativo (el perdedor pierde puntos)
            elo_change_loser = -k_loser * margin_multiplier * stats_factor * w_win_prob
            
            # Gestionar valores extremos con límites razonables
            elo_change_winner = min(50, max(1, elo_change_winner))
            elo_change_loser = max(-50, min(-1, elo_change_loser))
            
            # 14. Actualizar ratings generales con validación
            if winner_id not in self.player_ratings:
                self.player_ratings[winner_id] = self.initial_rating
            if loser_id not in self.player_ratings:
                self.player_ratings[loser_id] = self.initial_rating
            
            self.player_ratings[winner_id] += elo_change_winner
            self.player_ratings[loser_id] += elo_change_loser
            
            # 15. Actualizar ratings por superficie con validación
            if winner_id not in self.player_ratings_by_surface[surface]:
                self.player_ratings_by_surface[surface][winner_id] = self.initial_rating
            if loser_id not in self.player_ratings_by_surface[surface]:
                self.player_ratings_by_surface[surface][loser_id] = self.initial_rating
                
            # Actualizar con mayor especificidad para superficie
            try:
                surface_mult = self.match_analyzer.surface_specificity.get(surface, 1.0)
            except:
                surface_mult = 1.0  # Valor por defecto
                
            self.player_ratings_by_surface[surface][winner_id] += elo_change_winner * surface_mult
            self.player_ratings_by_surface[surface][loser_id] += elo_change_loser * surface_mult
            
            # 16. Actualizar fecha del último partido
            self.player_last_match[winner_id] = match_date
            self.player_last_match[loser_id] = match_date
            
            # 17. Actualizar contadores de partidos con validación
            # General
            if winner_id not in self.player_match_count:
                self.player_match_count[winner_id] = 0
            if loser_id not in self.player_match_count:
                self.player_match_count[loser_id] = 0
                
            self.player_match_count[winner_id] = self.player_match_count[winner_id] + 1
            self.player_match_count[loser_id] = self.player_match_count[loser_id] + 1
            
            # Por superficie
            if winner_id not in self.player_match_count_by_surface[surface]:
                self.player_match_count_by_surface[surface][winner_id] = 0
            if loser_id not in self.player_match_count_by_surface[surface]:
                self.player_match_count_by_surface[surface][loser_id] = 0
                
            self.player_match_count_by_surface[surface][winner_id] += 1
            self.player_match_count_by_surface[surface][loser_id] += 1
            
            # 18. Actualizar registro head-to-head
            self.player_stats.update_h2h_record(winner_id, loser_id)
            
            # 19. Actualizar forma reciente
            self.player_stats.update_player_form(winner_id, 1, surface)  # Victoria = 1
            self.player_stats.update_player_form(loser_id, 0, surface)   # Derrota = 0
            
            # 20. Actualizar incertidumbre de los ratings
            # Más partidos = menor incertidumbre
            w_matches = self.player_match_count.get(winner_id, 0)
            l_matches = self.player_match_count.get(loser_id, 0)
            
            # Fórmula simplificada: decae con cada partido, más rápido al principio
            new_w_uncertainty = 350 / (w_matches + 5)
            new_l_uncertainty = 350 / (l_matches + 5)
            
            self.player_rating_uncertainty[winner_id] = new_w_uncertainty
            self.player_rating_uncertainty[loser_id] = new_l_uncertainty
            
            # 21. Registrar historial de ratings para análisis
            # Crear un registro seguro con validaciones de tipo
            try:
                history_record = {
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
                    'expected_win_prob': w_win_prob,
                    'h2h_factor': h2h_factor,
                    'victory_impact': victory_impact,
                    'stats_factor': stats_factor,
                    'match_importance': match_importance,
                    'retirement': retirement
                }
                
                self.rating_history.append(history_record)
            except Exception as e:
                logger.warning(f"Error registrando historial: {str(e)}")
            
            # 22. Actualizar historial de partidos por jugador
            try:
                # Crear resumen del partido para el ganador
                match_summary_winner = {
                    'date': match_date,
                    'opponent_id': loser_id,
                    'surface': surface,
                    'result': 'win',
                    'score': score,
                    'elo_change': elo_change_winner,
                    'tourney_level': tourney_level
                }
                
                # Crear resumen del partido para el perdedor
                match_summary_loser = {
                    'date': match_date,
                    'opponent_id': winner_id,
                    'surface': surface,
                    'result': 'loss',
                    'score': score,
                    'elo_change': elo_change_loser,
                    'tourney_level': tourney_level
                }
                
                # Actualizar historiales
                self.player_stats.update_match_history(winner_id, match_summary_winner)
                self.player_stats.update_match_history(loser_id, match_summary_loser)
            except Exception as e:
                logger.warning(f"Error actualizando historial de partidos: {str(e)}")
            
            return elo_change_winner, elo_change_loser
        except Exception as e:
            # Capturar cualquier error no manejado para evitar que falle todo el proceso
            logger.error(f"Error en update_ratings: {str(e)}")
            logger.debug(traceback.format_exc())
            return 0.0, 0.0  # Devolver valores neutrales
    
    def predict_match(self, player1_id: str, player2_id: str, surface: str, 
                     tourney_level: str = 'O', round_name: str = 'R32') -> Dict:
        """
        Predice el resultado de un partido hipotético entre dos jugadores.
        
        Args:
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            surface: Superficie de juego
            tourney_level: Nivel del torneo (opcional)
            round_name: Ronda del torneo (opcional)
            
        Returns:
            Diccionario con predicciones detalladas
        """
        player1_id = str(player1_id)
        player2_id = str(player2_id)
        surface = normalize_surface(surface)
        
        # Obtener ratings con enfoque bayesiano
        p1_general_elo = self.get_player_rating(player1_id)
        p2_general_elo = self.get_player_rating(player2_id)
        p1_surface_elo = self.get_combined_surface_rating(player1_id, surface)
        p2_surface_elo = self.get_combined_surface_rating(player2_id, surface)
        
        # Obtener incertidumbres
        p1_uncertainty = self.rating_calculator.get_player_uncertainty(
            player1_id, surface, 
            self.player_match_count, 
            self.player_match_count_by_surface,
            self.player_rating_uncertainty
        )
        
        p2_uncertainty = self.rating_calculator.get_player_uncertainty(
            player2_id, surface,
            self.player_match_count,
            self.player_match_count_by_surface,
            self.player_rating_uncertainty
        )
        
        # Calcular probabilidades base
        p1_prob_general = self.rating_calculator.calculate_expected_win_probability(
            p1_general_elo, p2_general_elo, p1_uncertainty, p2_uncertainty
        )
        p1_prob_surface = self.rating_calculator.calculate_expected_win_probability(
            p1_surface_elo, p2_surface_elo, p1_uncertainty, p2_uncertainty
        )
        
        # Pesos según superficie y tipo de torneo
        surface_weight = self.tourney_surface_weights.get(
            normalize_tournament_level(tourney_level), 
            {}
        ).get(surface, 0.7)
        
        # Probabilidad ponderada base
        p1_prob_base = (1.0 - surface_weight) * p1_prob_general + surface_weight * p1_prob_surface
        
        # Factores contextuales
        h2h_factor = self.player_stats.get_h2h_advantage(player1_id, player2_id)
        p1_form = self.player_stats.get_player_form(player1_id, surface)
        p2_form = self.player_stats.get_player_form(player2_id, surface)
        form_ratio = p1_form / p2_form if p2_form > 0 else 1.0
        
        # Otros factores específicos
        style_factor = self.match_analyzer.get_style_compatibility_factor(player1_id, player2_id, surface)
        
        # Calcular factores de experiencia y confianza
        p1_matches = self.player_match_count.get(player1_id, 0)
        p2_matches = self.player_match_count.get(player2_id, 0)
        p1_surface_matches = self.player_match_count_by_surface[surface].get(player1_id, 0)
        p2_surface_matches = self.player_match_count_by_surface[surface].get(player2_id, 0)
        
        experience_ratio = (p1_matches / max(1, p2_matches)) if p2_matches > 0 else 1.0
        surface_exp_ratio = (p1_surface_matches / max(1, p2_surface_matches)) if p2_surface_matches > 0 else 1.0
        
        # Ajustar probabilidad con factores contextuales
        adjustment = ((h2h_factor - 1.0) * 0.3) + ((form_ratio - 1.0) * 0.2) + ((style_factor - 1.0) * 0.1)
        
        # Añadir un pequeño factor de experiencia
        exp_adjustment = 0
        if p1_matches > 0 and p2_matches > 0:
            if experience_ratio > 1.5:  # Si P1 tiene bastante más experiencia
                exp_adjustment += 0.01
            elif experience_ratio < 0.67:  # Si P2 tiene bastante más experiencia
                exp_adjustment -= 0.01
        
        # Añadir ajuste por experiencia en la superficie
        surface_exp_adjustment = 0
        if p1_surface_matches > 0 and p2_surface_matches > 0:
            if surface_exp_ratio > 2.0:  # Si P1 tiene mucha más experiencia en la superficie
                surface_exp_adjustment += 0.02
            elif surface_exp_ratio < 0.5:  # Si P2 tiene mucha más experiencia en la superficie
                surface_exp_adjustment -= 0.02
        
        # Aplicar todos los ajustes
        p1_final_prob = min(0.95, max(0.05, p1_prob_base + adjustment * 0.1 + exp_adjustment + surface_exp_adjustment))
        
        # Confianza en la predicción (inversamente proporcional a la incertidumbre)
        prediction_certainty = 1.0 - ((p1_uncertainty + p2_uncertainty) / 700)
        prediction_certainty = max(0.1, min(0.9, prediction_certainty))
        
        # Intervalo de confianza aproximado
        margin_of_error = (1.0 - prediction_certainty) * 0.2
        prob_lower = max(0.01, p1_final_prob - margin_of_error)
        prob_upper = min(0.99, p1_final_prob + margin_of_error)
        
        # Construir respuesta detallada
        return {
            'player1': {
                'id': player1_id,
                'name': self.player_stats.get_player_name(player1_id),
                'elo_general': p1_general_elo,
                'elo_surface': p1_surface_elo,
                'form': p1_form,
                'matches': p1_matches,
                'surface_matches': p1_surface_matches,
                'uncertainty': p1_uncertainty
            },
            'player2': {
                'id': player2_id,
                'name': self.player_stats.get_player_name(player2_id),
                'elo_general': p2_general_elo,
                'elo_surface': p2_surface_elo,
                'form': p2_form,
                'matches': p2_matches,
                'surface_matches': p2_surface_matches,
                'uncertainty': p2_uncertainty
            },
            'prediction': {
                'p1_win_probability': p1_final_prob,
                'p2_win_probability': 1.0 - p1_final_prob,
                'confidence_interval': [prob_lower, prob_upper],
                'prediction_certainty': prediction_certainty,
                'favorite': player1_id if p1_final_prob > 0.5 else player2_id,
                'favorite_name': self.player_stats.get_player_name(player1_id if p1_final_prob > 0.5 else player2_id)
            },
            'factors': {
                'h2h_factor': h2h_factor,
                'form_ratio': form_ratio,
                'style_factor': style_factor,
                'experience_ratio': experience_ratio,
                'surface_experience_ratio': surface_exp_ratio,
                'surface_weight': surface_weight
            },
            'context': {
                'surface': surface,
                'tourney_level': tourney_level,
                'round': round_name
            }
        }
    
    def process_matches_dataframe(self, matches_df: pd.DataFrame, chronological: bool = True,
                           include_stats: bool = True, batch_size: int = 1000) -> pd.DataFrame:
        """
        Procesa todos los partidos de un DataFrame calculando ELO progresivamente.
        Implementa procesamiento en lotes para mejorar rendimiento y memoria.
        
        Args:
            matches_df: DataFrame con datos de partidos
            chronological: Si es True, ordena los partidos cronológicamente primero
            include_stats: Si debe extraer estadísticas adicionales si están disponibles
            batch_size: Tamaño de los lotes para procesamiento
            
        Returns:
            DataFrame con columnas de ELO añadidas
        """
        if matches_df.empty:
            logger.warning("DataFrame de partidos vacío, no hay nada que procesar")
            return matches_df
        
        start_time = datetime.now()
        
        # Hacer copia para no modificar el original
        df = matches_df.copy()
        
        # Mapear nombres de columnas si son diferentes
        column_map = {
            'match_date': ['date', 'tourney_date'],
            'winner_id': ['player1_id', 'p1_id', 'w_id'],
            'loser_id': ['player2_id', 'p2_id', 'l_id'],
            'surface': ['surface_normalized', 'court_surface'],
            'tourney_level': ['tournament_level', 'tournament_category', 'tourney_type', 'level'],
            'round': ['round_name'],
            'score': ['match_score'],
            'retirement': ['is_retirement', 'ret', 'w_outcome']
        }
        
        # Verificar y renombrar columnas si es necesario
        for target, alternatives in column_map.items():
            if target not in df.columns:
                for alt in alternatives:
                    if alt in df.columns:
                        df[target] = df[alt]
                        logger.debug(f"Renombrando columna '{alt}' a '{target}'")
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
            df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
            # Eliminar filas con fechas inválidas
            invalid_dates = df['match_date'].isna()
            if invalid_dates.any():
                logger.warning(f"Eliminando {invalid_dates.sum()} filas con fechas inválidas")
                df = df.dropna(subset=['match_date'])
        
        # Convertir IDs a string y validar
        df['winner_id'] = df['winner_id'].astype(str)
        df['loser_id'] = df['loser_id'].astype(str)
        
        # Validar que winner_id y loser_id no sean iguales
        invalid_ids = df['winner_id'] == df['loser_id']
        if invalid_ids.any():
            logger.warning(f"Eliminando {invalid_ids.sum()} filas con winner_id = loser_id")
            df = df[~invalid_ids]
        
        # Normalizar superficie
        df['surface'] = df['surface'].apply(lambda x: normalize_surface(x) if not pd.isna(x) else 'hard')
        
        # Normalizar nivel de torneo
        if 'tourney_level' in df.columns:
            df['tourney_level'] = df['tourney_level'].apply(
                lambda x: normalize_tournament_level(x) if not pd.isna(x) else 'O'
            )
        
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
        
        # Crear columnas de match_id si no existe
        if 'match_id' not in df.columns:
            df['match_id'] = range(len(df))
        
        # Ordenar cronológicamente si se especifica
        if chronological:
            logger.info("Ordenando partidos cronológicamente...")
            df = df.sort_values('match_date')
        
        # Añadir columnas para ratings ELO y predicciones
        logger.info("Preparando columnas para análisis ELO...")
        elo_columns = [
            'winner_elo_before', 'loser_elo_before', 
            'winner_surface_elo_before', 'loser_surface_elo_before',
            'winner_elo_after', 'loser_elo_after', 
            'winner_surface_elo_after', 'loser_surface_elo_after',
            'elo_change_winner', 'elo_change_loser',
            'expected_win_prob', 'margin_multiplier', 
            'k_factor_winner', 'k_factor_loser',
            'match_importance', 'h2h_factor', 'form_factor'
        ]
        
        for col in elo_columns:
            df[col] = 0.0
        
        # Extraer estadísticas de partidos si se solicita y están disponibles
        match_stats = {}
        if include_stats and hasattr(self, 'match_stats_df') and not self.match_stats_df.empty:
            logger.info("Extrayendo estadísticas de partidos...")
            try:
                # Preparar lookup de estadísticas por match_id
                if 'match_id' in self.match_stats_df.columns and 'match_id' in df.columns:
                    # Asegurar que los tipos sean compatibles
                    match_stats_df_copy = self.match_stats_df.copy()
                    match_stats_df_copy['match_id'] = match_stats_df_copy['match_id'].astype(str)
                    df['match_id_str'] = df['match_id'].astype(str)
                    
                    # Mapear columnas comunes para estadísticas
                    stat_cols = [
                        'ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 
                        'SvGms', 'bpSaved', 'bpFaced'
                    ]
                    
                    # Construir diccionario de estadísticas
                    for _, row in match_stats_df_copy.iterrows():
                        if 'match_id' in row:
                            match_id = row['match_id']
                            
                            w_stats = {}
                            l_stats = {}
                            
                            for col in stat_cols:
                                w_col = f'w_{col}'
                                l_col = f'l_{col}'
                                
                                if w_col in row:
                                    w_stats[col] = row[w_col]
                                if l_col in row:
                                    l_stats[col] = row[l_col]
                            
                            match_stats[match_id] = {
                                'winner': w_stats,
                                'loser': l_stats
                            }
                elif not self.match_stats_df.empty:
                    # Intentar hacer match por combinación de IDs y fecha
                    logger.info("Intentando hacer match de estadísticas por jugadores y fecha...")
                    
                    for _, match_row in df.iterrows():
                        winner_id = str(match_row['winner_id'])
                        loser_id = str(match_row['loser_id'])
                        match_date = match_row['match_date']
                        match_id = match_row['match_id']
                        
                        # Buscar estadísticas para este partido
                        potential_stats = self.match_stats_df[
                            (self.match_stats_df['winner_id'].astype(str) == winner_id) & 
                            (self.match_stats_df['loser_id'].astype(str) == loser_id)
                        ]
                        
                        if 'match_date' in self.match_stats_df.columns:
                            if pd.api.types.is_datetime64_any_dtype(self.match_stats_df['match_date']):
                                potential_stats = potential_stats[
                                    (potential_stats['match_date'] - match_date).abs() < pd.Timedelta(days=1)
                                ]
                            else:
                                # Intentar convertir a datetime para comparar
                                try:
                                    match_stats_date = pd.to_datetime(potential_stats['match_date'])
                                    match_row_date = pd.to_datetime(match_date)
                                    date_diff = abs(match_stats_date - match_row_date)
                                    potential_stats = potential_stats[date_diff < pd.Timedelta(days=1)]
                                except:
                                    # Si falla la conversión, usar todos los resultados
                                    pass
                        
                        if not potential_stats.empty:
                            stats_row = potential_stats.iloc[0]
                            
                            w_stats = {}
                            l_stats = {}
                            
                            for col in self.match_stats_df.columns:
                                if col.startswith('w_'):
                                    base_col = col[2:]
                                    w_stats[base_col] = stats_row[col]
                                elif col.startswith('l_'):
                                    base_col = col[2:]
                                    l_stats[base_col] = stats_row[col]
                            
                            match_stats[match_id] = {
                                'winner': w_stats,
                                'loser': l_stats
                            }
            except Exception as e:
                logger.warning(f"Error extrayendo estadísticas: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Actualizar estadísticas
        self.stats['total_matches_processed'] = 0
        self.stats['invalid_matches_skipped'] = 0
        
        # Registrar rango de fechas
        if not df.empty:
            self.stats['dates_range'] = (df['match_date'].min(), df['match_date'].max())
        
        # Procesar partidos en lotes
        logger.info(f"Procesando {len(df)} partidos para cálculo de ELO...")
        
        # Dividir en lotes para procesamiento optimizado
        lotes = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
        
        for lote_num, lote_df in enumerate(lotes):
            logger.info(f"Procesando lote {lote_num+1}/{len(lotes)} ({len(lote_df)} partidos)...")
            
            # Usar tqdm para mostrar barra de progreso dentro del lote
            for idx in tqdm(lote_df.index, desc=f"Lote {lote_num+1}", total=len(lote_df)):
                try:
                    match = df.loc[idx]
                    
                    # Obtener datos del partido con validaciones
                    winner_id = str(match['winner_id'])
                    loser_id = str(match['loser_id'])
                    match_date = match['match_date']
                    surface = match['surface']
                    tourney_level = match['tourney_level']
                    round_name = match['round']
                    score = match['score']
                    retirement = match['retirement']
                    match_id = match['match_id']
                    
                    # Validar IDs básicos
                    if pd.isna(winner_id) or pd.isna(loser_id) or winner_id == loser_id:
                        logger.warning(f"Partido inválido (ID ganador = ID perdedor o valores nulos): {winner_id} vs {loser_id}")
                        self.stats['invalid_matches_skipped'] += 1
                        continue
                    
                    # Guardar ratings actuales antes de la actualización
                    df.at[idx, 'winner_elo_before'] = self.get_player_rating(winner_id)
                    df.at[idx, 'loser_elo_before'] = self.get_player_rating(loser_id)
                    df.at[idx, 'winner_surface_elo_before'] = self.get_player_rating(winner_id, surface)
                    df.at[idx, 'loser_surface_elo_before'] = self.get_player_rating(loser_id, surface)
                    
                    # Calcular factores adicionales con manejo de errores
                    h2h_factor = self.player_stats.get_h2h_advantage(winner_id, loser_id)
                    df.at[idx, 'h2h_factor'] = h2h_factor
                    
                    # Calcular factor de forma con validación
                    w_form = self.player_stats.get_player_form(winner_id, surface)
                    l_form = self.player_stats.get_player_form(loser_id, surface)
                    form_factor = w_form / l_form if l_form > 0 else 1.0
                    df.at[idx, 'form_factor'] = form_factor
                    
                    # Calcular importancia del partido con validación
                    match_importance = self.match_analyzer.get_match_importance_factor(
                        tourney_level, round_name, winner_id, loser_id,
                        self.get_player_rating(winner_id),
                        self.get_player_rating(loser_id)
                    )
                    df.at[idx, 'match_importance'] = match_importance
                    
                    # Calcular margen del partido con validación
                    margin_multiplier = self.match_analyzer.get_margin_multiplier(score, retirement)
                    df.at[idx, 'margin_multiplier'] = margin_multiplier
                    
                    # Obtener estadísticas específicas del partido si existen
                    match_specific_stats = match_stats.get(match_id, None)
                    
                    # Actualizar ratings
                    elo_change_winner, elo_change_loser = self.update_ratings(
                        winner_id, loser_id, match_date, surface, 
                        tourney_level, round_name, score, retirement,
                        match_specific_stats
                    )
                    
                    # Guardar resultados de la actualización
                    df.at[idx, 'elo_change_winner'] = elo_change_winner
                    df.at[idx, 'elo_change_loser'] = elo_change_loser
                    df.at[idx, 'winner_elo_after'] = self.get_player_rating(winner_id)
                    df.at[idx, 'loser_elo_after'] = self.get_player_rating(loser_id)
                    df.at[idx, 'winner_surface_elo_after'] = self.get_player_rating(winner_id, surface)
                    df.at[idx, 'loser_surface_elo_after'] = self.get_player_rating(loser_id, surface)
                    
                    # Recuperar valores desde el último registro en el historial
                    if self.rating_history:
                        last_record = self.rating_history[-1]
                        df.at[idx, 'expected_win_prob'] = last_record.get('expected_win_prob', 0.5)
                        df.at[idx, 'k_factor_winner'] = last_record.get('k_factor_winner', self.k_factor_base)
                        df.at[idx, 'k_factor_loser'] = last_record.get('k_factor_loser', self.k_factor_base)
                    
                    # Actualizar contador
                    self.stats['total_matches_processed'] += 1
                    
                    # Actualizar contador de superficies
                    self.stats['surface_distribution'][normalize_surface(surface)] += 1
                    
                except Exception as e:
                    logger.warning(f"Error procesando partido {idx}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    self.stats['invalid_matches_skipped'] += 1
                    continue
        
        # Actualizar estadística de jugadores con ratings
        self.stats['players_with_ratings'] = len(self.player_ratings)
        self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Procesamiento completado: {self.stats['total_matches_processed']} partidos procesados, "
                  f"{self.stats['invalid_matches_skipped']} partidos omitidos")
        logger.info(f"Tiempo total de procesamiento: {self.stats['processing_time']:.1f} segundos")
        
        return df
    
    def get_player_details(self, player_id: str) -> Dict:
        """
        Obtiene información detallada del jugador incluyendo ratings,
        estadísticas, forma reciente y más.
        
        Args:
            player_id: ID del jugador
            
        Returns:
            Diccionario con toda la información disponible del jugador
        """
        return self.player_stats.get_player_details(
            player_id, 
            player_ratings=self.player_ratings,
            player_ratings_by_surface=self.player_ratings_by_surface, 
            player_match_count=self.player_match_count,
            player_match_count_by_surface=self.player_match_count_by_surface,
            player_uncertainty=self.player_rating_uncertainty
        )

    def get_top_players(self, n: int = 20, surface: Optional[str] = None, 
                      min_matches: int = 10, date: Optional[datetime] = None,
                      include_details: bool = False) -> pd.DataFrame:
        """
        Obtiene los mejores jugadores según su rating ELO.
        
        Args:
            n: Número de jugadores a mostrar
            surface: Superficie específica (opcional)
            min_matches: Número mínimo de partidos jugados
            date: Fecha para la cual obtener los ratings (opcional, usa los actuales si no se especifica)
            include_details: Si debe incluir detalles adicionales como nombre, etc.
            
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
            top_players = filtered_ratings.sort_values('elo_rating', ascending=False).head(n)
            
            if include_details:
                # Añadir nombres de jugadores y más detalles
                top_players['player_name'] = top_players['player_id'].apply(self.player_stats.get_player_name)
                
                # Añadir incertidumbre
                top_players['uncertainty'] = top_players['player_id'].apply(
                    lambda x: self.rating_calculator.get_player_uncertainty(
                        str(x), surface, self.player_match_count, 
                        self.player_match_count_by_surface,
                        self.player_rating_uncertainty
                    )
                )
                
                # Si hay superficie, añadir partidos en superficie
                if surface:
                    normalized_surface = normalize_surface(surface)
                    top_players['surface_matches'] = top_players['player_id'].apply(
                        lambda x: self.player_match_count_by_surface[normalized_surface].get(str(x), 0)
                    )
            
            return top_players
            
        # Usar ratings actuales
        if surface:
            surface = normalize_surface(surface)
            ratings = self.player_ratings_by_surface.get(surface, {})
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
        if surface:
            df['matches_played'] = df['player_id'].map(
                lambda x: self.player_match_count_by_surface[surface].get(x, 0)
            )
        else:
            df['matches_played'] = df['player_id'].map(
                lambda x: self.player_match_count.get(x, 0)
            )
        
        # Filtrar jugadores con pocos partidos
        df = df[df['matches_played'] >= min_matches]
        
        # Añadir detalles adicionales si se solicitan
        if include_details:
            # Añadir nombres de jugadores
            df['player_name'] = df['player_id'].apply(self.player_stats.get_player_name)
            
            # Añadir incertidumbre
            df['uncertainty'] = df['player_id'].apply(
                lambda x: self.rating_calculator.get_player_uncertainty(
                    x, surface, self.player_match_count, 
                    self.player_match_count_by_surface,
                    self.player_rating_uncertainty
                )
            )
            
            # Añadir forma reciente
            df['form'] = df['player_id'].apply(
                lambda x: self.player_stats.get_player_form(x, surface)
            )
            
            # Si hay superficie, añadir partidos en superficie
            if surface:
                df['surface_matches'] = df['matches_played']
                df['total_matches'] = df['player_id'].map(
                    lambda x: self.player_match_count.get(x, 0)
                )
            
            # Última fecha de partido
            df['last_match'] = df['player_id'].map(
                lambda x: self.player_last_match.get(x, None)
            )
        
        # Ordenar y devolver los N mejores
        return df.sort_values('elo_rating', ascending=False).head(n)   
    
    def save_ratings(self, output_dir: str = 'data/processed/elo',
                   include_history: bool = True,
                   include_match_history: bool = True) -> None:
        """
        Guarda los ratings ELO actuales en archivos JSON y CSV.
        
        Args:
            output_dir: Directorio donde guardar los archivos
            include_history: Si debe guardar historial completo de ratings
            include_match_history: Si debe guardar historial de partidos por jugador
        """
        # Delegar a la clase de persistencia cuando esté implementada
        from tennis_elo.io.persistence import save_ratings_data
        
        save_ratings_data(
            output_dir=output_dir,
            player_ratings=self.player_ratings,
            player_ratings_by_surface=self.player_ratings_by_surface,
            player_match_count=self.player_match_count,
            player_match_count_by_surface=self.player_match_count_by_surface,
            player_rating_uncertainty=self.player_rating_uncertainty,
            player_last_match=self.player_last_match,
            h2h_records=self.player_stats.h2h_records,
            player_recent_form=self.player_stats.player_recent_form,
            rating_history=self.rating_history,
            player_match_history=self.player_stats.player_match_history,
            stats=self.stats,
            get_top_players_func=self.get_top_players,
            include_history=include_history,
            include_match_history=include_match_history
        )
    
    def load_ratings(self, input_dir: str = 'data/processed/elo',
                   load_history: bool = True,
                   load_match_history: bool = True) -> None:
        """
        Carga ratings ELO previamente guardados.
        
        Args:
            input_dir: Directorio donde están los archivos
            load_history: Si debe cargar historial completo de ratings
            load_match_history: Si debe cargar historial de partidos por jugador
        """
        # Delegar a la clase de persistencia cuando esté implementada
        from tennis_elo.io.persistence import load_ratings_data
        
        result = load_ratings_data(
            input_dir=input_dir,
            load_history=load_history,
            load_match_history=load_match_history
        )
        
        # Actualizar todas las estructuras de datos
        self.player_ratings = result.get('player_ratings', {})
        self.player_ratings_by_surface = result.get('player_ratings_by_surface', {
            'hard': {}, 'clay': {}, 'grass': {}, 'carpet': {}
        })
        self.player_match_count = result.get('player_match_count', {})
        self.player_match_count_by_surface = result.get('player_match_count_by_surface', {
            'hard': {}, 'clay': {}, 'grass': {}, 'carpet': {}
        })
        self.player_rating_uncertainty = result.get('player_rating_uncertainty', {})
        self.player_last_match = result.get('player_last_match', {})
        self.rating_history = result.get('rating_history', [])
        self.stats = result.get('stats', self.stats)
        
        # Actualizar componentes
        self.player_stats.player_recent_form = result.get('player_recent_form', {})
        self.player_stats.h2h_records = result.get('h2h_records', defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0})))
        self.player_stats.player_match_history = result.get('player_match_history', defaultdict(list))
        
        logger.info(f"Ratings ELO cargados desde {input_dir} con {len(self.player_ratings)} jugadores")

   