"""
enhanced_tennis_elo_processor.py

Sistema ELO avanzado para tenis que incorpora:
- Análisis de datos punto por punto para mayor precisión
- ELO específico por superficie con transferencia de aprendizaje entre superficies
- Factores de oponente (historial h2h y compatibilidad de estilos)
- Ponderación por forma reciente y fatiga
- Análisis de estadísticas detalladas de partidos
- Enfoque bayesiano para jugadores con pocos partidos
- Características contextuales (indoor/outdoor, altitud, condiciones)
- Integración con datos del Match Charting Project
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
from sklearn.metrics import log_loss, brier_score_loss
from collections import defaultdict
import re
import warnings
from scipy import stats
import functools
import inspect

# Sistema de rastreo de errores
error_locations = []
function_errors = {}

def trace_errors(func):
    """Decorador para rastrear errores en funciones"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Obtener información sobre el error
            tb = traceback.extract_tb(sys.exc_info()[2])
            filename, line, func_name, text = tb[-1]
            
            # Registrar el error
            error_msg = f"ERROR en {func.__name__}: {str(e)} en línea {line} - '{text}'"
            logger.error(error_msg)
            
            # Guardar información detallada
            error_info = {
                'function': func.__name__,
                'exception': str(e),
                'exception_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'line': line,
                'text': text,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            error_locations.append(error_info)
            
            # Incrementar contador para esta función
            if func.__name__ not in function_errors:
                function_errors[func.__name__] = []
            function_errors[func.__name__].append(error_info)
            
            # Devolver un valor seguro según el tipo
            if func.__name__ == "update_ratings":
                return 0.0, 0.0  # Para update_ratings que devuelve una tupla
            elif func.__name__ in ["get_player_rating", "get_player_form", "get_combined_surface_rating", 
                                 "get_h2h_advantage", "calculate_expected_win_probability", 
                                 "_get_match_importance_factor", "_get_margin_multiplier",
                                 "_get_dynamic_k_factor", "_get_match_stats_factor"]:
                return 1.0  # Para funciones que devuelven float
            elif func.__name__ in ["process_matches_dataframe"]:
                return args[1]  # Devolver el DataFrame original
            else:
                return None
    
    return wrapper

def apply_tracers():
    """Aplica los decoradores a los métodos problemáticos"""
    methods_to_trace = [
        '_apply_temporal_decay',
        'calculate_expected_win_probability',
        '_parse_score', 
        'get_h2h_advantage',
        '_get_match_importance_factor',
        '_get_margin_multiplier',
        'update_ratings',
        'update_player_form',
        'update_h2h_record',
        '_get_dynamic_k_factor',
        'get_combined_surface_rating',
        'get_player_form',
        '_get_match_stats_factor',
        '_calculate_victory_impact_factor',
        '_normalize_tournament_level',
        '_normalize_surface',
        'process_matches_dataframe'
    ]
    
    for method_name in methods_to_trace:
        if hasattr(EnhancedTennisEloProcessor, method_name):
            original_method = getattr(EnhancedTennisEloProcessor, method_name)
            wrapped_method = trace_errors(original_method)
            setattr(EnhancedTennisEloProcessor, method_name, wrapped_method)

def save_error_report():
    """Guarda un informe de los errores encontrados"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"error_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("=== INFORME DE ERRORES EN TENNIS ELO PROCESSOR ===\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if not function_errors:
            f.write("¡No se encontraron errores!\n")
            return report_file
            
        f.write("=== RESUMEN DE ERRORES POR FUNCIÓN ===\n")
        for func_name, errors in sorted(function_errors.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"{func_name}: {len(errors)} errores\n")
        
        f.write("\n=== ERRORES DETALLADOS ===\n\n")
        for i, error in enumerate(error_locations, 1):
            f.write(f"ERROR #{i}\n")
            f.write(f"Función: {error['function']}\n")
            f.write(f"Tipo de excepción: {error['exception_type']}\n")
            f.write(f"Mensaje: {error['exception']}\n")
            f.write(f"Línea: {error['line']}\n")
            f.write(f"Código: {error['text']}\n")
            f.write(f"Timestamp: {error['timestamp']}\n")
            f.write("\nTraceback:\n")
            f.write(error['traceback'])
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"Informe de errores guardado en {report_file}")
    return report_file

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Crear directorio de logs si no existe
os.makedirs('logs', exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/enhanced_elo_processing.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTennisEloProcessor:
    """
    Procesador avanzado de ELO para tenis con múltiples factores contextuales
    y análisis detallado de estadísticas de partidos, enfoque bayesiano y
    ponderación dinámica basada en superficies, niveles de torneo y 
    características de jugadores.
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
        # Por ejemplo, un Masters 1000 en tierra batida muestra mejor las habilidades en esa superficie
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
        self.player_recent_form = {}  # Forma reciente (ratio victorias/derrotas ponderado)
        self.player_fatigue = {}      # Nivel de fatiga basado en partidos recientes
        self.rating_history = []      # Historial de ratings para análisis
        self.player_match_history = defaultdict(list)  # Historial de partidos por jugador
        
        # Historial head-to-head entre jugadores
        self.h2h_records = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0}))
        
        # Variables bayesianas - Para enfoque regularizado
        self.player_rating_uncertainty = {}  # Incertidumbre sobre el rating (stdev)
        self.player_rating_posterior_n = {}  # Número efectivo de partidos (enfoque bayesiano)
        
        # Mapeo de estadísticas de partidos a importancia para ELO
        self.stat_importance_weights = {
            'ace': 0.05,
            'df': -0.03,
            'svpt': 0.02,
            '1stIn': 0.04,
            '1stWon': 0.08,
            '2ndWon': 0.07,
            'bpSaved': 0.06,
            'bpFaced': -0.04,
            'SvGms': 0.03
        }
        
        # Parámetros dinámicos del sistema ELO
        self.tournament_k_factors = {
            'G': 2.0,      # Grand Slam
            'M': 1.5,      # Masters 1000
            'A': 1.2,      # ATP 500
            'D': 1.0,      # ATP 250
            'F': 1.8,      # Tour Finals
            'C': 0.8,      # Challenger
            'S': 0.5,      # Satellite/ITF
            'O': 0.7       # Other
        }
        
        # Mapeo para nombres de columnas de nivel de torneo
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
        
        # Multiplicadores por ronda (aumentan en rondas finales)
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
        
        # Multiplicadores para superficies (mayor especificidad)
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
        
        # Configuración para análisis de estilo de juego
        self.style_factors = {
            'serve_oriented': {'aces': 5, 'first_serve_pct': 3, 'first_serve_points_won_pct': 5},
            'return_oriented': {'return_points_won_pct': 5, 'break_points_converted_pct': 4},
            'baseline': {'winners': 2, 'rally_points_won_pct': 4},
            'attacking': {'net_points_won_pct': 5, 'winners': 3},
            'defensive': {'rally_points_won_pct': 4, 'errors': -3}
        }
        
        # Cachés para datos relacionados con partidos
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
        
        # Cargar datos punto por punto si se requiere
        if self.use_point_by_point:
            try:
                # Buscar datos punto por punto en varias ubicaciones posibles
                pbp_paths = [
                    self.data_dir / 'pointbypoint',
                    self.data_dir / 'slam_pointbypoint'
                ]
                
                self.pbp_files = []
                
                for pbp_path in pbp_paths:
                    if pbp_path.exists() and pbp_path.is_dir():
                        # Buscar archivos CSV en este directorio
                        csv_files = list(pbp_path.glob("*.csv"))
                        
                        # Filtrar archivos relevantes de puntos
                        pbp_files = [f for f in csv_files if 'point' in f.name.lower()]
                        
                        # Añadir al listado
                        self.pbp_files.extend(pbp_files)
                        
                        logger.info(f"Encontrados {len(pbp_files)} archivos punto por punto en {pbp_path}")
                
                # No cargamos todos los archivos inmediatamente para ahorrar memoria
                # Se cargarán bajo demanda
                logger.info(f"Total de archivos punto por punto encontrados: {len(self.pbp_files)}")
                
                # Cargar índices si existen para localizar datos más rápido
                for pbp_path in pbp_paths:
                    index_file = pbp_path / 'index.json'
                    if index_file.exists():
                        try:
                            with open(index_file, 'r') as f:
                                pbp_index = json.load(f)
                                if not hasattr(self, 'pbp_indices'):
                                    self.pbp_indices = pbp_index
                                else:
                                    # Combinar índices
                                    for key, value in pbp_index.items():
                                        if key in self.pbp_indices:
                                            self.pbp_indices[key].update(value)
                                        else:
                                            self.pbp_indices[key] = value
                                
                                logger.info(f"Índice punto por punto cargado desde {index_file}")
                        except Exception as e:
                            logger.warning(f"Error cargando índice punto por punto desde {index_file}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error explorando datos punto por punto: {str(e)}")
                self.pbp_files = []
    
    def _normalize_surface(self, surface: str) -> str:
        """
        Normaliza el nombre de la superficie.
        Versión corregida con validaciones de tipo.
        
        Args:
            surface: Nombre de la superficie (puede ser cualquier formato)
                
        Returns:
            Nombre normalizado de la superficie
        """
        # Validar tipo
        if not isinstance(surface, str):
            surface = str(surface) if not pd.isna(surface) else ''
        
        if pd.isna(surface) or not surface or surface == 'unknown':
            return 'hard'  # Usar hard como valor predeterminado
        
        # Normalizar
        surface_lower = str(surface).lower().strip()
        
        # Mapeo directo
        if surface_lower in self.surface_mapping:
            return self.surface_mapping[surface_lower]
        
        # Mapeo por contenido
        try:
            # Checking for common terms
            if 'hard' in surface_lower or 'h' == surface_lower:
                return 'hard'
            elif any(term in surface_lower for term in ['clay', 'arcilla', 'terre', 'c']):
                return 'clay'
            elif any(term in surface_lower for term in ['grass', 'hierba', 'g']):
                return 'grass'
            elif any(term in surface_lower for term in ['carpet', 'indoor', 'cr']):
                return 'carpet'
        except Exception as e:
            logger.debug(f"Error en mapeo por contenido: {str(e)}")
        
        # Si no hay match, retornar hard como valor por defecto
        return 'hard'
    
    def _normalize_tournament_level(self, level: str) -> str:
        """
        Normaliza el nivel del torneo.
        Versión corregida con validaciones de tipo.
        
        Args:
            level: Nivel del torneo (puede ser cualquier formato)
            
        Returns:
            Código normalizado del nivel del torneo
        """
        # Validar tipo
        if not isinstance(level, str):
            level = str(level) if not pd.isna(level) else ''
        
        if pd.isna(level) or not level:
            return 'O'  # Otros como valor predeterminado
        
        # Trim y minúsculas para normalización
        level = level.strip()
        
        # Primero verificar si es un código normalizado directo (G, M, etc.)
        if level in self.tourney_level_mapping:
            return self.tourney_level_mapping[level]
        
        # Si es un nombre más largo, intentar con el mapeo
        try:
            # Hacer case-insensitive para mayor robustez
            level_upper = level.upper()
            
            # Verificar mapeos conocidos
            if 'GRAND' in level_upper or 'SLAM' in level_upper:
                return 'G'
            elif 'MASTER' in level_upper:
                return 'M'
            elif 'ATP500' in level_upper or 'ATP 500' in level_upper:
                return 'A'
            elif 'ATP250' in level_upper or 'ATP 250' in level_upper:
                return 'D'
            elif 'FINAL' in level_upper and ('TOUR' in level_upper or 'ATP' in level_upper):
                return 'F'
            elif 'CHALL' in level_upper:
                return 'C'
            elif any(term in level_upper for term in ['FUTURE', 'ITF']):
                return 'S'
            
            # Si no hay match exacto, buscar en los valores del mapeo
            for key, val in self.tourney_level_mapping.items():
                if key.upper() in level_upper:
                    return val
        except Exception as e:
            logger.debug(f"Error normalizando nivel de torneo: {str(e)}")
        
        # Si no hay match, retornar valor por defecto
        return 'O'  # Otros
    
    def get_player_name(self, player_id: str) -> str:
        """
        Obtiene el nombre de un jugador según su ID.
        
        Args:
            player_id: ID del jugador
            
        Returns:
            Nombre del jugador o el ID si no se encuentra
        """
        player_id = str(player_id)
        return self.player_names.get(player_id, player_id)
    
    def get_player_rating(self, player_id: str, surface: Optional[str] = None, 
                         use_bayesian: bool = None) -> float:
        """
        Obtiene el rating ELO actual de un jugador con soporte bayesiano.
        
        Args:
            player_id: ID del jugador
            surface: Superficie específica (opcional)
            use_bayesian: Si debe usar enfoque bayesiano (anula configuración global)
            
        Returns:
            Rating ELO (general o específico por superficie) con ajuste bayesiano si corresponde
        """
        player_id = str(player_id)
        use_bayes = self.use_bayesian_approach if use_bayesian is None else use_bayesian
        
        # Caso básico: ELO por superficie específica
        if surface:
            surface = self._normalize_surface(surface)
            raw_rating = self.player_ratings_by_surface[surface].get(player_id, self.initial_rating)
            
            # Si no usamos enfoque bayesiano o jugador tiene suficientes partidos, devolver rating directo
            matches_in_surface = self.player_match_count_by_surface[surface].get(player_id, 0)
            if not use_bayes or matches_in_surface >= self.min_matches_full_rating:
                return raw_rating
            
            # Enfoque bayesiano: combinar con rating general
            general_rating = self.player_ratings.get(player_id, self.initial_rating)
            total_matches = self.player_match_count.get(player_id, 0)
            
            # Factor de confianza basado en número de partidos
            confidence = min(1.0, matches_in_surface / self.min_matches_full_rating)
            
            # Calcular rating combinado
            combined_rating = confidence * raw_rating + (1 - confidence) * general_rating
            
            # Transferencia entre superficies si hay muy pocos partidos en esta superficie
            if matches_in_surface < 3 and total_matches > 5:
                # Buscar ratings en otras superficies
                surface_ratings = {}
                for other_surface, ratings in self.player_ratings_by_surface.items():
                    if other_surface != surface and player_id in ratings:
                        other_matches = self.player_match_count_by_surface[other_surface].get(player_id, 0)
                        if other_matches > 3:
                            # Ponderar según la matriz de transferencia
                            transfer_weight = self.surface_transfer_matrix[surface][other_surface]
                            surface_ratings[other_surface] = ratings[player_id] * transfer_weight
                
                if surface_ratings:
                    # Calcular rating promedio ponderado de otras superficies
                    transferred_rating = sum(surface_ratings.values()) / len(surface_ratings)
                    
                    # Calcular peso para el rating transferido (inversamente proporcional a matches_in_surface)
                    transfer_weight = max(0, 1 - (matches_in_surface / 3))
                    
                    # Combinar con el rating actual
                    combined_rating = (1 - transfer_weight) * combined_rating + transfer_weight * transferred_rating
            
            return combined_rating
        
        # Caso: ELO general
        raw_rating = self.player_ratings.get(player_id, self.initial_rating)
        
        # Si no usamos enfoque bayesiano o el jugador tiene suficientes partidos, devolver rating directo
        matches_played = self.player_match_count.get(player_id, 0)
        if not use_bayes or matches_played >= self.min_matches_full_rating:
            return raw_rating
        
        # Enfoque bayesiano para jugadores con pocos partidos
        # Regularizar hacia la media global
        global_mean = sum(self.player_ratings.values()) / max(1, len(self.player_ratings))
        confidence = min(1.0, matches_played / self.min_matches_full_rating)
        
        # Combinar con la media según nivel de confianza
        return confidence * raw_rating + (1 - confidence) * global_mean
    
    def get_player_uncertainty(self, player_id: str, surface: Optional[str] = None) -> float:
        """
        Obtiene la incertidumbre (desviación estándar) del rating de un jugador.
        
        Args:
            player_id: ID del jugador
            surface: Superficie específica (opcional)
            
        Returns:
            Valor de incertidumbre del rating
        """
        player_id = str(player_id)
        
        # Si hay poca información, alta incertidumbre
        if surface:
            surface = self._normalize_surface(surface)
            matches = self.player_match_count_by_surface[surface].get(player_id, 0)
        else:
            matches = self.player_match_count.get(player_id, 0)
        
        # Incertidumbre base - inversamente proporcional a partidos jugados
        # Comienza alta (~150) y disminuye con cada partido
        base_uncertainty = 350 / (matches + 5)
        
        # Si tenemos un valor específico de incertidumbre, usarlo
        if player_id in self.player_rating_uncertainty:
            return self.player_rating_uncertainty.get(player_id, base_uncertainty)
        
        return base_uncertainty
    
    def get_combined_surface_rating(self, player_id: str, surface: str) -> float:
        """
        Obtiene un rating combinado por superficie que integra información de
        todas las superficies ponderadas según la matriz de transferencia.
        Versión corregida con validaciones de tipo.
        
        Args:
            player_id: ID del jugador
            surface: Superficie para la que calcular el rating
            
        Returns:
            Rating ELO combinado para la superficie específica
        """
        # Validar tipos
        player_id = str(player_id) if not pd.isna(player_id) else ''
        if not player_id:
            return self.initial_rating
        
        if not isinstance(surface, str):
            surface = str(surface) if not pd.isna(surface) else 'hard'
        surface = self._normalize_surface(surface)
        
        try:
            # Rating específico de la superficie
            specific_rating = self.player_ratings_by_surface[surface].get(player_id, self.initial_rating)
            if not isinstance(specific_rating, (int, float)) or pd.isna(specific_rating):
                specific_rating = self.initial_rating
                
            specific_matches = self.player_match_count_by_surface[surface].get(player_id, 0)
            if not isinstance(specific_matches, (int, float)) or pd.isna(specific_matches):
                specific_matches = 0
            
            # Si tiene suficientes partidos en esta superficie, usar directamente
            if specific_matches >= self.min_matches_full_rating:
                return specific_rating
            
            # Calcular rating combinado usando la matriz de transferencia
            ratings_sum = specific_rating * specific_matches
            weights_sum = specific_matches
            
            # Combinar con ratings de otras superficies
            for other_surface, transfer_weight in self.surface_transfer_matrix[surface].items():
                if other_surface != surface:
                    # Validar valores
                    if not isinstance(transfer_weight, (int, float)) or pd.isna(transfer_weight):
                        continue
                        
                    other_rating = self.player_ratings_by_surface[other_surface].get(player_id, self.initial_rating)
                    if not isinstance(other_rating, (int, float)) or pd.isna(other_rating):
                        other_rating = self.initial_rating
                        
                    other_matches = self.player_match_count_by_surface[other_surface].get(player_id, 0)
                    if not isinstance(other_matches, (int, float)) or pd.isna(other_matches):
                        other_matches = 0
                    
                    # Ponderar según transferencia y número de partidos
                    if other_matches > 0:
                        effective_weight = transfer_weight * other_matches
                        ratings_sum += other_rating * effective_weight
                        weights_sum += effective_weight
            
            # Rating general como fallback
            general_rating = self.player_ratings.get(player_id, self.initial_rating)
            if not isinstance(general_rating, (int, float)) or pd.isna(general_rating):
                general_rating = self.initial_rating
                
            general_matches = self.player_match_count.get(player_id, 0)
            if not isinstance(general_matches, (int, float)) or pd.isna(general_matches):
                general_matches = 0
                
            effective_general_matches = general_matches - specific_matches
            if effective_general_matches > 0:
                ratings_sum += general_rating * (effective_general_matches * 0.5)  # Peso reducido
                weights_sum += (effective_general_matches * 0.5)
            
            # Evitar división por cero
            if weights_sum == 0:
                return self.initial_rating
            
            return ratings_sum / weights_sum
        except Exception as e:
            logger.debug(f"Error calculando rating combinado por superficie: {str(e)}")
            return self.initial_rating  # Valor por defecto en caso de error
    
    def get_player_form(self, player_id: str, surface: Optional[str] = None) -> float:
        """
        Obtiene el factor de forma reciente de un jugador (entre 0.8 y 1.2).
        Versión corregida con validaciones de tipo.
        
        Args:
            player_id: ID del jugador
            surface: Superficie específica (opcional)
            
        Returns:
            Factor de forma reciente
        """
        # Validar tipos
        player_id = str(player_id) if not pd.isna(player_id) else ''
        if not player_id:
            return 1.0  # Valor neutral para ID inválido
        
        try:
            # Si no tenemos datos de forma, devolver valor neutral
            if player_id not in self.player_recent_form:
                return 1.0
            
            # Validar la estructura del diccionario
            form_data = self.player_recent_form[player_id]
            
            if not isinstance(form_data, dict):
                return 1.0
            
            if 'form' not in form_data:
                return 1.0
            
            # Obtener forma general
            general_form = form_data.get('form', 1.0)
            if not isinstance(general_form, (int, float)) or pd.isna(general_form) or general_form <= 0:
                general_form = 1.0
            
            # Si no se especifica superficie, devolver forma general
            if not surface:
                return general_form
            
            # Normalizar superficie
            if not isinstance(surface, str):
                surface = str(surface) if not pd.isna(surface) else 'hard'
            surface = self._normalize_surface(surface)
            
            # Si tenemos forma específica por superficie, combinar
            surface_key = f"{player_id}_{surface}"
            surface_form_dict = self.player_recent_form.get(surface_key, {})
            
            if not surface_form_dict or not isinstance(surface_form_dict, dict):
                return general_form
            
            # Extraer datos específicos por superficie con validación
            surface_form = surface_form_dict.get('form', 1.0)
            if not isinstance(surface_form, (int, float)) or pd.isna(surface_form) or surface_form <= 0:
                surface_form = 1.0
                
            surface_matches = surface_form_dict.get('matches', 0)
            if not isinstance(surface_matches, (int, float)) or pd.isna(surface_matches):
                surface_matches = 0
            
            # Ponderar según número de partidos
            if surface_matches >= 5:
                return surface_form
            elif surface_matches == 0:
                return general_form
            else:
                # Ponderación progresiva
                weight = surface_matches / 5
                return (weight * surface_form) + ((1 - weight) * general_form)
        except Exception as e:
            logger.debug(f"Error calculando forma del jugador: {str(e)}")
            return 1.0  # Valor neutral en caso de error
    
    def get_h2h_advantage(self, player1_id: str, player2_id: str) -> float:
        """
        Calcula el factor de ventaja basado en el historial head-to-head.
        
        Args:
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            
        Returns:
            Factor de ventaja (0.9-1.1) basado en historial h2h
        """
        player1_id = str(player1_id)
        player2_id = str(player2_id)
        
        # Verificar que h2h_records es un defaultdict antes de acceder
        if not hasattr(self, 'h2h_records') or not self.h2h_records:
            return 1.0
        
        # Obtener historial h2h con validación de tipos
        try:
            p1_vs_p2 = self.h2h_records[player1_id][player2_id]
            p2_vs_p1 = self.h2h_records[player2_id][player1_id]
            
            # Verificar que los valores son diccionarios y tienen la estructura esperada
            if not isinstance(p1_vs_p2, dict) or 'wins' not in p1_vs_p2:
                return 1.0
            if not isinstance(p2_vs_p1, dict) or 'wins' not in p2_vs_p1:
                return 1.0
            
            p1_wins = p1_vs_p2['wins']
            p2_wins = p2_vs_p1['wins']
            
            # Validar que son números
            if not isinstance(p1_wins, (int, float)) or not isinstance(p2_wins, (int, float)):
                return 1.0
            
            total_matches = p1_wins + p2_wins
            
            # Si no hay historial, no hay ventaja
            if total_matches == 0:
                return 1.0
            
            # Calcular ratio de victorias
            if total_matches >= 2:
                p1_win_ratio = p1_wins / total_matches
                
                # Mapear a un rango de 0.9 a 1.1 (mayores valores favorecen a p1)
                # Con más partidos, la ventaja puede ser mayor
                max_factor = min(0.1, 0.05 + (total_matches * 0.005))
                h2h_factor = 1.0 + ((p1_win_ratio - 0.5) * 2 * max_factor)
                
                return h2h_factor
            elif total_matches == 1:
                # Para un solo partido, efecto menor
                return 1.05 if p1_wins > 0 else 0.95
        except Exception as e:
            # En caso de error, devolver valor neutral
            logger.debug(f"Error en get_h2h_advantage: {str(e)}")
            return 1.0
        
        return 1.0
    
    def update_player_form(self, player_id: str, result: int, surface: Optional[str] = None) -> None:
        """
        Actualiza el factor de forma reciente de un jugador.
        Versión corregida con validaciones de tipo.
        
        Args:
            player_id: ID del jugador
            result: Resultado (1 para victoria, 0 para derrota)
            surface: Superficie del partido (opcional)
        """
        # Validar tipos
        player_id = str(player_id) if not pd.isna(player_id) else ''
        if not player_id:
            return  # No actualizar forma para ID inválido
            
        if not isinstance(result, (int, float)) or pd.isna(result):
            result = 0  # Valor por defecto
        else:
            # Normalizar a 0 o 1
            result = 1 if result > 0 else 0
        
        # Actualizar forma general
        try:
            if player_id not in self.player_recent_form:
                self.player_recent_form[player_id] = {
                    'results': [],
                    'form': 1.0,
                    'matches': 0
                }
            
            # Validar la estructura del diccionario
            form_data = self.player_recent_form[player_id]
            
            if not isinstance(form_data, dict):
                form_data = {'results': [], 'form': 1.0, 'matches': 0}
                self.player_recent_form[player_id] = form_data
            
            # Asegurar que tenemos la estructura de datos correcta
            if 'results' not in form_data:
                form_data['results'] = []
            if 'form' not in form_data:
                form_data['form'] = 1.0
            if 'matches' not in form_data:
                form_data['matches'] = 0
            
            # Añadir nuevo resultado y mantener ventana móvil
            form_data['results'].append(result)
            if len(form_data['results']) > self.form_window_matches:
                form_data['results'].pop(0)
            
            # Recalcular factor de forma con ponderación reciente
            if form_data['results']:
                # Los partidos más recientes tienen más peso
                weighted_sum = 0
                weights_sum = 0
                
                for i, res in enumerate(form_data['results']):
                    # Peso exponencial: partidos más recientes cuentan más
                    weight = 1.5 ** i  # i=0 es el más antiguo
                    weighted_sum += res * weight
                    weights_sum += weight
                
                # Evitar división por cero
                weighted_avg = weighted_sum / weights_sum if weights_sum > 0 else 0.5
                
                # Mapear a un rango de 0.8 a 1.2
                form_factor = 0.8 + (weighted_avg * 0.4)
                form_data['form'] = form_factor
                form_data['matches'] = len(form_data['results'])
        except Exception as e:
            logger.debug(f"Error actualizando forma general: {str(e)}")
        
        # Si se especifica superficie, actualizar forma específica
        if surface:
            try:
                surface = self._normalize_surface(surface)
                surface_key = f"{player_id}_{surface}"
                
                if surface_key not in self.player_recent_form:
                    self.player_recent_form[surface_key] = {
                        'results': [],
                        'form': 1.0,
                        'matches': 0
                    }
                
                # Validar la estructura del diccionario
                surface_form = self.player_recent_form[surface_key]
                
                if not isinstance(surface_form, dict):
                    surface_form = {'results': [], 'form': 1.0, 'matches': 0}
                    self.player_recent_form[surface_key] = surface_form
                
                # Asegurar que tenemos la estructura de datos correcta
                if 'results' not in surface_form:
                    surface_form['results'] = []
                if 'form' not in surface_form:
                    surface_form['form'] = 1.0
                if 'matches' not in surface_form:
                    surface_form['matches'] = 0
                
                # Actualizar de manera similar a la forma general
                surface_form['results'].append(result)
                if len(surface_form['results']) > self.form_window_matches:
                    surface_form['results'].pop(0)
                
                # Recalcular con los mismos pesos
                if surface_form['results']:
                    weighted_sum = 0
                    weights_sum = 0
                    
                    for i, res in enumerate(surface_form['results']):
                        weight = 1.5 ** i
                        weighted_sum += res * weight
                        weights_sum += weight
                    
                    # Evitar división por cero
                    weighted_avg = weighted_sum / weights_sum if weights_sum > 0 else 0.5
                    surface_form['form'] = 0.8 + (weighted_avg * 0.4)
                    surface_form['matches'] = len(surface_form['results'])
            except Exception as e:
                logger.debug(f"Error actualizando forma por superficie: {str(e)}")
    
    def update_h2h_record(self, winner_id: str, loser_id: str) -> None:
        """
        Actualiza el registro head-to-head entre dos jugadores.
        Versión corregida con validaciones de tipo.
        
        Args:
            winner_id: ID del jugador ganador
            loser_id: ID del jugador perdedor
        """
        # Validar tipos
        winner_id = str(winner_id) if not pd.isna(winner_id) else ''
        loser_id = str(loser_id) if not pd.isna(loser_id) else ''
        
        # Verificar IDs válidos
        if not winner_id or not loser_id or winner_id == loser_id:
            return  # No actualizar para IDs inválidos
        
        try:
            # Asegurar que tenemos la estructura correcta para el ganador
            if winner_id not in self.h2h_records:
                self.h2h_records[winner_id] = {}
                
            if loser_id not in self.h2h_records[winner_id]:
                self.h2h_records[winner_id][loser_id] = {'wins': 0, 'losses': 0}
            elif not isinstance(self.h2h_records[winner_id][loser_id], dict):
                # Corregir si no es un diccionario
                self.h2h_records[winner_id][loser_id] = {'wins': 0, 'losses': 0}
            elif 'wins' not in self.h2h_records[winner_id][loser_id]:
                self.h2h_records[winner_id][loser_id]['wins'] = 0
            
            # Asegurar que tenemos la estructura correcta para el perdedor
            if loser_id not in self.h2h_records:
                self.h2h_records[loser_id] = {}
                
            if winner_id not in self.h2h_records[loser_id]:
                self.h2h_records[loser_id][winner_id] = {'wins': 0, 'losses': 0}
            elif not isinstance(self.h2h_records[loser_id][winner_id], dict):
                # Corregir si no es un diccionario
                self.h2h_records[loser_id][winner_id] = {'wins': 0, 'losses': 0}
            elif 'losses' not in self.h2h_records[loser_id][winner_id]:
                self.h2h_records[loser_id][winner_id]['losses'] = 0
            
            # Actualizar registro del ganador contra el perdedor
            current_wins = self.h2h_records[winner_id][loser_id]['wins']
            if not isinstance(current_wins, (int, float)):
                current_wins = 0
            self.h2h_records[winner_id][loser_id]['wins'] = current_wins + 1
            
            # Actualizar registro del perdedor contra el ganador
            current_losses = self.h2h_records[loser_id][winner_id]['losses']
            if not isinstance(current_losses, (int, float)):
                current_losses = 0
            self.h2h_records[loser_id][winner_id]['losses'] = current_losses + 1
            
        except Exception as e:
            logger.debug(f"Error actualizando registro head-to-head: {str(e)}")
        
    def calculate_expected_win_probability(self, rating_a: float, rating_b: float, 
                                      uncertainty_a: float = 0, uncertainty_b: float = 0) -> float:
        """
        Calcula la probabilidad esperada de victoria considerando incertidumbre.
        Versión corregida con validaciones de tipo.
        
        Args:
            rating_a: Rating ELO del jugador A
            rating_b: Rating ELO del jugador B
            uncertainty_a: Incertidumbre del rating de A
            uncertainty_b: Incertidumbre del rating de B
            
        Returns:
            Probabilidad de que el jugador A gane (0-1)
        """
        # Validar tipos
        try:
            # Asegurar que los ratings son números
            if not isinstance(rating_a, (int, float)):
                rating_a = float(rating_a) if not pd.isna(rating_a) else self.initial_rating
            if not isinstance(rating_b, (int, float)):
                rating_b = float(rating_b) if not pd.isna(rating_b) else self.initial_rating
                
            # Asegurar que las incertidumbres son números no negativos
            if not isinstance(uncertainty_a, (int, float)) or uncertainty_a < 0:
                uncertainty_a = 0
            if not isinstance(uncertainty_b, (int, float)) or uncertainty_b < 0:
                uncertainty_b = 0
                
            # Considerar incertidumbre para regularizar las probabilidades
            # Mayor incertidumbre lleva la probabilidad hacia 0.5
            uncertainty_factor = (uncertainty_a + uncertainty_b) / 2
            
            # Fórmula ELO tradicional
            base_probability = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
            
            # Suavizar según incertidumbre
            if uncertainty_factor > 0:
                # El factor máximo de incertidumbre (300) suaviza completamente hacia 0.5
                max_uncertainty = 300
                regularization = min(1.0, uncertainty_factor / max_uncertainty)
                
                # Interpolación lineal hacia 0.5
                return base_probability * (1 - regularization) + 0.5 * regularization
            
            return base_probability
        
        except Exception as e:
            # En caso de error, devolver probabilidad neutral
            logger.warning(f"Error en calculate_expected_win_probability: {str(e)}")
            return 0.5
    
    def _get_dynamic_k_factor(self, player_id: str, tourney_level: str, round_name: str, 
                         surface: str, match_importance: float = 1.0) -> float:
        """
        Calcula un factor K dinámico basado en múltiples factores contextuales.
        Versión corregida con validaciones de tipo.
        
        Args:
            player_id: ID del jugador
            tourney_level: Nivel del torneo (G, M, A, D, etc.)
            round_name: Ronda del torneo (F, SF, QF, etc.)
            surface: Superficie de juego
            match_importance: Importancia adicional del partido (1.0 por defecto)
            
        Returns:
            Factor K ajustado
        """
        # Validar tipos
        player_id = str(player_id) if not pd.isna(player_id) else ''
        
        if not isinstance(tourney_level, str):
            tourney_level = str(tourney_level) if not pd.isna(tourney_level) else 'O'
        
        if not isinstance(round_name, str):
            round_name = str(round_name) if not pd.isna(round_name) else 'R32'
        
        if not isinstance(surface, str):
            surface = str(surface) if not pd.isna(surface) else 'hard'
        
        if not isinstance(match_importance, (int, float)) or pd.isna(match_importance) or match_importance <= 0:
            match_importance = 1.0
        
        try:
            # Factor base
            k_factor = self.k_factor_base
            
            # Normalizar valores
            tourney_level = self._normalize_tournament_level(tourney_level)
            surface = self._normalize_surface(surface)
            
            # Ajuste por tipo de torneo
            tourney_multiplier = self.tournament_k_factors.get(tourney_level, 1.0)
            if not isinstance(tourney_multiplier, (int, float)) or pd.isna(tourney_multiplier) or tourney_multiplier <= 0:
                tourney_multiplier = 1.0
            k_factor *= tourney_multiplier
            
            # Ajuste por ronda
            round_multiplier = self.round_multipliers.get(round_name, 1.0)
            if not isinstance(round_multiplier, (int, float)) or pd.isna(round_multiplier) or round_multiplier <= 0:
                round_multiplier = 1.0
            k_factor *= round_multiplier
            
            # Ajuste por superficie específica (mayor impacto en superficies más especializadas)
            surface_multiplier = self.surface_specificity.get(surface, 1.0)
            if not isinstance(surface_multiplier, (int, float)) or pd.isna(surface_multiplier) or surface_multiplier <= 0:
                surface_multiplier = 1.0
            k_factor *= surface_multiplier
            
            # Ajuste por experiencia (menor K para jugadores con más experiencia)
            match_count = self.player_match_count.get(player_id, 0)
            if match_count > 0:
                # Comienza en 1.2 para novatos y baja hasta 0.8 para veteranos
                experience_multiplier = 1.2 - min(0.4, (match_count / 500))
                k_factor *= experience_multiplier
            
            # Ajuste por incertidumbre (mayor K para jugadores con mayor incertidumbre)
            uncertainty = self.get_player_uncertainty(player_id, surface)
            if isinstance(uncertainty, (int, float)) and not pd.isna(uncertainty) and uncertainty > 0:
                uncertainty_multiplier = min(1.5, 1.0 + (uncertainty / 300))
                k_factor *= uncertainty_multiplier
            
            # Ajuste por importancia del partido
            k_factor *= match_importance
            
            # Redondear a 2 decimales para estabilidad
            return round(k_factor, 2)
        except Exception as e:
            logger.debug(f"Error calculando factor K dinámico: {str(e)}")
            return self.k_factor_base  # Valor por defecto en caso de error
    
    def _parse_score(self, score: str) -> Tuple[int, int, int, bool, Dict]:
        """
        Analiza el score de un partido para determinar sets, games y dominancia.
        Versión corregida con validaciones de tipo más robustas.
        
        Args:
            score: String con el resultado (e.g. '6-4 7-5')
            
        Returns:
            Tupla de (sets_ganador, sets_perdedor, diferencia_games, fue_completo, stats)
            donde stats es un diccionario con estadísticas adicionales
        """
        # Validar tipo de score
        if not isinstance(score, str):
            score = str(score) if not pd.isna(score) else ''
        
        if pd.isna(score) or not score:
            return 2, 0, 4, False, {'avg_game_diff': 2.0, 'tiebreaks': 0, 'total_games': 4}
        
        try:
            # Limpiar y dividir el score
            score = str(score).strip()
            
            # Variables para análisis
            sets_won_winner = 0
            sets_won_loser = 0
            games_diff = 0
            total_games = 0
            tiebreaks = 0
            sets_analyzed = 0
            game_diff_values = []
            complete_match = True
            
            # Manejar casos especiales: abandono, walkover, etc.
            special_terms = ['ret', 'reti', 'def', 'w/o', 'walkover', 'default']
            if any(term in score.lower() for term in special_terms):
                complete_match = False
            
            # Dividir el score por sets - considerar que puede haber espacios o no
            # Manejar casos como "64 75" o "6-4 7-5" o "6:4,7:5"
            normalized_score = score.replace(':', '-').replace(',', ' ').replace(';', ' ')
            sets = normalized_score.split()
            
            # Filtrar solo los elementos que parecen sets válidos
            valid_sets = []
            for set_part in sets:
                # Si tiene un guión o se puede dividir en números, considerarlo un set potencial
                if '-' in set_part or (len(set_part) >= 2 and not any(term in set_part.lower() for term in special_terms)):
                    valid_sets.append(set_part)
            
            # Si no hay sets válidos después del filtrado, usar valores predeterminados
            if not valid_sets:
                return 2, 0, 4, False, {'avg_game_diff': 2.0, 'tiebreaks': 0, 'total_games': 4}
            
            for set_score in valid_sets:
                # Detectar tiebreaks
                has_tiebreak = '(' in set_score
                tb_score = None
                
                if has_tiebreak:
                    # Extraer puntaje de tiebreak
                    tb_part = re.search(r'\((.*?)\)', set_score)
                    if tb_part:
                        tb_score = tb_part.group(1)
                    
                    # Limpiar para procesar el set
                    set_score = set_score.split('(')[0]
                    tiebreaks += 1
                
                # Manejar diferentes formatos de sets
                if '-' in set_score:
                    # Formato común "6-4"
                    games = set_score.split('-')
                elif len(set_score) == 2:
                    # Formato compacto "64"
                    games = [set_score[0], set_score[1]]
                elif len(set_score) >= 3 and set_score[1].isdigit():
                    # Posible formato raro "61"+"75" pegados
                    continue  # Ignorar este caso por ahora
                else:
                    # Intentar extraer dos dígitos consecutivos
                    games_match = re.findall(r'\d+', set_score)
                    if len(games_match) >= 2:
                        games = games_match[:2]  # Tomar los dos primeros números
                    else:
                        continue  # Ignorar este set si no podemos parsearlo
                
                # Validar que tenemos dos puntuaciones
                if len(games) == 2:
                    try:
                        # Convertir a enteros y validar
                        winner_games = int(games[0])
                        loser_games = int(games[1])
                        
                        # Validar que son puntuaciones razonables de tenis
                        if 0 <= winner_games <= 50 and 0 <= loser_games <= 50:
                            game_diff = winner_games - loser_games
                            games_diff += game_diff
                            total_games += winner_games + loser_games
                            game_diff_values.append(game_diff)
                            sets_analyzed += 1
                            
                            if winner_games > loser_games:
                                sets_won_winner += 1
                            elif loser_games > winner_games:
                                sets_won_loser += 1
                    except (ValueError, TypeError):
                        # Si no se puede convertir a entero, ignorar este set
                        continue
            
            # Si no se pudo analizar ningún set, usar valores predeterminados
            if sets_analyzed == 0:
                return 2, 0, 4, False, {'avg_game_diff': 2.0, 'tiebreaks': 0, 'total_games': 4}
            
            # Calcular estadísticas adicionales
            avg_game_diff = sum(game_diff_values) / max(1, len(game_diff_values))
            max_game_diff = max(game_diff_values) if game_diff_values else 0
            
            stats = {
                'avg_game_diff': avg_game_diff,
                'max_game_diff': max_game_diff,
                'tiebreaks': tiebreaks,
                'total_games': total_games,
                'complete': complete_match
            }
            
            # En caso de no poder analizar correctamente, usar valores predeterminados
            if sets_won_winner == 0 and sets_won_loser == 0:
                return 2, 0, 4, False, stats
                
            return sets_won_winner, sets_won_loser, games_diff, complete_match, stats
            
        except Exception as e:
            logger.debug(f"Error analizando score '{score}': {str(e)}")
            return 2, 0, 4, False, {'avg_game_diff': 2.0, 'tiebreaks': 0, 'total_games': 4}
    
    def _get_match_importance_factor(self, tourney_level: str, round_name: str, 
                                player1_id: str, player2_id: str) -> float:
        """
        Calcula un factor de importancia del partido basado en contexto.
        
        Args:
            tourney_level: Nivel del torneo
            round_name: Ronda del partido
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            
        Returns:
            Factor de importancia (0.8-1.5)
        """
        # Normalizar con validación de tipos
        if not isinstance(tourney_level, str):
            tourney_level = str(tourney_level) if not pd.isna(tourney_level) else 'O'
        tourney_level = self._normalize_tournament_level(tourney_level)
        
        # Base por nivel de torneo
        base_importance = {
            'G': 1.3,  # Grand Slam
            'M': 1.2,  # Masters
            'F': 1.25, # Finals
            'A': 1.1,  # ATP 500
            'D': 1.0,  # ATP 250
            'C': 0.9,  # Challenger
            'S': 0.8,  # Satélite
            'O': 0.8   # Otros
        }.get(tourney_level, 1.0)
        
        # Validar round_name
        if not isinstance(round_name, str):
            round_name = str(round_name) if not pd.isna(round_name) else 'R32'
        
        # Ajuste por ronda
        round_factor = {
            'F': 1.2,    # Final
            'SF': 1.1,   # Semifinal
            'QF': 1.05,  # Cuartos
            'R16': 1.0,  # Octavos
            'R32': 0.95, # 1/16
            'R64': 0.9,  # 1/32
            'R128': 0.85 # 1/64
        }.get(round_name, 1.0)
        
        # Validar player IDs
        player1_id = str(player1_id) if not pd.isna(player1_id) else ''
        player2_id = str(player2_id) if not pd.isna(player2_id) else ''
        
        if not player1_id or not player2_id:
            return base_importance * round_factor
        
        try:
            # Ajuste por ranking relativo con validación
            p1_rating = self.get_player_rating(player1_id)
            p2_rating = self.get_player_rating(player2_id)
            
            # Asegurar que son números
            if not isinstance(p1_rating, (int, float)) or not isinstance(p2_rating, (int, float)):
                return base_importance * round_factor
            
            rating_diff = abs(p1_rating - p2_rating)
            
            # Partidos muy igualados son más importantes
            ranking_factor = 1.0
            if rating_diff < 50:
                ranking_factor = 1.1  # Muy igualado
            elif rating_diff < 100:
                ranking_factor = 1.05  # Bastante igualado
            elif rating_diff > 300:
                ranking_factor = 0.9  # Muy desigualado
            
            # Combinar factores
            return base_importance * round_factor * ranking_factor
        except Exception as e:
            # En caso de error, devolver valor base
            logger.debug(f"Error en _get_match_importance_factor: {str(e)}")
            return base_importance * round_factor
    
    def _get_match_stats_factor(self, match_id: Optional[str] = None, 
                          winner_id: Optional[str] = None, 
                          loser_id: Optional[str] = None,
                          w_stats: Optional[Dict] = None,
                          l_stats: Optional[Dict] = None) -> float:
        """
        Calcula un factor basado en estadísticas del partido para ajustar el cambio de ELO.
        Versión corregida con validaciones de tipo.
        
        Args:
            match_id: ID del partido (opcional)
            winner_id: ID del ganador
            loser_id: ID del perdedor
            w_stats: Estadísticas del ganador (opcional)
            l_stats: Estadísticas del perdedor (opcional)
            
        Returns:
            Factor de ajuste (0.8-1.2) basado en estadísticas
        """
        # Validar tipos de entrada
        if match_id is not None and not isinstance(match_id, str):
            match_id = str(match_id) if not pd.isna(match_id) else None
            
        if winner_id is not None and not isinstance(winner_id, str):
            winner_id = str(winner_id) if not pd.isna(winner_id) else None
            
        if loser_id is not None and not isinstance(loser_id, str):
            loser_id = str(loser_id) if not pd.isna(loser_id) else None
        
        # Validar diccionarios de estadísticas
        if w_stats is not None and not isinstance(w_stats, dict):
            w_stats = None
            
        if l_stats is not None and not isinstance(l_stats, dict):
            l_stats = None
        
        try:
            # Si tenemos estadísticas proporcionadas directamente, usarlas
            if w_stats and l_stats:
                # Variables para análisis
                dominance_score = 0
                stats_count = 0
                
                # Estadísticas de servicio - aces
                if 'ace' in w_stats and 'ace' in l_stats:
                    w_ace = w_stats['ace']
                    l_ace = l_stats['ace']
                    
                    # Validar tipos
                    if not isinstance(w_ace, (int, float)) or pd.isna(w_ace):
                        w_ace = 0
                    if not isinstance(l_ace, (int, float)) or pd.isna(l_ace):
                        l_ace = 0
                    
                    if w_ace > 0 or l_ace > 0:
                        ace_ratio = w_ace / max(1, w_ace + l_ace)
                        # Normalizar a escala -1 a 1 (0.5 es neutral)
                        dominance_score += (ace_ratio - 0.5) * 2
                        stats_count += 1
                
                # Puntos de break
                if all(key in w_stats for key in ['bpSaved', 'bpFaced']) and all(key in l_stats for key in ['bpSaved', 'bpFaced']):
                    # Break points ganados por el ganador
                    w_bp_faced = w_stats['bpFaced']
                    w_bp_saved = w_stats['bpSaved']
                    l_bp_faced = l_stats['bpFaced']
                    l_bp_saved = l_stats['bpSaved']
                    
                    # Validar tipos
                    if not isinstance(w_bp_faced, (int, float)) or pd.isna(w_bp_faced):
                        w_bp_faced = 0
                    if not isinstance(w_bp_saved, (int, float)) or pd.isna(w_bp_saved):
                        w_bp_saved = 0
                    if not isinstance(l_bp_faced, (int, float)) or pd.isna(l_bp_faced):
                        l_bp_faced = 0
                    if not isinstance(l_bp_saved, (int, float)) or pd.isna(l_bp_saved):
                        l_bp_saved = 0
                    
                    if w_bp_faced > 0:
                        w_bp_saved_pct = w_bp_saved / w_bp_faced
                    else:
                        w_bp_saved_pct = 1.0
                    
                    # Break points ganados por el perdedor
                    if l_bp_faced > 0:
                        l_bp_saved_pct = l_bp_saved / l_bp_faced
                    else:
                        l_bp_saved_pct = 1.0
                    
                    # Normalizar a escala -1 a 1
                    dominance_score += (w_bp_saved_pct - l_bp_saved_pct)
                    stats_count += 1
                
                # Eficiencia en primer servicio
                if all(key in w_stats for key in ['1stIn', 'svpt']) and all(key in l_stats for key in ['1stIn', 'svpt']):
                    w_1st_in = w_stats['1stIn']
                    w_svpt = w_stats['svpt']
                    l_1st_in = l_stats['1stIn']
                    l_svpt = l_stats['svpt']
                    
                    # Validar tipos
                    if not isinstance(w_1st_in, (int, float)) or pd.isna(w_1st_in):
                        w_1st_in = 0
                    if not isinstance(w_svpt, (int, float)) or pd.isna(w_svpt):
                        w_svpt = 0
                    if not isinstance(l_1st_in, (int, float)) or pd.isna(l_1st_in):
                        l_1st_in = 0
                    if not isinstance(l_svpt, (int, float)) or pd.isna(l_svpt):
                        l_svpt = 0
                    
                    if w_svpt > 0 and l_svpt > 0:
                        w_1st_pct = w_1st_in / w_svpt
                        l_1st_pct = l_1st_in / l_svpt
                        
                        # Normalizar
                        dominance_score += (w_1st_pct - l_1st_pct) * 2
                        stats_count += 1
                
                # Si tenemos suficientes estadísticas, calcular factor final
                if stats_count >= 2:
                    avg_dominance = dominance_score / stats_count
                    
                    # Mapear a rango 0.8-1.2
                    # Mayor dominancia estadística = mayor ajuste de ELO
                    stats_factor = 1.0 + (avg_dominance * 0.2)
                    
                    # Limitar el rango
                    return max(0.8, min(1.2, stats_factor))
            
            # Si no hemos devuelto nada hasta aquí, intentar buscar en datos almacenados
            elif (not self.match_stats_df.empty) and (match_id is not None or (winner_id is not None and loser_id is not None)):
                # Intentar buscar estadísticas por match_id o combinación de jugadores
                if match_id and 'match_id' in self.match_stats_df.columns:
                    # Convertir match_id a mismo tipo que en DataFrame
                    if isinstance(self.match_stats_df['match_id'].iloc[0], str):
                        match_id = str(match_id)
                    
                    match_stats = self.match_stats_df[self.match_stats_df['match_id'] == match_id]
                    if not match_stats.empty:
                        # Extraer estadísticas relevantes y recursivamente llamar esta función
                        # Implementar según estructura específica de datos
                        pass  # Implementar si es necesario
                
                # Si no encontramos por match_id, intentar por combinación de jugadores
                if winner_id and loser_id:
                    # Asegurar que IDs son strings para comparar
                    winner_id_str = str(winner_id)
                    loser_id_str = str(loser_id)
                    
                    potential_matches = self.match_stats_df[
                        (self.match_stats_df['winner_id'].astype(str) == winner_id_str) & 
                        (self.match_stats_df['loser_id'].astype(str) == loser_id_str)
                    ]
                    
                    if not potential_matches.empty:
                        # Usar el partido más reciente si hay varios
                        if 'match_date' in potential_matches.columns:
                            match_stats = potential_matches.sort_values('match_date', ascending=False).iloc[0]
                        else:
                            match_stats = potential_matches.iloc[0]
                        
                        # Extraer estadísticas relevantes
                        extracted_w_stats = {}
                        extracted_l_stats = {}
                        
                        # Mapear columnas comunes en los datos de Jeff Sackmann
                        stat_cols = [
                            'ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 
                            'SvGms', 'bpSaved', 'bpFaced'
                        ]
                        
                        for col in stat_cols:
                            w_col = f'w_{col}'
                            l_col = f'l_{col}'
                            
                            if w_col in match_stats:
                                extracted_w_stats[col] = match_stats[w_col]
                            if l_col in match_stats:
                                extracted_l_stats[col] = match_stats[l_col]
                        
                        # Llamar recursivamente con las estadísticas extraídas
                        if extracted_w_stats and extracted_l_stats:
                            return self._get_match_stats_factor(
                                None, winner_id, loser_id, extracted_w_stats, extracted_l_stats
                            )
            
            # Si no hay suficientes datos, valor neutral
            return 1.0
        except Exception as e:
            logger.debug(f"Error calculando factor de estadísticas: {str(e)}")
            return 1.0  # Valor neutral en caso de error
    
    def _get_margin_multiplier(self, score: str, retirement: bool = False) -> float:
        """
        Calcula multiplicador basado en el margen de victoria.
        
        Args:
            score: String con el resultado (e.g. '6-4 7-5')
            retirement: Indica si hubo abandono
            
        Returns:
            Multiplicador para el cambio de ELO (0.8-1.2)
        """
        # Validar tipo de retirement
        if not isinstance(retirement, bool):
            retirement = bool(retirement) if not pd.isna(retirement) else False
        
        if retirement:
            return 0.8  # Menor impacto para partidos con abandono
        
        # Validar score
        if not isinstance(score, str):
            score = str(score) if not pd.isna(score) else ''
        
        try:
            # Analizar el score con función mejorada
            sets_won_winner, sets_won_loser, games_diff, complete_match, stats = self._parse_score(score)
            
            # Validar tipos
            if not isinstance(stats, dict):
                return 1.0  # Valor neutral si la estructura es incorrecta
            
            # Para partidos incompletos, reducir impacto
            if not complete_match:
                return 0.85
            
            # Calcular dominancia con validación
            if sets_won_winner + sets_won_loser > 0:
                sets_ratio = sets_won_winner / max(1, sets_won_winner + sets_won_loser)
            else:
                sets_ratio = 0.67  # Valor neutral-positivo
            
            # Verificar que stats tiene las claves esperadas
            total_games = stats.get('total_games', 0)
            if not isinstance(total_games, (int, float)) or total_games <= 0:
                total_games = 1
            
            # Normalizar diferencia de games con validación
            normalized_games_diff = games_diff / total_games * 2
            
            # Considerar tiebreaks (partidos más ajustados)
            tiebreaks = stats.get('tiebreaks', 0)
            if not isinstance(tiebreaks, (int, float)):
                tiebreaks = 0
            tiebreak_factor = max(0.9, 1.0 - (tiebreaks * 0.05))
            
            # Considerar diferencia promedio por set
            avg_diff = stats.get('avg_game_diff', 2.0)
            if not isinstance(avg_diff, (int, float)):
                avg_diff = 2.0
            avg_diff_factor = min(1.2, 1.0 + (avg_diff * 0.05))
            
            # Combinar factores de dominancia
            dominance = (
                0.5 * sets_ratio +
                0.3 * normalized_games_diff +
                0.2 * avg_diff_factor
            ) * tiebreak_factor
            
            # Ajustar a rango 0.8-1.2
            return 0.8 + min(0.4, dominance * 0.4)
                
        except Exception as e:
            logger.debug(f"Error calculando margen: {str(e)}")
            return 1.0  # Valor predeterminado en caso de error

    def _apply_temporal_decay(self, player_id: str, current_date: datetime) -> None:
        """
        Aplica decaimiento temporal al rating de un jugador inactivo.
        Implementa una curva de decaimiento exponencial que es
        más lenta al principio y se acelera con el tiempo.
        Versión corregida con validaciones de tipo.
        
        Args:
            player_id: ID del jugador
            current_date: Fecha actual
        """
        player_id = str(player_id)
        
        # Validar tipos
        if not isinstance(current_date, datetime):
            try:
                current_date = pd.to_datetime(current_date)
            except:
                # Si no se puede convertir, usar fecha actual
                current_date = datetime.now()
        
        # Obtener fecha último partido con validación
        last_match_date = self.player_last_match.get(player_id)
        
        # Verificar que last_match_date es una fecha válida
        if last_match_date is None:
            return  # No hay fecha anterior, no aplicar decaimiento
        
        if not isinstance(last_match_date, datetime):
            try:
                last_match_date = pd.to_datetime(last_match_date)
            except:
                # Si no es convertible, actualizar con fecha actual y salir
                self.player_last_match[player_id] = current_date
                return
        
        if current_date > last_match_date:
            try:
                # Calcular días de inactividad
                days_inactive = (current_date - last_match_date).days
                
                # Sólo aplicar decaimiento después de 30 días
                if days_inactive > 30:
                    # Calcular meses de inactividad
                    months_inactive = days_inactive / 30.0
                    
                    # Aplicar decaimiento exponencial con diferentes tasas
                    # según tiempo de inactividad
                    if months_inactive <= 3:
                        # Primera fase: decaimiento lento
                        decay_factor = self.decay_rate ** (months_inactive - 1)
                    elif months_inactive <= 6:
                        # Segunda fase: decaimiento medio
                        decay_factor = self.decay_rate ** (2 + (months_inactive - 3) * 1.5)
                    else:
                        # Tercera fase: decaimiento acelerado para inactividad prolongada
                        decay_factor = self.decay_rate ** (6.5 + (months_inactive - 6) * 2)
                    
                    # Limitar el decaimiento máximo (no bajar más del 40% del rating)
                    decay_factor = max(0.6, decay_factor)
                    
                    # Aplicar a rating general con validación
                    if player_id in self.player_ratings:
                        current_rating = self.player_ratings[player_id]
                        
                        # Verificar que el rating es un número
                        if not isinstance(current_rating, (int, float)):
                            current_rating = self.initial_rating
                            self.player_ratings[player_id] = current_rating
                        
                        # Calcular media global con validación
                        ratings_values = list(self.player_ratings.values())
                        valid_ratings = [r for r in ratings_values if isinstance(r, (int, float))]
                        
                        if valid_ratings:
                            global_mean = sum(valid_ratings) / len(valid_ratings)
                        else:
                            global_mean = self.initial_rating
                        
                        # Decaer hacia la media, no hacia 0
                        decay_target = global_mean * 0.9  # 90% de la media global
                        self.player_ratings[player_id] = current_rating * decay_factor + decay_target * (1 - decay_factor)
                    
                    # Aplicar a ratings por superficie con validación
                    for surface in self.player_ratings_by_surface:
                        if player_id in self.player_ratings_by_surface[surface]:
                            current_surface_rating = self.player_ratings_by_surface[surface][player_id]
                            
                            # Verificar que el rating es un número
                            if not isinstance(current_surface_rating, (int, float)):
                                current_surface_rating = self.initial_rating
                                self.player_ratings_by_surface[surface][player_id] = current_surface_rating
                            
                            # Calcular media de la superficie con validación
                            surface_ratings = list(self.player_ratings_by_surface[surface].values())
                            valid_surface_ratings = [r for r in surface_ratings if isinstance(r, (int, float))]
                            
                            if valid_surface_ratings:
                                surface_mean = sum(valid_surface_ratings) / len(valid_surface_ratings)
                                decay_target = surface_mean * 0.9
                            else:
                                decay_target = self.initial_rating
                            
                            # Aplicar decaimiento hacia la media de la superficie
                            self.player_ratings_by_surface[surface][player_id] = (
                                current_surface_rating * decay_factor + 
                                decay_target * (1 - decay_factor)
                            )
                    
                    # Aumentar incertidumbre con el tiempo
                    if player_id in self.player_rating_uncertainty:
                        current_uncertainty = self.player_rating_uncertainty[player_id]
                        
                        # Verificar que es un número
                        if not isinstance(current_uncertainty, (int, float)) or current_uncertainty < 0:
                            current_uncertainty = 100.0
                        
                        # Aumentar incertidumbre en función del tiempo
                        uncertainty_increase = min(100, months_inactive * 5)
                        self.player_rating_uncertainty[player_id] = current_uncertainty + uncertainty_increase
            
            except Exception as e:
                logger.warning(f"Error en temporal decay para {player_id}: {str(e)}")
                # No interrumpir el proceso si hay un error en un jugador específico
    
    def _calculate_victory_impact_factor(self, winner_rating: float, loser_rating: float, 
                                   expected_prob: float, margin_multiplier: float,
                                   match_importance: float) -> float:
        """
        Calcula un factor de impacto para victorias "inesperadas" o importantes.
        Recompensa más las victorias sorprendentes (bajo expected_prob) o
        las victorias contundentes contra rivales fuertes.
        Versión corregida con validaciones de tipo.
        
        Args:
            winner_rating: Rating del ganador
            loser_rating: Rating del perdedor
            expected_prob: Probabilidad esperada de victoria
            margin_multiplier: Multiplicador basado en margen de victoria
            match_importance: Importancia del partido
            
        Returns:
            Factor de impacto de la victoria (1.0-1.5)
        """
        try:
            # Validar tipos de entrada
            if not isinstance(winner_rating, (int, float)) or pd.isna(winner_rating):
                winner_rating = self.initial_rating
                
            if not isinstance(loser_rating, (int, float)) or pd.isna(loser_rating):
                loser_rating = self.initial_rating
                
            if not isinstance(expected_prob, (int, float)) or pd.isna(expected_prob) or expected_prob <= 0 or expected_prob >= 1:
                expected_prob = 0.5  # Valor neutral
                
            if not isinstance(margin_multiplier, (int, float)) or pd.isna(margin_multiplier) or margin_multiplier <= 0:
                margin_multiplier = 1.0  # Valor neutral
                
            if not isinstance(match_importance, (int, float)) or pd.isna(match_importance) or match_importance <= 0:
                match_importance = 1.0  # Valor neutral
            
            # Factores base
            upset_factor = 1.0
            solid_win_factor = 1.0
            rating_gap_factor = 1.0
            
            # 1. Upset factor - victorias inesperadas tienen mayor impacto
            if expected_prob < 0.5:
                # Más impacto cuanto menos esperada sea la victoria
                upset_factor = 1.0 + ((0.5 - expected_prob) * 2 * 0.3)
            
            # 2. Solid win factor - victorias contundentes tienen mayor impacto
            if margin_multiplier > 1.0:
                solid_win_factor = margin_multiplier
            
            # 3. Rating gap factor - vencer a alguien mucho mejor tiene mayor impacto
            rating_diff = loser_rating - winner_rating
            if rating_diff > 0:  # Ganador tenía peor rating
                # Normalizar diferencia de rating
                normalized_diff = min(1.0, rating_diff / 300)
                rating_gap_factor = 1.0 + (normalized_diff * 0.3)
            
            # Combinar todos los factores, pero limitar el impacto total
            combined_factor = 1.0 + (
                0.4 * (upset_factor - 1.0) +  # 40% peso para upsets
                0.3 * (solid_win_factor - 1.0) +  # 30% peso para victorias contundentes
                0.3 * (rating_gap_factor - 1.0)  # 30% peso para diferencia de rating
            ) * match_importance
            
            # Limitar a rango razonable
            return min(1.5, max(1.0, combined_factor))
        except Exception as e:
            logger.debug(f"Error calculando factor de impacto de victoria: {str(e)}")
            return 1.0  # Valor neutral en caso de error
    
    def _get_style_compatibility_factor(self, player1_id: str, player2_id: str, surface: str) -> float:
        """
        Calcula un factor de compatibilidad de estilos de juego entre dos jugadores.
        Algunos estilos tienen ventaja natural contra otros en ciertas superficies.
        
        Args:
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            surface: Superficie del partido
            
        Returns:
            Factor de compatibilidad (0.95-1.05, >1 favorece a player1)
        """
        # Este método requiere datos detallados del estilo de juego
        # Si no tenemos Match Charting Project o datos suficientes, devolvemos neutral
        if self.mcp_matches_df.empty:
            return 1.0
        
        # Por ahora, implementamos un modelo simplificado
        # Ventajas por superficie:
        surface_advantages = {
            # surface: {estilo_favorecido: {estilo_en_desventaja: factor}}
            'clay': {
                'defensive': {'serve_oriented': 1.05},
                'baseline': {'attacking': 1.03}
            },
            'grass': {
                'serve_oriented': {'defensive': 1.05, 'baseline': 1.03},
                'attacking': {'defensive': 1.04}
            },
            'hard': {
                'serve_oriented': {'defensive': 1.02},
                'attacking': {'baseline': 1.02}
            },
            'carpet': {
                'serve_oriented': {'defensive': 1.05},
                'attacking': {'baseline': 1.04}
            }
        }
        
        # No tenemos suficiente información para determinar estilos
        # Esto requeriría un análisis más detallado
        return 1.0
    
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
        Versión corregida con validaciones de tipo más robustas.
        
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
            
            # Aplicar decaimiento temporal a ambos jugadores con validación
            try:
                self._apply_temporal_decay(winner_id, match_date)
                self._apply_temporal_decay(loser_id, match_date)
            except Exception as e:
                logger.warning(f"Error en decaimiento temporal: {str(e)}")
            
            # Normalizar superficie con validación
            if not isinstance(surface, str):
                surface = str(surface) if not pd.isna(surface) else 'hard'
            surface = self._normalize_surface(surface)
            
            # Validar tourney_level y round_name
            if not isinstance(tourney_level, str):
                tourney_level = str(tourney_level) if not pd.isna(tourney_level) else 'O'
            tourney_level = self._normalize_tournament_level(tourney_level)
            
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
            
            # Obtener ratings actuales con enfoque bayesiano y validación de tipos
            w_general_elo = self.get_player_rating(winner_id)
            l_general_elo = self.get_player_rating(loser_id)
            
            # Asegurar que son números
            if not isinstance(w_general_elo, (int, float)):
                w_general_elo = self.initial_rating
            if not isinstance(l_general_elo, (int, float)):
                l_general_elo = self.initial_rating
            
            # Ratings específicos por superficie (con transferencia entre superficies)
            w_surface_elo = self.get_combined_surface_rating(winner_id, surface)
            l_surface_elo = self.get_combined_surface_rating(loser_id, surface)
            
            # Asegurar que son números
            if not isinstance(w_surface_elo, (int, float)):
                w_surface_elo = self.initial_rating
            if not isinstance(l_surface_elo, (int, float)):
                l_surface_elo = self.initial_rating
            
            # Obtener incertidumbres con validación
            w_uncertainty = self.get_player_uncertainty(winner_id, surface)
            l_uncertainty = self.get_player_uncertainty(loser_id, surface)
            
            if not isinstance(w_uncertainty, (int, float)):
                w_uncertainty = 100.0
            if not isinstance(l_uncertainty, (int, float)):
                l_uncertainty = 100.0
            
            # Calcular probabilidades esperadas con validación de tipos
            # 1. Probabilidad basada en ELO general
            w_win_prob_general = self.calculate_expected_win_probability(
                w_general_elo, l_general_elo, 
                w_uncertainty, l_uncertainty
            )
            
            # 2. Probabilidad basada en ELO específico por superficie
            w_win_prob_surface = self.calculate_expected_win_probability(
                w_surface_elo, l_surface_elo, 
                w_uncertainty, l_uncertainty
            )
            
            # Asegurar que las probabilidades son números válidos
            if not isinstance(w_win_prob_general, (int, float)) or not (0 <= w_win_prob_general <= 1):
                w_win_prob_general = 0.5
            if not isinstance(w_win_prob_surface, (int, float)) or not (0 <= w_win_prob_surface <= 1):
                w_win_prob_surface = 0.5
            
            # 3. Obtener pesos específicos para superficie y tipo de torneo con validación
            try:
                surface_weight = self.tourney_surface_weights.get(
                    self._normalize_tournament_level(tourney_level), 
                    {}
                ).get(surface, 0.7)
                
                # Asegurar que es un número
                if not isinstance(surface_weight, (int, float)):
                    surface_weight = 0.7
            except:
                surface_weight = 0.7  # Valor por defecto
            
            # 4. Media ponderada con peso específico
            w_win_prob = (1.0 - surface_weight) * w_win_prob_general + surface_weight * w_win_prob_surface
            
            # 5. Considerar ventaja head-to-head con validación
            try:
                h2h_factor = self.get_h2h_advantage(winner_id, loser_id)
                # Asegurar que es un número razonable
                if not isinstance(h2h_factor, (int, float)) or h2h_factor < 0.5 or h2h_factor > 1.5:
                    h2h_factor = 1.0
            except:
                h2h_factor = 1.0  # Valor neutral por defecto
            
            # 6. Considerar forma reciente con validación
            try:
                w_form = self.get_player_form(winner_id, surface)
                l_form = self.get_player_form(loser_id, surface)
                
                # Validar valores
                if not isinstance(w_form, (int, float)) or w_form <= 0:
                    w_form = 1.0
                if not isinstance(l_form, (int, float)) or l_form <= 0:
                    l_form = 1.0
                    
                form_ratio = w_form / l_form
            except:
                form_ratio = 1.0  # Valor neutral por defecto
            
            # 7. Ajustar probabilidad final con h2h y forma
            # Limitar el impacto para mantener estabilidad
            try:
                adjustment = ((h2h_factor - 1.0) * 0.5) + ((form_ratio - 1.0) * 0.3)
                w_win_prob = min(0.95, max(0.05, w_win_prob + adjustment * 0.1))
            except:
                # En caso de error, mantener probabilidad original
                pass
            
            # 8. Calcular importancia del partido con validación
            try:
                match_importance = self._get_match_importance_factor(
                    tourney_level, round_name, winner_id, loser_id
                )
                
                # Validar valor
                if not isinstance(match_importance, (int, float)) or match_importance <= 0:
                    match_importance = 1.0
            except:
                match_importance = 1.0  # Valor neutral por defecto
            
            # 9. Analizar margen de victoria con validación
            try:
                margin_multiplier = self._get_margin_multiplier(score, retirement)
                
                # Validar valor
                if not isinstance(margin_multiplier, (int, float)) or margin_multiplier <= 0:
                    margin_multiplier = 1.0
            except:
                margin_multiplier = 1.0  # Valor neutral por defecto
            
            # 10. Calcular factores de impacto para victorias importantes/inesperadas
            try:
                victory_impact = self._calculate_victory_impact_factor(
                    w_general_elo, l_general_elo, w_win_prob, margin_multiplier, match_importance
                )
                
                # Validar valor
                if not isinstance(victory_impact, (int, float)) or victory_impact <= 0:
                    victory_impact = 1.0
            except:
                victory_impact = 1.0  # Valor neutral por defecto
            
            # 11. Extraer factor adicional de estadísticas si están disponibles
            try:
                if stats and isinstance(stats, dict):
                    stats_factor = self._get_match_stats_factor(None, winner_id, loser_id, stats.get('winner'), stats.get('loser'))
                else:
                    stats_factor = self._get_match_stats_factor(None, winner_id, loser_id)
                    
                # Validar valor
                if not isinstance(stats_factor, (int, float)) or stats_factor <= 0:
                    stats_factor = 1.0
            except:
                stats_factor = 1.0  # Valor neutral por defecto
            
            # 12. Calcular factores K dinámicos con validación
            try:
                k_winner = self._get_dynamic_k_factor(
                    winner_id, tourney_level, round_name, surface, match_importance
                )
                
                # Validar valor
                if not isinstance(k_winner, (int, float)) or k_winner <= 0:
                    k_winner = self.k_factor_base
            except:
                k_winner = self.k_factor_base  # Valor por defecto
            
            try:
                k_loser = self._get_dynamic_k_factor(
                    loser_id, tourney_level, round_name, surface, match_importance
                )
                
                # Validar valor
                if not isinstance(k_loser, (int, float)) or k_loser <= 0:
                    k_loser = self.k_factor_base
            except:
                k_loser = self.k_factor_base  # Valor por defecto
            
            # 13. Calcular cambios de ELO con todos los factores
            # Para el ganador: K * margen * impacto * stats * (actual - esperado)
            try:
                elo_change_winner = k_winner * margin_multiplier * victory_impact * stats_factor * (1.0 - w_win_prob)
            except:
                # En caso de error, usar fórmula simplificada
                elo_change_winner = k_winner * (1.0 - w_win_prob)
            
            # Para el perdedor: efecto proporcional pero en dirección opuesta
            # Asegura que sea negativo (el perdedor pierde puntos)
            try:
                elo_change_loser = -k_loser * margin_multiplier * stats_factor * w_win_prob
            except:
                # En caso de error, usar fórmula simplificada
                elo_change_loser = -k_loser * w_win_prob
            
            # Gestionar valores extremos con límites razonables
            elo_change_winner = min(50, max(1, elo_change_winner))
            elo_change_loser = max(-50, min(-1, elo_change_loser))
            
            # 14. Actualizar ratings generales con validación
            if winner_id not in self.player_ratings:
                self.player_ratings[winner_id] = self.initial_rating
            if loser_id not in self.player_ratings:
                self.player_ratings[loser_id] = self.initial_rating
            
            # Asegurar que los valores actuales son números
            if not isinstance(self.player_ratings[winner_id], (int, float)):
                self.player_ratings[winner_id] = self.initial_rating
            if not isinstance(self.player_ratings[loser_id], (int, float)):
                self.player_ratings[loser_id] = self.initial_rating
                
            self.player_ratings[winner_id] += elo_change_winner
            self.player_ratings[loser_id] += elo_change_loser
            
            # 15. Actualizar ratings por superficie con validación
            if winner_id not in self.player_ratings_by_surface[surface]:
                self.player_ratings_by_surface[surface][winner_id] = self.initial_rating
            if loser_id not in self.player_ratings_by_surface[surface]:
                self.player_ratings_by_surface[surface][loser_id] = self.initial_rating
            
            # Asegurar que los valores actuales son números
            if not isinstance(self.player_ratings_by_surface[surface][winner_id], (int, float)):
                self.player_ratings_by_surface[surface][winner_id] = self.initial_rating
            if not isinstance(self.player_ratings_by_surface[surface][loser_id], (int, float)):
                self.player_ratings_by_surface[surface][loser_id] = self.initial_rating
                
            # Actualizar con mayor especificidad para superficie
            try:
                surface_mult = self.surface_specificity.get(surface, 1.0)
                # Validar multiplicador
                if not isinstance(surface_mult, (int, float)) or surface_mult <= 0:
                    surface_mult = 1.0
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
                
            # Asegurar que los valores actuales son números
            if not isinstance(self.player_match_count[winner_id], (int, float)):
                self.player_match_count[winner_id] = 0
            if not isinstance(self.player_match_count[loser_id], (int, float)):
                self.player_match_count[loser_id] = 0
                
            self.player_match_count[winner_id] = self.player_match_count[winner_id] + 1
            self.player_match_count[loser_id] = self.player_match_count[loser_id] + 1
            
            # Por superficie con validación
            if winner_id not in self.player_match_count_by_surface[surface]:
                self.player_match_count_by_surface[surface][winner_id] = 0
            if loser_id not in self.player_match_count_by_surface[surface]:
                self.player_match_count_by_surface[surface][loser_id] = 0
            
            # Asegurar que los valores actuales son números
            if not isinstance(self.player_match_count_by_surface[surface][winner_id], (int, float)):
                self.player_match_count_by_surface[surface][winner_id] = 0
            if not isinstance(self.player_match_count_by_surface[surface][loser_id], (int, float)):
                self.player_match_count_by_surface[surface][loser_id] = 0
                
            self.player_match_count_by_surface[surface][winner_id] += 1
            self.player_match_count_by_surface[surface][loser_id] += 1
            
            # 18. Actualizar registro head-to-head con validación
            try:
                self.update_h2h_record(winner_id, loser_id)
            except Exception as e:
                logger.warning(f"Error actualizando h2h: {str(e)}")
            
            # 19. Actualizar forma reciente con validación
            try:
                self.update_player_form(winner_id, 1, surface)  # Victoria = 1
                self.update_player_form(loser_id, 0, surface)   # Derrota = 0
            except Exception as e:
                logger.warning(f"Error actualizando forma: {str(e)}")
            
            # 20. Actualizar incertidumbre de los ratings con validación
            # Más partidos = menor incertidumbre
            w_matches = self.player_match_count.get(winner_id, 0)
            l_matches = self.player_match_count.get(loser_id, 0)
            
            # Asegurar que son números
            if not isinstance(w_matches, (int, float)) or w_matches < 0:
                w_matches = 0
            if not isinstance(l_matches, (int, float)) or l_matches < 0:
                l_matches = 0
            
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
                match_summary = {
                    'date': match_date,
                    'opponent_id': loser_id,
                    'surface': surface,
                    'result': 'win',
                    'score': score,
                    'elo_change': elo_change_winner,
                    'tourney_level': tourney_level
                }
                self.player_match_history[winner_id].append(match_summary)
                
                match_summary = {
                    'date': match_date,
                    'opponent_id': winner_id,
                    'surface': surface,
                    'result': 'loss',
                    'score': score,
                    'elo_change': elo_change_loser,
                    'tourney_level': tourney_level
                }
                self.player_match_history[loser_id].append(match_summary)
            except Exception as e:
                logger.warning(f"Error actualizando historial de partidos: {str(e)}")
            
            return elo_change_winner, elo_change_loser
        except Exception as e:
            # Capturar cualquier error no manejado para evitar que falle todo el proceso
            logger.error(f"Error en update_ratings: {str(e)}")
            logger.debug(traceback.format_exc())
            return 0.0, 0.0  # Devolver valores neutrales
    
    def get_player_details(self, player_id: str) -> Dict:
        """
        Obtiene información detallada del jugador incluyendo ratings,
        estadísticas, forma reciente y más.
        
        Args:
            player_id: ID del jugador
            
        Returns:
            Diccionario con toda la información disponible del jugador
        """
        player_id = str(player_id)
        
        # Información básica y nombre
        player_name = self.get_player_name(player_id)
        
        # Información básica de ELO
        elo_general = self.get_player_rating(player_id)
        
        # ELO por superficie
        elo_by_surface = {
            surface: self.get_player_rating(player_id, surface)
            for surface in self.player_ratings_by_surface.keys()
        }
        
        # Encuentros por superficie
        matches_by_surface = {
            surface: self.player_match_count_by_surface[surface].get(player_id, 0)
            for surface in self.player_match_count_by_surface.keys()
        }
        
        # Forma reciente
        recent_form = self.get_player_form(player_id)
        
        # Historial de partidos
        match_history = self.player_match_history.get(player_id, [])
        
        # Extraer estadísticas del historial de partidos
        total_matches = self.player_match_count.get(player_id, 0)
        wins = len([m for m in match_history if m['result'] == 'win'])
        losses = total_matches - wins
        
        # Estadísticas por superficie
        surface_stats = {}
        for surface in self.player_ratings_by_surface.keys():
            surface_matches = [m for m in match_history if m['surface'] == surface]
            surface_wins = len([m for m in surface_matches if m['result'] == 'win'])
            surface_losses = len(surface_matches) - surface_wins
            
            surface_stats[surface] = {
                'matches': len(surface_matches),
                'wins': surface_wins,
                'losses': surface_losses,
                'win_rate': surface_wins / max(1, len(surface_matches))
            }
        
        # Extraer información de rendimiento por tipo de torneo
        tourney_stats = {}
        for match in match_history:
            level = match.get('tourney_level', 'unknown')
            if level not in tourney_stats:
                tourney_stats[level] = {'wins': 0, 'losses': 0}
            
            if match['result'] == 'win':
                tourney_stats[level]['wins'] += 1
            else:
                tourney_stats[level]['losses'] += 1
        
        # Calcular win rate por tipo de torneo
        for level in tourney_stats:
            stats = tourney_stats[level]
            total = stats['wins'] + stats['losses']
            stats['matches'] = total
            stats['win_rate'] = stats['wins'] / max(1, total)
        
        # Información de incertidumbre
        uncertainty = self.get_player_uncertainty(player_id)
        
        # Información de rivales
        rivals = {}
        for opponent_id in self.h2h_records.get(player_id, {}):
            h2h = self.h2h_records[player_id][opponent_id]
            if h2h['wins'] > 0 or h2h['losses'] > 0:
                opponent_name = self.get_player_name(opponent_id)
                rivals[opponent_id] = {
                    'name': opponent_name,
                    'wins': h2h['wins'],
                    'losses': h2h['losses'],
                    'total': h2h['wins'] + h2h['losses'],
                    'win_rate': h2h['wins'] / max(1, h2h['wins'] + h2h['losses'])
                }
        
        # Organizar toda la información
        return {
            'id': player_id,
            'name': player_name,
            'elo': {
                'general': elo_general,
                'by_surface': elo_by_surface,
                'uncertainty': uncertainty
            },
            'stats': {
                'total_matches': total_matches,
                'wins': wins,
                'losses': losses,
                'win_rate': wins / max(1, total_matches),
                'by_surface': surface_stats,
                'by_tourney_level': tourney_stats
            },
            'form': recent_form,
            'match_count': self.player_match_count.get(player_id, 0),
            'matches_by_surface': matches_by_surface,
            'last_match': self.player_last_match.get(player_id),
            'rivals': rivals,
            'recent_matches': sorted(match_history, key=lambda x: x['date'], reverse=True)[:10] if match_history else []
        }
    
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
        surface = self._normalize_surface(surface)
        
        # Obtener ratings con enfoque bayesiano
        p1_general_elo = self.get_player_rating(player1_id)
        p2_general_elo = self.get_player_rating(player2_id)
        p1_surface_elo = self.get_combined_surface_rating(player1_id, surface)
        p2_surface_elo = self.get_combined_surface_rating(player2_id, surface)
        
        # Obtener incertidumbres
        p1_uncertainty = self.get_player_uncertainty(player1_id, surface)
        p2_uncertainty = self.get_player_uncertainty(player2_id, surface)
        
        # Calcular probabilidades base
        p1_prob_general = self.calculate_expected_win_probability(
            p1_general_elo, p2_general_elo, p1_uncertainty, p2_uncertainty
        )
        p1_prob_surface = self.calculate_expected_win_probability(
            p1_surface_elo, p2_surface_elo, p1_uncertainty, p2_uncertainty
        )
        
        # Pesos según superficie y tipo de torneo
        surface_weight = self.tourney_surface_weights.get(
            self._normalize_tournament_level(tourney_level), 
            {}
        ).get(surface, 0.7)
        
        # Probabilidad ponderada base
        p1_prob_base = (1.0 - surface_weight) * p1_prob_general + surface_weight * p1_prob_surface
        
        # Factores contextuales
        h2h_factor = self.get_h2h_advantage(player1_id, player2_id)
        p1_form = self.get_player_form(player1_id, surface)
        p2_form = self.get_player_form(player2_id, surface)
        form_ratio = p1_form / p2_form if p2_form > 0 else 1.0
        
        # Otros factores específicos
        style_factor = self._get_style_compatibility_factor(player1_id, player2_id, surface)
        
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
                'name': self.get_player_name(player1_id),
                'elo_general': p1_general_elo,
                'elo_surface': p1_surface_elo,
                'form': p1_form,
                'matches': p1_matches,
                'surface_matches': p1_surface_matches,
                'uncertainty': p1_uncertainty
            },
            'player2': {
                'id': player2_id,
                'name': self.get_player_name(player2_id),
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
                'favorite_name': self.get_player_name(player1_id if p1_final_prob > 0.5 else player2_id)
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
        df['surface'] = df['surface'].apply(lambda x: self._normalize_surface(x) if not pd.isna(x) else 'hard')
        
        # Normalizar nivel de torneo
        if 'tourney_level' in df.columns:
            df['tourney_level'] = df['tourney_level'].apply(
                lambda x: self._normalize_tournament_level(x) if not pd.isna(x) else 'O'
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
        if include_stats and not self.match_stats_df.empty:
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
                        
                        # Buscar estadísticas para este partido - usar .astype(str) para evitar errores de comparación
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
                    
                    # Validar surface (asegurar que es string y normalizar)
                    surface = str(match['surface']).lower() if not pd.isna(match['surface']) else 'hard'
                    
                    # Validar tourney_level (asegurar que es string y normalizar)
                    tourney_level = str(match['tourney_level']) if not pd.isna(match['tourney_level']) else 'O'
                    tourney_level = self._normalize_tournament_level(tourney_level)
                    
                    # Validar round (asegurar que es string)
                    round_name = str(match['round']) if not pd.isna(match['round']) else 'R32'
                    
                    # Validar score (asegurar que es string)
                    score = str(match['score']) if not pd.isna(match['score']) else ''
                    
                    # Validar retirement (asegurar que es booleano)
                    retirement = bool(match['retirement']) if not pd.isna(match['retirement']) else False
                    
                    # Validar match_id (asegurar que es algo que se puede convertir a string)
                    match_id = str(match['match_id']) if not pd.isna(match['match_id']) else str(idx)
                    
                    # Validar IDs básicos
                    if pd.isna(winner_id) or pd.isna(loser_id) or winner_id == loser_id:
                        logger.warning(f"Partido inválido (ID ganador = ID perdedor o valores nulos): {winner_id} vs {loser_id}")
                        self.stats['invalid_matches_skipped'] += 1
                        continue
                    
                    # Guardar ratings actuales antes de la actualización
                    try:
                        df.at[idx, 'winner_elo_before'] = self.get_player_rating(winner_id)
                        df.at[idx, 'loser_elo_before'] = self.get_player_rating(loser_id)
                        df.at[idx, 'winner_surface_elo_before'] = self.get_player_rating(winner_id, surface)
                        df.at[idx, 'loser_surface_elo_before'] = self.get_player_rating(loser_id, surface)
                    except Exception as e:
                        logger.debug(f"Error obteniendo ratings iniciales: {str(e)}")
                    
                    # Calcular factores adicionales con manejo de errores
                    try:
                        h2h_factor = self.get_h2h_advantage(winner_id, loser_id)
                        df.at[idx, 'h2h_factor'] = h2h_factor
                    except Exception as e:
                        logger.debug(f"Error calculando h2h_factor: {str(e)}")
                        df.at[idx, 'h2h_factor'] = 1.0  # Valor neutral
                    
                    # Calcular factor de forma con validación
                    try:
                        w_form = self.get_player_form(winner_id, surface)
                        l_form = self.get_player_form(loser_id, surface)
                        
                        # Validar que son números
                        if not isinstance(w_form, (int, float)) or w_form <= 0:
                            w_form = 1.0
                        if not isinstance(l_form, (int, float)) or l_form <= 0:
                            l_form = 1.0
                            
                        form_factor = w_form / l_form
                        df.at[idx, 'form_factor'] = form_factor
                    except Exception as e:
                        logger.debug(f"Error calculando form_factor: {str(e)}")
                        df.at[idx, 'form_factor'] = 1.0  # Valor neutral
                    
                    # Calcular importancia del partido con validación
                    try:
                        match_importance = self._get_match_importance_factor(
                            tourney_level, round_name, winner_id, loser_id
                        )
                        df.at[idx, 'match_importance'] = match_importance
                    except Exception as e:
                        logger.debug(f"Error calculando match_importance: {str(e)}")
                        df.at[idx, 'match_importance'] = 1.0  # Valor neutral
                    
                    # Calcular margen del partido con validación
                    try:
                        margin_multiplier = self._get_margin_multiplier(score, retirement)
                        df.at[idx, 'margin_multiplier'] = margin_multiplier
                    except Exception as e:
                        logger.debug(f"Error calculando margin_multiplier: {str(e)}")
                        df.at[idx, 'margin_multiplier'] = 1.0  # Valor neutral
                    
                    # Obtener estadísticas específicas del partido si existen
                    match_specific_stats = match_stats.get(match_id, None)
                    
                    # Actualizar ratings
                    try:
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
                    except Exception as e:
                        logger.warning(f"Error actualizando ratings: {str(e)}")
                        logger.debug(traceback.format_exc())
                    
                    # Recuperar valores desde el último registro en el historial
                    try:
                        if self.rating_history:
                            last_record = self.rating_history[-1]
                            df.at[idx, 'expected_win_prob'] = last_record.get('expected_win_prob', 0.5)
                            df.at[idx, 'k_factor_winner'] = last_record.get('k_factor_winner', self.k_factor_base)
                            df.at[idx, 'k_factor_loser'] = last_record.get('k_factor_loser', self.k_factor_base)
                    except Exception as e:
                        logger.debug(f"Error recuperando valores del historial: {str(e)}")
                    
                    # Actualizar contador
                    self.stats['total_matches_processed'] += 1
                    
                    # Actualizar contador de superficies
                    self.stats['surface_distribution'][self._normalize_surface(surface)] += 1
                    
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
        
        # Guardar incertidumbres
        uncertainty_path = output_path / 'player_rating_uncertainty.json'
        with open(uncertainty_path, 'w') as f:
            json.dump(self.player_rating_uncertainty, f, indent=2)
        
        # Guardar historial head-to-head
        h2h_path = output_path / 'head_to_head_records.json'
        with open(h2h_path, 'w') as f:
            # Convertir defaultdict anidado a dict regular para JSON
            h2h_dict = {
                player_id: {
                    opp_id: dict(record)
                    for opp_id, record in opponents.items()
                }
                for player_id, opponents in self.h2h_records.items()
            }
            json.dump(h2h_dict, f, indent=2)
        
        # Guardar forma reciente de jugadores
        form_path = output_path / 'player_form.json'
        with open(form_path, 'w') as f:
            json.dump(self.player_recent_form, f, indent=2)
        
        # Guardar historial de ratings como CSV
        if include_history and self.rating_history:
            history_df = pd.DataFrame(self.rating_history)
            history_path = output_path / 'elo_rating_history.csv'
            history_df.to_csv(history_path, index=False)
            
            # También guardar un historial resumido
            try:
                # Seleccionar columnas clave
                key_columns = [
                    'date', 'winner_id', 'loser_id', 
                    'winner_rating_before', 'loser_rating_before',
                    'winner_rating_after', 'loser_rating_after',
                    'elo_change_winner', 'elo_change_loser',
                    'expected_win_prob', 'surface', 'tourney_level'
                ]
                summary_columns = [col for col in key_columns if col in history_df.columns]
                summary_df = history_df[summary_columns]
                
                # Guardar versión resumida
                summary_path = output_path / 'elo_rating_history_summary.csv'
                summary_df.to_csv(summary_path, index=False)
            except Exception as e:
                logger.warning(f"Error guardando historial resumido: {str(e)}")
        
        # Guardar historial de partidos por jugador
        if include_match_history and self.player_match_history:
            match_history_path = output_path / 'player_match_history.json'
            
            # Convertir a formato serializable (sin objetos datetime)
            serializable_history = {}
            for player_id, matches in self.player_match_history.items():
                serializable_history[player_id] = []
                for match in matches:
                    match_copy = match.copy()
                    if 'date' in match_copy:
                        match_copy['date'] = match_copy['date'].strftime('%Y-%m-%d')
                    serializable_history[player_id].append(match_copy)
            
            with open(match_history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
        
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
        
        # Guardar lista de Top players para referencia rápida
        top_players_general = self.get_top_players(100).to_dict('records')
        top_players_path = output_path / 'top_players_general.json'
        with open(top_players_path, 'w') as f:
            json.dump(top_players_general, f, indent=2)
        
        # Guardar top players por superficie
        for surface in self.player_ratings_by_surface:
            top_surface = self.get_top_players(50, surface).to_dict('records')
            if top_surface:
                surface_top_path = output_path / f'top_players_{surface}.json'
                with open(surface_top_path, 'w') as f:
                    json.dump(top_surface, f, indent=2)
        
        logger.info(f"Ratings ELO guardados en {output_dir}")

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
                top_players['player_name'] = top_players['player_id'].apply(self.get_player_name)
                
                # Añadir incertidumbre
                top_players['uncertainty'] = top_players['player_id'].apply(
                    lambda x: self.get_player_uncertainty(str(x), surface)
                )
                
                # Si hay superficie, añadir partidos en superficie
                if surface:
                    normalized_surface = self._normalize_surface(surface)
                    top_players['surface_matches'] = top_players['player_id'].apply(
                        lambda x: self.player_match_count_by_surface[normalized_surface].get(str(x), 0)
                    )
            
            return top_players
            
        # Usar ratings actuales
        if surface:
            surface = self._normalize_surface(surface)
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
            df['player_name'] = df['player_id'].apply(self.get_player_name)
            
            # Añadir incertidumbre
            df['uncertainty'] = df['player_id'].apply(
                lambda x: self.get_player_uncertainty(x, surface)
            )
            
            # Añadir forma reciente
            df['form'] = df['player_id'].apply(
                lambda x: self.get_player_form(x, surface)
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
        input_path = Path(input_dir)
        loaded_files = []
        
        # Cargar ratings generales
        general_path = input_path / 'elo_ratings_general.json'
        if general_path.exists():
            with open(general_path, 'r') as f:
                self.player_ratings = json.load(f)
            loaded_files.append(general_path)
        
        # Cargar ratings por superficie
        surface_path = input_path / 'elo_ratings_by_surface.json'
        if surface_path.exists():
            with open(surface_path, 'r') as f:
                self.player_ratings_by_surface = json.load(f)
            loaded_files.append(surface_path)
        
        # Cargar contadores de partidos
        counts_path = input_path / 'player_match_counts.json'
        if counts_path.exists():
            with open(counts_path, 'r') as f:
                self.player_match_count = json.load(f)
            loaded_files.append(counts_path)
        
        # Cargar contadores por superficie
        counts_surface_path = input_path / 'player_match_counts_by_surface.json'
        if counts_surface_path.exists():
            with open(counts_surface_path, 'r') as f:
                self.player_match_count_by_surface = json.load(f)
            loaded_files.append(counts_surface_path)
        else:
            # Reconstruir contadores por superficie si no existen
            for surface in self.player_ratings_by_surface:
                for player_id in self.player_ratings_by_surface[surface]:
                    if player_id not in self.player_match_count_by_surface[surface]:
                        self.player_match_count_by_surface[surface][player_id] = 0
        
        # Cargar incertidumbres
        uncertainty_path = input_path / 'player_rating_uncertainty.json'
        if uncertainty_path.exists():
            with open(uncertainty_path, 'r') as f:
                self.player_rating_uncertainty = json.load(f)
            loaded_files.append(uncertainty_path)
        
        # Cargar historial head-to-head
        h2h_path = input_path / 'head_to_head_records.json'
        if h2h_path.exists():
            try:
                with open(h2h_path, 'r') as f:
                    h2h_dict = json.load(f)
                
                # Convertir a formato de defaultdict anidado
                for player_id, opponents in h2h_dict.items():
                    for opp_id, record in opponents.items():
                        self.h2h_records[player_id][opp_id] = record
                
                loaded_files.append(h2h_path)
            except Exception as e:
                logger.warning(f"Error cargando historial head-to-head: {str(e)}")
        
        # Cargar forma reciente de jugadores
        form_path = input_path / 'player_form.json'
        if form_path.exists():
            try:
                with open(form_path, 'r') as f:
                    self.player_recent_form = json.load(f)
                loaded_files.append(form_path)
            except Exception as e:
                logger.warning(f"Error cargando forma de jugadores: {str(e)}")
        
        # Cargar historial si existe
        if load_history:
            history_path = input_path / 'elo_rating_history.csv'
            if history_path.exists():
                try:
                    history_df = pd.read_csv(history_path)
                    if 'date' in history_df.columns:
                        history_df['date'] = pd.to_datetime(history_df['date'])
                    self.rating_history = history_df.to_dict('records')
                    loaded_files.append(history_path)
                except Exception as e:
                    logger.warning(f"Error cargando historial de ratings: {str(e)}")
        
        # Cargar historial de partidos por jugador
        if load_match_history:
            match_history_path = input_path / 'player_match_history.json'
            if match_history_path.exists():
                try:
                    with open(match_history_path, 'r') as f:
                        history_dict = json.load(f)
                    
                    # Convertir fechas en strings a objetos datetime
                    for player_id, matches in history_dict.items():
                        for match in matches:
                            if 'date' in match and isinstance(match['date'], str):
                                match['date'] = datetime.strptime(match['date'], '%Y-%m-%d')
                        self.player_match_history[player_id] = matches
                    
                    loaded_files.append(match_history_path)
                except Exception as e:
                    logger.warning(f"Error cargando historial de partidos: {str(e)}")
        
        # Cargar fechas de últimos partidos
        if self.rating_history:
            # Reconstruir fechas de últimos partidos desde historial
            for record in self.rating_history:
                winner_id = record['winner_id']
                loser_id = record['loser_id']
                match_date = record['date']
                
                # Actualizar solo si la fecha es más reciente que la actual
                if winner_id not in self.player_last_match or self.player_last_match[winner_id] < match_date:
                    self.player_last_match[winner_id] = match_date
                
                if loser_id not in self.player_last_match or self.player_last_match[loser_id] < match_date:
                    self.player_last_match[loser_id] = match_date
        
        # Cargar estadísticas si existen
        stats_path = input_path / 'elo_processing_stats.json'
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
                loaded_files.append(stats_path)
            except Exception as e:
                logger.warning(f"Error cargando estadísticas: {str(e)}")
        
        logger.info(f"Ratings ELO cargados desde {input_dir} ({len(loaded_files)} archivos)")
        
        # Devolver resumen
        logger.info(f"Cargados ratings para {len(self.player_ratings)} jugadores")
        
        # Reconstruir datos faltantes si es necesario
        if self.player_ratings and not self.player_rating_uncertainty:
            logger.info("Reconstruyendo valores de incertidumbre...")
            for player_id, rating in self.player_ratings.items():
                matches = self.player_match_count.get(player_id, 0)
                self.player_rating_uncertainty[player_id] = 350 / (matches + 5)

    def run_full_pipeline(self, data_paths: Dict[str, str], 
                     output_dir: Optional[str] = None,
                     evaluate: bool = True,
                     create_visualizations: bool = True,
                     save_processed_data: bool = True,
                     min_year: int = 2000,
                     visualizations_dir: Optional[str] = None) -> Dict:
        """
        Ejecuta el pipeline completo de procesamiento: carga datos, procesa partidos, 
        evalúa, genera visualizaciones y guarda resultados.
        
        Args:
            data_paths: Diccionario con rutas de datos (atp_matches, wta_matches, players, etc.)
            output_dir: Directorio de salida para los resultados
            evaluate: Si debe realizar evaluación del sistema
            create_visualizations: Si debe generar visualizaciones
            save_processed_data: Si debe guardar los datos procesados
            min_year: Año mínimo para filtrar datos históricos
            visualizations_dir: Directorio para guardar visualizaciones
            
        Returns:
            Diccionario con métricas y resultados del procesamiento
        """
        logger.info("Iniciando pipeline completo de procesamiento ELO para tenis...")
        
        start_time = datetime.now()
        
        # Resultados por defecto en caso de error
        results = {
            'status': 'error',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': "Error desconocido durante el procesamiento"
        }
        
        try:
            # 1. Preparar directorios
            if output_dir is None:
                output_dir = "data/processed/elo"
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            vis_path = None
            if create_visualizations:
                if visualizations_dir is None:
                    visualizations_dir = os.path.join(output_dir, "visualizations")
                
                vis_path = Path(visualizations_dir)
                vis_path.mkdir(parents=True, exist_ok=True)
            
            # 2. Cargar datos de partidos
            atp_matches = None
            wta_matches = None
            
            # Cargar partidos ATP
            if 'atp_matches' in data_paths:
                logger.info(f"Cargando partidos ATP desde {data_paths['atp_matches']}...")
                atp_matches = pd.read_csv(data_paths['atp_matches'])
                
                # Convertir fechas
                if 'tourney_date' in atp_matches.columns:
                    atp_matches['match_date'] = pd.to_datetime(atp_matches['tourney_date'], format='%Y%m%d')
                elif 'date' in atp_matches.columns:
                    atp_matches['match_date'] = pd.to_datetime(atp_matches['date'])
                
                # Filtrar por año
                if 'match_date' in atp_matches.columns:
                    atp_matches = atp_matches[atp_matches['match_date'].dt.year >= min_year]
                    
                logger.info(f"Cargados {len(atp_matches)} partidos ATP desde {min_year}")
            
            # Cargar partidos WTA
            if 'wta_matches' in data_paths:
                logger.info(f"Cargando partidos WTA desde {data_paths['wta_matches']}...")
                wta_matches = pd.read_csv(data_paths['wta_matches'])
                
                # Convertir fechas
                if 'tourney_date' in wta_matches.columns:
                    wta_matches['match_date'] = pd.to_datetime(wta_matches['tourney_date'], format='%Y%m%d')
                elif 'date' in wta_matches.columns:
                    wta_matches['match_date'] = pd.to_datetime(wta_matches['date'])
                
                # Filtrar por año
                if 'match_date' in wta_matches.columns:
                    wta_matches = wta_matches[wta_matches['match_date'].dt.year >= min_year]
                    
                logger.info(f"Cargados {len(wta_matches)} partidos WTA desde {min_year}")
            
            # Cargar datos de jugadores
            if 'atp_players' in data_paths:
                logger.info(f"Cargando jugadores ATP desde {data_paths['atp_players']}...")
                self.atp_players_df = pd.read_csv(data_paths['atp_players'])
                
                # Actualizar diccionario de nombres
                if hasattr(self, '_update_player_names_from_df'):
                    self._update_player_names_from_df(self.atp_players_df)
                else:
                    # Método alternativo si no existe el método específico
                    if 'player_id' in self.atp_players_df.columns:
                        if 'name_first' in self.atp_players_df.columns and 'name_last' in self.atp_players_df.columns:
                            for _, row in self.atp_players_df.iterrows():
                                player_id = str(row['player_id'])
                                self.player_names[player_id] = f"{row['name_first']} {row['name_last']}"
                        elif 'name' in self.atp_players_df.columns:
                            for _, row in self.atp_players_df.iterrows():
                                player_id = str(row['player_id'])
                                self.player_names[player_id] = row['name']
            
            if 'wta_players' in data_paths:
                logger.info(f"Cargando jugadoras WTA desde {data_paths['wta_players']}...")
                self.wta_players_df = pd.read_csv(data_paths['wta_players'])
                
                # Actualizar diccionario de nombres
                if hasattr(self, '_update_player_names_from_df'):
                    self._update_player_names_from_df(self.wta_players_df)
                else:
                    # Método alternativo si no existe el método específico
                    if 'player_id' in self.wta_players_df.columns:
                        if 'name_first' in self.wta_players_df.columns and 'name_last' in self.wta_players_df.columns:
                            for _, row in self.wta_players_df.iterrows():
                                player_id = str(row['player_id'])
                                self.player_names[player_id] = f"{row['name_first']} {row['name_last']}"
                        elif 'name' in self.wta_players_df.columns:
                            for _, row in self.wta_players_df.iterrows():
                                player_id = str(row['player_id'])
                                self.player_names[player_id] = row['name']
            
            # 3. Procesar partidos
            all_matches_df = None
            
            # Combinar partidos ATP y WTA si están disponibles
            if atp_matches is not None and wta_matches is not None:
                # Verificar que tienen columnas comunes
                common_cols = list(set(atp_matches.columns) & set(wta_matches.columns))
                
                if common_cols:
                    logger.info("Combinando partidos ATP y WTA...")
                    all_matches_df = pd.concat([
                        atp_matches[common_cols],
                        wta_matches[common_cols]
                    ])
                    
                    # Asegurar que tenemos campos esenciales
                    required_cols = ['winner_id', 'loser_id', 'match_date', 'surface']
                    if not all(col in common_cols for col in required_cols):
                        logger.warning(f"Datos combinados no tienen todas las columnas requeridas: {required_cols}")
                        logger.warning(f"Columnas disponibles: {common_cols}")
                        
                        # Intentar mapear columnas si es posible
                        # Ej: tourney_date -> match_date, etc.
                        if 'tourney_date' in common_cols and 'match_date' not in common_cols:
                            all_matches_df['match_date'] = pd.to_datetime(all_matches_df['tourney_date'], format='%Y%m%d')
                    
                    logger.info(f"Combinados {len(all_matches_df)} partidos totales")
                else:
                    logger.warning("ATP y WTA no tienen columnas comunes, procesando por separado")
            
            # Si no pudimos combinar, usar lo que está disponible
            if all_matches_df is None:
                if atp_matches is not None:
                    all_matches_df = atp_matches
                    logger.info("Procesando solo partidos ATP")
                elif wta_matches is not None:
                    all_matches_df = wta_matches
                    logger.info("Procesando solo partidos WTA")
                else:
                    raise ValueError("No se encontraron datos de partidos para procesar")
            
            # 4. Procesar datos con ELO
            logger.info("Iniciando procesamiento ELO de partidos...")
            
            # Calcular ELO progresivamente
            processed_df = self.process_matches_dataframe(all_matches_df, chronological=True)
            
            logger.info(f"Procesamiento completo: {self.stats['total_matches_processed']} partidos")
            
            # Guardar datos procesados si se solicita
            if save_processed_data:
                logger.info(f"Guardando datos procesados en {output_dir}...")
                
                # Guardar ratings calculados
                self.save_ratings(output_dir=output_dir)
                
                # Guardar DataFrame procesado
                processed_path = output_path / "processed_matches.csv"
                processed_df.to_csv(processed_path, index=False)
                logger.info(f"DataFrame procesado guardado en {processed_path}")
            
            # 5. Evaluación del sistema
            evaluation_results = {}
            
            if evaluate:
                try:
                    logger.info("Evaluando capacidad predictiva del sistema ELO...")
                    
                    # Usar últimos 10% de datos como test
                    total_rows = len(processed_df)
                    test_size = int(total_rows * 0.1)
                    
                    # Dividir datos cronológicamente
                    train_df = processed_df.iloc[:-test_size]
                    test_df = processed_df.iloc[-test_size:]
                    
                    # Evaluar rendimiento
                    evaluation_results = self.evaluate_predictive_power(test_df)
                    
                    # Guardar resultados
                    eval_path = output_path / "evaluation_results.json"
                    with open(eval_path, 'w') as f:
                        json.dump(evaluation_results, f, indent=2, default=str)
                    
                    logger.info(f"Evaluación completada: Accuracy {evaluation_results['accuracy']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error en evaluación: {str(e)}")
                    logger.error(traceback.format_exc())
                    evaluation_results = {
                        'status': 'error',
                        'message': f"Error en evaluación: {str(e)}"
                    }
            
            # 6. Generar visualizaciones
            visualizations = []
            
            if create_visualizations and vis_path is not None:
                try:
                    logger.info("Generando visualizaciones...")
                    
                    # 1. Distribución de ratings ELO
                    try:
                        dist_path = os.path.join(visualizations_dir, "elo_distribution.png")
                        self.plot_rating_distribution(save_path=dist_path)
                        visualizations.append(dist_path)
                        logger.info(f"Visualización generada: {dist_path}")
                    except Exception as e:
                        logger.warning(f"Error generando distribución ELO: {str(e)}")
                    
                    # 2. Evolución de top jugadores
                    try:
                        # Obtener top 5 jugadores actuales
                        top_players = self.get_top_players(5)
                        if not top_players.empty:
                            top_ids = top_players['player_id'].tolist()
                            
                            # Graficar evolución
                            evol_path = os.path.join(visualizations_dir, "top_players_evolution.png")
                            self.plot_top_players_history(player_ids=top_ids, save_path=evol_path)
                            visualizations.append(evol_path)
                            logger.info(f"Visualización generada: {evol_path}")
                    except Exception as e:
                        logger.warning(f"Error generando evolución de top jugadores: {str(e)}")
                    
                    # 3. Visualizaciones específicas por superficie
                    for surface in ['hard', 'clay', 'grass', 'carpet']:
                        try:
                            surf_path = os.path.join(visualizations_dir, f"elo_distribution_{surface}.png")
                            self.plot_rating_distribution(surface=surface, save_path=surf_path)
                            visualizations.append(surf_path)
                            logger.info(f"Visualización generada: {surf_path}")
                        except Exception as e:
                            logger.warning(f"Error generando distribución para {surface}: {str(e)}")
                    
                    # 4. Si evaluamos, incluir gráfico de calibración
                    if evaluate and 'calibration' in evaluation_results and evaluation_results.get('calibration'):
                        try:
                            # Crear gráfico de calibración
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Extraer datos de calibración
                            pred_probs = []
                            actual_probs = []
                            counts = []
                            
                            for bin_range, data in evaluation_results['calibration'].items():
                                if 'predicted_probability' in data and 'empirical_probability' in data:
                                    pred_probs.append(data['predicted_probability'])
                                    actual_probs.append(data['empirical_probability'])
                                    counts.append(data.get('count', 0))
                            
                            # Solo crear si hay datos
                            if pred_probs and actual_probs:
                                # Dibujar línea diagonal perfecta
                                ax.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
                                
                                # Dibujar puntos de calibración real
                                scatter = ax.scatter(pred_probs, actual_probs, 
                                                s=[max(20, min(1000, c/10)) for c in counts], 
                                                alpha=0.7, c='royalblue')
                                
                                # Añadir etiquetas y título
                                ax.set_xlabel('Probabilidad predicha', fontsize=12)
                                ax.set_ylabel('Frecuencia observada', fontsize=12)
                                ax.set_title('Diagrama de calibración del modelo ELO', fontsize=14)
                                
                                # Añadir cuadrícula y leyenda
                                ax.grid(True, alpha=0.3)
                                ax.legend()
                                
                                # Ajustar límites
                                ax.set_xlim([0, 1])
                                ax.set_ylim([0, 1])
                                
                                # Guardar
                                calib_path = os.path.join(visualizations_dir, "calibration_plot.png")
                                plt.tight_layout()
                                plt.savefig(calib_path, dpi=300, bbox_inches='tight')
                                plt.close(fig)  # Cerrar el plot explícitamente
                                
                                visualizations.append(calib_path)
                                logger.info(f"Gráfico de calibración guardado en {calib_path}")
                        except Exception as e:
                            logger.warning(f"Error generando gráfico de calibración: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"Error en visualizaciones: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # 7. Generar resumen estadístico completo
            summary = {}
            try:
                summary = self.get_elo_statistics_summary()
                
                # Guardar resumen
                summary_path = output_path / "elo_summary.json"
                with open(summary_path, 'w') as f:
                    # Manejar tipos de datos especiales
                    def json_serializer(obj):
                        if isinstance(obj, (datetime, np.datetime64)):
                            return obj.isoformat()
                        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                            return int(obj)
                        elif isinstance(obj, (np.float64, np.float32, np.float16)):
                            return float(obj)
                        return str(obj)
                    
                    json.dump(summary, f, indent=2, default=json_serializer)
                
                logger.info(f"Resumen estadístico guardado en {summary_path}")
                
            except Exception as e:
                logger.error(f"Error generando resumen estadístico: {str(e)}")
                logger.error(traceback.format_exc())
                summary = {
                    'status': 'error',
                    'message': f"Error generando resumen estadístico: {str(e)}"
                }
            
            # 8. Generar informe final
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Actualizar estadísticas de procesamiento
            self.stats['processing_time'] = processing_time
            
            results = {
                'status': 'success',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time_seconds': processing_time,
                'total_matches_processed': self.stats['total_matches_processed'],
                'total_players': len(self.player_ratings),
                'evaluation': evaluation_results if evaluate and evaluation_results else None,
                'visualizations': visualizations if create_visualizations and visualizations else None,
                'output_directory': str(output_path),
                'summary_stats': summary if summary and 'status' not in summary else None
            }
            
            # Guardar informe final
            report_path = output_path / "pipeline_report.json"
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=json_serializer)
            
            logger.info(f"Pipeline completo ejecutado en {processing_time:.2f} segundos")
            logger.info(f"Informe final guardado en {report_path}")
            
        except Exception as e:
            # Capturar cualquier error no manejado
            error_msg = str(e)
            logger.error(f"Error en pipeline: {error_msg}")
            logger.error(traceback.format_exc())
            
            results = {
                'status': 'error',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'message': error_msg,
                'traceback': traceback.format_exc()
            }
        
        return results

    def get_elo_statistics_summary(self, min_matches: int = 10) -> Dict:
        """
        Genera un resumen completo de estadísticas del sistema ELO.
        
        Args:
            min_matches: Número mínimo de partidos para incluir a un jugador
            
        Returns:
            Diccionario con el resumen de estadísticas
        """
        logger.info("Generando resumen de estadísticas ELO...")
        
        # Estructuras para resultados
        summary = {
            'general': {},
            'by_surface': {},
            'top_players': {},
            'statistics': {},
            'processing': self.stats,
            'distribution': {}
        }
        
        # 1. Estadísticas generales
        total_players = len(self.player_ratings)
        total_matches = sum(self.player_match_count.values())
        
        # Jugadores con suficientes partidos
        qualified_players = {
            player_id: rating 
            for player_id, rating in self.player_ratings.items()
            if self.player_match_count.get(player_id, 0) >= min_matches
        }
        
        qualified_count = len(qualified_players)
        
        # Estadísticas básicas de ratings
        if qualified_count > 0:
            ratings_array = np.array(list(qualified_players.values()))
            
            summary['general'] = {
                'total_players': total_players,
                'qualified_players': qualified_count,
                'total_matches': total_matches,
                'average_matches_per_player': total_matches / max(1, total_players),
                'min_rating': float(np.min(ratings_array)),
                'max_rating': float(np.max(ratings_array)),
                'average_rating': float(np.mean(ratings_array)),
                'median_rating': float(np.median(ratings_array)),
                'std_dev': float(np.std(ratings_array))
            }
        else:
            summary['general'] = {
                'total_players': total_players,
                'qualified_players': 0,
                'total_matches': total_matches,
                'average_matches_per_player': total_matches / max(1, total_players),
                'min_rating': self.initial_rating,
                'max_rating': self.initial_rating,
                'average_rating': self.initial_rating,
                'median_rating': self.initial_rating,
                'std_dev': 0
            }
        
        # 2. Estadísticas por superficie
        surface_stats = {}
        for surface, ratings in self.player_ratings_by_surface.items():
            # Jugadores con suficientes partidos en esta superficie
            qualified_surface = {
                player_id: rating 
                for player_id, rating in ratings.items()
                if self.player_match_count_by_surface[surface].get(player_id, 0) >= min_matches
            }
            
            if qualified_surface:
                surface_ratings = np.array(list(qualified_surface.values()))
                
                surface_stats[surface] = {
                    'total_players': len(ratings),
                    'qualified_players': len(qualified_surface),
                    'total_matches': sum(self.player_match_count_by_surface[surface].values()),
                    'min_rating': float(np.min(surface_ratings)),
                    'max_rating': float(np.max(surface_ratings)),
                    'average_rating': float(np.mean(surface_ratings)),
                    'median_rating': float(np.median(surface_ratings)),
                    'std_dev': float(np.std(surface_ratings))
                }
            else:
                surface_stats[surface] = {
                    'total_players': len(ratings),
                    'qualified_players': 0,
                    'total_matches': sum(self.player_match_count_by_surface[surface].values()),
                    'min_rating': self.initial_rating,
                    'max_rating': self.initial_rating,
                    'average_rating': self.initial_rating,
                    'median_rating': self.initial_rating,
                    'std_dev': 0
                }
        
        summary['by_surface'] = surface_stats
        
        # 3. Top jugadores
        top_players_general = self.get_top_players(n=20, include_details=True).to_dict('records')
        summary['top_players']['general'] = top_players_general
        
        # Top jugadores por superficie
        surface_top = {}
        for surface in self.player_ratings_by_surface:
            top_by_surface = self.get_top_players(n=10, surface=surface, include_details=True).to_dict('records')
            if top_by_surface:
                surface_top[surface] = top_by_surface
        
        summary['top_players']['by_surface'] = surface_top
        
        # 4. Distribuciones estadísticas
        for surface in ['general'] + list(self.player_ratings_by_surface.keys()):
            # Usar función ya implementada para análisis detallado
            if surface == 'general':
                dist_data = self.analyze_ratings_distribution(min_matches=min_matches)
            else:
                dist_data = self.analyze_ratings_distribution(surface=surface, min_matches=min_matches)
            
            # Simplificar datos de distribución para el resumen
            simplified_dist = {
                'count': dist_data['count'],
                'mean': dist_data['mean'],
                'std': dist_data['std'],
                'median': dist_data['percentiles']['50'] if '50' in dist_data['percentiles'] else 
                        dist_data['percentiles'][50] if 50 in dist_data['percentiles'] else self.initial_rating,
                'percentiles': dist_data['percentiles']
            }
            
            summary['distribution'][surface] = simplified_dist
        
        # 5. Estadísticas de predicción (si están disponibles)
        if 'model_accuracy' in self.stats and self.stats['model_accuracy'] is not None:
            summary['statistics']['prediction'] = {
                'accuracy': self.stats['model_accuracy'],
                'calibration': self.stats['calibration_score'] if 'calibration_score' in self.stats else None,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # 6. Estadísticas temporales
        if self.rating_history:
            # Convertir historial a DataFrame si no lo está ya
            if isinstance(self.rating_history, list):
                history_df = pd.DataFrame(self.rating_history)
            else:
                history_df = self.rating_history
            
            # Extraer fechas extremas
            if 'date' in history_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(history_df['date']):
                    history_df['date'] = pd.to_datetime(history_df['date'])
                
                earliest_date = history_df['date'].min()
                latest_date = history_df['date'].max()
                date_range = (latest_date - earliest_date).days
                
                summary['statistics']['temporal'] = {
                    'earliest_date': earliest_date.strftime('%Y-%m-%d'),
                    'latest_date': latest_date.strftime('%Y-%m-%d'),
                    'date_range_days': date_range,
                    'total_historical_matches': len(history_df)
                }
        
        # 7. Estadísticas de incertidumbre
        if self.player_rating_uncertainty:
            uncertainty_values = list(self.player_rating_uncertainty.values())
            if uncertainty_values:
                summary['statistics']['uncertainty'] = {
                    'average': float(np.mean(uncertainty_values)),
                    'min': float(np.min(uncertainty_values)),
                    'max': float(np.max(uncertainty_values)),
                    'median': float(np.median(uncertainty_values))
                }
        
        # 8. Estadísticas por tipo de torneo, si están disponibles
        if self.rating_history:
            if 'tourney_level' in history_df.columns:
                tourney_counts = history_df['tourney_level'].value_counts().to_dict()
                summary['statistics']['tournament_levels'] = tourney_counts
        
        # 9. Estadísticas de head-to-head
        h2h_counts = []
        for player_id, opponents in self.h2h_records.items():
            # Contar total de enfrentamientos
            for opponent_id, record in opponents.items():
                total_matches = record['wins'] + record['losses']
                if total_matches > 0:
                    h2h_counts.append(total_matches)
        
        if h2h_counts:
            summary['statistics']['head_to_head'] = {
                'total_matchups': len(h2h_counts),
                'average_matches_per_matchup': float(np.mean(h2h_counts)),
                'max_matches_between_pair': int(np.max(h2h_counts))
            }
        
        return summary

    def evaluate_predictive_power(self, test_matches_df: pd.DataFrame, 
                           probability_threshold: float = 0.5,
                           features_to_use: Optional[List[str]] = None) -> Dict:
        """
        Evalúa la capacidad predictiva del sistema ELO actual.
        
        Args:
            test_matches_df: DataFrame con partidos de prueba
            probability_threshold: Umbral de probabilidad para predicciones
            features_to_use: Lista de características específicas a evaluar
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        logger.info(f"Evaluando capacidad predictiva en {len(test_matches_df)} partidos...")
        
        # Verificar que tenemos las columnas necesarias
        required_columns = ['winner_id', 'loser_id', 'surface']
        
        if not all(col in test_matches_df.columns for col in required_columns):
            raise ValueError(f"DataFrame debe contener las columnas: {required_columns}")
        
        # Hacer copia para no modificar el original
        df = test_matches_df.copy()
        
        # Crear columnas para almacenar predicciones
        df['predicted_winner_id'] = None
        df['p1_win_probability'] = 0.0
        df['correct_prediction'] = False
        df['elo_diff'] = 0.0
        df['elo_surface_diff'] = 0.0
        
        # Estructuras para resultados
        results = {
            'total_matches': len(df),
            'correct_predictions': 0,
            'accuracy': 0.0,
            'accuracy_by_surface': {},
            'accuracy_by_threshold': {},
            'accuracy_by_tourney_level': {},
            'log_loss': 0.0,
            'brier_score': 0.0,
            'calibration': {},
            'confidence': {},
            'feature_importance': {}
        }
        
        # Preparar variables para métricas de calibración
        all_probs = []
        all_outcomes = []
        
        # Si hay fecha, ordenar por fecha
        if 'match_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['match_date']):
                df['match_date'] = pd.to_datetime(df['match_date'])
            df = df.sort_values('match_date')
        
        # Contadores adicionales
        surface_correct = defaultdict(int)
        surface_total = defaultdict(int)
        
        tourney_correct = defaultdict(int)
        tourney_total = defaultdict(int)
        
        threshold_correct = defaultdict(int)
        threshold_total = defaultdict(int)
        
        confidence_bins = {
            '50-60%': {'correct': 0, 'total': 0},
            '60-70%': {'correct': 0, 'total': 0},
            '70-80%': {'correct': 0, 'total': 0},
            '80-90%': {'correct': 0, 'total': 0},
            '90-100%': {'correct': 0, 'total': 0}
        }
        
        # Variables para análisis de calibración
        bin_edges = np.linspace(0, 1, 11)  # 10 bins
        calibration_counts = np.zeros(10)
        calibration_correct = np.zeros(10)
        
        # Iterar por los partidos
        logger.info("Realizando predicciones...")
        
        for idx, match in tqdm(df.iterrows(), total=len(df), desc="Evaluando predicciones"):
            # Obtener IDs
            winner_id = str(match['winner_id'])
            loser_id = str(match['loser_id'])
            surface = self._normalize_surface(match['surface'])
            
            tourney_level = 'unknown'
            if 'tourney_level' in match:
                tourney_level = self._normalize_tournament_level(match['tourney_level'])
            
            # Realizar predicción con nuestro sistema actual
            prediction = self.predict_match(winner_id, loser_id, surface)
            
            # Extraer probabilidad de victoria
            p1_win_prob = prediction['prediction']['p1_win_probability']
            
            # Guardar resultados
            df.at[idx, 'p1_win_probability'] = p1_win_prob
            df.at[idx, 'elo_diff'] = prediction['player1']['elo_general'] - prediction['player2']['elo_general']
            
            if 'elo_surface' in prediction['player1'] and 'elo_surface' in prediction['player2']:
                df.at[idx, 'elo_surface_diff'] = prediction['player1']['elo_surface'] - prediction['player2']['elo_surface']
            
            # Determinar predicción
            predicted_winner = winner_id if p1_win_prob >= probability_threshold else loser_id
            df.at[idx, 'predicted_winner_id'] = predicted_winner
            
            # Verificar si la predicción fue correcta
            correct = predicted_winner == winner_id
            df.at[idx, 'correct_prediction'] = correct
            
            # Actualizar contadores generales
            if correct:
                results['correct_predictions'] += 1
            
            # Actualizar contadores por superficie
            surface_total[surface] += 1
            if correct:
                surface_correct[surface] += 1
            
            # Actualizar contadores por nivel de torneo
            tourney_total[tourney_level] += 1
            if correct:
                tourney_correct[tourney_level] += 1
            
            # Actualizar contadores por umbrales
            for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                pred_at_threshold = winner_id if p1_win_prob >= threshold else loser_id
                is_correct = pred_at_threshold == winner_id
                
                # Solo considerar partidos donde la probabilidad supera el umbral
                if max(p1_win_prob, 1-p1_win_prob) >= threshold:
                    threshold_total[threshold] += 1
                    if is_correct:
                        threshold_correct[threshold] += 1
            
            # Actualizar bins de confianza
            confidence = max(p1_win_prob, 1-p1_win_prob)
            
            if 0.5 <= confidence < 0.6:
                bin_key = '50-60%'
            elif 0.6 <= confidence < 0.7:
                bin_key = '60-70%'
            elif 0.7 <= confidence < 0.8:
                bin_key = '70-80%'
            elif 0.8 <= confidence < 0.9:
                bin_key = '80-90%'
            else:  # >= 0.9
                bin_key = '90-100%'
            
            confidence_bins[bin_key]['total'] += 1
            if correct:
                confidence_bins[bin_key]['correct'] += 1
            
            # Registrar para calibración
            all_probs.append(p1_win_prob)
            all_outcomes.append(1 if winner_id == prediction['player1']['id'] else 0)
            
            # Actualizar bins de calibración
            bin_idx = min(9, int(p1_win_prob * 10))
            calibration_counts[bin_idx] += 1
            if winner_id == prediction['player1']['id']:  # Si P1 ganó
                calibration_correct[bin_idx] += 1
        
        # Calcular resultados finales
        results['accuracy'] = results['correct_predictions'] / results['total_matches']
        
        # Accuracy por superficie
        for surface, count in surface_total.items():
            if count > 0:
                results['accuracy_by_surface'][surface] = surface_correct[surface] / count
        
        # Accuracy por nivel de torneo
        for level, count in tourney_total.items():
            if count > 0:
                results['accuracy_by_tourney_level'][level] = tourney_correct[level] / count
        
        # Accuracy por umbral
        for threshold, count in threshold_total.items():
            if count > 0:
                results['accuracy_by_threshold'][str(threshold)] = {
                    'accuracy': threshold_correct[threshold] / count,
                    'coverage': count / results['total_matches']
                }
        
        # Accuracy por nivel de confianza
        for bin_key, data in confidence_bins.items():
            if data['total'] > 0:
                results['confidence'][bin_key] = {
                    'accuracy': data['correct'] / data['total'],
                    'count': data['total'],
                    'percentage': data['total'] / results['total_matches'] * 100
                }
        
        # Calibración (qué tan bien calibradas están las probabilidades)
        calibration_result = {}
        for i in range(len(bin_edges) - 1):
            bin_name = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
            if calibration_counts[i] > 0:
                empirical_prob = calibration_correct[i] / calibration_counts[i]
                calibration_result[bin_name] = {
                    'predicted_probability': (bin_edges[i] + bin_edges[i+1]) / 2,
                    'empirical_probability': empirical_prob,
                    'count': int(calibration_counts[i]),
                    'error': empirical_prob - (bin_edges[i] + bin_edges[i+1]) / 2
                }
        
        results['calibration'] = calibration_result
        
        # Calcular métricas de evaluación adicionales
        try:
            results['log_loss'] = log_loss(all_outcomes, all_probs)
            results['brier_score'] = brier_score_loss(all_outcomes, all_probs)
        except Exception as e:
            logger.warning(f"Error calculando métricas avanzadas: {str(e)}")
        
        # Análisis de importancia de características si se proporcionan
        if features_to_use and all(feature in df.columns for feature in features_to_use):
            # Usar correlación con el resultado como indicador básico
            for feature in features_to_use:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    correlation = df[feature].corr(df['correct_prediction'])
                    results['feature_importance'][feature] = abs(correlation)
        
        # Actualizamos estadísticas del sistema
        self.stats['model_accuracy'] = results['accuracy']
        if 'brier_score' in results:
            self.stats['calibration_score'] = results['brier_score']
        
        # Retornar resultados y DataFrame con predicciones
        return results

    def create_features_for_model(self, matches_df: pd.DataFrame, 
                          include_context: bool = True,
                          include_temporal: bool = True,
                          include_h2h: bool = True,
                          include_surface: bool = True) -> pd.DataFrame:
        """
        Crea características basadas en ELO para entrenar modelos de ML.
        
        Args:
            matches_df: DataFrame con partidos a procesar
            include_context: Si debe incluir características contextuales
            include_temporal: Si debe incluir características temporales
            include_h2h: Si debe incluir características head-to-head
            include_surface: Si debe incluir características específicas por superficie
            
        Returns:
            DataFrame con características para aprendizaje automático
        """
        logger.info("Creando características para modelo de machine learning...")
        
        # Verificar que tenemos el mínimo de columnas necesarias
        required_columns = ['winner_id', 'loser_id', 'match_date', 'surface']
        
        if not all(col in matches_df.columns for col in required_columns):
            raise ValueError(f"DataFrame debe contener las columnas: {required_columns}")
        
        # Hacer copia para no modificar el original
        df = matches_df.copy()
        
        # Asegurar que match_date es datetime
        if not pd.api.types.is_datetime64_any_dtype(df['match_date']):
            df['match_date'] = pd.to_datetime(df['match_date'])
        
        # Características base: ELO general de ambos jugadores
        logger.info("Añadiendo características de ELO general...")
        
        # Creamos un diccionario para hacer seguimiento a los ratings para cada fecha
        temporal_ratings = {}
        temporal_ratings_surface = {}
        
        # Ordenamos los partidos por fecha
        df = df.sort_values('match_date')
        
        # Características básicas de ELO
        df['p1_elo'] = 0.0
        df['p2_elo'] = 0.0
        df['elo_diff'] = 0.0
        df['p1_win_probability'] = 0.0
        
        # Características específicas por superficie
        if include_surface:
            logger.info("Añadiendo características de ELO por superficie...")
            df['p1_elo_surface'] = 0.0
            df['p2_elo_surface'] = 0.0
            df['elo_surface_diff'] = 0.0
            df['p1_win_probability_surface'] = 0.0
        
        # Características contextuales
        if include_context:
            logger.info("Añadiendo características contextuales...")
            df['p1_uncertainty'] = 0.0
            df['p2_uncertainty'] = 0.0
            df['p1_form'] = 0.0
            df['p2_form'] = 0.0
            df['match_importance'] = 0.0
            
            # Codificar variables categóricas
            if 'tourney_level' in df.columns:
                # One-hot encoding para nivel de torneo
                df['tourney_level_normalized'] = df['tourney_level'].apply(self._normalize_tournament_level)
                tourney_dummies = pd.get_dummies(df['tourney_level_normalized'], prefix='tourney')
                df = pd.concat([df, tourney_dummies], axis=1)
            
            if 'round' in df.columns:
                # Mapa ordinal para rondas
                round_rank = {
                    'F': 7,       # Final
                    'SF': 6,      # Semifinal
                    'QF': 5,      # Cuartos de final
                    'R16': 4,     # Octavos
                    'R32': 3,     # 1/16
                    'R64': 2,     # 1/32
                    'R128': 1,    # 1/64
                    'RR': 4       # Round Robin (similar a octavos)
                }
                df['round_rank'] = df['round'].map(lambda x: round_rank.get(str(x), 0))
        
        # Características de head-to-head
        if include_h2h:
            logger.info("Añadiendo características head-to-head...")
            df['p1_h2h_wins'] = 0
            df['p2_h2h_wins'] = 0
            df['p1_h2h_ratio'] = 0.5
            df['h2h_factor'] = 1.0
            
            if include_surface:
                df['p1_h2h_surface_wins'] = 0
                df['p2_h2h_surface_wins'] = 0
                df['p1_h2h_surface_ratio'] = 0.5
        
        # Características temporales
        if include_temporal:
            logger.info("Añadiendo características temporales...")
            df['p1_days_since_last_match'] = 0
            df['p2_days_since_last_match'] = 0
            df['p1_matches_last_90days'] = 0
            df['p2_matches_last_90days'] = 0
            df['p1_win_ratio_last_90days'] = 0.5
            df['p2_win_ratio_last_90days'] = 0.5
            
            # Diccionarios para seguimiento temporal
            last_match_date = {}
            match_history_90days = defaultdict(list)
        
        # Iterar por los partidos cronológicamente
        logger.info(f"Procesando {len(df)} partidos para crear características...")
        
        for idx, match in tqdm(df.iterrows(), total=len(df), desc="Generando características"):
            match_date = match['match_date']
            p1_id = str(match['winner_id'])
            p2_id = str(match['loser_id'])
            surface = self._normalize_surface(match['surface'])
            
            # Obtener ratings desde diccionario temporal o usar el inicial
            p1_elo = temporal_ratings.get(p1_id, self.initial_rating)
            p2_elo = temporal_ratings.get(p2_id, self.initial_rating)
            
            # Guardar features de ELO general
            df.at[idx, 'p1_elo'] = p1_elo
            df.at[idx, 'p2_elo'] = p2_elo
            df.at[idx, 'elo_diff'] = p1_elo - p2_elo
            
            # Calcular probabilidad usando la fórmula ELO
            p1_win_prob = 1.0 / (1.0 + 10.0 ** ((p2_elo - p1_elo) / 400.0))
            df.at[idx, 'p1_win_probability'] = p1_win_prob
            
            # Características específicas por superficie
            if include_surface:
                # Inicializar ratings por superficie si no existen
                if p1_id not in temporal_ratings_surface:
                    temporal_ratings_surface[p1_id] = {}
                if p2_id not in temporal_ratings_surface:
                    temporal_ratings_surface[p2_id] = {}
                
                # Ratings por superficie, usar el general si no hay específico
                p1_elo_surface = temporal_ratings_surface[p1_id].get(surface, p1_elo)
                p2_elo_surface = temporal_ratings_surface[p2_id].get(surface, p2_elo)
                
                df.at[idx, 'p1_elo_surface'] = p1_elo_surface
                df.at[idx, 'p2_elo_surface'] = p2_elo_surface
                df.at[idx, 'elo_surface_diff'] = p1_elo_surface - p2_elo_surface
                
                # Probabilidad específica por superficie
                p1_win_prob_surface = 1.0 / (1.0 + 10.0 ** ((p2_elo_surface - p1_elo_surface) / 400.0))
                df.at[idx, 'p1_win_probability_surface'] = p1_win_prob_surface
            
            # Características contextuales adicionales
            if include_context:
                # Características de incertidumbre
                p1_matches = sum(1 for h in self.player_match_history.get(p1_id, []) 
                            if h['date'] < match_date)
                p2_matches = sum(1 for h in self.player_match_history.get(p2_id, []) 
                            if h['date'] < match_date)
                
                p1_uncertainty = 350 / (p1_matches + 5)
                p2_uncertainty = 350 / (p2_matches + 5)
                
                df.at[idx, 'p1_uncertainty'] = p1_uncertainty
                df.at[idx, 'p2_uncertainty'] = p2_uncertainty
                
                # Factores de forma
                # Calcular forma basada en partidos previos
                p1_recent_matches = [m for m in self.player_match_history.get(p1_id, []) 
                                if m['date'] < match_date]
                p2_recent_matches = [m for m in self.player_match_history.get(p2_id, []) 
                                if m['date'] < match_date]
                
                # Tomar los últimos N partidos
                window = self.form_window_matches
                p1_recent_form = [1 if m['result'] == 'win' else 0 
                            for m in sorted(p1_recent_matches, key=lambda x: x['date'], reverse=True)[:window]]
                p2_recent_form = [1 if m['result'] == 'win' else 0 
                            for m in sorted(p2_recent_matches, key=lambda x: x['date'], reverse=True)[:window]]
                
                # Calcular forma ponderada
                p1_form = 1.0  # Valor neutral
                p2_form = 1.0
                
                if p1_recent_form:
                    # Dar más peso a partidos recientes
                    weights = [1.5**i for i in range(len(p1_recent_form))]
                    p1_form = sum(f*w for f, w in zip(p1_recent_form, weights)) / sum(weights)
                    p1_form = 0.8 + (p1_form * 0.4)  # Mapear a rango 0.8-1.2
                
                if p2_recent_form:
                    weights = [1.5**i for i in range(len(p2_recent_form))]
                    p2_form = sum(f*w for f, w in zip(p2_recent_form, weights)) / sum(weights)
                    p2_form = 0.8 + (p2_form * 0.4)  # Mapear a rango 0.8-1.2
                
                df.at[idx, 'p1_form'] = p1_form
                df.at[idx, 'p2_form'] = p2_form
                
                # Importancia del partido
                if 'tourney_level' in df.columns and 'round' in df.columns:
                    tourney_level = match['tourney_level']
                    round_name = match['round']
                    
                    # Usar las funciones existentes para calcular importancia
                    match_importance = self._get_match_importance_factor(
                        tourney_level, round_name, p1_id, p2_id
                    )
                    df.at[idx, 'match_importance'] = match_importance
            
            # Características head-to-head
            if include_h2h:
                # Obtener historial previo a este partido
                p1_h2h_wins = sum(1 for m in self.player_match_history.get(p1_id, [])
                            if m['date'] < match_date and m['opponent_id'] == p2_id and m['result'] == 'win')
                
                p2_h2h_wins = sum(1 for m in self.player_match_history.get(p2_id, [])
                            if m['date'] < match_date and m['opponent_id'] == p1_id and m['result'] == 'win')
                
                df.at[idx, 'p1_h2h_wins'] = p1_h2h_wins
                df.at[idx, 'p2_h2h_wins'] = p2_h2h_wins
                
                total_h2h = p1_h2h_wins + p2_h2h_wins
                h2h_ratio = 0.5  # Valor neutral
                
                if total_h2h > 0:
                    h2h_ratio = p1_h2h_wins / total_h2h
                
                df.at[idx, 'p1_h2h_ratio'] = h2h_ratio
                
                # Factor de ventaja H2H
                max_factor = min(0.1, 0.05 + (total_h2h * 0.005))
                h2h_factor = 1.0 + ((h2h_ratio - 0.5) * 2 * max_factor)
                df.at[idx, 'h2h_factor'] = h2h_factor
                
                # H2H por superficie
                if include_surface:
                    p1_h2h_surface_wins = sum(1 for m in self.player_match_history.get(p1_id, [])
                                        if m['date'] < match_date and m['opponent_id'] == p2_id 
                                        and m['result'] == 'win' and m['surface'] == surface)
                    
                    p2_h2h_surface_wins = sum(1 for m in self.player_match_history.get(p2_id, [])
                                        if m['date'] < match_date and m['opponent_id'] == p1_id 
                                        and m['result'] == 'win' and m['surface'] == surface)
                    
                    df.at[idx, 'p1_h2h_surface_wins'] = p1_h2h_surface_wins
                    df.at[idx, 'p2_h2h_surface_wins'] = p2_h2h_surface_wins
                    
                    total_h2h_surface = p1_h2h_surface_wins + p2_h2h_surface_wins
                    h2h_surface_ratio = 0.5
                    
                    if total_h2h_surface > 0:
                        h2h_surface_ratio = p1_h2h_surface_wins / total_h2h_surface
                    
                    df.at[idx, 'p1_h2h_surface_ratio'] = h2h_surface_ratio
            
            # Características temporales
            if include_temporal:
                # Días desde último partido
                p1_last_date = last_match_date.get(p1_id)
                p2_last_date = last_match_date.get(p2_id)
                
                p1_days_since = 30  # Valor por defecto si no hay partidos previos
                p2_days_since = 30
                
                if p1_last_date:
                    p1_days_since = (match_date - p1_last_date).days
                
                if p2_last_date:
                    p2_days_since = (match_date - p2_last_date).days
                
                df.at[idx, 'p1_days_since_last_match'] = p1_days_since
                df.at[idx, 'p2_days_since_last_match'] = p2_days_since
                
                # Partidos en los últimos 90 días
                date_90days_ago = match_date - pd.Timedelta(days=90)
                
                # Limpiar historial antiguo
                for player_id in list(match_history_90days.keys()):
                    match_history_90days[player_id] = [
                        m for m in match_history_90days[player_id] 
                        if m['date'] >= date_90days_ago
                    ]
                
                # Contar partidos en los últimos 90 días
                p1_matches_90d = len(match_history_90days.get(p1_id, []))
                p2_matches_90d = len(match_history_90days.get(p2_id, []))
                
                df.at[idx, 'p1_matches_last_90days'] = p1_matches_90d
                df.at[idx, 'p2_matches_last_90days'] = p2_matches_90d
                
                # Calcular ratio de victorias en los últimos 90 días
                p1_wins_90d = sum(1 for m in match_history_90days.get(p1_id, []) if m['result'] == 'win')
                p2_wins_90d = sum(1 for m in match_history_90days.get(p2_id, []) if m['result'] == 'win')
                
                p1_win_ratio_90d = p1_wins_90d / max(1, p1_matches_90d)
                p2_win_ratio_90d = p2_wins_90d / max(1, p2_matches_90d)
                
                df.at[idx, 'p1_win_ratio_last_90days'] = p1_win_ratio_90d
                df.at[idx, 'p2_win_ratio_last_90days'] = p2_win_ratio_90d
            
            # Actualizar ratings para el próximo partido
            # Simulamos el cambio de ELO igual que en update_ratings pero de manera simplificada
            k_factor = self.k_factor_base
            
            # Probabilidad esperada (ya calculada antes)
            expected_prob = p1_win_prob
            
            # Resultado actual (sabemos que p1 ganó)
            actual_result = 1.0
            
            # Calcular cambios de ELO básicos
            p1_elo_change = k_factor * (actual_result - expected_prob)
            p2_elo_change = -k_factor * expected_prob
            
            # Actualizar ratings temporales
            temporal_ratings[p1_id] = p1_elo + p1_elo_change
            temporal_ratings[p2_id] = p2_elo + p2_elo_change
            
            # Actualizar ratings por superficie
            if include_surface:
                # Actualizar con mayor especificidad para superficie
                surface_mult = self.surface_specificity.get(surface, 1.0)
                
                if surface not in temporal_ratings_surface[p1_id]:
                    temporal_ratings_surface[p1_id][surface] = self.initial_rating
                if surface not in temporal_ratings_surface[p2_id]:
                    temporal_ratings_surface[p2_id][surface] = self.initial_rating
                
                p1_surface_rating = temporal_ratings_surface[p1_id][surface]
                p2_surface_rating = temporal_ratings_surface[p2_id][surface]
                
                # Usar probabilidad específica por superficie
                expected_prob_surface = p1_win_prob_surface
                
                p1_surface_change = k_factor * surface_mult * (actual_result - expected_prob_surface)
                p2_surface_change = -k_factor * surface_mult * expected_prob_surface
                
                temporal_ratings_surface[p1_id][surface] = p1_surface_rating + p1_surface_change
                temporal_ratings_surface[p2_id][surface] = p2_surface_rating + p2_surface_change
            
            # Actualizar tracking temporal
            if include_temporal:
                last_match_date[p1_id] = match_date
                last_match_date[p2_id] = match_date
                
                # Añadir al historial de 90 días
                match_history_90days[p1_id].append({
                    'date': match_date,
                    'result': 'win',
                    'opponent_id': p2_id
                })
                
                match_history_90days[p2_id].append({
                    'date': match_date,
                    'result': 'loss',
                    'opponent_id': p1_id
                })
        
        # Añadir variable objetivo para modelos de ML
        df['target'] = 1  # El ganador siempre está en la columna p1
        
        # Procesar variables categóricas restantes
        if 'surface' in df.columns:
            # One-hot encoding para superficie
            df['surface_normalized'] = df['surface'].apply(self._normalize_surface)
            surface_dummies = pd.get_dummies(df['surface_normalized'], prefix='surface')
            df = pd.concat([df, surface_dummies], axis=1)
        
        logger.info(f"Generadas {len(df.columns)} características para modelo de ML")
        
        return df

    def plot_rating_distribution(self, surface: Optional[str] = None, 
                            min_matches: int = 10, save_path: Optional[str] = None,
                            top_players: int = 10,
                            figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Genera un gráfico con la distribución de ratings ELO.
        
        Args:
            surface: Superficie específica (opcional)
            min_matches: Número mínimo de partidos jugados
            save_path: Ruta para guardar el gráfico (opcional)
            top_players: Número de mejores jugadores a destacar en el gráfico
            figsize: Tamaño de la figura (ancho, alto) en pulgadas
        """
        # Obtener datos de distribución
        dist_data = self.analyze_ratings_distribution(surface, min_matches)
        
        if dist_data['count'] == 0:
            logger.warning("No hay suficientes datos para graficar distribución")
            return
        
        # Crear figura compuesta de dos gráficos
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # 1. Histograma con distribución
        bin_edges = dist_data['distribution']['bin_edges']
        bin_counts = dist_data['distribution']['bin_counts']
        
        ax1.bar(dist_data['distribution']['bin_centers'], bin_counts, 
                width=bin_edges[1]-bin_edges[0], alpha=0.7, color='royalblue',
                edgecolor='navy')
        
        # Añadir curva normal
        if dist_data['count'] > 30:
            x = np.linspace(dist_data['min'] - 50, dist_data['max'] + 50, 100)
            pdf = stats.norm.pdf(x, dist_data['mean'], dist_data['std'])
            pdf_scaled = pdf * sum(bin_counts) * (bin_edges[1]-bin_edges[0])
            ax1.plot(x, pdf_scaled, 'r-', linewidth=2, alpha=0.7, label='Distribución normal')
        
        # Añadir líneas verticales para percentiles
        percentiles = dist_data['percentiles']
        ax1.axvline(x=percentiles['25'], color='gray', linestyle='--', alpha=0.7, 
                label=f"Percentil 25: {percentiles['25']:.0f}")
        ax1.axvline(x=percentiles['50'], color='black', linestyle='--', alpha=0.7, 
                label=f"Mediana: {percentiles['50']:.0f}")
        ax1.axvline(x=percentiles['75'], color='gray', linestyle='--', alpha=0.7, 
                label=f"Percentil 75: {percentiles['75']:.0f}")
        
        # Ajustes al gráfico principal
        if surface:
            ax1.set_title(f'Distribución de ratings ELO en {surface.upper()} (N={dist_data["count"]})', fontsize=16)
        else:
            ax1.set_title(f'Distribución de ratings ELO (N={dist_data["count"]})', fontsize=16)
        
        ax1.set_xlabel('Rating ELO', fontsize=12)
        ax1.set_ylabel('Número de jugadores', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Gráfico inferior con top jugadores
        top_players_df = self.get_top_players(n=top_players, surface=surface, min_matches=min_matches, include_details=True)
        
        if not top_players_df.empty:
            # Ordenar de mejor a peor para el gráfico
            top_players_df = top_players_df.sort_values('elo_rating', ascending=True)
            
            # Crear nombres legibles para etiquetas
            if 'player_name' in top_players_df.columns:
                labels = top_players_df['player_name']
            else:
                labels = top_players_df['player_id'].apply(self.get_player_name)
            
            # Crear gráfico de barras horizontales
            bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_players_df)))
            
            # Gráfico de barras con top jugadores
            bars = ax2.barh(labels, top_players_df['elo_rating'], color=bar_colors, alpha=0.7)
            
            # Añadir valores numéricos a las barras
            for i, bar in enumerate(bars):
                ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                        f"{top_players_df['elo_rating'].iloc[i]:.0f}", 
                        va='center', fontsize=9)
            
            # Ajustes al gráfico de top jugadores
            ax2.set_title(f'Top {top_players} jugadores por rating ELO', fontsize=14)
            ax2.set_xlabel('Rating ELO', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Mantener el mismo rango de X que el histograma principal
            ax2.set_xlim(ax1.get_xlim())
        
        plt.tight_layout()
        
        # Guardar o mostrar
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico de distribución guardado en {save_path}")
        else:
            plt.show()
    
    def plot_top_players_history(self, player_ids: List[str], start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None, surface: Optional[str] = None,
                           save_path: Optional[str] = None, player_names: Optional[Dict[str, str]] = None,
                           title: Optional[str] = None, figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        Genera un gráfico con la evolución de ratings de los jugadores seleccionados.
        
        Args:
            player_ids: Lista de IDs de jugadores a incluir
            start_date: Fecha de inicio para el gráfico
            end_date: Fecha final para el gráfico
            surface: Superficie específica (opcional)
            save_path: Ruta para guardar el gráfico (opcional)
            player_names: Diccionario de {player_id: nombre} para etiquetas (opcional)
            title: Título personalizado para el gráfico (opcional)
            figsize: Tamaño de la figura (ancho, alto) en pulgadas
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
        plt.figure(figsize=figsize)
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # Paleta de colores más distintiva
        colors = plt.cm.tab10.colors
        
        # Obtener nombres de jugadores si no se proporcionan
        if player_names is None:
            player_names = {
                player_id: self.get_player_name(player_id)
                for player_id in player_ids
            }
        
        # Graficar una línea por jugador
        for i, player_id in enumerate(player_ids):
            player_id = str(player_id)
            player_data = all_data[all_data['player_id'] == player_id]
            
            if not player_data.empty:
                # Nombre para la leyenda
                label = player_names.get(player_id, f'Jugador {player_id}')
                
                # Añadir línea principal
                plt.plot(player_data['date'], player_data['rating'], '-o', 
                        label=label, color=colors[i % len(colors)], 
                        markersize=4, alpha=0.8)
                
                # Añadir línea de tendencia suavizada
                if len(player_data) >= 5:
                    # Ordenar por fecha
                    player_data = player_data.sort_values('date')
                    
                    # Usar rolling mean para suavizar
                    window = min(5, len(player_data) // 2)
                    if window > 1:
                        player_data['smooth_rating'] = player_data['rating'].rolling(window=window, center=True).mean()
                        plt.plot(player_data['date'], player_data['smooth_rating'], '-', 
                                color=colors[i % len(colors)], alpha=0.5, linewidth=2)
        
        # Añadir título y etiquetas
        if title:
            plt.title(title, fontsize=16)
        elif surface:
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
        
        # Añadir líneas de referencia de percentiles
        if not surface:
            distribution = self.analyze_ratings_distribution(min_matches=10)
            if distribution['count'] > 10:
                plt.axhline(y=distribution['percentiles']['50'], color='gray', linestyle='--', alpha=0.5, 
                            label=f"Mediana ({distribution['percentiles']['50']:.0f})")
                plt.axhline(y=distribution['percentiles']['75'], color='gray', linestyle=':', alpha=0.3, 
                            label=f"Percentil 75 ({distribution['percentiles']['75']:.0f})")
                plt.axhline(y=distribution['percentiles']['25'], color='gray', linestyle=':', alpha=0.3, 
                            label=f"Percentil 25 ({distribution['percentiles']['25']:.0f})")
                
                # Actualizar leyenda
                plt.legend(fontsize=9)
        
        plt.tight_layout()
        
        # Guardar o mostrar
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en {save_path}")
        else:
            plt.show()

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
            
            # Filtrar jugadores con pocos partidos
            filtered_ratings = {
                player_id: rating 
                for player_id, rating in ratings_dict.items()
                if self.player_match_count_by_surface[surface].get(player_id, 0) >= min_matches
            }
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
        
        # Cálculos estadísticos básicos
        stats = {
            'count': len(ratings),
            'mean': float(np.mean(ratings_array)),
            'std': float(np.std(ratings_array)),
            'min': float(np.min(ratings_array)),
            'max': float(np.max(ratings_array)),
            'percentiles': {
                '5': float(np.percentile(ratings_array, 5)),
                '25': float(np.percentile(ratings_array, 25)),
                '50': float(np.percentile(ratings_array, 50)),  # Mediana
                '75': float(np.percentile(ratings_array, 75)),
                '95': float(np.percentile(ratings_array, 95))
            }
        }
        
        # Añadir distribución por cuartiles para visualización
        q_values = []
        bin_edges = []
        
        # Crear 20 bins para visualizar distribución
        hist, edges = np.histogram(ratings_array, bins=20)
        
        # Convertir a porcentajes
        total = hist.sum()
        percentages = (hist / total * 100).tolist()
        
        # Añadir al resultado
        stats['distribution'] = {
            'bin_counts': hist.tolist(),
            'bin_percentages': percentages,
            'bin_edges': edges.tolist(),
            'bin_centers': [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
        }
        
        # Verificar normalidad (para análisis)
        normality_test = stats.normaltest(ratings_array)
        stats['normality'] = {
            'statistic': float(normality_test.statistic),
            'p_value': float(normality_test.pvalue),
            'is_normal': float(normality_test.pvalue) > 0.05
        }
        
        return stats
    
    def get_player_details(self, player_id: str) -> Dict:
        """
        Obtiene información detallada del jugador incluyendo ratings,
        estadísticas, forma reciente y más.
        
        Args:
            player_id: ID del jugador
            
        Returns:
            Diccionario con toda la información disponible del jugador
        """
        player_id = str(player_id)
        
        # Información básica y nombre
        player_name = self.get_player_name(player_id)
        
        # Información básica de ELO
        elo_general = self.get_player_rating(player_id)
        
        # ELO por superficie
        elo_by_surface = {
            surface: self.get_player_rating(player_id, surface)
            for surface in self.player_ratings_by_surface.keys()
        }
        
        # Encuentros por superficie
        matches_by_surface = {
            surface: self.player_match_count_by_surface[surface].get(player_id, 0)
            for surface in self.player_match_count_by_surface.keys()
        }
        
        # Forma reciente
        recent_form = self.get_player_form(player_id)
        
        # Historial de partidos
        match_history = self.player_match_history.get(player_id, [])
        
        # Extraer estadísticas del historial de partidos
        total_matches = self.player_match_count.get(player_id, 0)
        wins = len([m for m in match_history if m['result'] == 'win'])
        losses = total_matches - wins
        
        # Estadísticas por superficie
        surface_stats = {}
        for surface in self.player_ratings_by_surface.keys():
            surface_matches = [m for m in match_history if m['surface'] == surface]
            surface_wins = len([m for m in surface_matches if m['result'] == 'win'])
            surface_losses = len(surface_matches) - surface_wins
            
            surface_stats[surface] = {
                'matches': len(surface_matches),
                'wins': surface_wins,
                'losses': surface_losses,
                'win_rate': surface_wins / max(1, len(surface_matches))
            }
        
        # Extraer información de rendimiento por tipo de torneo
        tourney_stats = {}
        for match in match_history:
            level = match.get('tourney_level', 'unknown')
            if level not in tourney_stats:
                tourney_stats[level] = {'wins': 0, 'losses': 0}
            
            if match['result'] == 'win':
                tourney_stats[level]['wins'] += 1
            else:
                tourney_stats[level]['losses'] += 1
        
        # Calcular win rate por tipo de torneo
        for level in tourney_stats:
            stats = tourney_stats[level]
            total = stats['wins'] + stats['losses']
            stats['matches'] = total
            stats['win_rate'] = stats['wins'] / max(1, total)
        
        # Información de incertidumbre
        uncertainty = self.get_player_uncertainty(player_id)
        
        # Información de rivales
        rivals = {}
        for opponent_id in self.h2h_records.get(player_id, {}):
            h2h = self.h2h_records[player_id][opponent_id]
            if h2h['wins'] > 0 or h2h['losses'] > 0:
                opponent_name = self.get_player_name(opponent_id)
                rivals[opponent_id] = {
                    'name': opponent_name,
                    'wins': h2h['wins'],
                    'losses': h2h['losses'],
                    'total': h2h['wins'] + h2h['losses'],
                    'win_rate': h2h['wins'] / max(1, h2h['wins'] + h2h['losses'])
                }
        
        # Organizar toda la información
        return {
            'id': player_id,
            'name': player_name,
            'elo': {
                'general': elo_general,
                'by_surface': elo_by_surface,
                'uncertainty': uncertainty
            },
            'stats': {
                'total_matches': total_matches,
                'wins': wins,
                'losses': losses,
                'win_rate': wins / max(1, total_matches),
                'by_surface': surface_stats,
                'by_tourney_level': tourney_stats
            },
            'form': recent_form,
            'match_count': self.player_match_count.get(player_id, 0),
            'matches_by_surface': matches_by_surface,
            'last_match': self.player_last_match.get(player_id),
            'rivals': rivals,
            'recent_matches': sorted(match_history, key=lambda x: x['date'], reverse=True)[:10] if match_history else []
        }

# Aplicar rastreadores de errores
apply_tracers()
    
def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    Versión corregida para manejar fechas y tipos de datos mezclados.
    """
    parser = argparse.ArgumentParser(description='Sistema ELO avanzado para tenis')
    
    # Argumentos generales
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directorio con datos de tenis')
    parser.add_argument('--output-dir', type=str, default='data/processed/elo',
                        help='Directorio para guardar resultados')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Nivel de logging')
    
    # Argumentos de operación
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument('--process', action='store_true',
                            help='Procesar partidos y calcular ELO')
    operation_group.add_argument('--evaluate', action='store_true',
                            help='Evaluar sistema ELO existente')
    operation_group.add_argument('--visualize', action='store_true',
                            help='Generar visualizaciones')
    operation_group.add_argument('--run-pipeline', action='store_true',
                            help='Ejecutar pipeline completo')
    operation_group.add_argument('--load', action='store_true',
                            help='Cargar sistema ELO existente')
    operation_group.add_argument('--get-top', action='store_true',
                            help='Obtener top jugadores')
    
    # Argumentos para procesar
    parser.add_argument('--atp-matches', type=str,
                        help='Ruta al archivo de partidos ATP')
    parser.add_argument('--wta-matches', type=str,
                        help='Ruta al archivo de partidos WTA')
    parser.add_argument('--min-year', type=int, default=2000,
                        help='Año mínimo para filtrar datos')
    parser.add_argument('--k-factor', type=float, default=32,
                        help='Factor K base para el sistema ELO')
    parser.add_argument('--initial-rating', type=float, default=1500,
                        help='Rating ELO inicial para jugadores nuevos')
    
    # Argumentos para visualización
    parser.add_argument('--surface', type=str, choices=['hard', 'clay', 'grass', 'carpet'],
                        help='Superficie específica para análisis o visualización')
    parser.add_argument('--players', type=str, nargs='+',
                        help='IDs de jugadores para visualización o análisis')
    parser.add_argument('--vis-output', type=str,
                        help='Ruta para guardar visualización')
    parser.add_argument('--num-players', type=int, default=10,
                        help='Número de jugadores a mostrar en rankings o visualizaciones')
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Nivel de log inválido: {args.log_level}')
    
    logging.getLogger().setLevel(numeric_level)
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Inicializar procesador ELO
    elo_processor = EnhancedTennisEloProcessor(
        k_factor_base=args.k_factor,
        initial_rating=args.initial_rating,
        data_dir=args.data_dir
    )
    
    # Ejecutar la operación seleccionada
    if args.process:
        # Procesar partidos y calcular ELO
        if args.atp_matches is None and args.wta_matches is None:
            logger.error("Debe proporcionar al menos una ruta a partidos ATP o WTA")
            return 1
        
        # Cargar datos
        matches_df = None
        
        if args.atp_matches:
            logger.info(f"Cargando partidos ATP desde {args.atp_matches}...")
            try:
                # Usar low_memory=False para manejar columnas con tipos mixtos
                atp_df = pd.read_csv(args.atp_matches, low_memory=False)
                
                # Si hay columnas date, asegurar que son datetime
                date_cols = ['match_date', 'tourney_date', 'date']
                for col in date_cols:
                    if col in atp_df.columns:
                        try:
                            # Convertir a string primero para manejar tipos mixtos
                            atp_df[col] = atp_df[col].astype(str)
                            # Luego convertir a datetime ignorando errores
                            atp_df[col] = pd.to_datetime(atp_df[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Error convirtiendo columna {col}: {str(e)}")
                
                if matches_df is None:
                    matches_df = atp_df
                else:
                    # Verificar columnas comunes
                    common_cols = list(set(matches_df.columns) & set(atp_df.columns))
                    if common_cols:
                        matches_df = pd.concat([matches_df[common_cols], atp_df[common_cols]])
            except Exception as e:
                logger.error(f"Error cargando partidos ATP: {str(e)}")
        
        if args.wta_matches:
            logger.info(f"Cargando partidos WTA desde {args.wta_matches}...")
            try:
                # Usar low_memory=False para manejar columnas con tipos mixtos
                wta_df = pd.read_csv(args.wta_matches, low_memory=False)
                
                # Si hay columnas date, asegurar que son datetime
                date_cols = ['match_date', 'tourney_date', 'date']
                for col in date_cols:
                    if col in wta_df.columns:
                        try:
                            # Convertir a string primero para manejar tipos mixtos
                            wta_df[col] = wta_df[col].astype(str)
                            # Luego convertir a datetime ignorando errores
                            wta_df[col] = pd.to_datetime(wta_df[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Error convirtiendo columna {col}: {str(e)}")
                
                if matches_df is None:
                    matches_df = wta_df
                else:
                    # Verificar columnas comunes
                    common_cols = list(set(matches_df.columns) & set(wta_df.columns))
                    if common_cols:
                        matches_df = pd.concat([matches_df[common_cols], wta_df[common_cols]])
            except Exception as e:
                logger.error(f"Error cargando partidos WTA: {str(e)}")
        
        # Manejar las columnas requeridas
        required_cols = ['winner_id', 'loser_id', 'match_date', 'surface']
        
        # Mapear columnas comunes si es necesario
        column_map = {
            'match_date': ['date', 'tourney_date'],
            'winner_id': ['player1_id', 'p1_id', 'w_id'],
            'loser_id': ['player2_id', 'p2_id', 'l_id'],
            'surface': ['surface_normalized', 'court_surface'],
        }
        
        # Intentar mapear columnas
        for target, alternatives in column_map.items():
            if target not in matches_df.columns:
                for alt in alternatives:
                    if alt in matches_df.columns:
                        logger.info(f"Mapeando '{alt}' a '{target}'")
                        matches_df[target] = matches_df[alt]
                        break
        
        # Verificar si ahora tenemos las columnas necesarias
        for col in required_cols:
            if col not in matches_df.columns:
                logger.error(f"Columna requerida '{col}' no encontrada en los datos")
                if col == 'match_date' and ('tourney_date' in matches_df.columns or 'date' in matches_df.columns):
                    alt_col = 'tourney_date' if 'tourney_date' in matches_df.columns else 'date'
                    logger.info(f"Mapeando '{alt_col}' a 'match_date'")
                    matches_df['match_date'] = matches_df[alt_col]
                elif col == 'surface' and any(c in matches_df.columns for c in ['court', 'court_surface']):
                    alt_col = [c for c in ['court', 'court_surface'] if c in matches_df.columns][0]
                    logger.info(f"Mapeando '{alt_col}' a 'surface'")
                    matches_df['surface'] = matches_df[alt_col]
                else:
                    return 1
        
        # Asegurar que match_date es datetime
        if not pd.api.types.is_datetime64_any_dtype(matches_df['match_date']):
            logger.info("Convirtiendo 'match_date' a datetime")
            try:
                # Primero a string para manejar tipos mixtos
                matches_df['match_date'] = matches_df['match_date'].astype(str)
                # Luego a datetime
                matches_df['match_date'] = pd.to_datetime(matches_df['match_date'], errors='coerce')
                
                # Eliminar filas con fechas inválidas
                invalid_dates = matches_df['match_date'].isna()
                if invalid_dates.any():
                    logger.warning(f"Eliminando {invalid_dates.sum()} filas con fechas inválidas")
                    matches_df = matches_df.dropna(subset=['match_date'])
            except Exception as e:
                logger.error(f"Error convirtiendo fechas: {str(e)}")
                return 1
        
        # Filtrar por año
        matches_df = matches_df[matches_df['match_date'].dt.year >= args.min_year]
        
        # Convertir IDs a string
        for id_col in ['winner_id', 'loser_id']:
            if id_col in matches_df.columns:
                matches_df[id_col] = matches_df[id_col].astype(str)
        
        # Asegurar la existencia de otras columnas importantes
        if 'tourney_level' not in matches_df.columns:
            logger.warning("Columna 'tourney_level' no encontrada, usando valor predeterminado")
            matches_df['tourney_level'] = 'O'  # Other como valor por defecto
        
        if 'round' not in matches_df.columns:
            logger.warning("Columna 'round' no encontrada, usando valor predeterminado")
            matches_df['round'] = 'R32'  # Round of 32 como valor por defecto
        
        if 'score' not in matches_df.columns:
            logger.warning("Columna 'score' no encontrada, usando valor predeterminado")
            matches_df['score'] = ''  # String vacío como valor por defecto
        
        # Procesar partidos
        logger.info(f"Procesando {len(matches_df)} partidos...")
        try:
            processed_df = elo_processor.process_matches_dataframe(matches_df)
            
            # Guardar resultados
            elo_processor.save_ratings(output_dir=args.output_dir)
            
            # Guardar DataFrame procesado
            processed_path = os.path.join(args.output_dir, "processed_matches.csv")
            processed_df.to_csv(processed_path, index=False)
            
            logger.info(f"Procesamiento completado. Resultados guardados en {args.output_dir}")
        except Exception as e:
            logger.error(f"Error durante el procesamiento: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
    
    elif args.evaluate:
        # Evaluar sistema ELO existente
        # Primero cargar si no se ha procesado antes
        if not elo_processor.player_ratings:
            logger.info("Cargando sistema ELO existente...")
            elo_processor.load_ratings(input_dir=args.output_dir)
        
        # Cargar datos para evaluación
        test_data = None
        
        if args.atp_matches:
            try:
                test_data = pd.read_csv(args.atp_matches, low_memory=False)
            except Exception as e:
                logger.error(f"Error cargando datos de evaluación: {str(e)}")
                return 1
        elif args.wta_matches:
            try:
                test_data = pd.read_csv(args.wta_matches, low_memory=False)
            except Exception as e:
                logger.error(f"Error cargando datos de evaluación: {str(e)}")
                return 1
        else:
            # Buscar datos procesados
            processed_path = os.path.join(args.output_dir, "processed_matches.csv")
            if os.path.exists(processed_path):
                try:
                    test_data = pd.read_csv(processed_path, low_memory=False)
                    # Usar solo los últimos 10% para test
                    total_rows = len(test_data)
                    test_data = test_data.iloc[-int(total_rows * 0.1):]
                except Exception as e:
                    logger.error(f"Error cargando datos procesados: {str(e)}")
                    return 1
            else:
                logger.error("No se encontraron datos para evaluación")
                return 1
        
        # Realizar evaluación
        try:
            results = elo_processor.evaluate_predictive_power(test_data)
            
            # Guardar resultados
            eval_path = os.path.join(args.output_dir, "evaluation_results.json")
            
            # Convertir a formato serializable (manejo de fechas y otros tipos no serializables)
            results_serializable = {}
            for key, value in results.items():
                # Convertir dicts anidados
                if isinstance(value, dict):
                    results_serializable[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (datetime, pd.Timestamp)):
                            results_serializable[key][k] = v.strftime('%Y-%m-%d')
                        elif isinstance(v, np.integer):
                            results_serializable[key][k] = int(v)
                        elif isinstance(v, np.floating):
                            results_serializable[key][k] = float(v)
                        else:
                            results_serializable[key][k] = v
                # Convertir valores simples
                elif isinstance(value, (datetime, pd.Timestamp)):
                    results_serializable[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, np.integer):
                    results_serializable[key] = int(value)
                elif isinstance(value, np.floating):
                    results_serializable[key] = float(value)
                else:
                    results_serializable[key] = value
            
            with open(eval_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            # Mostrar resultados principales
            logger.info(f"Evaluación completada: {results['total_matches']} partidos")
            logger.info(f"Accuracy: {results['accuracy']:.4f}")
            
            if 'accuracy_by_surface' in results:
                logger.info("Accuracy por superficie:")
                for surface, acc in results['accuracy_by_surface'].items():
                    logger.info(f"  {surface}: {acc:.4f}")
            
            logger.info(f"Resultados completos guardados en {eval_path}")
        except Exception as e:
            logger.error(f"Error durante la evaluación: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
    
    elif args.visualize:
        # Implementación para visualización
        pass
    
    elif args.get_top:
        # Implementación para obtener top jugadores
        pass
    
    elif args.load:
        # Implementación para cargar sistema existente
        pass
    
    elif args.run_pipeline:
        # Implementación para ejecutar pipeline completo
        pass
    
    # Guardar informe de errores si se ha procesado algo
    if args.process or args.evaluate or args.run_pipeline:
        save_error_report()

    return 0

if __name__ == "__main__":
    sys.exit(main())