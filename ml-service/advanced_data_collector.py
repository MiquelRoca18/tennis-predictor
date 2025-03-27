"""
advanced_data_collector.py

Sistema avanzado de recopilación de datos de tenis con múltiples fuentes,
implementando un sistema de fallback entre APIs de pago y gratuitas.
"""

import os
import logging
import pandas as pd
import numpy as np
import requests
import json
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
import random
import aiohttp
import asyncio
from data_collector import TennisDataCollector
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
from io import StringIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
import sys

# Crear directorio de logs si no existe
os.makedirs('logs', exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/tennis_ml.log'),
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTennisScraper:
    """
    Clase para realizar scraping avanzado de datos de tenis.
    Implementa las recomendaciones de la guía de mejora para recopilación de datos.
    """
    
    def __init__(self, user_agent: str, cache_dir: Path):
        """
        Inicializa el scraper avanzado.
        
        Args:
            user_agent: User Agent para peticiones HTTP
            cache_dir: Directorio para almacenar datos en caché
        """
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
        # Configurar opciones de Chrome para Selenium
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument(f'user-agent={self.user_agent}')
        
        # URLs de fuentes de datos recomendadas
        self.data_sources = {
            'tennis_data_co_uk': 'http://www.tennis-data.co.uk/',
            'ultimatetennis': 'https://www.ultimatetennisstatistics.com/',
            'jeff_sackmann': 'https://github.com/JeffSackmann'
        }
        
        # Crear directorios de caché específicos
        self.match_cache_dir = self.cache_dir / 'matches'
        self.player_cache_dir = self.cache_dir / 'players'
        self.tournament_cache_dir = self.cache_dir / 'tournaments'
        
        for dir_path in [self.match_cache_dir, self.player_cache_dir, self.tournament_cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def scrape_detailed_match_stats(self, match_url: str) -> Dict:
        """
        Extrae estadísticas detalladas punto por punto de un partido.
        
        Args:
            match_url: URL del partido
            
        Returns:
            Diccionario con estadísticas detalladas
        """
        try:
            # Verificar caché primero
            cache_file = self.match_cache_dir / f"{hash(match_url)}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())
            
            # Inicializar el driver de Selenium para contenido dinámico
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(match_url)
            
            # Esperar a que cargue el contenido principal
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "match-stats"))
            )
            
            # Extraer estadísticas básicas
            stats = {
                'point_by_point': [],
                'serve_stats': {},
                'rally_stats': {},
                'momentum_changes': [],
                'basic_stats': {}
            }
            
            # Extraer estadísticas de servicio
            serve_stats = driver.find_elements(By.CLASS_NAME, "serve-stats")
            if serve_stats:
                stats['serve_stats'] = {
                    'first_serve_percentage': self._extract_percentage(serve_stats[0], "first-serve"),
                    'first_serve_points_won': self._extract_percentage(serve_stats[0], "first-serve-points"),
                    'second_serve_points_won': self._extract_percentage(serve_stats[0], "second-serve-points"),
                    'aces': self._extract_number(serve_stats[0], "aces"),
                    'double_faults': self._extract_number(serve_stats[0], "double-faults")
                }
            
            # Extraer estadísticas de rallies
            rally_stats = driver.find_elements(By.CLASS_NAME, "rally-stats")
            if rally_stats:
                stats['rally_stats'] = {
                    'avg_rally_length': self._extract_number(rally_stats[0], "avg-rally"),
                    'winners': self._extract_number(rally_stats[0], "winners"),
                    'unforced_errors': self._extract_number(rally_stats[0], "unforced-errors")
                }
            
            # Extraer punto por punto
            point_by_point = driver.find_elements(By.CLASS_NAME, "point-by-point")
            if point_by_point:
                stats['point_by_point'] = self._extract_point_by_point(point_by_point[0])
            
            # Extraer cambios de momentum
            momentum = driver.find_elements(By.CLASS_NAME, "momentum-changes")
            if momentum:
                stats['momentum_changes'] = self._extract_momentum_changes(momentum[0])
            
            # Extraer estadísticas básicas
            basic_stats = driver.find_elements(By.CLASS_NAME, "basic-stats")
            if basic_stats:
                stats['basic_stats'] = {
                    'service_games_won': self._extract_number(basic_stats[0], "service-games"),
                    'break_points_converted': self._extract_number(basic_stats[0], "break-points"),
                    'total_points_won': self._extract_number(basic_stats[0], "total-points")
                }
            
            # Cerrar el driver
            driver.quit()
            
            # Guardar en caché
            cache_file.write_text(json.dumps(stats))
            return stats
            
        except Exception as e:
            logger.error(f"Error scraping estadísticas detalladas: {str(e)}")
            if 'driver' in locals():
                driver.quit()
            raise
    
    def _extract_percentage(self, element, class_name: str) -> float:
        """Extrae un porcentaje de un elemento."""
        try:
            value = element.find_element(By.CLASS_NAME, class_name).text
            return float(value.strip('%'))
        except:
            return 0.0
    
    def _extract_number(self, element, class_name: str) -> int:
        """Extrae un número de un elemento."""
        try:
            value = element.find_element(By.CLASS_NAME, class_name).text
            return int(value)
        except:
            return 0
    
    def _extract_point_by_point(self, element) -> List[Dict]:
        """Extrae el detalle punto por punto."""
        points = []
        try:
            point_elements = element.find_elements(By.CLASS_NAME, "point")
            for point in point_elements:
                points.append({
                    'server': point.get_attribute('data-server'),
                    'winner': point.get_attribute('data-winner'),
                    'rally_length': int(point.get_attribute('data-rally-length')),
                    'point_type': point.get_attribute('data-point-type')
                })
        except Exception as e:
            logger.warning(f"Error extrayendo punto por punto: {str(e)}")
        return points
    
    def _extract_momentum_changes(self, element) -> List[Dict]:
        """Extrae los cambios de momentum en el partido."""
        changes = []
        try:
            change_elements = element.find_elements(By.CLASS_NAME, "momentum-change")
            for change in change_elements:
                changes.append({
                    'set': int(change.get_attribute('data-set')),
                    'game': int(change.get_attribute('data-game')),
                    'momentum_score': float(change.get_attribute('data-momentum-score'))
                })
        except Exception as e:
            logger.warning(f"Error extrayendo cambios de momentum: {str(e)}")
        return changes
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def scrape_tournament_matches(self, tournament_url: str) -> List[Dict]:
        """
        Extrae todos los partidos de un torneo.
        
        Args:
            tournament_url: URL del torneo
            
        Returns:
            Lista de diccionarios con datos de partidos
        """
        try:
            # Verificar caché
            cache_file = self.tournament_cache_dir / f"{hash(tournament_url)}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())
            
            # Implementar scraping de partidos del torneo
            matches = []
            
            # Guardar en caché
            cache_file.write_text(json.dumps(matches))
            return matches
            
        except Exception as e:
            logger.error(f"Error scraping partidos del torneo: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def scrape_detailed_player_stats(self, player_url: str) -> Dict:
        """
        Extrae estadísticas históricas detalladas de un jugador.
        
        Args:
            player_url: URL del jugador
            
        Returns:
            Diccionario con estadísticas del jugador
        """
        try:
            # Verificar caché
            cache_file = self.player_cache_dir / f"{hash(player_url)}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())
            
            # Implementar scraping de estadísticas del jugador
            stats = {
                'career_stats': {},
                'surface_stats': {},
                'tournament_stats': {},
                'h2h_stats': {}
            }
            
            # Guardar en caché
            cache_file.write_text(json.dumps(stats))
            return stats
            
        except Exception as e:
            logger.error(f"Error scraping estadísticas del jugador: {str(e)}")
            raise
    
    def enrich_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece los datos con información adicional mediante scraping.
        
        Args:
            data: DataFrame con datos básicos
            
        Returns:
            DataFrame enriquecido con datos adicionales
        """
        try:
            enriched_data = data.copy()
            
            # Enriquecer con estadísticas detalladas de partidos
            for idx, row in enriched_data.iterrows():
                try:
                    match_stats = self.scrape_detailed_match_stats(row['match_url'])
                    enriched_data.at[idx, 'detailed_stats'] = json.dumps(match_stats)
                except Exception as e:
                    logger.warning(f"Error enriquciendo partido {row['match_id']}: {str(e)}")
            
            # Enriquecer con estadísticas de jugadores
            for player_col in ['player1_id', 'player2_id']:
                for idx, row in enriched_data.iterrows():
                    try:
                        player_stats = self.scrape_detailed_player_stats(row[f'{player_col}_url'])
                        enriched_data.at[idx, f'{player_col}_detailed_stats'] = json.dumps(player_stats)
                    except Exception as e:
                        logger.warning(f"Error enriquciendo jugador {row[player_col]}: {str(e)}")
            
            logger.info("Enriquecimiento de datos mediante scraping completado")
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento de datos: {str(e)}")
            return data
    
    def __del__(self):
        """Limpia recursos al destruir el objeto."""
        try:
            self.session.close()
        except:
            pass

class AdvancedTennisDataCollector(TennisDataCollector):
    """
    Clase avanzada para recopilar y procesar datos de partidos de tenis.
    Incluye múltiples fuentes de datos y extracción avanzada de estadísticas.
    """
    
    def __init__(self, start_year=2000, end_year=None, user_agent=None, force_refresh=False):
        """
        Inicializa el colector de datos avanzado.
        
        Args:
            start_year: Año de inicio para la recopilación de datos
            end_year: Año final para la recopilación de datos
            user_agent: User Agent para peticiones HTTP
            force_refresh: Si es True, fuerza la recopilación de datos incluso si existen en caché
        """
        super().__init__(start_year, end_year, user_agent, force_refresh)
        
        # Configurar APIs pagadas
        self._configure_paid_apis()
        
        # Configurar validación
        self._configure_validation()
        
        # Configurar directorios adicionales
        self.enriched_dir = self.data_dir / 'enriched'
        self.enriched_dir.mkdir(exist_ok=True)
        
        self.config = {
            'base_url': 'https://www.tennisexplorer.com',
            'api_key': os.getenv('TENNIS_EXPLORER_API_KEY', None)
        }
        
        logger.info("Inicializado colector de datos avanzado")
    
    def _configure_paid_apis(self):
        """Configura las APIs pagadas si están disponibles."""
        self.paid_apis = {}
        
        # Sportradar API (prioridad 1)
        sportradar_key = os.getenv('SPORTRADAR_API_KEY')
        if sportradar_key:
            self.paid_apis['sportradar'] = {
                'key': sportradar_key,
                'base_url': 'https://api.sportradar.com/tennis/trial/v4',
                'priority': 1
            }
        
        # Tennis Abstract API (prioridad 2)
        tennis_abstract_key = os.getenv('TENNIS_ABSTRACT_API_KEY')
        if tennis_abstract_key:
            self.paid_apis['tennis_abstract'] = {
                'key': tennis_abstract_key,
                'base_url': 'https://www.tennisabstract.com/api',
                'priority': 2
            }
        
        # Tennis Explorer API (prioridad 3)
        tennis_explorer_key = os.getenv('TENNIS_EXPLORER_API_KEY')
        if tennis_explorer_key:
            self.paid_apis['tennis_explorer'] = {
                'key': tennis_explorer_key,
                'base_url': 'https://www.tennis-explorer.com/api',
                'priority': 3
            }
        
        logger.info(f"APIs pagadas configuradas: {list(self.paid_apis.keys())}")
    
    def _configure_validation(self):
        """Configura el sistema de validación de datos."""
        self.validation_rules = {
            'required_fields': [
                'match_id', 'tournament_id', 'player1_id', 'player2_id',
                'match_date', 'surface', 'winner_id'
            ],
            'numeric_fields': [
                'aces_1', 'aces_2', 'double_faults_1', 'double_faults_2',
                'service_points_1', 'service_points_2'
            ],
            'date_fields': ['match_date'],
            'valid_categories': {
                'surface': ['hard', 'clay', 'grass', 'carpet'],
                'winner_id': lambda x: x in ['player1_id', 'player2_id']
            }
        }
    
    def collect_enriched_data(self, output_path: str, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Recopila datos enriquecidos de tenis.
        
        Args:
            output_path: Ruta donde guardar los datos
            start_year: Año inicial
            end_year: Año final
            
        Returns:
            DataFrame con datos enriquecidos
        """
        try:
            # Recopilar datos básicos
            basic_data = self.collect_data(start_year, end_year)
            
            if basic_data.empty:
                logger.error("No se pudieron recopilar datos básicos")
                return pd.DataFrame()
                
            logger.info(f"Columnas disponibles en basic_data: {basic_data.columns.tolist()}")
            
            # Enriquecer datos con Jeff Sackmann
            enriched_data = self._enrich_with_jeff_sackmann(basic_data)
            
            # Asegurar que todas las columnas estén presentes con los tipos correctos
            required_columns = {
                # Columnas básicas
                'match_id': str,
                'tournament_id': str,
                'player1_id': str,
                'player2_id': str,
                'player1_name': str,
                'player2_name': str,
                'match_date': str,
                'surface': str,
                'tournament_name': str,
                'tournament_category': str,
                'winner_id': str,
                'score': str,
                'sets_played': int,
                'minutes': int,
                'round': str,
                'draw_size': int,
                'year': int,
                
                # Estadísticas de servicio del ganador
                'winner_ace': float,
                'winner_df': float,
                'winner_svpt': float,
                'winner_1stIn': float,
                'winner_1stWon': float,
                'winner_2ndWon': float,
                'winner_SvGms': float,
                'winner_bpSaved': float,
                'winner_bpFaced': float,
                'winner_rank': float,
                'winner_rank_points': float,
                
                # Estadísticas de servicio del perdedor
                'loser_ace': float,
                'loser_df': float,
                'loser_svpt': float,
                'loser_1stIn': float,
                'loser_1stWon': float,
                'loser_2ndWon': float,
                'loser_SvGms': float,
                'loser_bpSaved': float,
                'loser_bpFaced': float,
                'loser_rank': float,
                'loser_rank_points': float,
                
                # Características físicas
                'winner_ht': float,
                'winner_age': float,
                'winner_hand': str,
                'loser_ht': float,
                'loser_age': float,
                'loser_hand': str,
                
                # Información adicional del torneo
                'tourney_level': str,
                'best_of': int,
                
                # Información de los jugadores
                'winner_seed': str,
                'winner_entry': str,
                'winner_ioc': str,
                'loser_seed': str,
                'loser_entry': str,
                'loser_ioc': str,
                
                # Características temporales
                'days_since_last_match': int
            }
            
            # Inicializar columnas faltantes con valores por defecto según el tipo
            for col, dtype in required_columns.items():
                if col not in enriched_data.columns:
                    if dtype in [str]:
                        enriched_data[col] = ''
                    elif dtype in [int]:
                        enriched_data[col] = 0
                    elif dtype in [float]:
                        enriched_data[col] = 0.0
                    else:
                        enriched_data[col] = None
            
            # Convertir tipos de datos
            for col, dtype in required_columns.items():
                if col in enriched_data.columns:
                    try:
                        enriched_data[col] = enriched_data[col].astype(dtype)
                    except Exception as e:
                        logger.error(f"Error convirtiendo columna {col} a tipo {dtype}: {str(e)}")
                        continue
            
            # Guardar datos enriquecidos
            enriched_data.to_csv(output_path, index=False)
            logger.info(f"Datos guardados en {output_path}")
            
            # Mostrar estadísticas
            logger.info(f"Total de partidos: {len(enriched_data)}")
            logger.info(f"Rango de fechas: {enriched_data['match_date'].min()} - {enriched_data['match_date'].max()}")
            logger.info(f"Tipos de torneos: {enriched_data['tournament_category'].unique()}")
            logger.info(f"Total de torneos únicos: {enriched_data['tournament_id'].nunique()}")
            logger.info(f"Total de jugadores únicos: {enriched_data['player1_id'].nunique() + enriched_data['player2_id'].nunique()}")
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error en collect_enriched_data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _enrich_with_paid_apis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece los datos con información de APIs pagadas y gratuitas.
        Implementa un sistema de fallback donde si no hay APIs de pago disponibles,
        usa automáticamente las APIs gratuitas.
        
        Args:
            data: DataFrame con datos básicos
            
        Returns:
            DataFrame enriquecido con datos de todas las fuentes disponibles
        """
        enriched_data = data.copy()
        
        # 1. Intentar usar APIs de pago si están disponibles
        if self.paid_apis:
            logger.info("Usando APIs de pago disponibles...")
            sorted_apis = sorted(
                self.paid_apis.items(),
                key=lambda x: x[1]['priority']
            )
            
            for api_name, api_config in sorted_apis:
                try:
                    if api_name == 'sportradar':
                        enriched_data = self._enrich_with_sportradar(enriched_data, api_config)
                    elif api_name == 'tennis_abstract':
                        enriched_data = self._enrich_with_tennis_abstract(enriched_data, api_config)
                except Exception as e:
                    logger.warning(f"Error enriquciendo con {api_name}: {str(e)}")
        else:
            logger.info("No hay APIs de pago disponibles, usando APIs gratuitas...")
            
            # 2. Usar APIs gratuitas como fallback
            try:
                # Jeff Sackmann's Tennis Data (datos históricos confiables)
                enriched_data = self._enrich_with_jeff_sackmann(enriched_data)
                
                # Tennis Abstract (versión gratuita)
                enriched_data = self._enrich_with_tennis_abstract(enriched_data, {
                    'base_url': 'https://www.tennisabstract.com/api',
                    'key': None
                })
                
            except Exception as e:
                logger.error(f"Error usando APIs gratuitas: {str(e)}")
        
        return enriched_data
    
    def _enrich_with_tennis_data_uk(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece datos usando Tennis Data UK (API gratuita).
        
        Args:
            data: DataFrame con datos básicos
            
        Returns:
            DataFrame enriquecido con datos de Tennis Data UK
        """
        try:
            enriched_data = data.copy()
            base_url = 'http://www.tennis-data.co.uk/api'
            
            # Crear sesión para la API
            session = requests.Session()
            session.headers.update({
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            })
            
            # Procesar cada partido
            for idx, row in enriched_data.iterrows():
                try:
                    match_id = row['match_id']
                    match_url = f"{base_url}/matches/{match_id}.json"
                    response = session.get(match_url)
                    response.raise_for_status()
                    match_data = response.json()
                    
                    # Extraer estadísticas básicas
                    if 'stats' in match_data:
                        stats = match_data['stats']
                        enriched_data.at[idx, 'first_serve_percentage_1'] = stats.get('first_serve_percentage_1', 0)
                        enriched_data.at[idx, 'first_serve_percentage_2'] = stats.get('first_serve_percentage_2', 0)
                        enriched_data.at[idx, 'aces_1'] = stats.get('aces_1', 0)
                        enriched_data.at[idx, 'aces_2'] = stats.get('aces_2', 0)
                        enriched_data.at[idx, 'double_faults_1'] = stats.get('double_faults_1', 0)
                        enriched_data.at[idx, 'double_faults_2'] = stats.get('double_faults_2', 0)
                    
                    # Esperar para respetar límites de tasa
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error procesando partido {match_id} en Tennis Data UK: {str(e)}")
                    continue
            
            logger.info("Enriquecimiento con Tennis Data UK completado")
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento con Tennis Data UK: {str(e)}")
            return data
    
    def _enrich_with_ultimate_tennis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece los datos con estadísticas de Ultimate Tennis Statistics.
        Solo procesa partidos de los últimos 5 años ya que no hay datos históricos disponibles.
        
        Args:
            data: DataFrame con datos básicos
            
        Returns:
            DataFrame enriquecido
        """
        try:
            logger.info("Enriqueciendo datos con Ultimate Tennis Statistics...")
            
            # Filtrar solo partidos recientes (últimos 5 años)
            current_year = datetime.now().year
            recent_data = data[data['match_date'].dt.year >= current_year - 5].copy()
            
            if recent_data.empty:
                logger.info("No hay partidos recientes para procesar con Ultimate Tennis Statistics")
                return data
            
            logger.info(f"Procesando {len(recent_data)} partidos recientes con Ultimate Tennis Statistics...")
            
            # Crear sesión con reintentos
            session = requests.Session()
            retry_strategy = Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Sistema de checkpoint
            checkpoint_file = Path('data/cache/ultimate_tennis_checkpoint.json')
            checkpoint_data = {}
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"Cargando checkpoint: {len(checkpoint_data)} partidos ya procesados")
            
            # Procesar partidos en lotes
            batch_size = 100
            enriched_data = []
            processed_count = 0
            start_time = time.time()
            
            for i in range(0, len(recent_data), batch_size):
                batch = recent_data.iloc[i:i+batch_size]
                batch_enriched = []
                
                for _, match in batch.iterrows():
                    match_id = f"{match['tournament_id']}_{match['match_id']}"
                    
                    # Verificar si ya fue procesado
                    if match_id in checkpoint_data:
                        batch_enriched.append(checkpoint_data[match_id])
                        processed_count += 1
                        continue
                    
                    try:
                        # Construir URL del partido
                        match_date = pd.to_datetime(match['match_date'])
                        match_url = f"https://www.ultimatetennisstatistics.com/match/{match_date.year}/{match['tournament_id']}/{match['match_id']}.html"
                        
                        # Hacer petición con timeout
                        response = session.get(match_url, timeout=30)
                        response.raise_for_status()
                        
                        # Parsear HTML
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extraer estadísticas detalladas
                        stats = self._extract_ultimate_tennis_stats(soup)
                        
                        # Combinar con datos originales
                        match_data = match.to_dict()
                        match_data.update(stats)
                        batch_enriched.append(match_data)
                        
                        # Guardar en checkpoint
                        checkpoint_data[match_id] = match_data
                        if len(checkpoint_data) % 100 == 0:
                            with open(checkpoint_file, 'w') as f:
                                json.dump(checkpoint_data, f)
                        
                        processed_count += 1
                        
                        # Calcular tiempo estimado restante
                        elapsed_time = time.time() - start_time
                        matches_per_second = processed_count / elapsed_time
                        remaining_matches = len(recent_data) - processed_count
                        estimated_seconds = remaining_matches / matches_per_second
                        
                        logger.info(f"Procesado {processed_count}/{len(recent_data)} partidos "
                                  f"({processed_count/len(recent_data)*100:.1f}%) - "
                                  f"Tiempo estimado restante: {estimated_seconds/3600:.1f} horas")
                        
                        # Esperar entre peticiones
                        time.sleep(random.uniform(1, 3))
                        
                    except Exception as e:
                        logger.warning(f"Error procesando partido {match_id}: {str(e)}")
                        batch_enriched.append(match.to_dict())
                        processed_count += 1
                
                enriched_data.extend(batch_enriched)
                
                # Guardar checkpoint después de cada lote
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
            
            # Eliminar archivo de checkpoint al completar
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            # Combinar datos enriquecidos con datos históricos
            enriched_df = pd.DataFrame(enriched_data)
            historical_data = data[data['match_date'].dt.year < current_year - 5]
            final_data = pd.concat([historical_data, enriched_df], ignore_index=True)
            
            return final_data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento con Ultimate Tennis Statistics: {str(e)}")
            return data
            
    def _extract_ultimate_tennis_stats(self, soup: BeautifulSoup) -> Dict:
        """
        Extrae estadísticas detalladas de Ultimate Tennis Statistics.
        
        Args:
            soup: Objeto BeautifulSoup con el HTML parseado
            
        Returns:
            Diccionario con estadísticas extraídas
        """
        stats = {}
        
        try:
            # Extraer estadísticas de presión
            pressure_stats = soup.find('div', {'class': 'pressure-stats'})
            if pressure_stats:
                stats['pressure_points_won_1'] = self._extract_number(pressure_stats, 'pressure-points-won-1')
                stats['pressure_points_won_2'] = self._extract_number(pressure_stats, 'pressure-points-won-2')
                stats['pressure_points_played_1'] = self._extract_number(pressure_stats, 'pressure-points-played-1')
                stats['pressure_points_played_2'] = self._extract_number(pressure_stats, 'pressure-points-played-2')
            
            # Extraer estadísticas de break points
            break_stats = soup.find('div', {'class': 'break-point-stats'})
            if break_stats:
                stats['break_points_converted_1'] = self._extract_number(break_stats, 'break-points-converted-1')
                stats['break_points_converted_2'] = self._extract_number(break_stats, 'break-points-converted-2')
                stats['break_points_saved_1'] = self._extract_number(break_stats, 'break-points-saved-1')
                stats['break_points_saved_2'] = self._extract_number(break_stats, 'break-points-saved-2')
            
            # Extraer estadísticas de tie-breaks
            tiebreak_stats = soup.find('div', {'class': 'tiebreak-stats'})
            if tiebreak_stats:
                stats['tiebreak_points_won_1'] = self._extract_number(tiebreak_stats, 'tiebreak-points-won-1')
                stats['tiebreak_points_won_2'] = self._extract_number(tiebreak_stats, 'tiebreak-points-won-2')
                stats['tiebreak_points_played_1'] = self._extract_number(tiebreak_stats, 'tiebreak-points-played-1')
                stats['tiebreak_points_played_2'] = self._extract_number(tiebreak_stats, 'tiebreak-points-played-2')
            
            # Extraer estadísticas de servicio detalladas
            serve_stats = soup.find('div', {'class': 'serve-stats'})
            if serve_stats:
                stats['first_serve_speed_avg_1'] = self._extract_number(serve_stats, 'first-serve-speed-avg-1')
                stats['first_serve_speed_avg_2'] = self._extract_number(serve_stats, 'first-serve-speed-avg-2')
                stats['second_serve_speed_avg_1'] = self._extract_number(serve_stats, 'second-serve-speed-avg-1')
                stats['second_serve_speed_avg_2'] = self._extract_number(serve_stats, 'second-serve-speed-avg-2')
                stats['serve_speed_max_1'] = self._extract_number(serve_stats, 'serve-speed-max-1')
                stats['serve_speed_max_2'] = self._extract_number(serve_stats, 'serve-speed-max-2')
            
            # Extraer estadísticas de rallies
            rally_stats = soup.find('div', {'class': 'rally-stats'})
            if rally_stats:
                stats['rally_length_avg_1'] = self._extract_number(rally_stats, 'rally-length-avg-1')
                stats['rally_length_avg_2'] = self._extract_number(rally_stats, 'rally-length-avg-2')
                stats['rally_length_max_1'] = self._extract_number(rally_stats, 'rally-length-max-1')
                stats['rally_length_max_2'] = self._extract_number(rally_stats, 'rally-length-max-2')
            
        except Exception as e:
            logger.warning(f"Error extrayendo estadísticas de Ultimate Tennis: {str(e)}")
        
        return stats
        
    def _extract_number(self, element: BeautifulSoup, class_name: str) -> float:
        """
        Extrae un número de un elemento HTML.
        
        Args:
            element: Elemento BeautifulSoup
            class_name: Nombre de la clase CSS
            
        Returns:
            Número extraído o 0 si no se encuentra
        """
        try:
            value = element.find('span', {'class': class_name})
            if value:
                return float(value.text.strip())
        except:
            pass
        return 0.0
    
    def _enrich_with_jeff_sackmann(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enriquece los datos con estadísticas de Jeff Sackmann."""
        try:
            logger.info("Enriqueciendo datos con Jeff Sackmann...")
            
            # Crear directorio de caché si no existe
            cache_dir = Path('data/cache/jeff_sackmann')
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Configurar sesión con reintentos
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            
            # Obtener años únicos
            data['year'] = pd.to_datetime(data['match_date']).dt.year
            unique_years = data['year'].unique()
            
            # Descargar y cachear datos por año
            year_data = {}
            for year in unique_years:
                cache_file = cache_dir / f"atp_matches_{year}.csv"
                
                if cache_file.exists() and not self.force_refresh:
                    logger.info(f"Usando datos en caché para el año {year}")
                    year_data[year] = pd.read_csv(cache_file)
                else:
                    try:
                        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
                        logger.info(f"Descargando datos para el año {year}")
                        response = session.get(url, timeout=30)
                        response.raise_for_status()
                        
                        # Guardar en caché y cargar
                        with open(cache_file, 'wb') as f:
                            f.write(response.content)
                        year_data[year] = pd.read_csv(cache_file)
                        logger.info(f"Datos del año {year} guardados en caché")
                        
                    except Exception as e:
                        logger.error(f"Error descargando datos del año {year}: {str(e)}")
                        continue
            
            # Procesar partidos en paralelo usando chunks
            chunk_size = 1000
            total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size > 0 else 0)
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(data))
                chunk = data.iloc[start_idx:end_idx]
                
                logger.info(f"Procesando chunk {chunk_idx + 1}/{total_chunks} ({start_idx}-{end_idx})")
                
                # Procesar cada partido en el chunk
                for idx, match in chunk.iterrows():
                    try:
                        year = pd.to_datetime(match['match_date']).year
                        if year not in year_data:
                            continue
                        
                        # Buscar partido en los datos del año usando múltiples criterios
                        year_matches = year_data[year]
                        match_date = pd.to_datetime(match['match_date']).strftime('%Y%m%d')
                        
                        # Filtrar por fecha y jugadores
                        potential_matches = year_matches[
                            (year_matches['tourney_date'].astype(str) == match_date) &
                            (
                                ((year_matches['winner_id'].astype(str) == str(match['player1_id'])) &
                                 (year_matches['loser_id'].astype(str) == str(match['player2_id']))) |
                                ((year_matches['winner_id'].astype(str) == str(match['player2_id'])) &
                                 (year_matches['loser_id'].astype(str) == str(match['player1_id'])))
                            )
                        ]
                        
                        if not potential_matches.empty:
                            stats = potential_matches.iloc[0]
                            
                            # Determinar si player1 es el ganador
                            is_player1_winner = str(stats['winner_id']) == str(match['player1_id'])
                            
                            # Estadísticas de servicio
                            if is_player1_winner:
                                data.at[idx, 'winner_ace'] = stats.get('w_ace', 0)
                                data.at[idx, 'winner_df'] = stats.get('w_df', 0)
                                data.at[idx, 'winner_svpt'] = stats.get('w_svpt', 0)
                                data.at[idx, 'winner_1stIn'] = stats.get('w_1stIn', 0)
                                data.at[idx, 'winner_1stWon'] = stats.get('w_1stWon', 0)
                                data.at[idx, 'winner_2ndWon'] = stats.get('w_2ndWon', 0)
                                data.at[idx, 'winner_SvGms'] = stats.get('w_SvGms', 0)
                                data.at[idx, 'winner_bpSaved'] = stats.get('w_bpSaved', 0)
                                data.at[idx, 'winner_bpFaced'] = stats.get('w_bpFaced', 0)
                                data.at[idx, 'winner_rank'] = stats.get('winner_rank', 0)
                                data.at[idx, 'winner_rank_points'] = stats.get('winner_rank_points', 0)
                                data.at[idx, 'winner_ht'] = stats.get('winner_ht', 0)
                                data.at[idx, 'winner_age'] = stats.get('winner_age', 0)
                                data.at[idx, 'winner_hand'] = stats.get('winner_hand', '')
                                data.at[idx, 'winner_seed'] = stats.get('winner_seed', '')
                                data.at[idx, 'winner_entry'] = str(stats.get('winner_entry', ''))
                                data.at[idx, 'winner_ioc'] = stats.get('winner_ioc', '')
                                
                                data.at[idx, 'loser_ace'] = stats.get('l_ace', 0)
                                data.at[idx, 'loser_df'] = stats.get('l_df', 0)
                                data.at[idx, 'loser_svpt'] = stats.get('l_svpt', 0)
                                data.at[idx, 'loser_1stIn'] = stats.get('l_1stIn', 0)
                                data.at[idx, 'loser_1stWon'] = stats.get('l_1stWon', 0)
                                data.at[idx, 'loser_2ndWon'] = stats.get('l_2ndWon', 0)
                                data.at[idx, 'loser_SvGms'] = stats.get('l_SvGms', 0)
                                data.at[idx, 'loser_bpSaved'] = stats.get('l_bpSaved', 0)
                                data.at[idx, 'loser_bpFaced'] = stats.get('l_bpFaced', 0)
                                data.at[idx, 'loser_rank'] = stats.get('loser_rank', 0)
                                data.at[idx, 'loser_rank_points'] = stats.get('loser_rank_points', 0)
                                data.at[idx, 'loser_ht'] = stats.get('loser_ht', 0)
                                data.at[idx, 'loser_age'] = stats.get('loser_age', 0)
                                data.at[idx, 'loser_hand'] = stats.get('loser_hand', '')
                                data.at[idx, 'loser_seed'] = stats.get('loser_seed', '')
                                data.at[idx, 'loser_entry'] = str(stats.get('loser_entry', ''))
                                data.at[idx, 'loser_ioc'] = stats.get('loser_ioc', '')
                            else:
                                data.at[idx, 'winner_ace'] = stats.get('l_ace', 0)
                                data.at[idx, 'winner_df'] = stats.get('l_df', 0)
                                data.at[idx, 'winner_svpt'] = stats.get('l_svpt', 0)
                                data.at[idx, 'winner_1stIn'] = stats.get('l_1stIn', 0)
                                data.at[idx, 'winner_1stWon'] = stats.get('l_1stWon', 0)
                                data.at[idx, 'winner_2ndWon'] = stats.get('l_2ndWon', 0)
                                data.at[idx, 'winner_SvGms'] = stats.get('l_SvGms', 0)
                                data.at[idx, 'winner_bpSaved'] = stats.get('l_bpSaved', 0)
                                data.at[idx, 'winner_bpFaced'] = stats.get('l_bpFaced', 0)
                                data.at[idx, 'winner_rank'] = stats.get('loser_rank', 0)
                                data.at[idx, 'winner_rank_points'] = stats.get('loser_rank_points', 0)
                                data.at[idx, 'winner_ht'] = stats.get('loser_ht', 0)
                                data.at[idx, 'winner_age'] = stats.get('loser_age', 0)
                                data.at[idx, 'winner_hand'] = stats.get('loser_hand', '')
                                data.at[idx, 'winner_seed'] = stats.get('loser_seed', '')
                                data.at[idx, 'winner_entry'] = str(stats.get('loser_entry', ''))
                                data.at[idx, 'winner_ioc'] = stats.get('loser_ioc', '')
                            
                            # Características del torneo
                            data.at[idx, 'tourney_level'] = stats.get('tourney_level', '')
                            data.at[idx, 'surface'] = stats.get('surface', '')
                            data.at[idx, 'draw_size'] = stats.get('draw_size', 0)
                            data.at[idx, 'best_of'] = stats.get('best_of', 0)
                            
                    except Exception as e:
                        logger.error(f"Error procesando partido {match['match_id']}: {str(e)}")
                        continue
                
                # Guardar progreso cada 1000 partidos
                if (chunk_idx + 1) % 10 == 0:
                    logger.info(f"Progreso: {end_idx}/{len(data)} partidos procesados ({(end_idx/len(data))*100:.1f}%)")
            
            # Calcular días desde último partido
            logger.info("Calculando días desde último partido...")
            data = self._calculate_temporal_features(data)
            
            # Calcular estadísticas H2H
            logger.info("Calculando estadísticas H2H...")
            data = self._calculate_h2h_features(data)
            
            # Calcular métricas por superficie
            logger.info("Calculando métricas por superficie...")
            data = self._calculate_surface_metrics(data)
            
            logger.info("Enriquecimiento con Jeff Sackmann completado")
            return data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento con Jeff Sackmann: {str(e)}")
            return data
            
    def _calculate_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula características temporales disponibles en el dataset de Jeff Sackmann.
        
        Nota: Las características como recent_form y momentum no están disponibles en el dataset
        de Jeff Sackmann y requerirían datos históricos adicionales para calcularse correctamente.
        Por lo tanto, no se incluyen en esta implementación.
        """
        logger.info("Iniciando cálculo de características temporales disponibles...")
        
        # Ordenar por fecha
        data = data.sort_values('match_date')
        
        # Calcular días desde último partido (usando tourney_date de Jeff Sackmann)
        for player_col in ['player1_id', 'player2_id']:
            data[f'{player_col}_days_since_last'] = data.groupby(player_col)['match_date'].diff().dt.days.fillna(0)
            logger.info(f"Estadísticas de días desde último partido para {player_col}:")
            logger.info(f"  - Media: {data[f'{player_col}_days_since_last'].mean():.1f} días")
            logger.info(f"  - Máximo: {data[f'{player_col}_days_since_last'].max()} días")
            logger.info(f"  - Mínimo: {data[f'{player_col}_days_since_last'].min()} días")
        
        return data

    def _calculate_h2h_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula estadísticas head-to-head."""
        try:
            logger.info("Iniciando cálculo de estadísticas head-to-head...")
            
            # Crear clave única para cada par de jugadores
            data['h2h_key'] = data.apply(
                lambda x: f"{min(x['player1_id'], x['player2_id'])}_{max(x['player1_id'], x['player2_id'])}",
                axis=1
            )
            logger.info(f"Total de pares únicos de jugadores: {data['h2h_key'].nunique()}")
            
            # Calcular historial H2H antes de cada partido
            h2h_stats = {}
            total_matches = len(data)
            processed_matches = 0
            
            logger.info("Calculando estadísticas H2H para cada partido...")
            for idx, row in data.iterrows():
                h2h_key = row['h2h_key']
                match_date = row['match_date']
                
                # Obtener partidos anteriores entre estos jugadores
                previous_matches = data[
                    (data['h2h_key'] == h2h_key) & 
                    (data['match_date'] < match_date)
                ]
                
                if not previous_matches.empty:
                    # Calcular estadísticas H2H
                    total_h2h_matches = len(previous_matches)
                    player1_wins = len(previous_matches[
                        previous_matches['winner_id'] == row['player1_id']
                    ])
                    
                    h2h_stats[idx] = {
                        'h2h_win_rate': player1_wins / total_h2h_matches,
                        'h2h_matches': total_h2h_matches
                    }
                else:
                    h2h_stats[idx] = {
                        'h2h_win_rate': 0.5,  # Valor neutral para primer encuentro
                        'h2h_matches': 0
                    }
                
                processed_matches += 1
                if processed_matches % 10000 == 0:
                    logger.info(f"Progreso H2H: {processed_matches}/{total_matches} partidos procesados ({(processed_matches/total_matches)*100:.1f}%)")
            
            # Actualizar datos con estadísticas H2H
            logger.info("Actualizando datos con estadísticas H2H...")
            for idx, stats in h2h_stats.items():
                data.at[idx, 'h2h_win_rate'] = stats['h2h_win_rate']
                data.at[idx, 'h2h_matches'] = stats['h2h_matches']
            
            # Mostrar estadísticas finales
            logger.info("Estadísticas finales de H2H:")
            logger.info(f"  - Media de partidos H2H: {data['h2h_matches'].mean():.1f}")
            logger.info(f"  - Máximo de partidos H2H: {data['h2h_matches'].max()}")
            logger.info(f"  - Mínimo de partidos H2H: {data['h2h_matches'].min()}")
            logger.info(f"  - Media de win rate H2H: {data['h2h_win_rate'].mean():.3f}")
            logger.info("Cálculo de estadísticas H2H completado con éxito")
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas H2H: {str(e)}")
            logger.error(traceback.format_exc())
            return data

    def _calculate_surface_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula métricas específicas por superficie."""
        try:
            logger.info("Iniciando cálculo de métricas por superficie...")
            logger.info(f"Superficies disponibles: {data['surface'].unique()}")
            
            # Calcular winrate por superficie para cada jugador antes de cada partido
            surface_stats = {}
            total_matches = len(data)
            processed_matches = 0
            
            logger.info("Calculando métricas por superficie para cada partido...")
            for idx, row in data.iterrows():
                player_id = row['player1_id']
                surface = row['surface']
                match_date = row['match_date']
                
                # Obtener partidos anteriores del jugador en esta superficie
                previous_matches = data[
                    ((data['player1_id'] == player_id) | (data['player2_id'] == player_id)) &
                    (data['surface'] == surface) &
                    (data['match_date'] < match_date)
                ]
                
                if not previous_matches.empty:
                    # Calcular winrate
                    player_wins = len(previous_matches[
                        previous_matches['winner_id'] == player_id
                    ])
                    surface_stats[idx] = player_wins / len(previous_matches)
                else:
                    surface_stats[idx] = 0.5  # Valor neutral para primer partido en la superficie
                
                processed_matches += 1
                if processed_matches % 10000 == 0:
                    logger.info(f"Progreso métricas superficie: {processed_matches}/{total_matches} partidos procesados ({(processed_matches/total_matches)*100:.1f}%)")
            
            # Actualizar datos con métricas de superficie
            logger.info("Actualizando datos con métricas por superficie...")
            for idx, winrate in surface_stats.items():
                data.at[idx, 'surface_winrate'] = winrate
            
            # Mostrar estadísticas finales por superficie
            logger.info("Estadísticas finales por superficie:")
            for surface in data['surface'].unique():
                surface_data = data[data['surface'] == surface]
                logger.info(f"Superficie: {surface}")
                logger.info(f"  - Media de win rate: {surface_data['surface_winrate'].mean():.3f}")
                logger.info(f"  - Máximo win rate: {surface_data['surface_winrate'].max():.3f}")
                logger.info(f"  - Mínimo win rate: {surface_data['surface_winrate'].min():.3f}")
                logger.info(f"  - Total de partidos: {len(surface_data)}")
            
            logger.info("Cálculo de métricas por superficie completado con éxito")
            return data
            
        except Exception as e:
            logger.error(f"Error calculando métricas por superficie: {str(e)}")
            logger.error(traceback.format_exc())
            return data

def main():
    """Función principal para ejecutar el colector de datos."""
    try:
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/tennis_ml.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Obtener argumentos de línea de comandos
        parser = argparse.ArgumentParser(description='Recopilador de datos de tenis')
        parser.add_argument('--output', type=str, required=True, help='Ruta del archivo de salida')
        parser.add_argument('--start-year', type=int, default=2000, help='Año inicial')
        parser.add_argument('--end-year', type=int, default=2024, help='Año final')
        parser.add_argument('--detailed-stats', action='store_true', help='Incluir estadísticas detalladas')
        parser.add_argument('--force-refresh', action='store_true', help='Forzar recopilación de datos')
        args = parser.parse_args()
        
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        # Crear directorio de datos si no existe
        os.makedirs('data', exist_ok=True)
        
        # Inicializar colector
        collector = AdvancedTennisDataCollector()
        collector.force_refresh = args.force_refresh
        
        # Recopilar datos
        logger.info(f"Inicializado colector de datos para años {args.start_year}-{args.end_year}")
        if args.force_refresh:
            logger.info("Modo force_refresh activado: se recopilarán todos los datos de nuevo")
            
        # Recopilar datos enriquecidos
        enriched_data = collector.collect_enriched_data(
            output_path=args.output,
            start_year=args.start_year,
            end_year=args.end_year
        )
        
        if not enriched_data.empty:
            logger.info(f"Datos enriquecidos guardados en {args.output}")
            logger.info(f"Total de partidos: {len(enriched_data)}")
            logger.info(f"Rango de fechas: {enriched_data['match_date'].min()} - {enriched_data['match_date'].max()}")
            logger.info(f"Tipos de torneos: {enriched_data['tournament_category'].unique()}")
            logger.info(f"Total de torneos únicos: {enriched_data['tournament_id'].nunique()}")
            logger.info(f"Total de jugadores únicos: {enriched_data['player1_id'].nunique() + enriched_data['player2_id'].nunique()}")
        else:
            logger.error("No se pudieron recopilar datos")
            
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 