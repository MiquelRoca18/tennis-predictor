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
    
    def collect_enriched_data(self, output_path: str) -> Optional[pd.DataFrame]:
        """
        Recopila y enriquece datos de partidos de tenis.
        
        Args:
            output_path: Ruta donde guardar los datos enriquecidos
            
        Returns:
            DataFrame con datos enriquecidos o None si hay error
        """
        try:
            logger.info("Iniciando recopilación de datos enriquecidos...")
            
            # Recopilar datos básicos
            basic_data = self.collect_data(self.start_year, self.end_year)
            if basic_data is None or basic_data.empty:
                logger.error("No se pudieron recopilar datos básicos")
                return None
            
            # Enriquecer datos con APIs pagadas
            enriched_data = self._enrich_with_paid_apis(basic_data)
            
            # Guardar datos enriquecidos
            enriched_data.to_csv(output_path, index=False)
            logger.info(f"Datos enriquecidos guardados en {output_path}")
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error en recopilación de datos enriquecidos: {str(e)}")
            return None
    
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
        Enriquece datos usando Ultimate Tennis Statistics (API gratuita).
        
        Args:
            data: DataFrame con datos básicos
            
        Returns:
            DataFrame enriquecido con datos de Ultimate Tennis Statistics
        """
        try:
            enriched_data = data.copy()
            base_url = 'https://www.ultimatetennisstatistics.com/api'
            
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
                    
                    # Extraer estadísticas avanzadas
                    if 'advanced_stats' in match_data:
                        stats = match_data['advanced_stats']
                        enriched_data.at[idx, 'avg_rally_length'] = stats.get('avg_rally_length', 0)
                        enriched_data.at[idx, 'winners_1'] = stats.get('winners_1', 0)
                        enriched_data.at[idx, 'winners_2'] = stats.get('winners_2', 0)
                        enriched_data.at[idx, 'unforced_errors_1'] = stats.get('unforced_errors_1', 0)
                        enriched_data.at[idx, 'unforced_errors_2'] = stats.get('unforced_errors_2', 0)
                    
                    # Esperar para respetar límites de tasa
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error procesando partido {match_id} en Ultimate Tennis: {str(e)}")
                    continue
            
            logger.info("Enriquecimiento con Ultimate Tennis Statistics completado")
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento con Ultimate Tennis Statistics: {str(e)}")
            return data
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Valida los datos según las reglas configuradas."""
        try:
            # Verificar campos requeridos
            for field in self.validation_rules['required_fields']:
                if field not in data.columns:
                    logger.error(f"Campo requerido no encontrado: {field}")
                    return False
            
            # Verificar campos numéricos
            for field in self.validation_rules['numeric_fields']:
                if field in data.columns:
                    if not pd.to_numeric(data[field], errors='coerce').notna().all():
                        logger.error(f"Campo numérico inválido: {field}")
                        return False
            
            # Verificar fechas
            for field in self.validation_rules['date_fields']:
                if field in data.columns:
                    if not pd.to_datetime(data[field], errors='coerce').notna().all():
                        logger.error(f"Campo de fecha inválido: {field}")
                        return False
            
            # Verificar categorías
            for field, valid_values in self.validation_rules['valid_categories'].items():
                if field in data.columns:
                    if callable(valid_values):
                        if not data[field].apply(valid_values).all():
                            logger.error(f"Valores inválidos en campo: {field}")
                            return False
                    else:
                        if not data[field].isin(valid_values).all():
                            logger.error(f"Valores inválidos en campo: {field}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error en validación de datos: {str(e)}")
            return False
    
    def _enrich_with_sportradar(self, data: pd.DataFrame, api_config: Dict) -> pd.DataFrame:
        """
        Enriquece datos con la API de Sportradar.
        
        Args:
            data: DataFrame con datos básicos
            api_config: Configuración de la API
            
        Returns:
            DataFrame enriquecido con datos de Sportradar
        """
        try:
            enriched_data = data.copy()
            base_url = api_config['base_url']
            api_key = api_config['key']
            
            # Crear sesión para la API
            session = requests.Session()
            session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json'
            })
            
            # Procesar cada partido
            for idx, row in enriched_data.iterrows():
                try:
                    # Obtener detalles del partido
                    match_id = row['match_id']
                    match_url = f"{base_url}/matches/{match_id}/summary.json"
                    response = session.get(match_url)
                    response.raise_for_status()
                    match_data = response.json()
                    
                    # Extraer estadísticas detalladas
                    if 'statistics' in match_data:
                        stats = match_data['statistics']
                        
                        # Estadísticas de servicio
                        enriched_data.at[idx, 'first_serve_percentage_1'] = stats.get('first_serve_percentage_1', 0)
                        enriched_data.at[idx, 'first_serve_percentage_2'] = stats.get('first_serve_percentage_2', 0)
                        enriched_data.at[idx, 'aces_1'] = stats.get('aces_1', 0)
                        enriched_data.at[idx, 'aces_2'] = stats.get('aces_2', 0)
                        enriched_data.at[idx, 'double_faults_1'] = stats.get('double_faults_1', 0)
                        enriched_data.at[idx, 'double_faults_2'] = stats.get('double_faults_2', 0)
                        
                        # Estadísticas de puntos
                        enriched_data.at[idx, 'service_points_won_1'] = stats.get('service_points_won_1', 0)
                        enriched_data.at[idx, 'service_points_won_2'] = stats.get('service_points_won_2', 0)
                        enriched_data.at[idx, 'return_points_won_1'] = stats.get('return_points_won_1', 0)
                        enriched_data.at[idx, 'return_points_won_2'] = stats.get('return_points_won_2', 0)
                        
                        # Estadísticas de break
                        enriched_data.at[idx, 'break_points_won_1'] = stats.get('break_points_won_1', 0)
                        enriched_data.at[idx, 'break_points_won_2'] = stats.get('break_points_won_2', 0)
                        enriched_data.at[idx, 'break_points_faced_1'] = stats.get('break_points_faced_1', 0)
                        enriched_data.at[idx, 'break_points_faced_2'] = stats.get('break_points_faced_2', 0)
                    
                    # Obtener datos de los jugadores
                    for player_num in [1, 2]:
                        player_id = row[f'player{player_num}_id']
                        player_url = f"{base_url}/players/{player_id}/profile.json"
                        response = session.get(player_url)
                        response.raise_for_status()
                        player_data = response.json()
                        
                        # Estadísticas del jugador
                        if 'statistics' in player_data:
                            player_stats = player_data['statistics']
                            enriched_data.at[idx, f'player{player_num}_rank'] = player_stats.get('rank', 0)
                            enriched_data.at[idx, f'player{player_num}_win_rate'] = player_stats.get('win_rate', 0)
                            enriched_data.at[idx, f'player{player_num}_surface_stats'] = json.dumps(player_stats.get('surface_stats', {}))
                    
                    # Obtener datos del torneo
                    tournament_id = row['tournament_id']
                    tournament_url = f"{base_url}/tournaments/{tournament_id}/info.json"
                    response = session.get(tournament_url)
                    response.raise_for_status()
                    tournament_data = response.json()
                    
                    # Información del torneo
                    if 'tournament' in tournament_data:
                        tournament = tournament_data['tournament']
                        enriched_data.at[idx, 'tournament_name'] = tournament.get('name', '')
                        enriched_data.at[idx, 'tournament_category'] = tournament.get('category', '')
                        enriched_data.at[idx, 'tournament_surface'] = tournament.get('surface', '')
                        enriched_data.at[idx, 'tournament_draw_size'] = tournament.get('draw_size', 0)
                    
                    # Esperar para respetar límites de tasa
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error procesando partido {match_id}: {str(e)}")
                    continue
            
            logger.info("Enriquecimiento con Sportradar completado")
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento con Sportradar: {str(e)}")
            return data
    
    def _enrich_with_tennis_abstract(self, data: pd.DataFrame, api_config: Dict) -> pd.DataFrame:
        """
        Enriquece datos con la API de Tennis Abstract.
        Obtiene estadísticas avanzadas de partidos y jugadores.
        Solo procesa datos recientes (últimos 5 años) ya que la API no tiene datos históricos.
        
        Args:
            data: DataFrame con datos básicos
            api_config: Configuración de la API
            
        Returns:
            DataFrame enriquecido con datos de Tennis Abstract
        """
        try:
            enriched_data = data.copy()
            base_url = api_config['base_url']
            api_key = api_config.get('key')
            
            # Crear sesión para la API
            session = requests.Session()
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            }
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            session.headers.update(headers)
            
            # Filtrar solo datos recientes (últimos 5 años)
            current_year = datetime.now().year
            recent_data = enriched_data[enriched_data['match_date'].dt.year >= current_year - 5].copy()
            
            if recent_data.empty:
                logger.info("No hay datos recientes para procesar con Tennis Abstract")
                return enriched_data
            
            logger.info(f"Procesando {len(recent_data)} partidos recientes con Tennis Abstract...")
            
            # Procesar cada partido
            for idx, row in recent_data.iterrows():
                try:
                    # Construir ID de partido en formato correcto
                    match_date = row['match_date']
                    tournament_id = row['tournament_id']
                    match_num = row['match_id']
                    
                    # Formato: YYYY-TournamentID-MatchNum
                    match_id = f"{match_date.year}-{tournament_id}-{match_num}"
                    
                    match_url = f"{base_url}/matches/{match_id}.json"
                    
                    # Verificar si ya tenemos los datos en caché
                    cache_file = Path('data/cache/tennis_abstract') / f"{match_id}.json"
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    if cache_file.exists() and not self.force_refresh:
                        with open(cache_file, 'r') as f:
                            match_data = json.load(f)
                    else:
                        response = session.get(match_url)
                        if response.status_code == 404:
                            logger.debug(f"Partido {match_id} no encontrado en Tennis Abstract")
                            continue
                        response.raise_for_status()
                        match_data = response.json()
                        
                        # Guardar en caché
                        with open(cache_file, 'w') as f:
                            json.dump(match_data, f)
                    
                    # Extraer estadísticas avanzadas
                    if 'advanced_stats' in match_data:
                        stats = match_data['advanced_stats']
                        
                        # Estadísticas de servicio avanzadas
                        enriched_data.at[idx, 'first_serve_speed_1'] = stats.get('first_serve_speed_1', 0)
                        enriched_data.at[idx, 'first_serve_speed_2'] = stats.get('first_serve_speed_2', 0)
                        enriched_data.at[idx, 'second_serve_speed_1'] = stats.get('second_serve_speed_1', 0)
                        enriched_data.at[idx, 'second_serve_speed_2'] = stats.get('second_serve_speed_2', 0)
                        
                        # Estadísticas de puntos avanzadas
                        enriched_data.at[idx, 'net_points_won_1'] = stats.get('net_points_won_1', 0)
                        enriched_data.at[idx, 'net_points_won_2'] = stats.get('net_points_won_2', 0)
                        enriched_data.at[idx, 'net_points_played_1'] = stats.get('net_points_played_1', 0)
                        enriched_data.at[idx, 'net_points_played_2'] = stats.get('net_points_played_2', 0)
                        
                        # Estadísticas de rallies
                        enriched_data.at[idx, 'rally_length_0_4_1'] = stats.get('rally_length_0_4_1', 0)
                        enriched_data.at[idx, 'rally_length_0_4_2'] = stats.get('rally_length_0_4_2', 0)
                        enriched_data.at[idx, 'rally_length_5_8_1'] = stats.get('rally_length_5_8_1', 0)
                        enriched_data.at[idx, 'rally_length_5_8_2'] = stats.get('rally_length_5_8_2', 0)
                        enriched_data.at[idx, 'rally_length_9_plus_1'] = stats.get('rally_length_9_plus_1', 0)
                        enriched_data.at[idx, 'rally_length_9_plus_2'] = stats.get('rally_length_9_plus_2', 0)
                    
                    # Obtener datos de los jugadores
                    for player_num in [1, 2]:
                        player_id = row[f'player{player_num}_id']
                        player_url = f"{base_url}/players/{player_id}/stats.json"
                        
                        # Verificar caché de jugador
                        player_cache_file = Path('data/cache/tennis_abstract/players') / f"{player_id}.json"
                        player_cache_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        if player_cache_file.exists() and not self.force_refresh:
                            with open(player_cache_file, 'r') as f:
                                player_data = json.load(f)
                        else:
                            response = session.get(player_url)
                            if response.status_code == 404:
                                logger.debug(f"Jugador {player_id} no encontrado en Tennis Abstract")
                                continue
                            response.raise_for_status()
                            player_data = response.json()
                            
                            # Guardar en caché
                            with open(player_cache_file, 'w') as f:
                                json.dump(player_data, f)
                        
                        # Estadísticas históricas del jugador
                        if 'historical_stats' in player_data:
                            hist_stats = player_data['historical_stats']
                            enriched_data.at[idx, f'player{player_num}_career_win_rate'] = hist_stats.get('career_win_rate', 0)
                            enriched_data.at[idx, f'player{player_num}_career_serve_win_rate'] = hist_stats.get('career_serve_win_rate', 0)
                            enriched_data.at[idx, f'player{player_num}_career_return_win_rate'] = hist_stats.get('career_return_win_rate', 0)
                            enriched_data.at[idx, f'player{player_num}_career_break_point_conversion'] = hist_stats.get('career_break_point_conversion', 0)
                            
                            # Estadísticas por superficie
                            surface_stats = hist_stats.get('surface_stats', {})
                            for surface in ['hard', 'clay', 'grass']:
                                if surface in surface_stats:
                                    enriched_data.at[idx, f'player{player_num}_{surface}_win_rate'] = surface_stats[surface].get('win_rate', 0)
                                    enriched_data.at[idx, f'player{player_num}_{surface}_serve_win_rate'] = surface_stats[surface].get('serve_win_rate', 0)
                                    enriched_data.at[idx, f'player{player_num}_{surface}_return_win_rate'] = surface_stats[surface].get('return_win_rate', 0)
                    
                    # Esperar para respetar límites de tasa
                    time.sleep(1)
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        logger.debug(f"Recurso no encontrado en Tennis Abstract: {str(e)}")
                    else:
                        logger.warning(f"Error HTTP procesando partido {match_id}: {str(e)}")
                    continue
                except Exception as e:
                    logger.warning(f"Error procesando partido {match_id}: {str(e)}")
                    continue
            
            logger.info("Enriquecimiento con Tennis Abstract completado")
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento con Tennis Abstract: {str(e)}")
            return data
    
    def _enrich_with_jeff_sackmann(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece los datos usando Jeff Sackmann's Tennis Data.
        
        Args:
            data: DataFrame con datos básicos
            
        Returns:
            DataFrame enriquecido
        """
        try:
            logger.info("Enriqueciendo datos con Jeff Sackmann's Tennis Data...")
            
            # Crear directorio de caché si no existe
            cache_dir = Path('data/cache/jeff_sackmann')
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Convertir match_date a datetime si no lo es ya
            if not pd.api.types.is_datetime64_any_dtype(data['match_date']):
                try:
                    data['match_date'] = pd.to_datetime(data['match_date'].astype(str), format='%Y%m%d')
                except Exception as e:
                    logger.error(f"Error convirtiendo fechas: {str(e)}")
                    return data
            
            # Verificar si ya tenemos los datos enriquecidos en caché
            enriched_cache_file = cache_dir / 'enriched_data.csv'
            if enriched_cache_file.exists() and not self.force_refresh:
                logger.info("Usando datos enriquecidos en caché...")
                return pd.read_csv(enriched_cache_file)
            
            # Agrupar partidos por año para procesarlos más eficientemente
            for year in data['match_date'].dt.year.unique():
                try:
                    logger.info(f"Procesando datos del año {year}...")
                    
                    # Filtrar partidos del año actual
                    year_data = data[data['match_date'].dt.year == year].copy()
                    
                    # Construir URL del año
                    url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
                    
                    # Verificar caché
                    cache_file = cache_dir / f"atp_matches_{year}.csv"
                    
                    if cache_file.exists() and not self.force_refresh:
                        logger.info(f"Usando datos en caché para el año {year}")
                        year_matches = pd.read_csv(cache_file)
                    else:
                        # Intentar descargar datos con reintentos
                        max_retries = 3
                        retry_delay = 5  # segundos
                        
                        for attempt in range(max_retries):
                            try:
                                response = self.session.get(url)
                                response.raise_for_status()
                                break
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    logger.warning(f"Intento {attempt + 1} fallido para el año {year}: {str(e)}")
                                    logger.info(f"Esperando {retry_delay} segundos antes de reintentar...")
                                    time.sleep(retry_delay)
                                    retry_delay *= 2  # Aumentar el retraso exponencialmente
                                else:
                                    raise
                        
                        # Leer CSV directamente
                        content = response.content.decode('utf-8')
                        year_matches = pd.read_csv(StringIO(content))
                        
                        # Guardar en caché
                        year_matches.to_csv(cache_file, index=False)
                        logger.info(f"Datos del año {year} guardados en caché")
                    
                    # Procesar todos los partidos del año de una vez
                    for idx, row in year_data.iterrows():
                        try:
                            # Filtrar partido específico
                            match_data = year_matches[
                                (year_matches['tourney_id'] == str(row['tournament_id'])) &
                                (year_matches['match_num'] == str(row['match_id']))
                            ]
                            
                            if not match_data.empty:
                                # Actualizar datos del partido
                                data.at[idx, 'aces_1'] = match_data['w_ace'].iloc[0]
                                data.at[idx, 'aces_2'] = match_data['l_ace'].iloc[0]
                                data.at[idx, 'double_faults_1'] = match_data['w_df'].iloc[0]
                                data.at[idx, 'double_faults_2'] = match_data['l_df'].iloc[0]
                                data.at[idx, 'service_points_1'] = match_data['w_svpt'].iloc[0]
                                data.at[idx, 'service_points_2'] = match_data['l_svpt'].iloc[0]
                                data.at[idx, 'first_serves_in_1'] = match_data['w_1stIn'].iloc[0]
                                data.at[idx, 'first_serves_in_2'] = match_data['l_1stIn'].iloc[0]
                                data.at[idx, 'first_serve_points_won_1'] = match_data['w_1stWon'].iloc[0]
                                data.at[idx, 'first_serve_points_won_2'] = match_data['l_1stWon'].iloc[0]
                                data.at[idx, 'second_serve_points_won_1'] = match_data['w_2ndWon'].iloc[0]
                                data.at[idx, 'second_serve_points_won_2'] = match_data['l_2ndWon'].iloc[0]
                                data.at[idx, 'service_games_1'] = match_data['w_SvGms'].iloc[0]
                                data.at[idx, 'service_games_2'] = match_data['l_SvGms'].iloc[0]
                                data.at[idx, 'break_points_saved_1'] = match_data['w_BpSaved'].iloc[0]
                                data.at[idx, 'break_points_saved_2'] = match_data['l_BpSaved'].iloc[0]
                                data.at[idx, 'break_points_faced_1'] = match_data['w_BpFaced'].iloc[0]
                                data.at[idx, 'break_points_faced_2'] = match_data['l_BpFaced'].iloc[0]
                            
                        except Exception as e:
                            logger.warning(f"Error procesando partido {idx} del año {year}: {str(e)}")
                            continue
                    
                except Exception as e:
                    logger.warning(f"Error procesando año {year}: {str(e)}")
                    continue
            
            # Guardar datos enriquecidos en caché
            data.to_csv(enriched_cache_file, index=False)
            logger.info("Datos enriquecidos guardados en caché")
            
            logger.info("Enriquecimiento con Jeff Sackmann completado")
            return data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento con Jeff Sackmann: {str(e)}")
            return data
    
    def _enrich_with_tennis_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece datos usando Tennis Stats API (API gratuita).
        Obtiene estadísticas avanzadas de rendimiento.
        
        Args:
            data: DataFrame con datos básicos
            
        Returns:
            DataFrame enriquecido con datos de Tennis Stats
        """
        try:
            enriched_data = data.copy()
            base_url = 'https://tennis-stats-api.herokuapp.com/api'
            
            # Crear sesión para la API
            session = requests.Session()
            session.headers.update({
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            })
            
            # Procesar cada partido
            for idx, row in enriched_data.iterrows():
                try:
                    # Verificar que tenemos los datos necesarios
                    if 'match_id' not in row or pd.isna(row['match_id']):
                        logger.warning(f"Partido {idx} no tiene match_id válido")
                        continue
                        
                    match_id = str(row['match_id'])
                    match_url = f"{base_url}/matches/{match_id}/stats.json"
                    
                    # Resto del código...
                    
                except Exception as e:
                    logger.warning(f"Error procesando partido {idx}: {str(e)}")
                    continue
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento con Tennis Stats: {str(e)}")
            return data 

def main():
    """
    Función principal para recopilar y enriquecer datos de tenis.
    """
    try:
        # Crear directorios necesarios
        Path('data/enriched').mkdir(parents=True, exist_ok=True)
        Path('data/cache').mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/tennis_ml.log')
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Inicializar colector de datos avanzado
        start_year = 2000
        end_year = datetime.now().year
        force_refresh = True  # Forzar recopilación de datos
        
        collector = AdvancedTennisDataCollector(
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_refresh
        )
        
        # Recopilar y enriquecer datos
        logger.info(f"Iniciando recopilación de datos desde {start_year} hasta {end_year}")
        enriched_data = collector.collect_enriched_data(
            output_path='data/enriched/tennis_data_enriched.csv'
        )
        
        if enriched_data is not None and not enriched_data.empty:
            logger.info(f"Total de partidos procesados: {len(enriched_data)}")
        else:
            logger.error("No se pudieron recopilar datos enriquecidos")
            
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 