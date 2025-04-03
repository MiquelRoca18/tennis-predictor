"""
tennis_data_collector_improved.py

Script optimizado que recopila TODOS los datos disponibles de los repositorios de Jeff Sackmann:
- Partidos (ATP y WTA)
- Datos de jugadores
- Rankings consolidados por década (formato correcto)
- Partidos de Challenger y Futures
- Datos punto por punto de Grand Slam
- Datos del Match Charting Project
- Datos generales punto por punto
"""

import pandas as pd
import numpy as np
import logging
import os
import requests
import time
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Set
from pathlib import Path
import random
from io import StringIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
import sys
from tqdm import tqdm
import glob

# Crear directorio de logs si no existe
os.makedirs('logs', exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/tennis_data_collection.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)

# Lista de user agents de respaldo
FALLBACK_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15'
]

def get_random_user_agent() -> str:
    """Obtiene un user agent aleatorio de la lista de respaldo."""
    return random.choice(FALLBACK_USER_AGENTS)

class TennisDataCollector:
    """
    Clase optimizada para recopilar absolutamente todos los datos disponibles
    de los repositorios de Jeff Sackmann, incluyendo partidos, rankings, datos de jugadores,
    datos punto por punto y del Match Charting Project.
    """
    
    def __init__(self, 
            start_year: int = 1968, 
            end_year: Optional[int] = None,
            tours: List[str] = ['atp', 'wta'],
            include_challengers: bool = True,
            include_futures: bool = True,
            include_rankings: bool = True,
            include_pointbypoint: bool = True,
            include_slam_pointbypoint: bool = True,
            include_match_charting: bool = True,
            force_refresh: bool = False,
            delay_between_requests: float = 0.2):
        """
        Inicializa el colector mejorado de datos de tenis.
        """
        # Inicialización básica
        self.start_year = start_year
        self.end_year = end_year or datetime.now().year
        self.tours = tours
        self.include_challengers = include_challengers
        self.include_futures = include_futures
        self.include_rankings = include_rankings
        self.include_pointbypoint = include_pointbypoint
        self.include_slam_pointbypoint = include_slam_pointbypoint
        self.include_match_charting = include_match_charting
        self.force_refresh = force_refresh
        self.delay_between_requests = delay_between_requests
        
        # User agent y configuración de sesión HTTP
        self.user_agent = get_random_user_agent()
        self.session = self._create_robust_session()
        
        # Configurar directorios
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir = self.data_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir = self.data_dir / 'processed'
        self.output_dir.mkdir(exist_ok=True)
        
        # Crear directorios específicos para cada tour
        for tour in tours:
            (self.cache_dir / tour).mkdir(exist_ok=True)
            (self.output_dir / tour).mkdir(exist_ok=True)
        
        # Crear directorios para datos adicionales
        if self.include_pointbypoint:
            (self.cache_dir / 'pointbypoint').mkdir(exist_ok=True)
            (self.output_dir / 'pointbypoint').mkdir(exist_ok=True)
        
        if self.include_slam_pointbypoint:
            (self.cache_dir / 'slam_pointbypoint').mkdir(exist_ok=True)
            (self.output_dir / 'slam_pointbypoint').mkdir(exist_ok=True)
        
        if self.include_match_charting:
            (self.cache_dir / 'match_charting').mkdir(exist_ok=True)
            (self.output_dir / 'match_charting').mkdir(exist_ok=True)
        
        # Estructura de repositorios de Jeff Sackmann
        self.repo_base_url = "https://raw.githubusercontent.com/JeffSackmann"
        self.repo_structure = {
            'atp': {
                'main': '/tennis_atp/master',
                'rankings_dir': '/tennis_atp/master/rankings',
                'matches_pattern': 'atp_matches_{year}.csv',
                'players_file': 'atp_players.csv',
                # Nuevo formato para rankings consolidados por década
                'rankings_decades': {
                    '1970-1979': 'atp_rankings_70s.csv',
                    '1980-1989': 'atp_rankings_80s.csv',
                    '1990-1999': 'atp_rankings_90s.csv',
                    '2000-2009': 'atp_rankings_00s.csv',
                    '2010-2019': 'atp_rankings_10s.csv',
                    '2020-2029': 'atp_rankings_20s.csv',
                    'current': 'atp_rankings_current.csv'
                }
            },
            'wta': {
                'main': '/tennis_wta/master',
                'rankings_dir': '/tennis_wta/master/rankings',
                'matches_pattern': 'wta_matches_{year}.csv',
                'players_file': 'wta_players.csv',
                # Nuevo formato para rankings consolidados por década
                'rankings_decades': {
                    '1970-1979': 'wta_rankings_70s.csv',
                    '1980-1989': 'wta_rankings_80s.csv',
                    '1990-1999': 'wta_rankings_90s.csv',
                    '2000-2009': 'wta_rankings_00s.csv',
                    '2010-2019': 'wta_rankings_10s.csv',
                    '2020-2029': 'wta_rankings_20s.csv',
                    'current': 'wta_rankings_current.csv'
                }
            }
        }
        
        # Estructura de archivos Challenger/Futures
        self.challenger_patterns = {
            'atp': 'atp_matches_qual_chall_{year}.csv',
            'wta': 'wta_matches_qual_chall_{year}.csv'
        }
        self.futures_patterns = {
            'atp': 'atp_matches_futures_{year}.csv',
            'wta': 'wta_matches_futures_{year}.csv'
        }
        
        # Nuevas estructuras para repositorios adicionales
        self.slam_pointbypoint_structure = {
            'repo': '/tennis_slam_pointbypoint/master',
            'tournaments': {
                'us_open': 'usopen_points_{year}.csv',
                'australian_open': 'ausopen_points_{year}.csv',
                'wimbledon': 'wimbledon_points_{year}.csv',
                'french_open': 'frenchopen_points_{year}.csv'
            },
            # Directorios alternativos
            'tournament_dirs': {
                'usopen': ['usopen', 'us_open', 'us-open', 'uso'],
                'ausopen': ['ausopen', 'australian_open', 'aus_open', 'australian-open', 'ao'],
                'wimbledon': ['wimbledon', 'wim'],
                'frenchopen': ['frenchopen', 'french_open', 'french-open', 'roland_garros', 'rg', 'fo']
            },
            # Patrones alternativos de archivos
            'file_patterns': [
                '{tournament}_points_{year}.csv',
                '{tournament}_{year}_points.csv',
                '{tournament}_{year}.csv',
                'points_{year}.csv',
                '{year}.csv',
                '{year}_points.csv'
            ]
        }
        
        self.match_charting_structure = {
            'repo': '/tennis_MatchChartingProject/master',
            'files': {
                'matches': 'matches.csv',
                'shots': 'shots.csv',
                'charting_m_stats': 'charting-m-stats.csv',
                'charting_w_stats': 'charting-w-stats.csv'
            },
            # Archivos alternativos que también buscaremos
            'alt_files': [
                'match_stats.csv',
                'match_charting.csv',
                'charting_matches.csv',
                'charting_shots.csv',
                'matches_all.csv',
                'wta_matches_charted.csv',
                'atp_matches_charted.csv',
                'all_matches.csv',
                'all_shots.csv',
                'match_data.csv'
            ],
            # Posibles subdirectorios
            'subdirs': ['data', 'csv', 'stats', 'charting']
        }
        
        self.pointbypoint_structure = {
            'repo': '/tennis_pointbypoint/master',
            # Este repositorio tiene una estructura más compleja y variada
            'known_patterns': [
                '{year}/{tournament}/{match_id}.csv',
                '{tournament}/{year}/{match_id}.csv',
                '{year}_{tournament}/{match_id}.csv',
                '{tournament}_{year}/{match_id}.csv',
                '{year}/{match_id}.csv'
            ]
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
        
        # Estructura para rankings por década
        self.rankings_by_decade = {
            '70s': (1970, 1979),
            '80s': (1980, 1989),
            '90s': (1990, 1999),
            '00s': (2000, 2009),
            '10s': (2010, 2019),
            '20s': (2020, 2029)
        }

        # Patrones adicionales para rankings históricos
        self.ranking_patterns = [
            "{tour}_rankings_{year}.csv",
            "{tour}_rankings_{month}_{year}.csv",
            "{year}_{month}_{day}.csv",
            "{year}{month}{day}.csv",
            "{tour}_rankings_{decade}.csv"
        ]
        
        # AHORA verificamos los repositorios disponibles (después de definir repo_base_url)
        self.available_repos = self._check_available_repositories()
        logger.info(f"Repositorios disponibles: {', '.join(self.available_repos)}")
        
        # Actualizar flags basados en disponibilidad real
        if include_slam_pointbypoint and 'tennis_slam_pointbypoint' not in self.available_repos:
            logger.info("El repositorio 'tennis_slam_pointbypoint' no está disponible - se omitirán datos punto por punto de Grand Slam")
            self.include_slam_pointbypoint = False
        
        if include_match_charting and 'tennis_MatchChartingProject' not in self.available_repos:
            logger.info("El repositorio 'tennis_MatchChartingProject' no está disponible - se omitirán datos del Match Charting Project")
            self.include_match_charting = False
        
        # Verificar disponibilidad real de datos por tour
        self.available_data_types = self._check_available_data_types()
        
        # Filtrar opciones según disponibilidad real
        if include_challengers:
            for tour in self.tours:
                if tour not in self.available_data_types.get('challenger', []):
                    logger.info(f"Los datos de Challenger para {tour.upper()} no están disponibles - se omitirán")
        
        if include_futures:
            for tour in self.tours:
                if tour not in self.available_data_types.get('futures', []):
                    logger.info(f"Los datos de Futures para {tour.upper()} no están disponibles - se omitirán")
        
        logger.info(f"Inicializado colector mejorado de datos de tenis")
        if self.include_challengers:
            logger.info("Incluyendo partidos de Challenger")
        if self.include_futures:
            logger.info("Incluyendo partidos de Futures/ITF")
        if self.include_rankings:
            logger.info("Incluyendo datos de rankings")
        if self.include_pointbypoint:
            logger.info("Incluyendo datos punto por punto generales")
        if self.include_slam_pointbypoint:
            logger.info("Incluyendo datos punto por punto de Grand Slam")
        if self.include_match_charting:
            logger.info("Incluyendo datos del Match Charting Project")
        if self.force_refresh:
            logger.info("Modo force_refresh activado: se recopilarán todos los datos de nuevo")
    
    def _check_available_repositories(self) -> List[str]:
        """
        Verifica qué repositorios de Jeff Sackmann están realmente disponibles.
        
        Returns:
            Lista de nombres de repositorios disponibles
        """
        repos_to_check = [
            'tennis_atp',
            'tennis_wta',
            'tennis_slam_pointbypoint',
            'tennis_MatchChartingProject',
            'tennis_pointbypoint'
        ]
        
        available_repos = []
        
        for repo in repos_to_check:
            # Verificar si el repositorio existe probando múltiples archivos que pueden existir
            test_urls = []
            
            if repo == 'tennis_atp':
                test_urls = [
                    f"{self.repo_base_url}/{repo}/master/atp_matches_2000.csv",
                    f"{self.repo_base_url}/{repo}/master/atp_matches_2019.csv",
                    f"{self.repo_base_url}/{repo}/master/README.md"
                ]
            elif repo == 'tennis_wta':
                test_urls = [
                    f"{self.repo_base_url}/{repo}/master/wta_matches_2000.csv",
                    f"{self.repo_base_url}/{repo}/master/wta_matches_2019.csv",
                    f"{self.repo_base_url}/{repo}/master/README.md"
                ]
            elif repo == 'tennis_slam_pointbypoint':
                test_urls = [
                    f"{self.repo_base_url}/{repo}/master/README.md",
                    f"{self.repo_base_url}/{repo}/master/usopen_points_2019.csv",
                    f"{self.repo_base_url}/{repo}/master/ausopen/points_2019.csv",
                    f"{self.repo_base_url}/{repo}/master/wimbledon/2019.csv"
                ]
            elif repo == 'tennis_MatchChartingProject':
                test_urls = [
                    f"{self.repo_base_url}/{repo}/master/README.md",
                    f"{self.repo_base_url}/{repo}/master/matches.csv",
                    f"{self.repo_base_url}/{repo}/master/charting-m-stats.csv",
                    f"{self.repo_base_url}/{repo}/master/data/matches.csv"
                ]
            elif repo == 'tennis_pointbypoint':
                test_urls = [
                    f"{self.repo_base_url}/{repo}/master/README.md",
                    f"{self.repo_base_url}/{repo}/master/2019/usopen/1.csv",
                    f"{self.repo_base_url}/{repo}/master/usopen/2019/1.csv"
                ]
            
            # Verificar si alguna de las URLs existe
            for url in test_urls:
                if self._check_file_exists(url):
                    available_repos.append(repo)
                    logger.debug(f"Repositorio {repo} disponible (verificado con {url})")
                    break
                
        return available_repos

    def _check_rankings_structure(self, tour: str) -> Dict[str, str]:
        """
        Verifica la estructura real del directorio de rankings para un tour específico.
        
        Args:
            tour: Tour de tenis ('atp' o 'wta')
            
        Returns:
            Diccionario con la estructura de archivos de rankings disponibles
        """
        rankings_structure = {}
        
        # Verificar directamente el archivo de rankings actuales en la raíz
        current_rankings_url = f"{self.repo_base_url}{self.repo_structure[tour]['main']}/{tour}_rankings_current.csv"
        if self._check_file_exists(current_rankings_url):
            rankings_structure['format'] = 'root'
            rankings_structure['filename'] = f"{tour}_rankings_current.csv"
            rankings_structure['url'] = current_rankings_url
            logger.info(f"Rankings {tour.upper()} actuales disponibles en la raíz: {tour}_rankings_current.csv")
            return rankings_structure
        
        # El resto del código puede permanecer igual, ya que es una buena red de seguridad
        # para encontrar rankings en otras ubicaciones posibles
        
        # Comprobar si existe el directorio de rankings
        rankings_dir = f"{self.repo_base_url}{self.repo_structure[tour]['rankings_dir']}"
        
        # Intentar diferentes estructuras posibles
        
        # 1. Archivos semanales formato fecha (YYYYMMDD.csv)
        # Para comprobar esto, probar con algunas fechas recientes conocidas
        sample_dates = ['20240101', '20230101', '20220101', '20210101']
        weekly_format_exists = False
        
        for date in sample_dates:
            url = f"{rankings_dir}/{date}.csv"
            if self._check_file_exists(url):
                weekly_format_exists = True
                rankings_structure['format'] = 'weekly'
                logger.info(f"Rankings {tour.upper()} disponibles en formato semanal (YYYYMMDD.csv)")
                break
        
        # 2. Archivos por década
        decade_files = {
            '1970-1979': f"{tour}_rankings_70s.csv",
            '1980-1989': f"{tour}_rankings_80s.csv",
            '1990-1999': f"{tour}_rankings_90s.csv",
            '2000-2009': f"{tour}_rankings_00s.csv",
            '2010-2019': f"{tour}_rankings_10s.csv",
            '2020-2029': f"{tour}_rankings_20s.csv"
        }
        
        found_decades = []
        for decade, filename in decade_files.items():
            url = f"{rankings_dir}/{filename}"
            if self._check_file_exists(url):
                found_decades.append(decade)
        
        if found_decades:
            rankings_structure['format'] = 'decade'
            rankings_structure['decades'] = found_decades
            logger.info(f"Rankings {tour.upper()} disponibles en formato por década: {', '.join(found_decades)}")
        
        # 3. Archivo agregado o consolidado
        consolidated_formats = [
            f"{tour}_rankings.csv",
            f"{tour}_rankings_all.csv",
            "rankings.csv",
            "rankings_all.csv",
            f"{tour}_rankings_current.csv",
            "rankings_archive.csv",
            f"{tour}_rankings_archive.csv",
            f"{tour}_rankings_history.csv"
        ]

        additional_urls = [
            f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/rankings_archive/{tour}_rankings_all.csv",
            f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/rankings/{tour}_rankings_all.csv",
            f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/rankings/current.csv"
        ]

        for url in additional_urls:
            if self._check_file_exists(url):
                rankings_structure['format'] = 'external'
                rankings_structure['url'] = url
                logger.info(f"Rankings {tour.upper()} disponibles en URL externa: {url}")
                break
        
        for filename in consolidated_formats:
            url = f"{rankings_dir}/{filename}"
            if self._check_file_exists(url):
                rankings_structure['format'] = 'consolidated'
                rankings_structure['filename'] = filename
                logger.info(f"Rankings {tour.upper()} disponibles en formato consolidado: {filename}")
                break
        
        # Si no se detectó ningún formato
        if not rankings_structure:
            # Probar directamente el README para ver si hay información
            readme_url = f"{rankings_dir}/README.md"
            if self._check_file_exists(readme_url):
                logger.info(f"Encontrado README en directorio de rankings {tour.upper()}. Revisar manualmente para más información.")
            else:
                # Último intento: buscar en la raíz del repositorio
                for filename in consolidated_formats:
                    url = f"{self.repo_base_url}{self.repo_structure[tour]['main']}/{filename}"
                    if self._check_file_exists(url):
                        rankings_structure['format'] = 'root'
                        rankings_structure['filename'] = filename
                        logger.info(f"Rankings {tour.upper()} disponibles en la raíz del repositorio: {filename}")
                        break
        
        if not rankings_structure:
            logger.warning(f"No se pudo determinar el formato de rankings para {tour.upper()}")
        
        return rankings_structure

    def _check_available_data_types(self) -> Dict[str, List[str]]:
        """
        Verifica qué tipos de datos están realmente disponibles para cada tour.
        
        Returns:
            Diccionario con tipos de datos disponibles por tour
        """
        available_types = {
            'main': [],
            'challenger': [],
            'futures': [],
            'rankings': []
        }
        
        # Verificamos un año reciente para mayor probabilidad de encontrar datos
        sample_years = [2022, 2020, 2018, 2015]
        
        for tour in self.tours:
            # Verificar partidos principales (casi seguro que existen)
            main_found = False
            for year in sample_years:
                main_pattern = self.repo_structure[tour]['matches_pattern']
                main_filename = main_pattern.format(year=year)
                main_url = f"{self.repo_base_url}{self.repo_structure[tour]['main']}/{main_filename}"
                
                if self._check_file_exists(main_url):
                    available_types['main'].append(tour)
                    main_found = True
                    break
            
            if not main_found:
                logger.warning(f"¡No se encontraron archivos de partidos principales para {tour.upper()}!")
            
            # Verificar partidos challenger
            if self.include_challengers:
                chall_found = False
                for year in sample_years:
                    chall_pattern = self.challenger_patterns[tour]
                    chall_filename = chall_pattern.format(year=year)
                    chall_url = f"{self.repo_base_url}{self.repo_structure[tour]['main']}/{chall_filename}"
                    
                    if self._check_file_exists(chall_url):
                        available_types['challenger'].append(tour)
                        chall_found = True
                        break
                
                if not chall_found:
                    logger.info(f"Partidos Challenger no disponibles para {tour.upper()} (verificado con múltiples años)")
            
            # Verificar partidos futures
            if self.include_futures:
                futures_found = False
                for year in sample_years:
                    futures_pattern = self.futures_patterns[tour]
                    futures_filename = futures_pattern.format(year=year)
                    futures_url = f"{self.repo_base_url}{self.repo_structure[tour]['main']}/{futures_filename}"
                    
                    if self._check_file_exists(futures_url):
                        available_types['futures'].append(tour)
                        futures_found = True
                        break
                
                if not futures_found:
                    logger.info(f"Partidos Futures no disponibles para {tour.upper()} (verificado con múltiples años)")
            
            # Verificar rankings - solo verificamos si el directorio existe, la estructura detallada se verificará más tarde
            rankings_dir_url = f"{self.repo_base_url}{self.repo_structure[tour]['rankings_dir']}"
            readme_url = f"{rankings_dir_url}/README.md"
            
            if self._check_file_exists(readme_url):
                available_types['rankings'].append(tour)
                logger.debug(f"Directorio de rankings encontrado para {tour.upper()}")
            else:
                # Intentar con algunos archivos posibles en el directorio principal
                possible_ranking_files = [
                    f"{tour}_rankings.csv",
                    "rankings.csv",
                    f"{tour}_rankings_current.csv"
                ]
                
                for filename in possible_ranking_files:
                    url = f"{self.repo_base_url}{self.repo_structure[tour]['main']}/{filename}"
                    if self._check_file_exists(url):
                        available_types['rankings'].append(tour)
                        logger.debug(f"Archivo de rankings encontrado para {tour.upper()} en la raíz del repositorio")
                        break
        
        return available_types
    

    def _create_robust_session(self) -> requests.Session:
        """Crea una sesión HTTP con reintentos automáticos y tiempos de espera optimizados."""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({'User-Agent': self.user_agent})
        return session
    
    def _make_request(self, url: str, timeout: int = 30) -> requests.Response:
        """
        Realiza una solicitud HTTP con control de retrasos para evitar limitaciones.
        
        Args:
            url: URL a solicitar
            timeout: Tiempo de espera máximo para la solicitud
            
        Returns:
            Objeto Response de la solicitud
        """
        response = self.session.get(url, timeout=timeout)
        time.sleep(self.delay_between_requests)  # Pausa para evitar limitaciones
        return response
    
    def _download_file(self, url: str, cache_file: Path) -> pd.DataFrame:
        """
        Descarga un archivo y lo guarda en caché.
        
        Args:
            url: URL del archivo a descargar
            cache_file: Ruta donde guardar el archivo descargado
            
        Returns:
            DataFrame con los datos descargados
        """
        try:
            # Verificar caché
            if cache_file.exists() and not self.force_refresh:
                logger.debug(f"Usando archivo en caché: {cache_file}")
                return pd.read_csv(cache_file, low_memory=False)
            
            # Asegurar que el directorio de caché existe
            cache_file.parent.mkdir(exist_ok=True)
            
            # Descargar archivo
            logger.debug(f"Descargando {url}...")
            response = self._make_request(url)
            response.raise_for_status()
            
            # Analizar CSV
            content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(content), low_memory=False)
            
            # Guardar en caché
            df.to_csv(cache_file, index=False)
            logger.debug(f"Archivo descargado y guardado en caché: {cache_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error descargando {url}: {str(e)}")
            raise
    
    def _get_decade_for_year(self, year: int) -> str:
        """
        Determina la clave de década para un año específico.
        
        Args:
            year: Año para el que se busca la década
            
        Returns:
            Clave de década en el formato '1970-1979', etc.
        """
        # Calcular la década basada en el año
        decade_start = (year // 10) * 10
        decade_end = decade_start + 9
        return f"{decade_start}-{decade_end}"
    
    def _check_file_exists(self, url: str) -> bool:
        """
        Verifica si un archivo existe en el repositorio remoto.
        
        Args:
            url: URL del archivo a verificar
            
        Returns:
            True si el archivo existe, False si no
        """
        try:
            response = self.session.head(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def _search_files_recursively(self, base_url: str, patterns: List[str]) -> List[str]:
        """
        Busca archivos recursivamente en diferentes patrones posibles.
        
        Args:
            base_url: URL base para la búsqueda
            patterns: Lista de patrones a probar
            
        Returns:
            Lista de URLs encontradas
        """
        found_urls = []
        
        for pattern in patterns:
            url = f"{base_url}/{pattern}"
            if self._check_file_exists(url):
                found_urls.append(url)
        
        return found_urls

    def _download_yearly_matches(self, year: int, tour: str, match_type: str = 'main') -> pd.DataFrame:
        """
        Descarga datos de partidos de un año específico.
        
        Args:
            year: Año de los datos a descargar
            tour: Tour de tenis ('atp' o 'wta')
            match_type: Tipo de partidos ('main', 'challenger', 'futures')
            
        Returns:
            DataFrame con los datos del año
        """
        if tour not in self.repo_structure:
            logger.error(f"Tour no soportado: {tour}")
            return pd.DataFrame()
        
        try:
            # Determinar patrón de archivo según tipo de partidos
            if match_type == 'main':
                file_pattern = self.repo_structure[tour]['matches_pattern']
            elif match_type == 'challenger':
                file_pattern = self.challenger_patterns[tour]
            elif match_type == 'futures':
                file_pattern = self.futures_patterns[tour]
            else:
                logger.error(f"Tipo de partidos no soportado: {match_type}")
                return pd.DataFrame()
            
            # Formar nombre de archivo
            filename = file_pattern.format(year=year)
            
            # Formar URL
            base_path = self.repo_structure[tour]['main']
            url = f"{self.repo_base_url}{base_path}/{filename}"
            
            # Para WTA, verificar si los archivos de challenger/futures existen antes de intentar descargarlos
            if tour == 'wta' and match_type in ['challenger', 'futures']:
                if not self._check_file_exists(url):
                    logger.debug(f"Archivo {match_type} {tour.upper()} {year} no disponible: {url}")
                    return pd.DataFrame()
            
            # Formar ruta de caché
            cache_file = self.cache_dir / tour / f"{filename}"
            
            # Descargar datos
            logger.info(f"Descargando {match_type} {tour.upper()} {year}...")
            df = self._download_file(url, cache_file)
            
            # Añadir columnas útiles
            df['tour'] = tour
            df['match_type'] = match_type
            df['year'] = year
            
            # Verificar si hay datos
            if df.empty:
                logger.warning(f"No hay datos para {match_type} {tour.upper()} {year}")
                return pd.DataFrame()
            
            # Información básica
            logger.info(f"Datos de {match_type} {tour.upper()} {year} descargados: {len(df)} partidos")
            
            return df
            
        except Exception as e:
            logger.warning(f"Error descargando {match_type} {tour.upper()} {year}: {str(e)}")
            return pd.DataFrame()
        
    
    def _download_rankings_data(self, tour: str) -> pd.DataFrame:
        """
        Descarga y procesa datos de rankings directamente desde la raíz del repositorio.
        """
        if not self.include_rankings:
            logger.info(f"Descarga de rankings desactivada para {tour.upper()}")
            return pd.DataFrame()
        
        logger.info(f"=== Descargando archivos de rankings para {tour.upper()} ===")
        
        # URL base del repositorio
        base_url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master"
        
        # DataFrame para almacenar todos los rankings
        all_rankings = pd.DataFrame()
        
        # Lista de archivos de décadas que debemos intentar descargar
        decade_files = [
            f"{tour}_rankings_00s.csv",
            f"{tour}_rankings_10s.csv",
            f"{tour}_rankings_20s.csv",
            f"{tour}_rankings_current.csv"
        ]
        
        # Descargar cada archivo de rankings
        for decade_file in decade_files:
            url = f"{base_url}/{decade_file}"
            
            if self._check_file_exists(url):
                logger.info(f"Descargando archivo de rankings: {decade_file}")
                
                # Crear ruta de caché
                cache_file = self.cache_dir / tour / 'rankings' / decade_file
                cache_file.parent.mkdir(exist_ok=True)
                
                try:
                    # Descargar el archivo directamente sin pasar por _download_file para manejo especializado
                    response = self._make_request(url)
                    content = response.content.decode('utf-8')
                    
                    # Guardar copia en caché
                    with open(cache_file, 'w') as f:
                        f.write(content)
                    
                    # Leer el CSV
                    df = pd.read_csv(StringIO(content))
                    
                    if not df.empty:
                        # Asegurarse de que todos los datos tienen la columna 'tour'
                        df['tour'] = tour
                        
                        # IMPORTANTE: No intentar convertir o filtrar por fecha todavía
                        # Simplemente guardar todos los datos
                        all_rankings = pd.concat([all_rankings, df], ignore_index=True)
                        logger.info(f"Añadidos {len(df)} registros de rankings de {decade_file}")
                except Exception as e:
                    logger.warning(f"Error procesando {decade_file}: {str(e)}")
        
        # Guardar el archivo sin filtrar primero
        if not all_rankings.empty:
            # Guardar una copia completa sin filtrar
            output_file_all = self.output_dir / tour / f"{tour}_rankings_all.csv"
            all_rankings.to_csv(output_file_all, index=False)
            logger.info(f"Guardados {len(all_rankings)} registros de rankings completos para {tour.upper()} en {output_file_all}")
            
            # Intentar filtrar por fecha ahora, pero si falla, al menos tenemos el archivo completo
            try:
                if 'ranking_date' in all_rankings.columns:
                    # Primero verificar el formato de fecha que tienen los archivos
                    sample_date = all_rankings['ranking_date'].iloc[0] if len(all_rankings) > 0 else None
                    logger.info(f"Muestra de fecha en rankings: {sample_date}")
                    
                    # Probamos diferentes formatos de fecha
                    filtered_rankings = pd.DataFrame()
                    
                    # Intentar varias estrategias de filtrado por año
                    # 1. Si la fecha ya está en formato YYYY-MM-DD
                    if re.match(r'\d{4}-\d{2}-\d{2}', str(sample_date)):
                        all_rankings['ranking_date'] = pd.to_datetime(all_rankings['ranking_date'])
                        filtered_rankings = all_rankings[(all_rankings['ranking_date'].dt.year >= self.start_year) & 
                                                        (all_rankings['ranking_date'].dt.year <= self.end_year)]
                        logger.info(f"Filtrado por formato YYYY-MM-DD")
                    
                    # 2. Si la fecha está en otro formato como YYYYMMDD
                    elif re.match(r'\d{8}', str(sample_date)):
                        all_rankings['year'] = all_rankings['ranking_date'].astype(str).str[:4].astype(int)
                        filtered_rankings = all_rankings[(all_rankings['year'] >= self.start_year) & 
                                                        (all_rankings['year'] <= self.end_year)]
                        logger.info(f"Filtrado por formato YYYYMMDD")
                    
                    # 3. Si la fecha está en otro formato o es un string, intentar extraer el año
                    else:
                        # Intentar varios formatos de fecha comunes
                        date_formats = ['%Y%m%d', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y']
                        for date_format in date_formats:
                            try:
                                all_rankings['date_temp'] = pd.to_datetime(all_rankings['ranking_date'], format=date_format)
                                all_rankings['year'] = all_rankings['date_temp'].dt.year
                                filtered_rankings = all_rankings[(all_rankings['year'] >= self.start_year) & 
                                                                (all_rankings['year'] <= self.end_year)]
                                if not filtered_rankings.empty:
                                    logger.info(f"Filtrado por formato {date_format}")
                                    break
                            except:
                                continue
                    
                    # Si ninguno de los formatos anteriores funcionó, intentar extraer el año usando regex
                    if filtered_rankings.empty:
                        all_rankings['year'] = all_rankings['ranking_date'].astype(str).str.extract(r'(\d{4})').astype(int)
                        filtered_rankings = all_rankings[(all_rankings['year'] >= self.start_year) & 
                                                        (all_rankings['year'] <= self.end_year)]
                        logger.info(f"Filtrado por extracción de año usando regex")
                    
                    # Si aún así no funciona, guardar todo sin filtrar
                    if filtered_rankings.empty:
                        logger.warning(f"No se pudo filtrar por año, guardando todos los rankings disponibles")
                        filtered_rankings = all_rankings
                    
                    # Guardar el archivo filtrado
                    output_file = self.output_dir / tour / f"{tour}_rankings_{self.start_year}_{self.end_year}.csv"
                    filtered_rankings.to_csv(output_file, index=False)
                    logger.info(f"Guardados {len(filtered_rankings)} registros de rankings filtrados para {tour.upper()} en {output_file}")
                    
                    return filtered_rankings
                else:
                    logger.warning(f"No hay columna 'ranking_date' en los rankings, guardando todos sin filtrar")
                    return all_rankings
            except Exception as e:
                logger.warning(f"Error al filtrar rankings por fecha: {str(e)}")
                logger.info(f"Guardando todos los rankings sin filtrar")
                return all_rankings
        else:
            logger.warning(f"No se encontraron rankings para {tour.upper()}")
        
        return all_rankings

    def _download_weekly_rankings(self, year: int, tour: str) -> pd.DataFrame:
        """
        Descarga rankings semanales para un año específico.
        
        Args:
            year: Año de los rankings a descargar
            tour: Tour de tenis ('atp' o 'wta')
            
        Returns:
            DataFrame con rankings del año
        """
        # Encontrar fechas disponibles (lunes) para este año
        dates = []
        date = datetime(year, 1, 1)
        while date.year == year:
            if date.weekday() == 0:  # Lunes
                date_str = date.strftime("%Y%m%d")
                dates.append(date_str)
            date += timedelta(days=1)
        
        # Intentar descargar rankings para cada fecha
        rankings_frames = []
        
        for date_str in dates:
            url = f"{self.repo_base_url}{self.repo_structure[tour]['rankings_dir']}/{date_str}.csv"
            cache_file = self.cache_dir / tour / 'rankings' / f"{date_str}.csv"
            cache_file.parent.mkdir(exist_ok=True)
            
            try:
                if self._check_file_exists(url):
                    df = self._download_file(url, cache_file)
                    df['ranking_date'] = date_str
                    df['tour'] = tour
                    rankings_frames.append(df)
            except Exception as e:
                logger.debug(f"No se pudo descargar rankings para {tour.upper()} {date_str}: {str(e)}")
        
        # Combinar todos los marcos
        if rankings_frames:
            return pd.concat(rankings_frames, ignore_index=True)
        
        return pd.DataFrame()

    def _download_decade_rankings(self, decade: str, tour: str) -> pd.DataFrame:
        """
        Descarga rankings consolidados de una década específica.
        
        Args:
            decade: Década en formato '1970-1979', etc.
            tour: Tour de tenis ('atp' o 'wta')
            
        Returns:
            DataFrame con rankings de la década
        """
        if tour not in self.repo_structure:
            logger.error(f"Tour no soportado: {tour}")
            return pd.DataFrame()
        
        try:
            # Determinar el archivo a descargar
            if decade not in self.repo_structure[tour]['rankings_decades']:
                logger.warning(f"Década no soportada para rankings: {decade}")
                return pd.DataFrame()
            
            filename = self.repo_structure[tour]['rankings_decades'][decade]
            
            # Formar URL
            rankings_dir = self.repo_structure[tour]['rankings_dir']
            url = f"{self.repo_base_url}{rankings_dir}/{filename}"
            
            # Formar ruta de caché
            cache_file = self.cache_dir / tour / 'rankings' / f"{filename}"
            cache_file.parent.mkdir(exist_ok=True)
            
            # Descargar datos
            logger.info(f"Descargando rankings de {tour.upper()} para la década {decade}...")
            df = self._download_file(url, cache_file)
            
            # Añadir columna útil
            df['tour'] = tour
            
            # Verificar si hay datos
            if df.empty:
                logger.warning(f"No hay rankings para {tour.upper()} década {decade}")
                return pd.DataFrame()
            
            # Información básica
            dates_count = df['ranking_date'].nunique() if 'ranking_date' in df.columns else 0
            logger.info(f"Rankings de {tour.upper()} década {decade} descargados: {len(df)} registros, {dates_count} fechas")
            
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"No hay rankings disponibles para {tour.upper()} década {decade}")
            else:
                logger.warning(f"Error HTTP descargando rankings de {tour.upper()} década {decade}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error descargando rankings de {tour.upper()} década {decade}: {str(e)}")
            return pd.DataFrame()
    
    def _download_decade_rankings_and_filter(self, year: int, decade: str, tour: str) -> pd.DataFrame:
        """
        Descarga rankings por década y filtra para un año específico.
        
        Args:
            year: Año a filtrar
            decade: Década en formato '1970-1979'
            tour: Tour de tenis ('atp' o 'wta')
            
        Returns:
            DataFrame con rankings del año
        """
        # Determinar el nombre del archivo
        if decade == '1970-1979':
            filename = f"{tour}_rankings_70s.csv"
        elif decade == '1980-1989':
            filename = f"{tour}_rankings_80s.csv"
        elif decade == '1990-1999':
            filename = f"{tour}_rankings_90s.csv"
        elif decade == '2000-2009':
            filename = f"{tour}_rankings_00s.csv"
        elif decade == '2010-2019':
            filename = f"{tour}_rankings_10s.csv"
        elif decade == '2020-2029':
            filename = f"{tour}_rankings_20s.csv"
        else:
            return pd.DataFrame()
        
        # Descargar archivo de década
        url = f"{self.repo_base_url}{self.repo_structure[tour]['rankings_dir']}/{filename}"
        cache_file = self.cache_dir / tour / 'rankings' / f"{filename}"
        cache_file.parent.mkdir(exist_ok=True)
        
        try:
            df = self._download_file(url, cache_file)
            df['tour'] = tour
            
            # Filtrar por año
            year_str = str(year)
            if 'ranking_date' in df.columns:
                # Asegurarse de que ranking_date sea una columna de strings
                df['ranking_date'] = df['ranking_date'].astype(str)
                
                # Ahora filtramos por año
                return df[df['ranking_date'].str.startswith(year_str)]
            else:
                logger.warning(f"No se pudo filtrar rankings por año: columna 'ranking_date' no encontrada")
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error descargando rankings por década para {tour.upper()} {decade}: {str(e)}")
            return pd.DataFrame()

    def _download_consolidated_rankings_and_filter(self, year: int, tour: str, format_type: str) -> pd.DataFrame:
        """
        Descarga rankings consolidados y filtra para un año específico.
        
        Args:
            year: Año a filtrar
            tour: Tour de tenis ('atp' o 'wta')
            format_type: Tipo de formato ('consolidated' o 'root')
            
        Returns:
            DataFrame con rankings del año
        """
        # Obtener el nombre del archivo
        filename = self.rankings_structure[tour].get('filename')
        if not filename:
            return pd.DataFrame()
        
        # Determinar la URL según el formato
        if format_type == 'consolidated':
            url = f"{self.repo_base_url}{self.repo_structure[tour]['rankings_dir']}/{filename}"
        else:  # 'root'
            url = f"{self.repo_base_url}{self.repo_structure[tour]['main']}/{filename}"
        
        cache_file = self.cache_dir / tour / 'rankings' / f"{filename}"
        cache_file.parent.mkdir(exist_ok=True)
        
        try:
            df = self._download_file(url, cache_file)
            df['tour'] = tour
            
            # Filtrar por año
            year_str = str(year)
            if 'ranking_date' in df.columns:
                # Asegurarse de que ranking_date sea una columna de strings
                df['ranking_date'] = df['ranking_date'].astype(str)
                
                # Ahora filtramos por año
                return df[df['ranking_date'].str.startswith(year_str)]
            else:
                logger.warning(f"No se pudo filtrar rankings por año: columna 'ranking_date' no encontrada")
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error descargando rankings consolidados para {tour.upper()}: {str(e)}")
            return pd.DataFrame()

    def _search_rankings_advanced(self, tour: str, year: int) -> pd.DataFrame:
        """
        Búsqueda avanzada de rankings para un año específico.
        Prueba múltiples formatos y ubicaciones posibles.
        
        Args:
            tour: Tour de tenis ('atp' o 'wta')
            year: Año a buscar
            
        Returns:
            DataFrame con rankings del año
        """
        logger.info(f"Iniciando búsqueda avanzada de rankings para {tour.upper()} {year}...")
        
        # Generar fechas para este año (lunes)
        dates = []
        date = datetime(year, 1, 1)
        while date.year == year:
            if date.weekday() == 0:  # Lunes
                dates.append(date)
            date += timedelta(days=1)
        
        # Formatear fechas en diversos formatos
        date_formats = [
            lambda d: d.strftime("%Y%m%d"),              # 20240101
            lambda d: d.strftime("%Y-%m-%d"),            # 2024-01-01
            lambda d: d.strftime("%Y_%m_%d"),            # 2024_01_01
            lambda d: d.strftime("%d%m%Y"),              # 01012024
            lambda d: d.strftime("%d-%m-%Y"),            # 01-01-2024
            lambda d: d.strftime("%d_%m_%Y"),            # 01_01_2024
            lambda d: f"{d.year}_{d.month:02d}_{d.day:02d}",  # 2024_01_01
            lambda d: f"{d.year}{d.month:02d}{d.day:02d}",    # 20240101
            lambda d: f"{d.day:02d}{d.month:02d}{d.year}",    # 01012024
        ]
        
        # Construir lista de posibles directorios
        directories = [
            f"{self.repo_structure[tour]['rankings_dir']}",
            f"{self.repo_structure[tour]['main']}/rankings",
            f"{self.repo_structure[tour]['main']}/{year}/rankings",
            f"{self.repo_structure[tour]['main']}/rankings/{year}",
            f"{self.repo_structure[tour]['main']}",
        ]
        
        # Década para este año
        decade = self._get_decade_for_year(year)
        decade_short = f"{str(decade)[:4]}s"  # Convertir "1970-1979" a "1970s"
        
        # Patrones de archivos por década
        decade_patterns = [
            f"{tour}_rankings_{decade_short}.csv",
            f"{tour}_rankings_{str(decade)[:4]}_{str(decade)[5:]}.csv",
            f"{tour}_{decade_short}_rankings.csv",
            f"rankings_{decade_short}.csv",
            f"rankings_{tour}_{decade_short}.csv",
        ]
        
        # Patrones de archivos por año
        year_patterns = [
            f"{tour}_rankings_{year}.csv",
            f"{tour}_{year}_rankings.csv",
            f"rankings_{tour}_{year}.csv",
            f"rankings_{year}.csv",
            f"{year}_rankings.csv",
            f"{year}_{tour}_rankings.csv",
        ]
        
        # Buscar archivos por fecha (semanal/mensual)
        rankings_frames = []
        
        # 1. Primero intentar con archivos por fecha
        for directory in directories:
            base_url = f"{self.repo_base_url}{directory}"
            
            for date in dates:
                for date_format in date_formats:
                    date_str = date_format(date)
                    url = f"{base_url}/{date_str}.csv"
                    
                    if self._check_file_exists(url):
                        logger.info(f"Encontrado archivo de rankings por fecha: {url}")
                        
                        try:
                            cache_file = self.cache_dir / tour / 'rankings' / f"{date_str}.csv"
                            cache_file.parent.mkdir(exist_ok=True)
                            
                            df = self._download_file(url, cache_file)
                            if not df.empty:
                                # Añadir fecha si no existe
                                if 'ranking_date' not in df.columns:
                                    df['ranking_date'] = date.strftime("%Y-%m-%d")
                                df['tour'] = tour
                                rankings_frames.append(df)
                        except Exception as e:
                            logger.warning(f"Error descargando {url}: {str(e)}")
            
            # Si encontramos al menos un archivo de rankings por fecha, detenemos la búsqueda
            if rankings_frames:
                break
        
        # 2. Si no encontramos rankings por fecha, intentar con archivos por año
        if not rankings_frames:
            for directory in directories:
                base_url = f"{self.repo_base_url}{directory}"
                
                for pattern in year_patterns:
                    url = f"{base_url}/{pattern}"
                    
                    if self._check_file_exists(url):
                        logger.info(f"Encontrado archivo de rankings por año: {url}")
                        
                        try:
                            cache_file = self.cache_dir / tour / 'rankings' / pattern
                            cache_file.parent.mkdir(exist_ok=True)
                            
                            df = self._download_file(url, cache_file)
                            if not df.empty:
                                # Filtrar por año si hay fechas
                                if 'ranking_date' in df.columns:
                                    df['ranking_date'] = pd.to_datetime(df['ranking_date'])
                                    df = df[df['ranking_date'].dt.year == year]
                                df['tour'] = tour
                                rankings_frames.append(df)
                        except Exception as e:
                            logger.warning(f"Error descargando {url}: {str(e)}")
                
                # Si encontramos al menos un archivo de rankings por año, detenemos la búsqueda
                if rankings_frames:
                    break
        
        # 3. Si no encontramos rankings por año, intentar con archivos por década y filtrar
        if not rankings_frames:
            for directory in directories:
                base_url = f"{self.repo_base_url}{directory}"
                
                for pattern in decade_patterns:
                    url = f"{base_url}/{pattern}"
                    
                    if self._check_file_exists(url):
                        logger.info(f"Encontrado archivo de rankings por década: {url}")
                        
                        try:
                            cache_file = self.cache_dir / tour / 'rankings' / pattern
                            cache_file.parent.mkdir(exist_ok=True)
                            
                            df = self._download_file(url, cache_file)
                            if not df.empty:
                                # Filtrar por año si hay fechas
                                if 'ranking_date' in df.columns:
                                    # Asegurarse de que ranking_date sea datetime
                                    df['ranking_date'] = pd.to_datetime(df['ranking_date'])
                                    df = df[df['ranking_date'].dt.year == year]
                                df['tour'] = tour
                                rankings_frames.append(df)
                        except Exception as e:
                            logger.warning(f"Error descargando {url}: {str(e)}")
                
                # Si encontramos al menos un archivo de rankings por década, detenemos la búsqueda
                if rankings_frames:
                    break
        
        # 4. Intentar con archivos de rankings actuales o anuales conocidos
        if not rankings_frames:
            known_files = [
                f"{tour}_rankings_current.csv",
                f"{tour}_rankings_all.csv",
                f"{tour}_rankings.csv",
                "rankings_current.csv",
                "rankings_all.csv",
                "rankings.csv"
            ]
            
            for directory in directories:
                base_url = f"{self.repo_base_url}{directory}"
                
                for filename in known_files:
                    url = f"{base_url}/{filename}"
                    
                    if self._check_file_exists(url):
                        logger.info(f"Encontrado archivo de rankings conocido: {url}")
                        
                        try:
                            cache_file = self.cache_dir / tour / 'rankings' / filename
                            cache_file.parent.mkdir(exist_ok=True)
                            
                            df = self._download_file(url, cache_file)
                            if not df.empty:
                                # Filtrar por año si hay fechas
                                if 'ranking_date' in df.columns:
                                    # Asegurarse de que ranking_date sea datetime
                                    df['ranking_date'] = pd.to_datetime(df['ranking_date'])
                                    df = df[df['ranking_date'].dt.year == year]
                                elif 'date' in df.columns:
                                    df['date'] = pd.to_datetime(df['date'])
                                    df = df[df['date'].dt.year == year]
                                    df['ranking_date'] = df['date']
                                
                                df['tour'] = tour
                                rankings_frames.append(df)
                        except Exception as e:
                            logger.warning(f"Error descargando {url}: {str(e)}")
                
                # Si encontramos al menos un archivo de rankings conocido, detenemos la búsqueda
                if rankings_frames:
                    break
        
        # Combinar todos los rankings y eliminar duplicados
        if rankings_frames:
            combined_rankings = pd.concat(rankings_frames, ignore_index=True)
            
            # Eliminar duplicados si hay una columna de fecha
            if 'ranking_date' in combined_rankings.columns and 'player_id' in combined_rankings.columns:
                combined_rankings = combined_rankings.drop_duplicates(subset=['ranking_date', 'player_id'])
            
            logger.info(f"Se encontraron {len(combined_rankings)} registros de rankings para {tour.upper()} {year} (búsqueda avanzada)")
            return combined_rankings
        
        logger.warning(f"No se encontraron rankings para {tour.upper()} {year} en búsqueda avanzada")
        return pd.DataFrame()

    def _download_players_data(self, tour: str) -> pd.DataFrame:
        """
        Descarga datos de jugadores para un tour específico.
        
        Args:
            tour: Tour de tenis ('atp' o 'wta')
            
        Returns:
            DataFrame con datos de jugadores
        """
        if tour not in self.repo_structure:
            logger.error(f"Tour no soportado: {tour}")
            return pd.DataFrame()
        
        try:
            # Formar nombre de archivo
            filename = self.repo_structure[tour]['players_file']
            
            # Formar URL
            base_path = self.repo_structure[tour]['main']
            url = f"{self.repo_base_url}{base_path}/{filename}"
            
            # Formar ruta de caché
            cache_file = self.cache_dir / tour / f"{filename}"
            
            # Descargar datos
            logger.info(f"Descargando datos de jugadores de {tour.upper()}...")
            df = self._download_file(url, cache_file)
            
            # Añadir columna útil
            df['tour'] = tour
            
            # Información básica
            logger.info(f"Datos de jugadores de {tour.upper()} descargados: {len(df)} jugadores")
            
            # Guardar en directorio de salida
            output_file = self.output_dir / tour / f"{filename}"
            df.to_csv(output_file, index=False)
            logger.info(f"Datos de jugadores de {tour.upper()} guardados en {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error descargando datos de jugadores de {tour.upper()}: {str(e)}")
            return pd.DataFrame()
    
    def _generate_summary(self, results: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """
        Genera un resumen de los datos recopilados.
        
        Args:
            results: Diccionario con los DataFrames recopilados
        """
        try:
            summary = {
                'data_collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'year_range': f"{self.start_year}-{self.end_year}",
                'tours': self.tours,
                'include_challengers': self.include_challengers,
                'include_futures': self.include_futures,
                'include_rankings': self.include_rankings,
                'include_pointbypoint': self.include_pointbypoint,
                'include_slam_pointbypoint': self.include_slam_pointbypoint,
                'include_match_charting': self.include_match_charting,
                'counts': {}
            }
            
            # Contar partidos principales
            for tour, df in results['main_matches'].items():
                summary['counts'][f'{tour}_main_matches'] = len(df)
            
            # Contar partidos de Challenger
            for tour, df in results['challenger_matches'].items():
                summary['counts'][f'{tour}_challenger_matches'] = len(df)
            
            # Contar partidos de Futures
            for tour, df in results['futures_matches'].items():
                summary['counts'][f'{tour}_futures_matches'] = len(df)
            
            # Contar jugadores
            for tour, df in results['players'].items():
                summary['counts'][f'{tour}_players'] = len(df)
            
            # Contar registros de rankings
            for tour, df in results['rankings'].items():
                summary['counts'][f'{tour}_ranking_records'] = len(df)
                summary['counts'][f'{tour}_ranking_dates'] = df['ranking_date'].nunique() if 'ranking_date' in df.columns else 0
            
            # Contar datos de Match Charting Project
            match_charting_total = 0
            for data_type, df in results['match_charting'].items():
                summary['counts'][f'match_charting_{data_type}'] = len(df)
                match_charting_total += len(df)
            if match_charting_total > 0:
                summary['counts']['match_charting_total'] = match_charting_total
            
            # Contar datos punto por punto de Grand Slam
            slam_pointbypoint_total = 0
            for tournament, frames_list in results['slam_pointbypoint'].items():
                if isinstance(frames_list, list):
                    tournament_total = sum(len(df) for df in frames_list)
                else:
                    tournament_total = len(frames_list)
                summary['counts'][f'slam_pointbypoint_{tournament}'] = tournament_total
                slam_pointbypoint_total += tournament_total
            if slam_pointbypoint_total > 0:
                summary['counts']['slam_pointbypoint_total'] = slam_pointbypoint_total
            
            # Calcular totales
            total_matches = sum([
                len(df) for df in results['main_matches'].values()
            ]) + sum([
                len(df) for df in results['challenger_matches'].values()
            ]) + sum([
                len(df) for df in results['futures_matches'].values()
            ])
            
            summary['counts']['total_matches'] = total_matches
            
            # Crear sección de datos faltantes y problemas
            summary['missing_data'] = {}
            
            # Verificar faltantes en rankings
            for tour in self.tours:
                if tour in results['rankings']:
                    if results['rankings'][tour].empty:
                        summary['missing_data'][f'{tour}_rankings'] = "No se pudieron encontrar rankings"
                    else:
                        missing_years = []
                        for year in range(self.start_year, self.end_year + 1):
                            year_str = str(year)
                            if 'ranking_date' in results['rankings'][tour].columns:
                                if not any(year_str in str(date) for date in results['rankings'][tour]['ranking_date'].unique()):
                                    missing_years.append(year)
                        if missing_years:
                            summary['missing_data'][f'{tour}_rankings_years'] = missing_years
            
            # Verificar faltantes en datos punto por punto
            if self.include_slam_pointbypoint and not results['slam_pointbypoint']:
                summary['missing_data']['slam_pointbypoint'] = "No se encontraron datos punto por punto de Grand Slam"
            elif self.include_slam_pointbypoint:
                missing_tournaments = []
                expected_tournaments = ['us_open', 'australian_open', 'wimbledon', 'french_open']
                for tournament in expected_tournaments:
                    if tournament not in results['slam_pointbypoint']:
                        missing_tournaments.append(tournament)
                if missing_tournaments:
                    summary['missing_data']['slam_pointbypoint_tournaments'] = missing_tournaments
            
            # Verificar faltantes en Match Charting Project
            if self.include_match_charting and not results['match_charting']:
                summary['missing_data']['match_charting'] = "No se encontraron datos del Match Charting Project"
            
            # Guardar resumen como JSON
            with open(self.output_dir / 'data_collection_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Guardar informe detallado con información sobre repositorios
            detailed_report = {
                'available_repositories': self.available_repos,
                'available_data_types': self.available_data_types,
                'data_collection_summary': summary,
                'data_structures': {
                    'rankings_structure': self.rankings_structure if hasattr(self, 'rankings_structure') else {}
                }
            }
            
            with open(self.output_dir / 'data_collection_detailed_report.json', 'w') as f:
                json.dump(detailed_report, f, indent=4)
            
            # Mostrar resumen
            logger.info("=== RESUMEN DE LA RECOPILACIÓN DE DATOS ===")
            logger.info(f"Período: {summary['year_range']}")
            logger.info(f"Tours: {', '.join(summary['tours'])}")
            for key, count in summary['counts'].items():
                logger.info(f"{key}: {count:,}")
            
            # Mostrar datos faltantes
            if summary['missing_data']:
                logger.info("=== DATOS FALTANTES ===")
                for key, value in summary['missing_data'].items():
                    logger.info(f"{key}: {value}")
            
            logger.info("==========================================")
            
        except Exception as e:
            logger.error(f"Error generando resumen: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _create_combined_files(self, results: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """
        Crea archivos combinados a partir de los datos recopilados.
        
        Args:
            results: Diccionario con los resultados de la recopilación
        """
        try:
            # 1. Combinar todos los partidos principales
            if results['main_matches']:
                all_main_matches = pd.concat(results['main_matches'].values(), ignore_index=True)
                output_file = self.output_dir / f"all_main_matches_{self.start_year}_{self.end_year}.csv"
                all_main_matches.to_csv(output_file, index=False)
                logger.info(f"Todos los partidos principales guardados en {output_file}: {len(all_main_matches)} partidos")
            
            # 2. Combinar todos los partidos (principales, challenger, futures)
            all_matches_frames = []
            
            # Añadir partidos principales
            for tour, df in results['main_matches'].items():
                all_matches_frames.append(df)
            
            # Añadir partidos Challenger
            for tour, df in results['challenger_matches'].items():
                all_matches_frames.append(df)
            
            # Añadir partidos Futures
            for tour, df in results['futures_matches'].items():
                all_matches_frames.append(df)
            
            if all_matches_frames:
                all_matches = pd.concat(all_matches_frames, ignore_index=True)
                output_file = self.output_dir / f"all_matches_{self.start_year}_{self.end_year}.csv"
                all_matches.to_csv(output_file, index=False)
                logger.info(f"Todos los partidos guardados en {output_file}: {len(all_matches)} partidos")
            
            # 3. Crear archivos combinados para datos adicionales
            
            # Match Charting Project
            if results['match_charting']:
                # Identificar los DataFrames principales
                matches_dfs = []
                shots_dfs = []
                stats_dfs = []
                other_dfs = {}
                
                for name, df in results['match_charting'].items():
                    if 'matches' in name.lower():
                        matches_dfs.append(df)
                    elif 'shots' in name.lower():
                        shots_dfs.append(df)
                    elif 'stats' in name.lower():
                        stats_dfs.append(df)
                    else:
                        other_dfs[name] = df
                
                # Combinar y guardar los datos de partidos
                if matches_dfs:
                    matches_combined = pd.concat(matches_dfs, ignore_index=True)
                    output_file = self.output_dir / 'match_charting' / f"all_matches_{self.start_year}_{self.end_year}.csv"
                    matches_combined.to_csv(output_file, index=False)
                    logger.info(f"Datos combinados de partidos Match Charting guardados en {output_file}: {len(matches_combined)} registros")
                
                # Combinar y guardar los datos de tiros
                if shots_dfs:
                    shots_combined = pd.concat(shots_dfs, ignore_index=True)
                    output_file = self.output_dir / 'match_charting' / f"all_shots_{self.start_year}_{self.end_year}.csv"
                    shots_combined.to_csv(output_file, index=False)
                    logger.info(f"Datos combinados de tiros Match Charting guardados en {output_file}: {len(shots_combined)} registros")
                
                # Combinar y guardar los datos de estadísticas
                if stats_dfs:
                    stats_combined = pd.concat(stats_dfs, ignore_index=True)
                    output_file = self.output_dir / 'match_charting' / f"all_stats_{self.start_year}_{self.end_year}.csv"
                    stats_combined.to_csv(output_file, index=False)
                    logger.info(f"Datos combinados de estadísticas Match Charting guardados en {output_file}: {len(stats_combined)} registros")
            
            # Datos punto por punto (ya combinados por torneo en collect_all_data)
            
            # 4. Generar resumen final
            self._generate_summary(results)
            
        except Exception as e:
            logger.error(f"Error creando archivos combinados: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def collect_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Método principal que recopila todos los tipos de datos solicitados.
        
        Returns:
            Diccionario con los resultados de la recopilación
        """
        results = {
            'main_matches': {},
            'challenger_matches': {},
            'futures_matches': {},
            'players': {},
            'rankings': {},
            'slam_pointbypoint': {},
            'match_charting': {},
            'pointbypoint': {}
        }
        
        # Para cada tour
        for tour in self.tours:
            logger.info(f"=== Recopilando datos para {tour.upper()} ===")
            
            # 1. Descargar datos de jugadores
            results['players'][tour] = self._download_players_data(tour)
            
            # 2. Recopilar partidos principales
            main_matches_frames = []
            
            for year in range(self.start_year, self.end_year + 1):
                df = self._download_yearly_matches(year, tour, 'main')
                if not df.empty:
                    main_matches_frames.append(df)
            
            if main_matches_frames:
                results['main_matches'][tour] = pd.concat(main_matches_frames, ignore_index=True)
                
                # Guardar partidos principales
                output_file = self.output_dir / tour / f"{tour}_matches_main_{self.start_year}_{self.end_year}.csv"
                results['main_matches'][tour].to_csv(output_file, index=False)
                logger.info(f"Partidos principales de {tour.upper()} guardados en {output_file}: {len(results['main_matches'][tour])} partidos")
            
            # 3. Recopilar partidos de Challenger si se solicitaron y están disponibles para este tour
            if self.include_challengers and tour in self.available_data_types.get('challenger', []):
                challenger_frames = []
                
                for year in range(self.start_year, self.end_year + 1):
                    df = self._download_yearly_matches(year, tour, 'challenger')
                    if not df.empty:
                        challenger_frames.append(df)
                
                if challenger_frames:
                    results['challenger_matches'][tour] = pd.concat(challenger_frames, ignore_index=True)
                    
                    # Guardar partidos de Challenger
                    output_file = self.output_dir / tour / f"{tour}_matches_challenger_{self.start_year}_{self.end_year}.csv"
                    results['challenger_matches'][tour].to_csv(output_file, index=False)
                    logger.info(f"Partidos Challenger de {tour.upper()} guardados en {output_file}: {len(results['challenger_matches'][tour])} partidos")
            
            # 4. Recopilar partidos de Futures si se solicitaron y están disponibles para este tour
            if self.include_futures and tour in self.available_data_types.get('futures', []):
                futures_frames = []
                
                for year in range(self.start_year, self.end_year + 1):
                    df = self._download_yearly_matches(year, tour, 'futures')
                    if not df.empty:
                        futures_frames.append(df)
                
                if futures_frames:
                    results['futures_matches'][tour] = pd.concat(futures_frames, ignore_index=True)
                    
                    # Guardar partidos de Futures
                    output_file = self.output_dir / tour / f"{tour}_matches_futures_{self.start_year}_{self.end_year}.csv"
                    results['futures_matches'][tour].to_csv(output_file, index=False)
                    logger.info(f"Partidos Futures de {tour.upper()} guardados en {output_file}: {len(results['futures_matches'][tour])} partidos")
            
            # 5. Recopilar rankings si se solicitaron
            if self.include_rankings:
                # Usar el método mejorado para obtener todos los tipos de archivos de ranking
                for tour in self.tours:
                    rankings_df = self._download_rankings_data(tour)
                    
                    if not rankings_df.empty:
                        results['rankings'][tour] = rankings_df
                        
                        # Guardar rankings
                        output_file = self.output_dir / tour / f"{tour}_rankings_all_{self.start_year}_{self.end_year}.csv"
                        results['rankings'][tour].to_csv(output_file, index=False)
                        
                        # Total de fechas únicas
                        unique_dates = rankings_df['ranking_date'].nunique() if 'ranking_date' in rankings_df.columns else 0
                        
                        logger.info(f"Rankings de {tour.upper()} guardados en {output_file}: {len(rankings_df)} registros, {unique_dates} fechas únicas")
                    else:
                        logger.warning(f"No se encontraron rankings para {tour.upper()} en el rango de años {self.start_year}-{self.end_year}")
        
        # 6. Recopilar datos del Match Charting Project si se solicitaron y el repositorio está disponible
        if self.include_match_charting:
            logger.info("=== Recopilando datos del Match Charting Project de forma exhaustiva ===")
            # Usar el nuevo método mejorado en lugar del anterior
            match_charting_results = self.collect_match_charting_data()
            results['match_charting'] = match_charting_results
            
            # Calcular estadísticas para el resumen
            if match_charting_results:
                # Categorizar archivos por tipo para mostrar en el resumen
                matches_count = sum(1 for name in match_charting_results.keys() 
                                if 'match' in name.lower() and 'stat' not in name.lower())
                shots_count = sum(1 for name in match_charting_results.keys() 
                                if 'shot' in name.lower())
                stats_count = sum(1 for name in match_charting_results.keys() 
                                if 'stat' in name.lower())
                other_count = len(match_charting_results) - matches_count - shots_count - stats_count
                
                logger.info(f"Recopilados {len(match_charting_results)} archivos del Match Charting Project:")
                logger.info(f"  - Archivos de partidos: {matches_count}")
                logger.info(f"  - Archivos de tiros: {shots_count}")
                logger.info(f"  - Archivos de estadísticas: {stats_count}")
                logger.info(f"  - Otros archivos: {other_count}")
                
                # Verificar si se crearon archivos combinados
                combined_files = [name for name in match_charting_results.keys() 
                                if name.startswith('all_') and name.endswith('_combined')]
                if combined_files:
                    logger.info(f"Se crearon {len(combined_files)} archivos combinados")
                    for name in combined_files:
                        logger.info(f"  - {name}: {len(match_charting_results[name])} registros")
            else:
                logger.warning("No se pudieron encontrar datos del Match Charting Project")

        # 7. Para datos punto por punto de Grand Slam
        if self.include_slam_pointbypoint:
            logger.info("=== Recopilando datos punto por punto de Grand Slam de manera exhaustiva ===")
            # Usar el método mejorado que implementa múltiples estrategias de búsqueda
            slam_pointbypoint_results = self._download_slam_pointbypoint_data()
            results['slam_pointbypoint'] = slam_pointbypoint_results
            
            # Mostrar estadísticas relevantes
            if slam_pointbypoint_results:
                # Contar archivos por torneo
                tournament_counts = {}
                year_counts = {}
                total_records = 0
                
                for name, df in slam_pointbypoint_results.items():
                    total_records += len(df)
                    
                    # Contar por torneo
                    tournament = None
                    if 'tournament' in df.columns and len(df) > 0:
                        tournament = df['tournament'].iloc[0]
                    elif 'slam' in df.columns and len(df) > 0:
                        tournament = df['slam'].iloc[0]
                    else:
                        # Intentar extraer del nombre
                        for t in ["australian_open", "french_open", "wimbledon", "us_open"]:
                            if t in name:
                                tournament = t
                                break
                    
                    if tournament:
                        if tournament not in tournament_counts:
                            tournament_counts[tournament] = 0
                        tournament_counts[tournament] += 1
                    
                    # Contar por año
                    year = None
                    if 'year' in df.columns and len(df) > 0:
                        year = str(df['year'].iloc[0])
                    else:
                        # Intentar extraer del nombre
                        for y in range(1990, 2025):
                            if str(y) in name:
                                year = str(y)
                                break
                    
                    if year:
                        if year not in year_counts:
                            year_counts[year] = 0
                        year_counts[year] += 1
                
                # Nombres legibles para torneos
                tournament_readable = {
                    "australian_open": "Australian Open",
                    "french_open": "French Open (Roland Garros)",
                    "wimbledon": "Wimbledon",
                    "us_open": "US Open"
                }
                
                # Mostrar resumen completo
                logger.info(f"Recopilados {len(slam_pointbypoint_results)} archivos punto por punto de Grand Slam con {total_records:,} registros totales")
                
                # Mostrar por torneo
                if tournament_counts:
                    logger.info("Archivos por torneo:")
                    for tournament, count in tournament_counts.items():
                        readable = tournament_readable.get(tournament, tournament)
                        logger.info(f"  - {readable}: {count} archivos")
                
                # Mostrar por año
                if year_counts:
                    sorted_years = sorted(year_counts.keys())
                    years_info = ", ".join(f"{y}: {year_counts[y]}" for y in sorted_years)
                    logger.info(f"Archivos por año: {years_info}")
                
                # Verificar si se crearon archivos combinados
                combined_files = [name for name in slam_pointbypoint_results.keys() if 'combined' in name.lower()]
                if combined_files:
                    logger.info(f"Se crearon {len(combined_files)} archivos combinados de datos punto por punto de Grand Slam")
            else:
                logger.warning("No se pudieron encontrar datos punto por punto de Grand Slam")

        # 8. Para datos punto por punto
        if self.include_pointbypoint:
            logger.info("=== Recopilando datos punto por punto de manera exhaustiva ===")
            # Usar el método mejorado que implementa múltiples estrategias de búsqueda
            pointbypoint_results = self._download_pointbypoint_data()
            results['pointbypoint'] = pointbypoint_results
            
            # Mostrar estadísticas relevantes
            if pointbypoint_results:
                # Contar tipos de archivos para el resumen
                points_files = sum(1 for name in pointbypoint_results.keys() if 'point' in name.lower())
                matches_files = sum(1 for name in pointbypoint_results.keys() if 'match' in name.lower() and 'point' not in name.lower())
                stats_files = sum(1 for name in pointbypoint_results.keys() if 'stat' in name.lower())
                rally_files = sum(1 for name in pointbypoint_results.keys() if 'rally' in name.lower())
                other_files = len(pointbypoint_results) - points_files - matches_files - stats_files - rally_files
                
                # Contar registros totales
                total_records = sum(len(df) for df in pointbypoint_results.values())
                
                # Identificar torneos y años disponibles
                tournaments = set()
                years = set()
                
                for df in pointbypoint_results.values():
                    if 'tournament' in df.columns and len(df) > 0:
                        tournaments.update(df['tournament'].unique())
                    if 'year' in df.columns and len(df) > 0:
                        years.update(df['year'].unique())
                
                logger.info(f"Recopilados {len(pointbypoint_results)} archivos punto por punto con {total_records:,} registros totales:")
                logger.info(f"  - Archivos de puntos: {points_files}")
                logger.info(f"  - Archivos de partidos: {matches_files}")
                logger.info(f"  - Archivos de estadísticas: {stats_files}")
                logger.info(f"  - Archivos de rallies: {rally_files}")
                logger.info(f"  - Otros archivos: {other_files}")
                
                if tournaments:
                    logger.info(f"  - Torneos disponibles: {', '.join(str(t) for t in sorted(tournaments))}")
                if years:
                    logger.info(f"  - Años disponibles: {', '.join(str(y) for y in sorted(years))}")
                
                # Verificar si se crearon archivos combinados
                combined_files = [name for name in pointbypoint_results.keys() if 'combined' in name.lower()]
                if combined_files:
                    logger.info(f"Se crearon {len(combined_files)} archivos combinados de datos punto por punto")
            else:
                logger.warning("No se pudieron encontrar datos punto por punto")
            
        # 9. Crear archivos combinados
        self._create_combined_files(results)
            
        return results
    
    def _download_slam_pointbypoint_data(self) -> Dict[str, pd.DataFrame]:
        """
        Descarga datos punto por punto de torneos Grand Slam de manera exhaustiva.
        
        Implementa múltiples estrategias para encontrar todos los archivos disponibles
        en el repositorio tennis_slam_pointbypoint, considerando diferentes estructuras
        y convenciones de nomenclatura.
        
        Returns:
            Dict[str, pd.DataFrame]: Diccionario con todos los datos recopilados
        """
        results = {}
        
        # URL base del repositorio oficial
        base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master"
        
        logger.info("=== Iniciando recopilación exhaustiva de datos punto por punto de Grand Slam ===")
        
        # Verificar si podemos acceder al repositorio
        readme_url = f"{base_url}/README.md"
        if not self._check_file_exists(readme_url):
            logger.warning("No se puede acceder al repositorio de datos punto por punto de Grand Slam.")
            return results
        
        # 1. ESTRUCTURA DEL REPOSITORIO
        # Este repositorio tiene varias posibles estructuras:
        # - Archivos principales en la raíz: /usopen_points_YYYY.csv
        # - Por torneo: /usopen/points_YYYY.csv o /usopen/YYYY.csv o /usopen/YYYY_points.csv
        # - Por torneo/año: /usopen/2019/points.csv
        # - Por año/torneo: /2019/usopen/points.csv
        
        # 1.1 Mapeo de variantes de nombres de torneos a nombres canónicos
        tournament_mappings = {
            # Australian Open
            "ausopen": "australian_open",
            "australian_open": "australian_open",
            "australian-open": "australian_open", 
            "ao": "australian_open",
            "australia": "australian_open",
            "melbourne": "australian_open",
            # French Open / Roland Garros
            "frenchopen": "french_open",
            "french_open": "french_open",
            "french-open": "french_open",
            "fo": "french_open",
            "rg": "french_open",
            "roland_garros": "french_open",
            "roland-garros": "french_open",
            "rolandgarros": "french_open",
            "paris": "french_open",
            # Wimbledon
            "wimbledon": "wimbledon",
            "wim": "wimbledon",
            "london": "wimbledon",
            # US Open
            "usopen": "us_open",
            "us_open": "us_open",
            "us-open": "us_open",
            "uso": "us_open",
            "newyork": "us_open",
            "flushing": "us_open"
        }
        
        # 1.2 Torneo a nombre legible
        tournament_readable = {
            "australian_open": "Australian Open",
            "french_open": "French Open (Roland Garros)",
            "wimbledon": "Wimbledon",
            "us_open": "US Open"
        }
        
        # 1.3 Años para buscar
        years = list(range(2000, 2025))  # Ajustar según necesidad
        
        # 2. BUSCAR ARCHIVOS EN LA RAÍZ DEL REPOSITORIO
        logger.info("Buscando archivos en la raíz del repositorio...")
        
        # 2.1 Patrones de archivos en la raíz
        root_patterns = [
            "{tournament}_points_{year}.csv",
            "{tournament}_{year}_points.csv",
            "{tournament}_{year}.csv",
            "{year}_{tournament}_points.csv",
            "{year}_{tournament}.csv",
            "slam_pointbypoint_{tournament}_{year}.csv"
        ]
        
        for tournament_code in tournament_mappings.keys():
            canonical_name = tournament_mappings[tournament_code]
            
            for year in years:
                for pattern in root_patterns:
                    # Generar nombre de archivo según el patrón
                    file_name = pattern.format(tournament=tournament_code, year=year)
                    file_url = f"{base_url}/{file_name}"
                    
                    if self._check_file_exists(file_url):
                        logger.info(f"Encontrado archivo en raíz: {file_name}")
                        
                        # Descargar el archivo
                        output_name = f"{canonical_name}_{year}_points"
                        cache_file = self.cache_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                        output_file = self.output_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                        
                        try:
                            response = self._make_request(file_url)
                            content = response.content.decode('utf-8')
                            df = pd.read_csv(StringIO(content))
                            
                            # Añadir metadatos si no existen
                            if 'tournament' not in df.columns:
                                df['tournament'] = canonical_name
                            if 'year' not in df.columns:
                                df['year'] = year
                            if 'slam' not in df.columns:
                                df['slam'] = canonical_name
                            
                            # Guardar archivos
                            cache_file.parent.mkdir(exist_ok=True)
                            output_file.parent.mkdir(exist_ok=True)
                            
                            df.to_csv(cache_file, index=False)
                            df.to_csv(output_file, index=False)
                            
                            results[output_name] = df
                            logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                        except Exception as e:
                            logger.warning(f"Error procesando {file_url}: {str(e)}")
        
        # 3. BUSCAR ARCHIVOS EN ESTRUCTURA DE DIRECTORIO POR TORNEO
        logger.info("Buscando archivos en estructura de directorios por torneo...")
        
        # 3.1 Patrones de archivos dentro de directorios de torneo
        tournament_patterns = [
            "points_{year}.csv",
            "{year}_points.csv",
            "{year}.csv",
            "points/{year}.csv",
            "data/{year}.csv",
            "{year}/points.csv",
            "{year}/matches.csv",
            "{year}/all_points.csv"
        ]
        
        for tournament_code in tournament_mappings.keys():
            canonical_name = tournament_mappings[tournament_code]
            tournament_dir = f"{tournament_code}/"
            tournament_url = f"{base_url}/{tournament_dir}"
            
            # Verificar si existe el directorio de torneo con algún archivo conocido
            dir_exists = False
            for test_file in ["README.md", "index.html", "points.csv"]:
                if self._check_file_exists(f"{tournament_url}{test_file}"):
                    dir_exists = True
                    logger.info(f"Encontrado directorio para torneo: {tournament_code}")
                    break
                    
            if dir_exists:
                # Buscar archivos por año en este directorio de torneo
                for year in years:
                    for pattern in tournament_patterns:
                        # Generar nombre de archivo según el patrón
                        file_path = pattern.format(year=year)
                        file_url = f"{tournament_url}{file_path}"
                        
                        if self._check_file_exists(file_url):
                            logger.info(f"Encontrado archivo: {tournament_dir}{file_path}")
                            
                            # Descargar el archivo
                            output_name = f"{canonical_name}_{year}_points_v2"
                            cache_file = self.cache_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                            output_file = self.output_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                            
                            try:
                                response = self._make_request(file_url)
                                content = response.content.decode('utf-8')
                                df = pd.read_csv(StringIO(content))
                                
                                # Añadir metadatos si no existen
                                if 'tournament' not in df.columns:
                                    df['tournament'] = canonical_name
                                if 'year' not in df.columns:
                                    df['year'] = year
                                if 'slam' not in df.columns:
                                    df['slam'] = canonical_name
                                
                                # Guardar archivos
                                cache_file.parent.mkdir(exist_ok=True)
                                output_file.parent.mkdir(exist_ok=True)
                                
                                df.to_csv(cache_file, index=False)
                                df.to_csv(output_file, index=False)
                                
                                results[output_name] = df
                                logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                            except Exception as e:
                                logger.warning(f"Error procesando {file_url}: {str(e)}")
                
                # Buscar subdirectorios por año
                for year in years:
                    year_dir = f"{tournament_dir}{year}/"
                    year_url = f"{base_url}/{year_dir}"
                    
                    for file_name in ["points.csv", "matches.csv", "all.csv", "stats.csv"]:
                        file_url = f"{year_url}{file_name}"
                        
                        if self._check_file_exists(file_url):
                            logger.info(f"Encontrado archivo: {year_dir}{file_name}")
                            
                            # Descargar el archivo
                            output_name = f"{canonical_name}_{year}_{file_name.replace('.csv', '')}_v3"
                            cache_file = self.cache_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                            output_file = self.output_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                            
                            try:
                                response = self._make_request(file_url)
                                content = response.content.decode('utf-8')
                                df = pd.read_csv(StringIO(content))
                                
                                # Añadir metadatos si no existen
                                if 'tournament' not in df.columns:
                                    df['tournament'] = canonical_name
                                if 'year' not in df.columns:
                                    df['year'] = year
                                if 'slam' not in df.columns:
                                    df['slam'] = canonical_name
                                
                                # Guardar archivos
                                cache_file.parent.mkdir(exist_ok=True)
                                output_file.parent.mkdir(exist_ok=True)
                                
                                df.to_csv(cache_file, index=False)
                                df.to_csv(output_file, index=False)
                                
                                results[output_name] = df
                                logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                            except Exception as e:
                                logger.warning(f"Error procesando {file_url}: {str(e)}")
        
        # 4. BUSCAR ARCHIVOS EN ESTRUCTURA DE DIRECTORIO POR AÑO PRIMERO
        logger.info("Buscando archivos en estructura de directorios por año/torneo...")
        
        for year in years:
            year_dir = f"{year}/"
            year_url = f"{base_url}/{year_dir}"
            
            # Verificar si existe el directorio de año
            dir_exists = False
            for test_file in ["README.md", "index.html", "points.csv"]:
                if self._check_file_exists(f"{year_url}{test_file}"):
                    dir_exists = True
                    logger.info(f"Encontrado directorio para año: {year}")
                    break
            
            if dir_exists:
                # Buscar subdirectorios de torneos en este año
                for tournament_code in tournament_mappings.keys():
                    canonical_name = tournament_mappings[tournament_code]
                    tournament_dir = f"{year_dir}{tournament_code}/"
                    tournament_url = f"{base_url}/{tournament_dir}"
                    
                    for file_name in ["points.csv", "matches.csv", "all.csv", "stats.csv"]:
                        file_url = f"{tournament_url}{file_name}"
                        
                        if self._check_file_exists(file_url):
                            logger.info(f"Encontrado archivo: {tournament_dir}{file_name}")
                            
                            # Descargar el archivo
                            output_name = f"{canonical_name}_{year}_{file_name.replace('.csv', '')}_v4"
                            cache_file = self.cache_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                            output_file = self.output_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                            
                            try:
                                response = self._make_request(file_url)
                                content = response.content.decode('utf-8')
                                df = pd.read_csv(StringIO(content))
                                
                                # Añadir metadatos si no existen
                                if 'tournament' not in df.columns:
                                    df['tournament'] = canonical_name
                                if 'year' not in df.columns:
                                    df['year'] = year
                                if 'slam' not in df.columns:
                                    df['slam'] = canonical_name
                                
                                # Guardar archivos
                                cache_file.parent.mkdir(exist_ok=True)
                                output_file.parent.mkdir(exist_ok=True)
                                
                                df.to_csv(cache_file, index=False)
                                df.to_csv(output_file, index=False)
                                
                                results[output_name] = df
                                logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                            except Exception as e:
                                logger.warning(f"Error procesando {file_url}: {str(e)}")
                
                # Buscar archivos directamente en el directorio año (posibles consolidados)
                for file_pattern in ["all_points.csv", "all_matches.csv", "points_summary.csv", "stats.csv"]:
                    file_url = f"{year_url}{file_pattern}"
                    
                    if self._check_file_exists(file_url):
                        logger.info(f"Encontrado archivo consolidado: {year_dir}{file_pattern}")
                        
                        # Descargar el archivo
                        output_name = f"all_slams_{year}_{file_pattern.replace('.csv', '')}"
                        cache_file = self.cache_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                        output_file = self.output_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                        
                        try:
                            response = self._make_request(file_url)
                            content = response.content.decode('utf-8')
                            df = pd.read_csv(StringIO(content))
                            
                            # Añadir metadatos si no existen
                            if 'year' not in df.columns:
                                df['year'] = year
                            
                            # Guardar archivos
                            cache_file.parent.mkdir(exist_ok=True)
                            output_file.parent.mkdir(exist_ok=True)
                            
                            df.to_csv(cache_file, index=False)
                            df.to_csv(output_file, index=False)
                            
                            results[output_name] = df
                            logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                        except Exception as e:
                            logger.warning(f"Error procesando {file_url}: {str(e)}")
        
        # 5. BUSCAR ARCHIVOS CONSOLIDADOS EN DIRECTORIOS ESPECIALES
        logger.info("Buscando archivos consolidados en directorios especiales...")
        
        special_dirs = ["data/", "combined/", "all/", "full/", "complete/"]
        
        for special_dir in special_dirs:
            special_url = f"{base_url}/{special_dir}"
            
            # 5.1 Buscar archivos por torneo
            for tournament_code in tournament_mappings.keys():
                canonical_name = tournament_mappings[tournament_code]
                
                # Patrones para archivos consolidados por torneo
                consolidated_patterns = [
                    f"{tournament_code}_all.csv",
                    f"{tournament_code}_all_points.csv",
                    f"{tournament_code}_points.csv",
                    f"all_{tournament_code}.csv",
                    f"all_{tournament_code}_points.csv"
                ]
                
                for pattern in consolidated_patterns:
                    file_url = f"{special_url}{pattern}"
                    
                    if self._check_file_exists(file_url):
                        logger.info(f"Encontrado archivo consolidado: {special_dir}{pattern}")
                        
                        # Descargar el archivo
                        output_name = f"{canonical_name}_all_consolidated"
                        cache_file = self.cache_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                        output_file = self.output_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                        
                        try:
                            response = self._make_request(file_url)
                            content = response.content.decode('utf-8')
                            df = pd.read_csv(StringIO(content))
                            
                            # Añadir metadatos si no existen
                            if 'tournament' not in df.columns:
                                df['tournament'] = canonical_name
                            if 'slam' not in df.columns:
                                df['slam'] = canonical_name
                            
                            # Guardar archivos
                            cache_file.parent.mkdir(exist_ok=True)
                            output_file.parent.mkdir(exist_ok=True)
                            
                            df.to_csv(cache_file, index=False)
                            df.to_csv(output_file, index=False)
                            
                            results[output_name] = df
                            logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                        except Exception as e:
                            logger.warning(f"Error procesando {file_url}: {str(e)}")
            
            # 5.2 Buscar archivos consolidados generales
            consolidated_files = [
                "all_slam_points.csv", 
                "all_points.csv", 
                "all_slams.csv", 
                "slam_pointbypoint.csv",
                "slam_points_all.csv"
            ]
            
            for file_name in consolidated_files:
                file_url = f"{special_url}{file_name}"
                
                if self._check_file_exists(file_url):
                    logger.info(f"Encontrado archivo consolidado general: {special_dir}{file_name}")
                    
                    # Descargar el archivo
                    output_name = f"all_slams_consolidated_{file_name.replace('.csv', '')}"
                    cache_file = self.cache_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                    output_file = self.output_dir / 'slam_pointbypoint' / f"{output_name}.csv"
                    
                    try:
                        response = self._make_request(file_url)
                        content = response.content.decode('utf-8')
                        df = pd.read_csv(StringIO(content))
                        
                        # Guardar archivos
                        cache_file.parent.mkdir(exist_ok=True)
                        output_file.parent.mkdir(exist_ok=True)
                        
                        df.to_csv(cache_file, index=False)
                        df.to_csv(output_file, index=False)
                        
                        results[output_name] = df
                        logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                    except Exception as e:
                        logger.warning(f"Error procesando {file_url}: {str(e)}")
        
        # 6. SI AÚN NO SE HAN ENCONTRADO ARCHIVOS, INTENTAR CON LA API DE GITHUB
        if not results:
            logger.info("No se encontraron archivos con métodos de búsqueda estándar, intentando con API de GitHub...")
            github_results = self._download_slam_pointbypoint_via_github()
            results.update(github_results)
        
        # 7. PROCESAMIENTO FINAL Y RESUMEN
        if results:
            # Número total de archivos y registros
            total_files = len(results)
            total_records = sum(len(df) for df in results.values())
            
            logger.info(f"=== Recopilación de datos punto por punto de Grand Slam completada ===")
            logger.info(f"Total de archivos descargados: {total_files}")
            logger.info(f"Total de registros: {total_records:,}")
            
            # Agrupar archivos por torneo y año para análisis
            tournament_files = {}
            year_files = {}
            
            for name, df in results.items():
                # Identificar torneo
                tournament = None
                if 'tournament' in df.columns and len(df) > 0:
                    tournament = df['tournament'].iloc[0]
                else:
                    # Intentar extraer del nombre
                    for t_code, t_name in tournament_mappings.items():
                        if t_code in name or t_name in name:
                            tournament = t_name
                            break
                
                # Identificar año
                year = None
                if 'year' in df.columns and len(df) > 0:
                    year = df['year'].iloc[0]
                else:
                    # Intentar extraer del nombre
                    for y in years:
                        if str(y) in name:
                            year = y
                            break
                
                # Registrar en las estructuras
                if tournament:
                    if tournament not in tournament_files:
                        tournament_files[tournament] = []
                    tournament_files[tournament].append(name)
                
                if year:
                    year_str = str(year)
                    if year_str not in year_files:
                        year_files[year_str] = []
                    year_files[year_str].append(name)
            
            # Crear archivo de metadatos
            try:
                metadata = {
                    "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_files": total_files,
                    "total_records": total_records,
                    "tournaments": {},
                    "years": {},
                    "files": {}
                }
                
                # Información por torneo
                for tournament, files in tournament_files.items():
                    metadata["tournaments"][tournament] = {
                        "files": files,
                        "count": len(files),
                        "readable_name": tournament_readable.get(tournament, tournament)
                    }
                
                # Información por año
                for year, files in year_files.items():
                    metadata["years"][year] = {
                        "files": files,
                        "count": len(files)
                    }
                
                # Información por archivo
                for name, df in results.items():
                    metadata["files"][name] = {
                        "records": len(df),
                        "columns": list(df.columns),
                        "memory_size_kb": int(df.memory_usage(deep=True).sum() / 1024)
                    }
                
                # Guardar archivo de metadatos
                metadata_file = self.output_dir / 'slam_pointbypoint' / 'metadata.json'
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Creado archivo de metadatos con información de {total_files} archivos")
                
                # Crear archivos combinados por torneo para facilitar el análisis
                for tournament, files in tournament_files.items():
                    if len(files) > 1:
                        try:
                            # Preparar lista de DataFrames a combinar
                            dfs_to_combine = []
                            for file_name in files:
                                dfs_to_combine.append(results[file_name])
                            
                            combined_df = pd.concat(dfs_to_combine, ignore_index=True)
                            
                            # Eliminar duplicados si es posible
                            if 'point_id' in combined_df.columns:
                                orig_len = len(combined_df)
                                combined_df.drop_duplicates(subset=['point_id'], inplace=True)
                                if len(combined_df) < orig_len:
                                    logger.info(f"Eliminados {orig_len - len(combined_df)} duplicados en {tournament}")
                            
                            # Guardar archivo combinado
                            readable_name = tournament_readable.get(tournament, tournament)
                            output_file = self.output_dir / 'slam_pointbypoint' / f"{tournament}_all_combined.csv"
                            combined_df.to_csv(output_file, index=False)
                            
                            logger.info(f"Creado archivo combinado para {readable_name}: {len(combined_df)} registros")
                        except Exception as e:
                            logger.warning(f"Error al crear archivo combinado para {tournament}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error al crear archivo de metadatos: {str(e)}")
        else:
            logger.warning("No se pudieron encontrar archivos de datos punto por punto de Grand Slam.")
        
        return results

    def _download_slam_pointbypoint_via_github(self) -> Dict[str, pd.DataFrame]:
        """
        Descarga datos punto por punto de Grand Slam utilizando la API de GitHub.
        Esta función obtiene la lista completa de archivos CSV en el repositorio
        sin depender de adivinar rutas o nombres.
        
        Returns:
            Dict[str, pd.DataFrame]: Diccionario con los DataFrames descargados
        """
        results = {}
        
        logger.info("Iniciando descarga de datos punto por punto de Grand Slam mediante API de GitHub...")
        
        # Repositorio a explorar
        repo = "JeffSackmann/tennis_slam_pointbypoint"
        
        # Mapeo de torneos para enriquecimiento de datos
        tournament_mappings = {
            "ausopen": "australian_open",
            "ao": "australian_open",
            "australian": "australian_open",
            "frenchopen": "french_open",
            "fo": "french_open",
            "rg": "french_open",
            "roland": "french_open",
            "wimbledon": "wimbledon",
            "wim": "wimbledon",
            "usopen": "us_open",
            "uso": "us_open"
        }
        
        try:
            # Obtener todos los archivos CSV del repositorio (profundidad máxima de 3 niveles)
            csv_files = self._download_github_repo_files(repo, ['.csv'], max_depth=3)
            
            if not csv_files:
                logger.warning(f"No se encontraron archivos CSV en {repo} vía GitHub API")
                return results
            
            logger.info(f"Encontrados {len(csv_files)} archivos CSV en {repo} vía GitHub API")
            
            # Descargar todos los archivos encontrados
            for file_path, download_url in csv_files.items():
                # Crear un nombre único para este archivo
                # Reemplazar / por _ para evitar problemas con rutas
                safe_name = file_path.replace('/', '_').replace('-', '_').replace('.csv', '')
                
                # Si el nombre generado es demasiado largo, truncarlo de manera inteligente
                if len(safe_name) > 100:
                    # Extraer partes importantes (primero y últimos componentes)
                    parts = safe_name.split('_')
                    if len(parts) > 4:
                        # Tomar el primer componente (generalmente el torneo) y los últimos componentes
                        safe_name = f"{parts[0]}_{parts[1]}__{parts[-2]}_{parts[-1]}"
                
                # Rutas de archivo
                cache_file = self.cache_dir / 'slam_pointbypoint' / f"{safe_name}.csv"
                output_file = self.output_dir / 'slam_pointbypoint' / f"{safe_name}.csv"
                
                try:
                    logger.info(f"Descargando {file_path} desde GitHub API")
                    response = self._make_request(download_url)
                    content = response.content.decode('utf-8')
                    
                    # Intentar procesar como CSV
                    try:
                        df = pd.read_csv(StringIO(content))
                        
                        # Verificar si el archivo parece contener datos de puntos
                        # Algunas columnas comunes en archivos punto por punto
                        point_columns = ['point_no', 'point_id', 'server', 'receiver', 'winner', 'p1_score', 'p2_score', 
                                        'set_no', 'game_no', 'point_victor', 'serve_no']
                        
                        is_point_data = any(col in df.columns for col in point_columns)
                        
                        if is_point_data or len(df.columns) >= 5:  # Archivos punto por punto suelen tener varias columnas
                            # Extraer información de la ruta del archivo para enriquecer los datos
                            parts = file_path.split('/')
                            
                            # Tratar de identificar torneo y año de la ruta
                            tournament = None
                            year = None
                            
                            for part in parts:
                                # Detectar año (números entre 1990-2025)
                                if part.isdigit() and 1990 <= int(part) <= 2025:
                                    year = int(part)
                                
                                # Detectar torneo
                                for t_code in tournament_mappings.keys():
                                    if t_code in part.lower():
                                        tournament = tournament_mappings[t_code]
                                        break
                            
                            # Si no se detectó en partes, buscar en el nombre del archivo
                            if tournament is None:
                                filename = os.path.basename(file_path)
                                for t_code in tournament_mappings.keys():
                                    if t_code in filename.lower():
                                        tournament = tournament_mappings[t_code]
                                        break
                            
                            # Añadir metadatos útiles si no existen
                            if 'tournament' not in df.columns and tournament:
                                df['tournament'] = tournament
                            if 'year' not in df.columns and year:
                                df['year'] = year
                            if 'slam' not in df.columns and tournament:
                                df['slam'] = tournament
                            
                            # Añadir ruta original como referencia
                            df['source_path'] = file_path
                            
                            # Guardar archivos
                            cache_file.parent.mkdir(exist_ok=True)
                            output_file.parent.mkdir(exist_ok=True)
                            
                            df.to_csv(cache_file, index=False)
                            df.to_csv(output_file, index=False)
                            
                            results[safe_name] = df
                            logger.info(f"Descargado archivo {safe_name}.csv vía GitHub API: {len(df)} registros")
                        else:
                            logger.debug(f"Omitiendo {file_path} - no parece contener datos punto por punto")
                    except pd.errors.EmptyDataError:
                        logger.warning(f"Archivo vacío: {file_path}")
                    except pd.errors.ParserError:
                        logger.warning(f"Error al parsear CSV: {file_path} - posiblemente no es un CSV válido")
                except Exception as e:
                    logger.warning(f"Error procesando {download_url}: {str(e)}")
        
            # Procesar los resultados
            if results:
                # Contar archivos por torneo
                tournaments_count = {}
                for name, df in results.items():
                    tournament = None
                    if 'tournament' in df.columns and len(df) > 0:
                        tournament = df['tournament'].iloc[0]
                    elif 'slam' in df.columns and len(df) > 0:
                        tournament = df['slam'].iloc[0]
                    else:
                        # Intentar extraer del nombre
                        for t_code, t_name in tournament_mappings.items():
                            if t_code in name or t_name in name:
                                tournament = t_name
                                break
                    
                    if tournament:
                        if tournament not in tournaments_count:
                            tournaments_count[tournament] = 0
                        tournaments_count[tournament] += 1
                
                # Mostrar resumen
                logger.info(f"Descarga vía GitHub API completada: {len(results)} archivos descargados")
                for tournament, count in tournaments_count.items():
                    logger.info(f"  - {tournament}: {count} archivos")
        
        except Exception as e:
            logger.warning(f"Error explorando {repo} vía GitHub API: {str(e)}")
        
        return results

    def _download_pointbypoint_data(self) -> Dict[str, pd.DataFrame]:
        """
        Descarga datos punto por punto directamente del repositorio oficial de manera exhaustiva.
        Implementa múltiples estrategias para encontrar todos los archivos disponibles.
        
        Returns:
            Dict[str, pd.DataFrame]: Diccionario con todos los datos punto por punto recopilados
        """
        results = {}
        
        # URL base del repositorio oficial
        base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_pointbypoint/master"
        
        logger.info("=== Iniciando recopilación exhaustiva de datos punto por punto ===")
        
        # Verificar si podemos acceder al repositorio
        readme_url = f"{base_url}/README.md"
        if not self._check_file_exists(readme_url):
            logger.warning("No se puede acceder al repositorio de datos punto por punto.")
            return results
        
        # 1. ESTRUCTURA DEL REPOSITORIO
        # El repositorio tennis_pointbypoint tiene una estructura más compleja y variable
        # que los otros repositorios. Los datos pueden estar organizados de diferentes maneras:
        # - Por año y luego por torneo: /2019/ausopen/...
        # - Por torneo y luego por año: /ausopen/2019/...
        # - Por torneo y año combinados: /ausopen_2019/...
        # - Por año solo con puntos de varios torneos: /2019/points.csv
        
        # Torneos principales
        tournaments = [
            "ausopen", "ao", "australian_open", "australian-open",
            "frenchopen", "fo", "rg", "roland_garros", "roland-garros", "french_open", "french-open",
            "usopen", "uso", "us_open", "us-open",
            "wimbledon", "wim", "wm",
            "masters", "atp_finals", "wta_finals",
            "master1000", "m1000", "atp1000",
            "master500", "m500", "atp500",
            "master250", "m250", "atp250",
            "davis", "davis_cup", "fed", "fed_cup", "bjk", "bjk_cup",
            "olympics", "olympic"
        ]
        
        # Años para buscar (ajustar según necesidad)
        years = list(range(2000, 2025))
        
        # 2. BÚSQUEDA POR ESTRUCTURA AÑO/TORNEO
        logger.info("Buscando en estructura de directorios por año/torneo...")
        for year in years:
            year_str = str(year)
            year_dir = f"{year_str}/"
            
            # Verificar si existe el directorio del año
            test_url = f"{base_url}/{year_dir}"
            
            # Para ser más confiables, verificamos si existe algún archivo o subdirectorio conocido
            test_files = ["index.html", "README.md", "matches.csv", "points.csv"]
            year_dir_exists = False
            
            for test_file in test_files:
                if self._check_file_exists(f"{test_url}/{test_file}"):
                    year_dir_exists = True
                    logger.info(f"Encontrado directorio para año {year}")
                    break
                    
            if year_dir_exists:
                # Buscar subdirectorios de torneos en este año
                for tournament in tournaments:
                    tournament_dir = f"{year_dir}{tournament}/"
                    tournament_url = f"{base_url}/{tournament_dir}"
                    
                    # Verificar si existe algún archivo en este directorio
                    for test_file in ["1.csv", "points.csv", "matches.csv"]:
                        file_url = f"{tournament_url}/{test_file}"
                        
                        if self._check_file_exists(file_url):
                            logger.info(f"Encontrado directorio {tournament} para año {year}")
                            
                            # Ahora buscar todos los archivos CSV en este directorio
                            # Típicamente son archivos numerados (1.csv, 2.csv, etc.) para partidos individuales
                            for match_num in range(1, 501):  # Un límite razonable
                                match_file = f"{match_num}.csv"
                                match_url = f"{tournament_url}/{match_file}"
                                
                                if self._check_file_exists(match_url):
                                    logger.info(f"Encontrado archivo de partido: {tournament_dir}{match_file}")
                                    
                                    # Descargar y procesar el archivo
                                    output_name = f"{year}_{tournament}_match_{match_num}"
                                    cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                                    output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                                    
                                    try:
                                        response = self._make_request(match_url)
                                        content = response.content.decode('utf-8')
                                        df = pd.read_csv(StringIO(content))
                                        
                                        # Añadir metadatos útiles si no existen
                                        if 'tournament' not in df.columns:
                                            df['tournament'] = tournament
                                        if 'year' not in df.columns:
                                            df['year'] = year
                                        if 'match_num' not in df.columns:
                                            df['match_num'] = match_num
                                        
                                        # Guardar archivo
                                        cache_file.parent.mkdir(exist_ok=True)
                                        output_file.parent.mkdir(exist_ok=True)
                                        
                                        df.to_csv(cache_file, index=False)
                                        df.to_csv(output_file, index=False)
                                        
                                        results[output_name] = df
                                        logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                                    except Exception as e:
                                        logger.warning(f"Error procesando {match_url}: {str(e)}")
                                else:
                                    # Si no encontramos este número de partido, probablemente llegamos al final
                                    if match_num > 1:
                                        break
                            
                            # Buscar también archivos consolidados de puntos o partidos
                            for consolidated_file in ["points.csv", "matches.csv", "stats.csv", "all_points.csv", "all_matches.csv"]:
                                consolidated_url = f"{tournament_url}/{consolidated_file}"
                                
                                if self._check_file_exists(consolidated_url):
                                    logger.info(f"Encontrado archivo consolidado: {tournament_dir}{consolidated_file}")
                                    
                                    output_name = f"{year}_{tournament}_{consolidated_file.replace('.csv', '')}"
                                    cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                                    output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                                    
                                    try:
                                        response = self._make_request(consolidated_url)
                                        content = response.content.decode('utf-8')
                                        df = pd.read_csv(StringIO(content))
                                        
                                        # Añadir metadatos útiles si no existen
                                        if 'tournament' not in df.columns:
                                            df['tournament'] = tournament
                                        if 'year' not in df.columns:
                                            df['year'] = year
                                        
                                        # Guardar archivo
                                        cache_file.parent.mkdir(exist_ok=True)
                                        output_file.parent.mkdir(exist_ok=True)
                                        
                                        df.to_csv(cache_file, index=False)
                                        df.to_csv(output_file, index=False)
                                        
                                        results[output_name] = df
                                        logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                                    except Exception as e:
                                        logger.warning(f"Error procesando {consolidated_url}: {str(e)}")
                            
                            # Solo necesitamos verificar un archivo para confirmar que existe el directorio
                            break
                
                # Buscar también archivos directamente en el directorio del año (sin subdirectorio de torneo)
                for file_pattern in ["points.csv", "matches.csv", "stats.csv", "all_points.csv", "all_matches.csv"]:
                    file_url = f"{base_url}/{year_dir}{file_pattern}"
                    
                    if self._check_file_exists(file_url):
                        logger.info(f"Encontrado archivo en directorio de año: {year_dir}{file_pattern}")
                        
                        output_name = f"{year}_{file_pattern.replace('.csv', '')}"
                        cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                        output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                        
                        try:
                            response = self._make_request(file_url)
                            content = response.content.decode('utf-8')
                            df = pd.read_csv(StringIO(content))
                            
                            # Añadir metadatos útiles si no existen
                            if 'year' not in df.columns:
                                df['year'] = year
                            
                            # Guardar archivo
                            cache_file.parent.mkdir(exist_ok=True)
                            output_file.parent.mkdir(exist_ok=True)
                            
                            df.to_csv(cache_file, index=False)
                            df.to_csv(output_file, index=False)
                            
                            results[output_name] = df
                            logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                        except Exception as e:
                            logger.warning(f"Error procesando {file_url}: {str(e)}")
        
        # 3. BÚSQUEDA POR ESTRUCTURA TORNEO/AÑO
        logger.info("Buscando en estructura de directorios por torneo/año...")
        for tournament in tournaments:
            tournament_dir = f"{tournament}/"
            tournament_url = f"{base_url}/{tournament_dir}"
            
            # Verificar si existe el directorio del torneo
            test_files = ["index.html", "README.md", "matches.csv", "points.csv"]
            tournament_dir_exists = False
            
            for test_file in test_files:
                if self._check_file_exists(f"{tournament_url}/{test_file}"):
                    tournament_dir_exists = True
                    logger.info(f"Encontrado directorio para torneo {tournament}")
                    break
                    
            if tournament_dir_exists:
                # Buscar subdirectorios de años en este torneo
                for year in years:
                    year_str = str(year)
                    year_dir = f"{tournament_dir}{year_str}/"
                    year_url = f"{base_url}/{year_dir}"
                    
                    # Verificar si existe algún archivo en este directorio
                    for test_file in ["1.csv", "points.csv", "matches.csv"]:
                        file_url = f"{year_url}/{test_file}"
                        
                        if self._check_file_exists(file_url):
                            logger.info(f"Encontrado directorio {year} para torneo {tournament}")
                            
                            # Ahora buscar todos los archivos CSV en este directorio
                            for match_num in range(1, 501):  # Un límite razonable
                                match_file = f"{match_num}.csv"
                                match_url = f"{year_url}/{match_file}"
                                
                                if self._check_file_exists(match_url):
                                    logger.info(f"Encontrado archivo de partido: {year_dir}{match_file}")
                                    
                                    # Descargar y procesar el archivo
                                    output_name = f"{tournament}_{year}_match_{match_num}"
                                    cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                                    output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                                    
                                    try:
                                        response = self._make_request(match_url)
                                        content = response.content.decode('utf-8')
                                        df = pd.read_csv(StringIO(content))
                                        
                                        # Añadir metadatos útiles si no existen
                                        if 'tournament' not in df.columns:
                                            df['tournament'] = tournament
                                        if 'year' not in df.columns:
                                            df['year'] = year
                                        if 'match_num' not in df.columns:
                                            df['match_num'] = match_num
                                        
                                        # Guardar archivo
                                        cache_file.parent.mkdir(exist_ok=True)
                                        output_file.parent.mkdir(exist_ok=True)
                                        
                                        df.to_csv(cache_file, index=False)
                                        df.to_csv(output_file, index=False)
                                        
                                        results[output_name] = df
                                        logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                                    except Exception as e:
                                        logger.warning(f"Error procesando {match_url}: {str(e)}")
                                else:
                                    # Si no encontramos este número de partido, probablemente llegamos al final
                                    if match_num > 1:
                                        break
                            
                            # Buscar también archivos consolidados de puntos o partidos
                            for consolidated_file in ["points.csv", "matches.csv", "stats.csv", "all_points.csv", "all_matches.csv"]:
                                consolidated_url = f"{year_url}/{consolidated_file}"
                                
                                if self._check_file_exists(consolidated_url):
                                    logger.info(f"Encontrado archivo consolidado: {year_dir}{consolidated_file}")
                                    
                                    output_name = f"{tournament}_{year}_{consolidated_file.replace('.csv', '')}"
                                    cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                                    output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                                    
                                    try:
                                        response = self._make_request(consolidated_url)
                                        content = response.content.decode('utf-8')
                                        df = pd.read_csv(StringIO(content))
                                        
                                        # Añadir metadatos útiles si no existen
                                        if 'tournament' not in df.columns:
                                            df['tournament'] = tournament
                                        if 'year' not in df.columns:
                                            df['year'] = year
                                        
                                        # Guardar archivo
                                        cache_file.parent.mkdir(exist_ok=True)
                                        output_file.parent.mkdir(exist_ok=True)
                                        
                                        df.to_csv(cache_file, index=False)
                                        df.to_csv(output_file, index=False)
                                        
                                        results[output_name] = df
                                        logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                                    except Exception as e:
                                        logger.warning(f"Error procesando {consolidated_url}: {str(e)}")
                            
                            # Solo necesitamos verificar un archivo para confirmar que existe el directorio
                            break
                
                # Buscar también archivos directamente en el directorio del torneo (sin subdirectorio de año)
                for file_pattern in ["points.csv", "matches.csv", "stats.csv", "all_points.csv", "all_matches.csv"]:
                    file_url = f"{base_url}/{tournament_dir}{file_pattern}"
                    
                    if self._check_file_exists(file_url):
                        logger.info(f"Encontrado archivo en directorio de torneo: {tournament_dir}{file_pattern}")
                        
                        output_name = f"{tournament}_{file_pattern.replace('.csv', '')}"
                        cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                        output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                        
                        try:
                            response = self._make_request(file_url)
                            content = response.content.decode('utf-8')
                            df = pd.read_csv(StringIO(content))
                            
                            # Añadir metadatos útiles si no existen
                            if 'tournament' not in df.columns:
                                df['tournament'] = tournament
                            
                            # Guardar archivo
                            cache_file.parent.mkdir(exist_ok=True)
                            output_file.parent.mkdir(exist_ok=True)
                            
                            df.to_csv(cache_file, index=False)
                            df.to_csv(output_file, index=False)
                            
                            results[output_name] = df
                            logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                        except Exception as e:
                            logger.warning(f"Error procesando {file_url}: {str(e)}")
        
        # 4. BÚSQUEDA POR ESTRUCTURA TORNEO_AÑO
        logger.info("Buscando en estructura de directorios torneo_año...")
        for tournament in tournaments:
            for year in years:
                year_str = str(year)
                combined_dir = f"{tournament}_{year_str}/"
                combined_url = f"{base_url}/{combined_dir}"
                
                # Verificar si existe el directorio combinado
                for test_file in ["1.csv", "points.csv", "matches.csv"]:
                    file_url = f"{combined_url}/{test_file}"
                    
                    if self._check_file_exists(file_url):
                        logger.info(f"Encontrado directorio combinado {combined_dir}")
                        
                        # Ahora buscar todos los archivos CSV en este directorio
                        for match_num in range(1, 501):  # Un límite razonable
                            match_file = f"{match_num}.csv"
                            match_url = f"{combined_url}/{match_file}"
                            
                            if self._check_file_exists(match_url):
                                logger.info(f"Encontrado archivo de partido: {combined_dir}{match_file}")
                                
                                # Descargar y procesar el archivo
                                output_name = f"{tournament}_{year}_match_{match_num}"
                                cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                                output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                                
                                try:
                                    response = self._make_request(match_url)
                                    content = response.content.decode('utf-8')
                                    df = pd.read_csv(StringIO(content))
                                    
                                    # Añadir metadatos útiles si no existen
                                    if 'tournament' not in df.columns:
                                        df['tournament'] = tournament
                                    if 'year' not in df.columns:
                                        df['year'] = year
                                    if 'match_num' not in df.columns:
                                        df['match_num'] = match_num
                                    
                                    # Guardar archivo
                                    cache_file.parent.mkdir(exist_ok=True)
                                    output_file.parent.mkdir(exist_ok=True)
                                    
                                    df.to_csv(cache_file, index=False)
                                    df.to_csv(output_file, index=False)
                                    
                                    results[output_name] = df
                                    logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                                except Exception as e:
                                    logger.warning(f"Error procesando {match_url}: {str(e)}")
                            else:
                                # Si no encontramos este número de partido, probablemente llegamos al final
                                if match_num > 1:
                                    break
                        
                        # Buscar también archivos consolidados de puntos o partidos
                        for consolidated_file in ["points.csv", "matches.csv", "stats.csv", "all_points.csv", "all_matches.csv"]:
                            consolidated_url = f"{combined_url}/{consolidated_file}"
                            
                            if self._check_file_exists(consolidated_url):
                                logger.info(f"Encontrado archivo consolidado: {combined_dir}{consolidated_file}")
                                
                                output_name = f"{tournament}_{year}_{consolidated_file.replace('.csv', '')}"
                                cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                                output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                                
                                try:
                                    response = self._make_request(consolidated_url)
                                    content = response.content.decode('utf-8')
                                    df = pd.read_csv(StringIO(content))
                                    
                                    # Añadir metadatos útiles si no existen
                                    if 'tournament' not in df.columns:
                                        df['tournament'] = tournament
                                    if 'year' not in df.columns:
                                        df['year'] = year
                                    
                                    # Guardar archivo
                                    cache_file.parent.mkdir(exist_ok=True)
                                    output_file.parent.mkdir(exist_ok=True)
                                    
                                    df.to_csv(cache_file, index=False)
                                    df.to_csv(output_file, index=False)
                                    
                                    results[output_name] = df
                                    logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                                except Exception as e:
                                    logger.warning(f"Error procesando {consolidated_url}: {str(e)}")
                        
                        # Solo necesitamos verificar un archivo para confirmar que existe el directorio
                        break
                
                # También probar la estructura AÑO_TORNEO
                combined_dir_alt = f"{year_str}_{tournament}/"
                combined_url_alt = f"{base_url}/{combined_dir_alt}"
                
                for test_file in ["1.csv", "points.csv", "matches.csv"]:
                    file_url = f"{combined_url_alt}/{test_file}"
                    
                    if self._check_file_exists(file_url):
                        logger.info(f"Encontrado directorio combinado alternativo {combined_dir_alt}")
                        
                        # Implementar la misma lógica que arriba para este patrón alternativo
                        # (código similar al bloque anterior, adaptado para combined_url_alt)
                        # ...
                        
                        # Solo necesitamos verificar un archivo para confirmar que existe el directorio
                        break
        
        # 5. BUSCAR DIRECTORIOS/ARCHIVOS ESPECIALES
        logger.info("Buscando directorios y archivos especiales...")
        
        # Directorio de recuento de rallies
        rallycount_dir = "rallycount/"
        rallycount_url = f"{base_url}/{rallycount_dir}"
        
        for file_pattern in ["all_points_with_rally_length.csv", "rally_stats.csv", "rally_data.csv"]:
            file_url = f"{rallycount_url}/{file_pattern}"
            
            if self._check_file_exists(file_url):
                logger.info(f"Encontrado archivo especial de rallycount: {rallycount_dir}{file_pattern}")
                
                output_name = f"rallycount_{file_pattern.replace('.csv', '')}"
                cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                
                try:
                    response = self._make_request(file_url)
                    content = response.content.decode('utf-8')
                    df = pd.read_csv(StringIO(content))
                    
                    # Guardar archivo
                    cache_file.parent.mkdir(exist_ok=True)
                    output_file.parent.mkdir(exist_ok=True)
                    
                    df.to_csv(cache_file, index=False)
                    df.to_csv(output_file, index=False)
                    
                    results[output_name] = df
                    logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                except Exception as e:
                    logger.warning(f"Error procesando {file_url}: {str(e)}")
        
        # Otros directorios especiales posibles
        special_dirs = ["data/", "csv/", "analysis/", "stats/", "combined/", "processed/"]
        
        for special_dir in special_dirs:
            special_url = f"{base_url}/{special_dir}"
            
            for file_pattern in ["all_points.csv", "all_matches.csv", "points_summary.csv", "match_summary.csv"]:
                file_url = f"{special_url}/{file_pattern}"
                
                if self._check_file_exists(file_url):
                    logger.info(f"Encontrado archivo en directorio especial: {special_dir}{file_pattern}")
                    
                    output_name = f"{special_dir.replace('/', '')}_{file_pattern.replace('.csv', '')}"
                    cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                    output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                    
                    try:
                        response = self._make_request(file_url)
                        content = response.content.decode('utf-8')
                        df = pd.read_csv(StringIO(content))
                        
                        # Guardar archivo
                        cache_file.parent.mkdir(exist_ok=True)
                        output_file.parent.mkdir(exist_ok=True)
                        
                        df.to_csv(cache_file, index=False)
                        df.to_csv(output_file, index=False)
                        
                        results[output_name] = df
                        logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                    except Exception as e:
                        logger.warning(f"Error procesando {file_url}: {str(e)}")
        
        # 6. BUSCAR ARCHIVOS ESPECÍFICOS EN LA RAÍZ
        logger.info("Buscando archivos específicos en la raíz del repositorio...")
        root_files = [
            "all_points.csv", "all_matches.csv", "combined_data.csv", 
            "points_master.csv", "matches_master.csv", "points_database.csv",
            "pointbypoint_data.csv", "pointbypoint_matches.csv"
        ]
        
        for file_name in root_files:
            file_url = f"{base_url}/{file_name}"
            
            if self._check_file_exists(file_url):
                logger.info(f"Encontrado archivo en la raíz: {file_name}")
                
                output_name = f"root_{file_name.replace('.csv', '')}"
                cache_file = self.cache_dir / 'pointbypoint' / f"{output_name}.csv"
                output_file = self.output_dir / 'pointbypoint' / f"{output_name}.csv"
                
                try:
                    response = self._make_request(file_url)
                    content = response.content.decode('utf-8')
                    df = pd.read_csv(StringIO(content))
                    
                    # Guardar archivo
                    cache_file.parent.mkdir(exist_ok=True)
                    output_file.parent.mkdir(exist_ok=True)
                    
                    df.to_csv(cache_file, index=False)
                    df.to_csv(output_file, index=False)
                    
                    results[output_name] = df
                    logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                except Exception as e:
                    logger.warning(f"Error procesando {file_url}: {str(e)}")
        
        # 7. SI AÚN NO SE HAN ENCONTRADO ARCHIVOS, INTENTAR CON LA API DE GITHUB
        if not results:
            logger.info("No se encontraron archivos con métodos de búsqueda estándar, intentando con API de GitHub...")
            github_results = self._download_pointbypoint_via_github()
            results.update(github_results)
        
        # 8. RESUMEN Y PROCESAMIENTO FINAL
        if results:
            # Número total de archivos y registros
            total_files = len(results)
            total_records = sum(len(df) for df in results.values())
            
            logger.info(f"=== Recopilación de datos punto por punto completada ===")
            logger.info(f"Total de archivos descargados: {total_files}")
            logger.info(f"Total de registros: {total_records:,}")
            
            # Crear archivo de metadatos
            try:
                metadata = {
                    "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_files": total_files,
                    "total_records": total_records,
                    "files": {}
                }
                
                for name, df in results.items():
                    # Extraer información del nombre para categorizar
                    parts = name.split('_')
                    category = "other"
                    
                    if "match" in name:
                        category = "match"
                    elif "point" in name:
                        category = "point"
                    elif "rally" in name:
                        category = "rally"
                    elif "stat" in name:
                        category = "stat"
                    
                    # Extraer torneo y año si están disponibles en las columnas
                    tournament = df['tournament'].iloc[0] if 'tournament' in df.columns and len(df) > 0 else "unknown"
                    year = df['year'].iloc[0] if 'year' in df.columns and len(df) > 0 else "unknown"
                    
                    metadata["files"][name] = {
                        "category": category,
                        "records": len(df),
                        "columns": list(df.columns),
                        "tournament": tournament,
                        "year": year,
                        "memory_size_kb": int(df.memory_usage(deep=True).sum() / 1024)
                    }
                
                # Guardar archivo de metadatos
                metadata_file = self.output_dir / 'pointbypoint' / 'metadata.json'
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Creado archivo de metadatos con información de {total_files} archivos")
                
                # Crear índice de partidos y puntos por torneo/año
                try:
                    # Agrupar archivos por torneo y año
                    tournament_year_files = {}
                    
                    for name, df in results.items():
                        # Extraer torneo y año de las columnas o del nombre
                        if 'tournament' in df.columns and 'year' in df.columns and len(df) > 0:
                            tournament = df['tournament'].iloc[0]
                            year = df['year'].iloc[0]
                        else:
                            # Intentar extraer del nombre
                            parts = name.split('_')
                            tournament = parts[0] if len(parts) > 0 else "unknown"
                            year = None
                            for part in parts:
                                if part.isdigit() and 1990 <= int(part) <= 2025:
                                    year = int(part)
                                    break
                            if year is None:
                                year = "unknown"
                        
                        key = f"{tournament}_{year}"
                        if key not in tournament_year_files:
                            tournament_year_files[key] = []
                        
                        tournament_year_files[key].append(name)
                    
                    # Crear y guardar el índice
                    index = {
                        "tournaments": {},
                        "years": {}
                    }
                    
                    for key, files in tournament_year_files.items():
                        tournament, year = key.split('_')
                        
                        # Actualizar índice de torneos
                        if tournament not in index["tournaments"]:
                            index["tournaments"][tournament] = {}
                        
                        if year not in index["tournaments"][tournament]:
                            index["tournaments"][tournament][year] = files
                        else:
                            index["tournaments"][tournament][year].extend(files)
                        
                        # Actualizar índice de años
                        if year not in index["years"]:
                            index["years"][year] = {}
                        
                        if tournament not in index["years"][year]:
                            index["years"][year][tournament] = files
                        else:
                            index["years"][year][tournament].extend(files)
                    
                    # Guardar índice
                    index_file = self.output_dir / 'pointbypoint' / 'index.json'
                    with open(index_file, 'w') as f:
                        json.dump(index, f, indent=2)
                    
                    logger.info(f"Creado índice de archivos por torneo y año")
                except Exception as e:
                    logger.warning(f"Error al crear índice por torneo/año: {str(e)}")
                
                # Intentar combinar datos de puntos por torneo/año para análisis más fácil
                try:
                    combined_points_by_tournament = {}
                    
                    for name, df in results.items():
                        if 'point' in name.lower() and len(df) > 0:
                            # Determinar el torneo
                            tournament = df['tournament'].iloc[0] if 'tournament' in df.columns else "unknown"
                            
                            if tournament not in combined_points_by_tournament:
                                combined_points_by_tournament[tournament] = []
                            
                            combined_points_by_tournament[tournament].append(df)
                    
                    # Combinar y guardar por torneo
                    for tournament, dfs in combined_points_by_tournament.items():
                        if len(dfs) > 1:
                            try:
                                combined_df = pd.concat(dfs, ignore_index=True)
                                output_file = self.output_dir / 'pointbypoint' / f"{tournament}_all_points_combined.csv"
                                combined_df.to_csv(output_file, index=False)
                                logger.info(f"Creado archivo combinado para {tournament}: {len(combined_df)} registros")
                            except Exception as e:
                                logger.warning(f"Error al combinar puntos para {tournament}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error al intentar combinar datos de puntos por torneo: {str(e)}")
            
            except Exception as e:
                logger.warning(f"Error al crear archivo de metadatos: {str(e)}")
        else:
            logger.warning("No se pudieron encontrar archivos de datos punto por punto.")
        
        return results
    
    def _download_pointbypoint_via_github(self) -> Dict[str, pd.DataFrame]:
        """
        Descarga datos punto por punto utilizando la API de GitHub.
        Esta función obtiene la lista completa de archivos CSV en el repositorio 
        sin depender de adivinar rutas o nombres.
        
        Returns:
            Dict[str, pd.DataFrame]: Diccionario con los DataFrames descargados
        """
        results = {}
        
        logger.info("Iniciando descarga de datos punto por punto mediante API de GitHub...")
        
        # Repositorio a explorar
        repo = "JeffSackmann/tennis_pointbypoint"
        
        try:
            # Obtener todos los archivos CSV del repositorio (profundidad máxima de 4 niveles)
            # ya que este repositorio puede tener una estructura más anidada
            csv_files = self._download_github_repo_files(repo, ['.csv'], max_depth=4)
            
            if not csv_files:
                logger.warning(f"No se encontraron archivos CSV en {repo} vía GitHub API")
                return results
            
            logger.info(f"Encontrados {len(csv_files)} archivos CSV en {repo} vía GitHub API")
            
            # Descargar todos los archivos encontrados
            for file_path, download_url in csv_files.items():
                # Crear un nombre único para este archivo
                # Reemplazar / por _ para evitar problemas con rutas
                safe_name = file_path.replace('/', '_').replace('-', '_').replace('.csv', '')
                
                # Si el nombre generado es demasiado largo, truncarlo de manera inteligente
                if len(safe_name) > 100:
                    # Extraer partes importantes (primero y últimos componentes)
                    parts = safe_name.split('_')
                    if len(parts) > 4:
                        # Tomar el primer componente (generalmente el año o torneo) y los dos últimos
                        safe_name = f"{parts[0]}_{parts[1]}__{parts[-2]}_{parts[-1]}"
                
                # Rutas de archivo
                cache_file = self.cache_dir / 'pointbypoint' / f"{safe_name}.csv"
                output_file = self.output_dir / 'pointbypoint' / f"{safe_name}.csv"
                
                try:
                    logger.info(f"Descargando {file_path} desde GitHub API")
                    response = self._make_request(download_url)
                    content = response.content.decode('utf-8')
                    
                    # Intentar procesar como CSV
                    try:
                        df = pd.read_csv(StringIO(content))
                        
                        # Verificar si el archivo parece ser realmente datos de puntos
                        # Algunas columnas comunes en archivos punto por punto
                        point_columns = ['point_no', 'point_id', 'server', 'receiver', 'winner', 'p1_score', 'p2_score', 
                                        'set_no', 'game_no', 'point_victor', 'serve_no']
                        
                        is_point_data = any(col in df.columns for col in point_columns)
                        
                        if is_point_data or len(df.columns) >= 5:  # Arquivos punto por punto suelen tener varias columnas
                            # Extraer información del camino del archivo para enriquecer los datos
                            parts = file_path.split('/')
                            
                            # Tratar de identificar torneo y año de la ruta
                            tournament = None
                            year = None
                            
                            for part in parts:
                                # Detectar año
                                if part.isdigit() and 1990 <= int(part) <= 2025:
                                    year = int(part)
                                
                                # Detectar torneo
                                tournaments = ["ausopen", "ao", "australian", "frenchopen", "fo", "rg", "roland", 
                                            "usopen", "uso", "us_open", "wimbledon", "wim"]
                                for t in tournaments:
                                    if t in part.lower():
                                        tournament = part
                                        break
                            
                            # Añadir metadatos útiles si no existen
                            if 'tournament' not in df.columns and tournament:
                                df['tournament'] = tournament
                            if 'year' not in df.columns and year:
                                df['year'] = year
                            
                            # Añadir ruta original como referencia
                            df['source_path'] = file_path
                            
                            # Guardar archivos
                            cache_file.parent.mkdir(exist_ok=True)
                            output_file.parent.mkdir(exist_ok=True)
                            
                            df.to_csv(cache_file, index=False)
                            df.to_csv(output_file, index=False)
                            
                            results[safe_name] = df
                            logger.info(f"Descargado archivo {safe_name}.csv vía GitHub API: {len(df)} registros")
                        else:
                            logger.debug(f"Omitiendo {file_path} - no parece contener datos punto por punto")
                    except pd.errors.EmptyDataError:
                        logger.warning(f"Archivo vacío: {file_path}")
                    except pd.errors.ParserError:
                        logger.warning(f"Error al parsear CSV: {file_path} - posiblemente no es un CSV válido")
                except Exception as e:
                    logger.warning(f"Error procesando {download_url}: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Error explorando {repo} vía GitHub API: {str(e)}")
        
        logger.info(f"Descarga vía GitHub API completada: {len(results)} archivos descargados")
        return results

    def _find_slam_pointbypoint_files(self) -> List[Dict[str, str]]:
        files = []
        base_url = f"{self.repo_base_url}{self.slam_pointbypoint_structure['repo']}"
        
        # Añadir verificación directa de archivos conocidos
        known_files = [
            {"tournament": "ausopen", "years": [2017, 2018, 2019, 2020]},
            {"tournament": "frenchopen", "years": [2017, 2018, 2019, 2020]},
            {"tournament": "usopen", "years": [2017, 2018, 2019, 2020, 2021]},
            {"tournament": "wimbledon", "years": [2017, 2018, 2019]}
        ]
        
        for tournament_info in known_files:
            tournament = tournament_info["tournament"]
            for year in tournament_info["years"]:
                # Verificar varios patrones específicos que se sabe que existen
                file_patterns = [
                    f"{tournament}/{year}/points.csv",
                    f"{year}/{tournament}/points.csv",
                    f"{tournament}_{year}/points.csv",
                    f"{tournament}/{year}.csv",
                    f"{tournament}/points_{year}.csv"
                ]
                
                for pattern in file_patterns:
                    file_url = f"{base_url}/{pattern}"
                    if self._check_file_exists(file_url):
                        files.append({
                            'name': f"{tournament}_{year}",
                            'url': file_url
                        })
                        logger.info(f"Encontrado archivo punto por punto: {pattern}")
                        break
        
        return files

    def _download_match_charting_data(self) -> Dict[str, pd.DataFrame]:
        """
        Descarga todos los datos disponibles del Match Charting Project desde el repositorio oficial.
        Implementa una búsqueda exhaustiva para encontrar todos los archivos y formatos posibles.
        
        Returns:
            Dict[str, pd.DataFrame]: Diccionario con los DataFrames descargados
        """
        results = {}
        
        # URL base del repositorio oficial
        base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
        
        logger.info("=== Descargando datos completos del Match Charting Project ===")
        
        # 1. Primero verificar que podemos acceder al repositorio
        readme_url = f"{base_url}/README.md"
        if not self._check_file_exists(readme_url):
            logger.warning("No se puede acceder al repositorio Match Charting Project. Verificando URL alternativa...")
            alt_url = "https://raw.githubusercontent.com/JeffSackmann/tennis-atp-charting/master/README.md"
            if not self._check_file_exists(alt_url):
                logger.error("No se puede acceder a ningún repositorio de Match Charting Project.")
                return results
            else:
                # Si la URL alternativa funciona, actualizar la base
                base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis-atp-charting/master"
                logger.info(f"Usando repositorio alternativo: {base_url}")
        
        # 2. Estructurar la búsqueda para ser más exhaustiva
        
        # 2.1 Definir todas las posibles ubicaciones de archivos
        locations = [
            "",  # Directorio raíz
            "data/",
            "csv/",
            "matches/",
            "stats/",
            "charting/"
        ]
        
        # 2.2 Definir todos los posibles nombres de archivos
        file_patterns = [
            # Archivos principales
            "matches.csv",
            "shots.csv",
            "charting-m-stats.csv",
            "charting-w-stats.csv",
            "charting-m-matches.csv",
            "charting-w-matches.csv",
            
            # Posibles variantes
            "match_stats.csv",
            "match_charting.csv",
            "charting_matches.csv",
            "charting_shots.csv",
            "matches_all.csv",
            "shots_all.csv",
            "match_data.csv",
            "player_stats.csv",
            "rally_stats.csv",
            "serve_stats.csv",
            "return_stats.csv",
            
            # Subdivisiones posibles
            "atp_matches_charted.csv",
            "wta_matches_charted.csv",
            "atp_shots.csv",
            "wta_shots.csv",
            "mens_matches.csv",
            "womens_matches.csv",
            "mens_shots.csv",
            "womens_shots.csv",
            
            # Combinaciones adicionales
            "all_matches.csv",
            "all_shots.csv",
            "complete_dataset.csv",
            "full_matches.csv",
            "all_match_stats.csv"
        ]
        
        # 3. Buscar sistemáticamente en cada ubicación
        for location in locations:
            logger.info(f"Explorando directorio: {base_url}/{location}")
            location_found_files = 0
            
            for file_pattern in file_patterns:
                url = f"{base_url}/{location}{file_pattern}"
                try:
                    if self._check_file_exists(url):
                        logger.info(f"Encontrado archivo: {location}{file_pattern}")
                        
                        # Crear un nombre único para el archivo de salida
                        # Reemplazar guiones con guiones bajos para consistencia
                        output_name = file_pattern.replace('-', '_').replace('.csv', '')
                        if location:
                            # Incluir la ruta en el nombre para evitar colisiones
                            output_name = f"{location.replace('/', '_')}{output_name}"
                        
                        # Descargar y procesar el archivo
                        cache_file = self.cache_dir / 'match_charting' / f"{output_name}.csv"
                        output_file = self.output_dir / 'match_charting' / f"{output_name}.csv"
                        
                        try:
                            response = self._make_request(url)
                            content = response.content.decode('utf-8')
                            df = pd.read_csv(StringIO(content))
                            
                            # Guardar el archivo en caché y en el directorio de salida
                            cache_file.parent.mkdir(exist_ok=True)
                            output_file.parent.mkdir(exist_ok=True)
                            
                            df.to_csv(cache_file, index=False)
                            df.to_csv(output_file, index=False)
                            
                            results[output_name] = df
                            location_found_files += 1
                            logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                        except Exception as e:
                            logger.warning(f"Error procesando {url}: {str(e)}")
                except Exception as e:
                    # No loggear errores de verificación para reducir ruido
                    pass
            
            logger.info(f"Encontrados {location_found_files} archivos en {location if location else 'directorio raíz'}")
        
        # 4. Buscar en posibles directorios estructurados por año
        years = range(2010, 2025)  # Ajustar rango según sea necesario
        
        for year in years:
            year_dir = f"{year}/"
            year_url = f"{base_url}/{year_dir}"
            
            # Verificar si existe un subdirectorio para este año
            for test_file in ["index.html", "README.md", "matches.csv"]:
                test_url = f"{year_url}{test_file}"
                if self._check_file_exists(test_url):
                    logger.info(f"Encontrado directorio para año {year}")
                    
                    # Buscar archivos en ese directorio
                    for file_pattern in file_patterns:
                        file_url = f"{year_url}{file_pattern}"
                        if self._check_file_exists(file_url):
                            logger.info(f"Encontrado archivo para año {year}: {file_pattern}")
                            
                            output_name = f"{year}_{file_pattern.replace('-', '_').replace('.csv', '')}"
                            cache_file = self.cache_dir / 'match_charting' / f"{output_name}.csv"
                            output_file = self.output_dir / 'match_charting' / f"{output_name}.csv"
                            
                            try:
                                response = self._make_request(file_url)
                                content = response.content.decode('utf-8')
                                df = pd.read_csv(StringIO(content))
                                
                                cache_file.parent.mkdir(exist_ok=True)
                                output_file.parent.mkdir(exist_ok=True)
                                
                                df.to_csv(cache_file, index=False)
                                df.to_csv(output_file, index=False)
                                
                                results[output_name] = df
                                logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                            except Exception as e:
                                logger.warning(f"Error procesando {file_url}: {str(e)}")
                    
                    # Solo necesitamos verificar un archivo para confirmar que existe el directorio
                    break
        
        # 5. Buscar en URL alternativa si aún no se han encontrado suficientes archivos
        if len(results) < 3:  # Si hemos encontrado menos de 3 archivos, intentar con URL alternativa
            alt_base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis-atp-charting/master"
            logger.info(f"Buscando archivos en repositorio alternativo: {alt_base_url}")
            
            for file_pattern in file_patterns:
                alt_url = f"{alt_base_url}/{file_pattern}"
                if self._check_file_exists(alt_url):
                    logger.info(f"Encontrado archivo en repositorio alternativo: {file_pattern}")
                    
                    output_name = f"alt_{file_pattern.replace('-', '_').replace('.csv', '')}"
                    cache_file = self.cache_dir / 'match_charting' / f"{output_name}.csv"
                    output_file = self.output_dir / 'match_charting' / f"{output_name}.csv"
                    
                    try:
                        response = self._make_request(alt_url)
                        content = response.content.decode('utf-8')
                        df = pd.read_csv(StringIO(content))
                        
                        cache_file.parent.mkdir(exist_ok=True)
                        output_file.parent.mkdir(exist_ok=True)
                        
                        df.to_csv(cache_file, index=False)
                        df.to_csv(output_file, index=False)
                        
                        results[output_name] = df
                        logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                    except Exception as e:
                        logger.warning(f"Error procesando {alt_url}: {str(e)}")
        
        # 6. Comprobar proyecto en GitHub directamente si es necesario
        if len(results) < 2:
            logger.info("Intentando obtener información del repositorio directamente vía GitHub API...")
            github_url = "https://api.github.com/repos/JeffSackmann/tennis_MatchChartingProject/contents"
            
            try:
                response = self._make_request(github_url)
                if response.status_code == 200:
                    contents = response.json()
                    logger.info(f"Obtenida estructura del repositorio: {len(contents)} elementos")
                    
                    # Analizar la estructura e identificar archivos CSV y directorios
                    for item in contents:
                        if item['type'] == 'file' and item['name'].endswith('.csv'):
                            logger.info(f"Encontrado archivo via GitHub API: {item['name']}")
                            
                            # Descargar el archivo usando la URL raw
                            file_url = item['download_url']
                            output_name = f"gh_{item['name'].replace('-', '_').replace('.csv', '')}"
                            
                            cache_file = self.cache_dir / 'match_charting' / f"{output_name}.csv"
                            output_file = self.output_dir / 'match_charting' / f"{output_name}.csv"
                            
                            try:
                                file_response = self._make_request(file_url)
                                content = file_response.content.decode('utf-8')
                                df = pd.read_csv(StringIO(content))
                                
                                cache_file.parent.mkdir(exist_ok=True)
                                output_file.parent.mkdir(exist_ok=True)
                                
                                df.to_csv(cache_file, index=False)
                                df.to_csv(output_file, index=False)
                                
                                results[output_name] = df
                                logger.info(f"Descargado archivo {output_name}.csv: {len(df)} registros")
                            except Exception as e:
                                logger.warning(f"Error procesando {file_url}: {str(e)}")
                        
                        elif item['type'] == 'dir':
                            # Registrar directorios pero no los exploramos por ahora
                            logger.info(f"Encontrado directorio via GitHub API: {item['name']}")
            except Exception as e:
                logger.warning(f"Error accediendo a GitHub API: {str(e)}")
        
        # 7. Intentar método de último recurso: verificar archivos conocidos específicos
        # que pueden existir en ubicaciones no estándar
        last_resort_files = [
            "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/match_charting/matches.csv",
            "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/match_charting/matches.csv",
            "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/main/matches.csv",  # Rama main en lugar de master
            "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/data/All_Matches.csv",
            "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/Match_Stats/all_matches.csv",
            "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/charting/charting-m-stats.csv"
        ]
        
        for url in last_resort_files:
            if self._check_file_exists(url):
                logger.info(f"Encontrado archivo de último recurso: {url}")
                
                # Extraer nombre de archivo de la URL
                file_name = os.path.basename(url)
                output_name = f"lr_{file_name.replace('-', '_').replace('.csv', '')}"
                
                cache_file = self.cache_dir / 'match_charting' / f"{output_name}.csv"
                output_file = self.output_dir / 'match_charting' / f"{output_name}.csv"
                
                try:
                    response = self._make_request(url)
                    content = response.content.decode('utf-8')
                    df = pd.read_csv(StringIO(content))
                    
                    cache_file.parent.mkdir(exist_ok=True)
                    output_file.parent.mkdir(exist_ok=True)
                    
                    df.to_csv(cache_file, index=False)
                    df.to_csv(output_file, index=False)
                    
                    results[output_name] = df
                    logger.info(f"Descargado archivo de último recurso {output_name}.csv: {len(df)} registros")
                except Exception as e:
                    logger.warning(f"Error procesando {url}: {str(e)}")
        
        # 8. Combinar archivos similares si es necesario
        try:
            # Identificar archivos que son del mismo tipo
            match_dfs = []
            shots_dfs = []
            stats_dfs = []
            
            for name, df in results.items():
                if 'match' in name.lower() and 'stat' not in name.lower() and 'shot' not in name.lower():
                    match_dfs.append((name, df))
                elif 'shot' in name.lower():
                    shots_dfs.append((name, df))
                elif 'stat' in name.lower():
                    stats_dfs.append((name, df))
            
            # Combinar matches si hay más de uno
            if len(match_dfs) > 1:
                logger.info(f"Combinando {len(match_dfs)} archivos de partidos...")
                match_names = [name for name, _ in match_dfs]
                combined_matches = pd.concat([df for _, df in match_dfs], ignore_index=True)
                
                # Eliminar duplicados si los hay
                if 'match_id' in combined_matches.columns:
                    orig_len = len(combined_matches)
                    combined_matches = combined_matches.drop_duplicates(subset=['match_id'])
                    if len(combined_matches) < orig_len:
                        logger.info(f"Eliminados {orig_len - len(combined_matches)} duplicados por match_id")
                
                # Guardar el archivo combinado
                output_file = self.output_dir / 'match_charting' / f"all_matches_combined.csv"
                combined_matches.to_csv(output_file, index=False)
                
                results['all_matches_combined'] = combined_matches
                logger.info(f"Creado archivo combinado de partidos: {len(combined_matches)} registros")
            
            # Combinar shots si hay más de uno
            if len(shots_dfs) > 1:
                logger.info(f"Combinando {len(shots_dfs)} archivos de tiros...")
                shot_names = [name for name, _ in shots_dfs]
                combined_shots = pd.concat([df for _, df in shots_dfs], ignore_index=True)
                
                # Guardar el archivo combinado
                output_file = self.output_dir / 'match_charting' / f"all_shots_combined.csv"
                combined_shots.to_csv(output_file, index=False)
                
                results['all_shots_combined'] = combined_shots
                logger.info(f"Creado archivo combinado de tiros: {len(combined_shots)} registros")
            
            # Combinar stats si hay más de uno
            if len(stats_dfs) > 1:
                logger.info(f"Combinando {len(stats_dfs)} archivos de estadísticas...")
                stat_names = [name for name, _ in stats_dfs]
                combined_stats = pd.concat([df for _, df in stats_dfs], ignore_index=True)
                
                # Guardar el archivo combinado
                output_file = self.output_dir / 'match_charting' / f"all_stats_combined.csv"
                combined_stats.to_csv(output_file, index=False)
                
                results['all_stats_combined'] = combined_stats
                logger.info(f"Creado archivo combinado de estadísticas: {len(combined_stats)} registros")
        
        except Exception as e:
            logger.warning(f"Error al intentar combinar archivos similares: {str(e)}")
        
        # Resumen final
        if results:
            logger.info(f"Se descargaron {len(results)} archivos del Match Charting Project.")
            
            # Crear un archivo de resumen para facilitar la comprensión de los datos
            try:
                summary = {
                    "total_files": len(results),
                    "files": {}
                }
                
                for name, df in results.items():
                    summary["files"][name] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "size_kb": df.memory_usage(deep=True).sum() / 1024
                    }
                
                summary_file = self.output_dir / 'match_charting' / 'summary.json'
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                logger.info(f"Creado archivo de resumen: {summary_file}")
            except Exception as e:
                logger.warning(f"Error al crear archivo de resumen: {str(e)}")
        else:
            logger.warning("No se pudieron encontrar archivos del Match Charting Project.")
        
        return results
    
    def _get_github_repo_contents(self, repo_path: str, directory: str = "") -> List[Dict]:
        """
        Obtiene la lista de archivos y directorios en un repositorio GitHub usando la API.
        
        Args:
            repo_path: Ruta del repositorio (formato: 'usuario/repo')
            directory: Directorio dentro del repositorio (opcional)
        
        Returns:
            Lista de diccionarios con información de los archivos/directorios
        """
        url = f"https://api.github.com/repos/{repo_path}/contents"
        if directory:
            url = f"{url}/{directory}"
        
        logger.info(f"Obteniendo contenidos de GitHub: {url}")
        
        try:
            # Usar headers para evitar límites de tasa
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": self.user_agent
            }
            
            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            contents = response.json()
            
            # Si es un solo archivo (no una lista), convertirlo a lista
            if not isinstance(contents, list):
                contents = [contents]
            
            return contents
            
        except Exception as e:
            logger.warning(f"Error obteniendo contenidos de GitHub: {str(e)}")
            return []

    def _download_github_repo_files(self, repo_path: str, file_types: List[str] = ['.csv'], 
                                max_depth: int = 2, directory: str = "", current_depth: int = 0) -> Dict[str, str]:
        """
        Descarga todos los archivos de ciertos tipos de un repositorio GitHub recursivamente.
        
        Args:
            repo_path: Ruta del repositorio (formato: 'usuario/repo')
            file_types: Lista de extensiones de archivo a buscar (por defecto solo CSV)
            max_depth: Profundidad máxima de recursión
            directory: Directorio actual dentro del repositorio
            current_depth: Profundidad actual de recursión
        
        Returns:
            Diccionario con rutas de archivos y sus URLs
        """
        if current_depth > max_depth:
            return {}
        
        result = {}
        contents = self._get_github_repo_contents(repo_path, directory)
        
        for item in contents:
            # Archivos
            if item['type'] == 'file':
                file_name = item['name']
                file_path = f"{directory}/{file_name}" if directory else file_name
                
                # Verificar si la extensión del archivo coincide con alguna de las buscadas
                if any(file_name.endswith(ext) for ext in file_types):
                    result[file_path] = item['download_url']
                    logger.debug(f"Encontrado archivo {file_path}")
            
            # Directorios (recursión)
            elif item['type'] == 'dir' and current_depth < max_depth:
                dir_name = item['name']
                dir_path = f"{directory}/{dir_name}" if directory else dir_name
                
                # Obtener archivos del subdirectorio
                sub_files = self._download_github_repo_files(
                    repo_path, 
                    file_types, 
                    max_depth, 
                    dir_path, 
                    current_depth + 1
                )
                
                # Añadir archivos del subdirectorio al resultado
                result.update(sub_files)
        
        return result

    def _download_match_charting_via_github(self) -> Dict[str, pd.DataFrame]:
        """
        Descarga todos los archivos CSV del Match Charting Project directamente usando la API de GitHub.
        Esta función no depende de adivinar nombres de archivos, sino que obtiene la lista completa.
        
        Returns:
            Dict[str, pd.DataFrame]: Diccionario con los DataFrames descargados
        """
        results = {}
        
        # Repositorios a verificar
        repos = [
            "JeffSackmann/tennis_MatchChartingProject",
            "JeffSackmann/tennis-atp-charting",
            "JeffSackmann/tennis-wta-charting"  # Por si acaso existe
        ]
        
        total_files_found = 0
        
        for repo in repos:
            logger.info(f"Explorando repositorio GitHub: {repo}")
            
            try:
                # Obtener todos los archivos CSV del repositorio (profundidad máxima de 3 niveles)
                csv_files = self._download_github_repo_files(repo, ['.csv'], max_depth=3)
                
                if not csv_files:
                    logger.info(f"No se encontraron archivos CSV en {repo}")
                    continue
                
                logger.info(f"Encontrados {len(csv_files)} archivos CSV en {repo}")
                
                # Descargar cada archivo encontrado
                for file_path, download_url in csv_files.items():
                    # Crear un nombre único para este archivo
                    safe_name = file_path.replace('/', '_').replace('-', '_').replace('.csv', '')
                    repo_prefix = repo.split('/')[1].replace('-', '_')
                    output_name = f"{repo_prefix}_{safe_name}"
                    
                    # Rutas de archivo
                    cache_file = self.cache_dir / 'match_charting' / f"{output_name}.csv"
                    output_file = self.output_dir / 'match_charting' / f"{output_name}.csv"
                    
                    try:
                        logger.info(f"Descargando {file_path} desde {download_url}")
                        response = self._make_request(download_url)
                        content = response.content.decode('utf-8')
                        df = pd.read_csv(StringIO(content))
                        
                        # Guardar archivos
                        cache_file.parent.mkdir(exist_ok=True)
                        output_file.parent.mkdir(exist_ok=True)
                        
                        df.to_csv(cache_file, index=False)
                        df.to_csv(output_file, index=False)
                        
                        results[output_name] = df
                        total_files_found += 1
                        
                        logger.info(f"Descargado {file_path}: {len(df)} registros")
                    except Exception as e:
                        logger.warning(f"Error procesando {download_url}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error explorando {repo}: {str(e)}")
        
        logger.info(f"Total de archivos descargados vía GitHub API: {total_files_found}")
        return results
    
    def collect_match_charting_data(self) -> Dict[str, pd.DataFrame]:
        """
        Función principal para recopilar todos los datos del Match Charting Project.
        Combina múltiples estrategias para garantizar la captura completa de datos.
        
        Returns:
            Dict[str, pd.DataFrame]: Diccionario con todos los datos recopilados
        """
        logger.info("=== Iniciando recopilación exhaustiva de datos del Match Charting Project ===")
        
        results = {}
        
        # 1. Primer intento: método tradicional optimizado
        traditional_results = self._download_match_charting_data()
        if traditional_results:
            logger.info(f"Método tradicional encontró {len(traditional_results)} archivos")
            results.update(traditional_results)
        
        # 2. Segundo intento: usando API de GitHub (más completo)
        if len(results) < 5:  # Si el primer método no encontró suficientes archivos
            logger.info("Intentando método de API de GitHub para descubrimiento completo de archivos")
            github_results = self._download_match_charting_via_github()
            
            # Añadir solo archivos que no se hayan encontrado antes
            for name, df in github_results.items():
                if name not in results:
                    results[name] = df
                    logger.info(f"Añadido archivo {name} vía GitHub API: {len(df)} registros")
        
        # 3. Análisis y procesamiento de los datos recopilados
        if results:
            # Categorizar archivos por tipo
            matches_files = {}
            shots_files = {}
            stats_files = {}
            other_files = {}
            
            for name, df in results.items():
                name_lower = name.lower()
                
                if 'match' in name_lower and 'stat' not in name_lower:
                    matches_files[name] = df
                elif 'shot' in name_lower:
                    shots_files[name] = df
                elif 'stat' in name_lower:
                    stats_files[name] = df
                else:
                    other_files[name] = df
            
            logger.info(f"Archivos categorizados - Partidos: {len(matches_files)}, Tiros: {len(shots_files)}, "
                    f"Estadísticas: {len(stats_files)}, Otros: {len(other_files)}")
            
            # Combinación de archivos por categoría
            
            # Combinar archivos de partidos
            if len(matches_files) > 0:
                try:
                    # Si hay varios archivos, intentar combinarlos
                    if len(matches_files) > 1:
                        combined_matches = pd.concat(matches_files.values(), ignore_index=True)
                        
                        # Eliminar duplicados si es posible
                        if 'match_id' in combined_matches.columns:
                            orig_len = len(combined_matches)
                            combined_matches.drop_duplicates(subset=['match_id'], inplace=True)
                            if len(combined_matches) < orig_len:
                                logger.info(f"Eliminados {orig_len - len(combined_matches)} duplicados de partidos")
                        
                        # Guardar archivo combinado
                        output_file = self.output_dir / 'match_charting' / 'all_matches_combined.csv'
                        combined_matches.to_csv(output_file, index=False)
                        
                        results['all_matches_combined'] = combined_matches
                        logger.info(f"Creado archivo combinado de partidos: {len(combined_matches)} registros")
                    else:
                        # Si solo hay un archivo, usarlo como archivo principal
                        key, df = next(iter(matches_files.items()))
                        logger.info(f"Usando {key} como archivo principal de partidos: {len(df)} registros")
                except Exception as e:
                    logger.warning(f"Error al combinar archivos de partidos: {str(e)}")
            
            # Combinar archivos de tiros/shots
            if len(shots_files) > 0:
                try:
                    # Si hay varios archivos, intentar combinarlos
                    if len(shots_files) > 1:
                        combined_shots = pd.concat(shots_files.values(), ignore_index=True)
                        
                        # Eliminar duplicados si es posible
                        if 'shot_id' in combined_shots.columns:
                            orig_len = len(combined_shots)
                            combined_shots.drop_duplicates(subset=['shot_id'], inplace=True)
                            if len(combined_shots) < orig_len:
                                logger.info(f"Eliminados {orig_len - len(combined_shots)} duplicados de tiros")
                        elif 'point_id' in combined_shots.columns and 'shot_no' in combined_shots.columns:
                            orig_len = len(combined_shots)
                            combined_shots.drop_duplicates(subset=['point_id', 'shot_no'], inplace=True)
                            if len(combined_shots) < orig_len:
                                logger.info(f"Eliminados {orig_len - len(combined_shots)} duplicados de tiros")
                        
                        # Guardar archivo combinado
                        output_file = self.output_dir / 'match_charting' / 'all_shots_combined.csv'
                        combined_shots.to_csv(output_file, index=False)
                        
                        results['all_shots_combined'] = combined_shots
                        logger.info(f"Creado archivo combinado de tiros: {len(combined_shots)} registros")
                    else:
                        # Si solo hay un archivo, usarlo como archivo principal
                        key, df = next(iter(shots_files.items()))
                        logger.info(f"Usando {key} como archivo principal de tiros: {len(df)} registros")
                except Exception as e:
                    logger.warning(f"Error al combinar archivos de tiros: {str(e)}")
            
            # Combinar archivos de estadísticas
            if len(stats_files) > 0:
                try:
                    # Si hay varios archivos, intentar combinarlos
                    if len(stats_files) > 1:
                        # Revisar si los archivos tienen estructuras compatibles
                        column_sets = [set(df.columns) for df in stats_files.values()]
                        common_columns = set.intersection(*column_sets) if column_sets else set()
                        
                        if common_columns:
                            logger.info(f"Archivos de estadísticas tienen {len(common_columns)} columnas en común")
                            
                            # Intentar combinar usando solo columnas comunes
                            combined_stats = pd.concat([df[list(common_columns)] for df in stats_files.values()], 
                                                    ignore_index=True)
                            
                            # Eliminar duplicados si es posible
                            if 'match_id' in combined_stats.columns and 'player_id' in combined_stats.columns:
                                orig_len = len(combined_stats)
                                combined_stats.drop_duplicates(subset=['match_id', 'player_id'], inplace=True)
                                if len(combined_stats) < orig_len:
                                    logger.info(f"Eliminados {orig_len - len(combined_stats)} duplicados de estadísticas")
                            
                            # Guardar archivo combinado
                            output_file = self.output_dir / 'match_charting' / 'all_stats_combined.csv'
                            combined_stats.to_csv(output_file, index=False)
                            
                            results['all_stats_combined'] = combined_stats
                            logger.info(f"Creado archivo combinado de estadísticas: {len(combined_stats)} registros")
                        else:
                            logger.warning("No se pudieron combinar archivos de estadísticas: estructuras incompatibles")
                            
                            # Guardar cada archivo por separado con nombres descriptivos
                            for i, (key, df) in enumerate(stats_files.items()):
                                output_file = self.output_dir / 'match_charting' / f"stats_group_{i+1}.csv"
                                df.to_csv(output_file, index=False)
                                logger.info(f"Guardado archivo de estadísticas {i+1}: {len(df)} registros")
                    else:
                        # Si solo hay un archivo, usarlo como archivo principal
                        key, df = next(iter(stats_files.items()))
                        logger.info(f"Usando {key} como archivo principal de estadísticas: {len(df)} registros")
                except Exception as e:
                    logger.warning(f"Error al combinar archivos de estadísticas: {str(e)}")
            
            # 4. Crear archivo de metadatos para facilitar el uso posterior
            try:
                metadata = {
                    "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_files": len(results),
                    "categories": {
                        "matches": len(matches_files),
                        "shots": len(shots_files),
                        "stats": len(stats_files),
                        "other": len(other_files)
                    },
                    "combined_files": [],
                    "file_details": {}
                }
                
                # Registrar archivos combinados
                if 'all_matches_combined' in results:
                    metadata["combined_files"].append({
                        "name": "all_matches_combined.csv",
                        "records": len(results['all_matches_combined']),
                        "source_files": list(matches_files.keys())
                    })
                
                if 'all_shots_combined' in results:
                    metadata["combined_files"].append({
                        "name": "all_shots_combined.csv",
                        "records": len(results['all_shots_combined']),
                        "source_files": list(shots_files.keys())
                    })
                
                if 'all_stats_combined' in results:
                    metadata["combined_files"].append({
                        "name": "all_stats_combined.csv",
                        "records": len(results['all_stats_combined']),
                        "source_files": list(stats_files.keys())
                    })
                
                # Registrar detalles de cada archivo individual
                for name, df in results.items():
                    if name not in ['all_matches_combined', 'all_shots_combined', 'all_stats_combined']:
                        metadata["file_details"][name] = {
                            "records": len(df),
                            "columns": list(df.columns),
                            "memory_size_kb": int(df.memory_usage(deep=True).sum() / 1024)
                        }
                
                # Guardar archivo de metadatos
                metadata_file = self.output_dir / 'match_charting' / 'metadata.json'
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Creado archivo de metadatos con información de {len(results)} archivos")
            except Exception as e:
                logger.warning(f"Error al crear archivo de metadatos: {str(e)}")
        
        else:
            logger.warning("No se encontraron archivos del Match Charting Project usando ningún método")
        
        # Resumen final
        logger.info(f"=== Recopilación de datos del Match Charting Project completada ===")
        logger.info(f"Total de archivos recopilados: {len(results)}")
        
        return results

    def _find_match_charting_files(self) -> List[Dict[str, str]]:
        files = []
        base_url = f"{self.repo_base_url}{self.match_charting_structure['repo']}"
        
        # Añadir verificación directa de archivos conocidos que existen actualmente
        known_files = [
            "charting-m-stats.csv",
            "charting-w-stats.csv", 
            "matches.csv",
            "shots.csv"
        ]
        
        for filename in known_files:
            file_url = f"{base_url}/{filename}"
            if self._check_file_exists(file_url):
                files.append({
                    'name': filename.replace('.csv', ''),
                    'url': file_url
                })
                logger.info(f"Encontrado archivo principal de Match Charting Project: {filename}")
        
        # En caso de que los archivos estén en el directorio raíz
        if not files:
            logger.info("Verificando archivos de Match Charting en directorio raíz")
            file_url = f"{self.repo_base_url}/tennis_MatchChartingProject/master/matches.csv"
            if self._check_file_exists(file_url):
                files.append({
                    'name': 'matches',
                    'url': file_url
                })
                logger.info("Encontrado archivo matches.csv en directorio raíz")
        
        return files

def main():
        """
        Función principal para ejecutar el script de recopilación de datos.
        Procesa argumentos de línea de comandos y ejecuta el colector.
        
        Ejemplo de uso:
        python data_collector.py --start-year 2000 --end-year 2024 --tours atp wta
        """
        parser = argparse.ArgumentParser(description='Recopilador de datos de tenis de Jeff Sackmann.')
        
        # Argumentos obligatorios
        parser.add_argument('--start-year', type=int, default=2000, 
                            help='Año inicial para la recopilación (default: 2000)')
        parser.add_argument('--end-year', type=int, default=datetime.now().year,
                            help=f'Año final para la recopilación (default: {datetime.now().year})')
        parser.add_argument('--tours', nargs='+', choices=['atp', 'wta'], default=['atp', 'wta'],
                            help='Tours a recopilar (default: atp wta)')
        
        # Argumentos opcionales
        parser.add_argument('--skip-challengers', action='store_true',
                            help='No incluir datos de torneos Challenger')
        parser.add_argument('--skip-futures', action='store_true',
                            help='No incluir datos de torneos Futures/ITF')
        parser.add_argument('--skip-rankings', action='store_true',
                            help='No incluir datos de rankings')
        parser.add_argument('--skip-pointbypoint', action='store_true',
                            help='No incluir datos punto por punto generales')
        parser.add_argument('--skip-slam-pointbypoint', action='store_true',
                            help='No incluir datos punto por punto de Grand Slam')
        parser.add_argument('--skip-match-charting', action='store_true',
                            help='No incluir datos del Match Charting Project')
        parser.add_argument('--force-refresh', action='store_true',
                            help='Forzar la descarga de todos los datos (ignorar caché)')
        parser.add_argument('--delay', type=float, default=0.2,
                            help='Demora entre solicitudes en segundos (default: 0.2)')
        
        args = parser.parse_args()
        
        # Configurar logger para mostrar un mensaje de inicio
        logger.info("====== INICIANDO RECOPILACIÓN DE DATOS DE TENIS ======")
        logger.info(f"Período: {args.start_year}-{args.end_year}")
        logger.info(f"Tours: {', '.join(args.tours)}")
        
        # Crear y configurar el colector
        collector = TennisDataCollector(
            start_year=args.start_year,
            end_year=args.end_year,
            tours=args.tours,
            include_challengers=not args.skip_challengers,
            include_futures=not args.skip_futures,
            include_rankings=not args.skip_rankings,
            include_pointbypoint=not args.skip_pointbypoint,
            include_slam_pointbypoint=not args.skip_slam_pointbypoint,
            include_match_charting=not args.skip_match_charting,
            force_refresh=args.force_refresh,
            delay_between_requests=args.delay
        )
        
        # Ejecutar la recopilación
        try:
            logger.info("Iniciando proceso de recopilación...")
            results = collector.collect_all_data()
            logger.info("¡Recopilación de datos completada con éxito!")
            
            # Mostrar breve resumen
            tours_str = ', '.join(args.tours).upper()
            logger.info(f"Datos recopilados para {tours_str} desde {args.start_year} hasta {args.end_year}")
            logger.info("Los datos se han guardado en el directorio 'data/processed/'")
            logger.info("Para más detalles, consulta los archivos de resumen en ese directorio")
            
        except Exception as e:
            logger.error(f"Error durante la recopilación de datos: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
        
        return 0

if __name__ == "__main__":
        sys.exit(main())