"""
data_collector.py

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
            }
        }
        
        self.match_charting_structure = {
            'repo': '/tennis_MatchChartingProject/master',
            'files': {
                'matches': 'matches.csv',
                'shots': 'shots.csv',
                'charting_m_stats': 'charting-m-stats.csv',
                'charting_w_stats': 'charting-w-stats.csv'
            }
        }
        
        self.pointbypoint_structure = {
            'repo': '/tennis_pointbypoint/master',
            # Este repositorio tiene una estructura más compleja y variada
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
            # Verificar si el repositorio existe probando un archivo que casi seguramente existe
            test_url = ""
            if repo == 'tennis_atp':
                test_url = f"{self.repo_base_url}/{repo}/master/atp_matches_2000.csv"
            elif repo == 'tennis_wta':
                test_url = f"{self.repo_base_url}/{repo}/master/wta_matches_2000.csv"
            elif repo == 'tennis_slam_pointbypoint':
                test_url = f"{self.repo_base_url}/{repo}/master/README.md"
            elif repo == 'tennis_MatchChartingProject':
                test_url = f"{self.repo_base_url}/{repo}/master/README.md"
            elif repo == 'tennis_pointbypoint':
                test_url = f"{self.repo_base_url}/{repo}/master/README.md"
            
            if test_url and self._check_file_exists(test_url):
                available_repos.append(repo)
                
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
            f"{tour}_rankings_current.csv"
        ]
        
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
    
    def _download_rankings_for_year(self, year: int, tour: str) -> pd.DataFrame:
        """
        Descarga rankings para un año específico usando la estructura detectada.
        
        Args:
            year: Año de los rankings a descargar
            tour: Tour de tenis ('atp' o 'wta')
            
        Returns:
            DataFrame con rankings del año
        """
        if not self.include_rankings:
            return pd.DataFrame()
        
        if tour not in self.repo_structure:
            logger.error(f"Tour no soportado: {tour}")
            return pd.DataFrame()
        
        try:
            # Verificar la estructura de rankings si aún no lo hemos hecho
            if not hasattr(self, 'rankings_structure'):
                self.rankings_structure = {}
            
            if tour not in self.rankings_structure:
                self.rankings_structure[tour] = self._check_rankings_structure(tour)
            
            # Si no se pudo determinar la estructura, retornar DataFrame vacío
            if not self.rankings_structure[tour]:
                logger.warning(f"No se pudo determinar la estructura de rankings para {tour.upper()}")
                return pd.DataFrame()
            
            # Intentar descargar rankings según el formato detectado
            format_type = self.rankings_structure[tour].get('format')
            
            if format_type == 'weekly':
                # Para formato semanal, necesitamos encontrar fechas disponibles para el año
                year_rankings = self._download_weekly_rankings(year, tour)
            elif format_type == 'decade':
                # Para formato por década, necesitamos encontrar el archivo apropiado y filtrar
                decade = self._get_decade_for_year(year)
                if decade in self.rankings_structure[tour].get('decades', []):
                    year_rankings = self._download_decade_rankings_and_filter(year, decade, tour)
                else:
                    logger.warning(f"No hay rankings disponibles para la década {decade} ({tour.upper()})")
                    return pd.DataFrame()
            elif format_type == 'consolidated' or format_type == 'root':
                # Para formato consolidado, descargamos todo y filtramos
                year_rankings = self._download_consolidated_rankings_and_filter(year, tour, format_type)
            else:
                logger.warning(f"Formato de rankings no reconocido para {tour.upper()}")
                return pd.DataFrame()
            
            # Si no hay rankings para este año
            if year_rankings.empty:
                logger.warning(f"No se encontraron rankings específicamente para {tour.upper()} {year}")
                return pd.DataFrame()
            
            # Guardar en directorio de salida
            output_file = self.output_dir / tour / f"{tour}_rankings_{year}.csv"
            year_rankings.to_csv(output_file, index=False)
            
            # Información
            dates_count = year_rankings['ranking_date'].nunique() if 'ranking_date' in year_rankings.columns else 0
            logger.info(f"Rankings de {tour.upper()} {year} guardados: {len(year_rankings)} registros de {dates_count} fechas")
            
            return year_rankings
            
        except Exception as e:
            logger.error(f"Error descargando rankings de {tour.upper()} {year}: {str(e)}")
            return pd.DataFrame()

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
    
    def _download_slam_pointbypoint(self, year: int, tournament: str) -> pd.DataFrame:
        """
        Descarga datos punto por punto de Grand Slam para un año y torneo específicos.
        
        Args:
            year: Año de los datos a descargar
            tournament: Torneo ('us_open', 'australian_open', 'wimbledon', 'french_open')
            
        Returns:
            DataFrame con datos punto por punto
        """
        if not self.include_slam_pointbypoint:
            return pd.DataFrame()
        
        if tournament not in self.slam_pointbypoint_structure['tournaments']:
            logger.error(f"Torneo no soportado: {tournament}")
            return pd.DataFrame()
        
        try:
            # Formar nombre de archivo
            file_pattern = self.slam_pointbypoint_structure['tournaments'][tournament]
            filename = file_pattern.format(year=year)
            
            # Formar URL
            base_path = self.slam_pointbypoint_structure['repo']
            url = f"{self.repo_base_url}{base_path}/{filename}"
            
            # Formar ruta de caché
            cache_file = self.cache_dir / 'slam_pointbypoint' / f"{filename}"
            
            # Descargar datos
            logger.info(f"Descargando datos punto por punto de {tournament.replace('_', ' ').title()} {year}...")
            df = self._download_file(url, cache_file)
            
            # Añadir columnas útiles
            df['tournament'] = tournament
            df['year'] = year
            
            # Verificar si hay datos
            if df.empty:
                logger.warning(f"No hay datos punto por punto para {tournament.replace('_', ' ').title()} {year}")
                return pd.DataFrame()
            
            # Información básica
            logger.info(f"Datos punto por punto de {tournament.replace('_', ' ').title()} {year} descargados: {len(df)} puntos")
            
            return df
            
        except Exception as e:
            logger.warning(f"Error descargando datos punto por punto de {tournament.replace('_', ' ').title()} {year}: {str(e)}")
            return pd.DataFrame()
    
    def _download_match_charting_data(self, data_type: str) -> pd.DataFrame:
        """
        Descarga datos del Match Charting Project.
        
        Args:
            data_type: Tipo de datos ('matches', 'shots', 'charting_m_stats', 'charting_w_stats')
            
        Returns:
            DataFrame con datos del Match Charting Project
        """
        if not self.include_match_charting:
            return pd.DataFrame()
        
        if data_type not in self.match_charting_structure['files']:
            logger.error(f"Tipo de datos no soportado: {data_type}")
            return pd.DataFrame()
        
        try:
            # Formar nombre de archivo
            filename = self.match_charting_structure['files'][data_type]
            
            # Formar URL
            base_path = self.match_charting_structure['repo']
            url = f"{self.repo_base_url}{base_path}/{filename}"
            
            # Formar ruta de caché
            cache_file = self.cache_dir / 'match_charting' / f"{filename}"
            
            # Descargar datos
            logger.info(f"Descargando datos de {data_type.replace('_', ' ').title()} del Match Charting Project...")
            df = self._download_file(url, cache_file)
            
            # Verificar si hay datos
            if df.empty:
                logger.warning(f"No hay datos de {data_type.replace('_', ' ').title()} del Match Charting Project")
                return pd.DataFrame()
            
            # Información básica
            logger.info(f"Datos de {data_type.replace('_', ' ').title()} del Match Charting Project descargados: {len(df)} registros")
            
            # Guardar en directorio de salida
            output_file = self.output_dir / 'match_charting' / f"{filename}"
            output_file.parent.mkdir(exist_ok=True)
            df.to_csv(output_file, index=False)
            
            logger.info(f"Datos de {data_type.replace('_', ' ').title()} del Match Charting Project guardados en {output_file}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Error descargando datos de {data_type.replace('_', ' ').title()} del Match Charting Project: {str(e)}")
            return pd.DataFrame()
    
    def collect_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Método principal que recopila todos los tipos de datos solicitados.
        
        Returns:
            Diccionario con todos los DataFrames recopilados
        """
        results = {
            'main_matches': {},
            'challenger_matches': {},
            'futures_matches': {},
            'players': {},
            'rankings': {},
            'slam_pointbypoint': {},
            'match_charting': {}
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
                rankings_frames = []
                
                for year in range(self.start_year, self.end_year + 1):
                    df = self._download_rankings_for_year(year, tour)
                    if not df.empty:
                        rankings_frames.append(df)
                
                if rankings_frames:
                    results['rankings'][tour] = pd.concat(rankings_frames, ignore_index=True)
                    
                    # Guardar rankings
                    output_file = self.output_dir / tour / f"{tour}_rankings_all_{self.start_year}_{self.end_year}.csv"
                    results['rankings'][tour].to_csv(output_file, index=False)
                    logger.info(f"Rankings de {tour.upper()} guardados en {output_file}: {len(results['rankings'][tour])} registros")
        
        # 6. Recopilar datos del Match Charting Project si se solicitaron y el repositorio está disponible
        if self.include_match_charting and 'tennis_MatchChartingProject' in self.available_repos:
            logger.info("=== Recopilando datos del Match Charting Project ===")
            
            # Verificar primero estructura real del repositorio
            mcp_files = self._find_match_charting_files()
            
            if mcp_files:
                for file_info in mcp_files:
                    df = self._download_match_charting_data_robust(file_info['name'], file_info['url'])
                    if not df.empty:
                        results['match_charting'][file_info['name']] = df
            else:
                logger.warning("No se encontraron archivos del Match Charting Project")
        
        # 7. Recopilar datos punto por punto de Grand Slam si se solicitaron y el repositorio está disponible
        if self.include_slam_pointbypoint and 'tennis_slam_pointbypoint' in self.available_repos:
            logger.info("=== Recopilando datos punto por punto de Grand Slam ===")
            
            # Verificar primero estructura real del repositorio
            slam_files = self._find_slam_pointbypoint_files()
            
            if slam_files:
                for file_info in slam_files:
                    df = self._download_slam_pointbypoint_robust(file_info['name'], file_info['url'])
                    if not df.empty:
                        # Extraer torneo y año del nombre
                        parts = file_info['name'].split('_')
                        if len(parts) >= 3:
                            tournament = parts[0]
                            if tournament not in results['slam_pointbypoint']:
                                results['slam_pointbypoint'][tournament] = []
                            results['slam_pointbypoint'][tournament].append(df)
                
                # Consolidar datos por torneo
                for tournament, frames in results['slam_pointbypoint'].items():
                    if frames:
                        combined = pd.concat(frames, ignore_index=True)
                        # Guardar datos punto por punto
                        output_file = self.output_dir / 'slam_pointbypoint' / f"{tournament}_{self.start_year}_{self.end_year}.csv"
                        output_file.parent.mkdir(exist_ok=True)
                        combined.to_csv(output_file, index=False)
                        logger.info(f"Datos punto por punto de {tournament.replace('_', ' ').title()} guardados en {output_file}: {len(combined)} puntos")
            else:
                logger.warning("No se encontraron archivos de datos punto por punto de Grand Slam")
        
        # 8. Crear archivos combinados
        self._create_combined_files(results)
        
        return results
    
    def _find_match_charting_files(self) -> List[Dict[str, str]]:
        """
        Encuentra archivos disponibles en el repositorio Match Charting Project.
        
        Returns:
            Lista de diccionarios con información de archivos
        """
        files = []
        
        # Intentar con archivos principales esperados
        base_url = f"{self.repo_base_url}{self.match_charting_structure['repo']}"
        
        # Comprobar si hay README para entender la estructura
        readme_url = f"{base_url}/README.md"
        if self._check_file_exists(readme_url):
            logger.debug("Encontrado README en repositorio Match Charting Project")
        
        # Probar con archivos comunes
        common_files = [
            "matches.csv",
            "shots.csv",
            "charting-m-stats.csv",
            "charting-w-stats.csv",
            "match_stats.csv",
            "match_charting.csv"
        ]
        
        for filename in common_files:
            file_url = f"{base_url}/{filename}"
            if self._check_file_exists(file_url):
                files.append({
                    'name': filename.replace('.csv', ''),
                    'url': file_url
                })
                logger.info(f"Encontrado archivo de Match Charting Project: {filename}")
        
        return files

    def _download_match_charting_data_robust(self, data_name: str, url: str) -> pd.DataFrame:
        """
        Descarga datos del Match Charting Project con manejo robusto de errores.
        
        Args:
            data_name: Nombre del conjunto de datos
            url: URL del archivo a descargar
            
        Returns:
            DataFrame con datos del Match Charting Project
        """
        try:
            # Formar ruta de caché
            filename = os.path.basename(url)
            cache_file = self.cache_dir / 'match_charting' / f"{filename}"
            
            # Descargar datos
            logger.info(f"Descargando datos de {data_name.replace('_', ' ').title()} del Match Charting Project...")
            df = self._download_file(url, cache_file)
            
            # Verificar si hay datos
            if df.empty:
                logger.warning(f"No hay datos de {data_name.replace('_', ' ').title()} del Match Charting Project")
                return pd.DataFrame()
            
            # Información básica
            logger.info(f"Datos de {data_name.replace('_', ' ').title()} del Match Charting Project descargados: {len(df)} registros")
            
            # Guardar en directorio de salida
            output_file = self.output_dir / 'match_charting' / f"{filename}"
            output_file.parent.mkdir(exist_ok=True)
            df.to_csv(output_file, index=False)
            
            logger.info(f"Datos de {data_name.replace('_', ' ').title()} del Match Charting Project guardados en {output_file}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Error descargando datos de {data_name.replace('_', ' ').title()} del Match Charting Project: {str(e)}")
            return pd.DataFrame()

    def _find_slam_pointbypoint_files(self) -> List[Dict[str, str]]:
        """
        Encuentra archivos disponibles en el repositorio de datos punto por punto de Grand Slam.
        
        Returns:
            Lista de diccionarios con información de archivos
        """
        files = []
        
        # Intentar con archivos principales esperados
        base_url = f"{self.repo_base_url}{self.slam_pointbypoint_structure['repo']}"
        
        # Comprobar si hay README para entender la estructura
        readme_url = f"{base_url}/README.md"
        if self._check_file_exists(readme_url):
            logger.debug("Encontrado README en repositorio de datos punto por punto de Grand Slam")
        
        # Buscar archivos por torneo y año
        tournaments = ['usopen', 'ausopen', 'wimbledon', 'frenchopen']
        years = range(max(self.start_year, 2000), self.end_year + 1)  # Usamos 2000 como límite inferior seguro
        
        for tournament in tournaments:
            for year in years:
                filename = f"{tournament}_points_{year}.csv"
                file_url = f"{base_url}/{filename}"
                
                if self._check_file_exists(file_url):
                    files.append({
                        'name': f"{tournament}_{year}",
                        'url': file_url
                    })
                    logger.info(f"Encontrado archivo de datos punto por punto: {filename}")
        
        # Búsqueda alternativa: verificar si hay directorios por torneo
        for tournament in tournaments:
            tournament_dir_url = f"{base_url}/{tournament}"
            # No podemos verificar directamente si existe un directorio, pero podemos probar con un archivo README
            readme_url = f"{tournament_dir_url}/README.md"
            if self._check_file_exists(readme_url):
                logger.debug(f"Encontrado directorio para {tournament}")
                
                # Buscar archivos por año dentro del directorio del torneo
                for year in years:
                    filename = f"points_{year}.csv"
                    file_url = f"{tournament_dir_url}/{filename}"
                    
                    if self._check_file_exists(file_url):
                        files.append({
                            'name': f"{tournament}_{year}",
                            'url': file_url
                        })
                        logger.info(f"Encontrado archivo de datos punto por punto: {tournament}/{filename}")
        
        return files

    def _download_slam_pointbypoint_robust(self, data_name: str, url: str) -> pd.DataFrame:
        """
        Descarga datos punto por punto de Grand Slam con manejo robusto de errores.
        
        Args:
            data_name: Nombre del conjunto de datos (en formato torneo_año)
            url: URL del archivo a descargar
            
        Returns:
            DataFrame con datos punto por punto
        """
        try:
            # Extraer torneo y año del nombre
            parts = data_name.split('_')
            if len(parts) < 2:
                logger.warning(f"Formato de nombre no válido: {data_name}")
                return pd.DataFrame()
                
            tournament = parts[0]
            year = int(parts[1])
            
            # Formar ruta de caché
            filename = os.path.basename(url)
            cache_file = self.cache_dir / 'slam_pointbypoint' / f"{filename}"
            
            # Descargar datos
            logger.info(f"Descargando datos punto por punto de {tournament.replace('_', ' ').title()} {year}...")
            df = self._download_file(url, cache_file)
            
            # Añadir columnas útiles
            df['tournament'] = tournament
            df['year'] = year
            
            # Verificar si hay datos
            if df.empty:
                logger.warning(f"No hay datos punto por punto para {tournament.replace('_', ' ').title()} {year}")
                return pd.DataFrame()
            
            # Información básica
            logger.info(f"Datos punto por punto de {tournament.replace('_', ' ').title()} {year} descargados: {len(df)} puntos")
            
            return df
            
        except Exception as e:
            logger.warning(f"Error descargando datos punto por punto de {data_name.replace('_', ' ').title()}: {str(e)}")
            return pd.DataFrame()
    
    def _create_combined_files(self, results: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """
        Crea archivos combinados a partir de los datos recopilados.
        
        Args:
            results: Diccionario con los DataFrames recopilados
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
            
            # 3. Generar resumen final
            self._generate_summary(results)
            
        except Exception as e:
            logger.error(f"Error creando archivos combinados: {str(e)}")

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
            for data_type, df in results['match_charting'].items():
                summary['counts'][f'match_charting_{data_type}'] = len(df)
            
            # Contar datos punto por punto de Grand Slam
            for tournament, df in results['slam_pointbypoint'].items():
                summary['counts'][f'slam_pointbypoint_{tournament}'] = len(df)
            
            # Calcular totales
            total_matches = sum([
                len(df) for df in results['main_matches'].values()
            ]) + sum([
                len(df) for df in results['challenger_matches'].values()
            ]) + sum([
                len(df) for df in results['futures_matches'].values()
            ])
            
            summary['counts']['total_matches'] = total_matches
            
            # Guardar resumen
            with open(self.output_dir / 'data_collection_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Mostrar resumen
            logger.info("=== RESUMEN DE LA RECOPILACIÓN DE DATOS ===")
            logger.info(f"Período: {summary['year_range']}")
            logger.info(f"Tours: {', '.join(summary['tours'])}")
            for key, count in summary['counts'].items():
                logger.info(f"{key}: {count:,}")
            logger.info("==========================================")
            
        except Exception as e:
            logger.error(f"Error generando resumen: {str(e)}")


def main():
    """Función principal para ejecutar el colector de datos."""
    parser = argparse.ArgumentParser(description='Colector mejorado de datos de tenis')
    parser.add_argument('--start-year', type=int, default=2000, help='Año inicial (default: 2000)')
    parser.add_argument('--end-year', type=int, default=datetime.now().year, help='Año final (default: año actual)')
    parser.add_argument('--tours', type=str, nargs='+', default=['atp', 'wta'], choices=['atp', 'wta'], 
                        help='Tours a recopilar (default: atp wta)')
    parser.add_argument('--no-challengers', action='store_true', help='Excluir partidos de Challenger')
    parser.add_argument('--no-futures', action='store_true', help='Excluir partidos de Futures/ITF')
    parser.add_argument('--no-rankings', action='store_true', help='Excluir datos de rankings')
    parser.add_argument('--no-pointbypoint', action='store_true', help='Excluir datos punto por punto generales')
    parser.add_argument('--no-slam-pointbypoint', action='store_true', help='Excluir datos punto por punto de Grand Slam')
    parser.add_argument('--no-match-charting', action='store_true', help='Excluir datos del Match Charting Project')
    parser.add_argument('--force-refresh', action='store_true', help='Forzar recopilación de datos incluso si existen en caché')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Directorio para archivos de salida')
    parser.add_argument('--delay', type=float, default=0.2, help='Retraso entre solicitudes (segundos)')
    parser.add_argument('--clear-cache', action='store_true', help='Borrar caché antes de iniciar')
        
    args = parser.parse_args()
        
    try:
        # Configurar directorio de salida personalizado si se especificó
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
            
        # Borrar caché si se solicitó
        if args.clear_cache:
            import shutil
            cache_dir = Path('data/cache')
            if cache_dir.exists():
                logger.info(f"Borrando directorio de caché: {cache_dir}")
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Inicializar colector
        collector = TennisDataCollector(
            start_year=args.start_year,
            end_year=args.end_year,
            tours=args.tours,
            include_challengers=not args.no_challengers,
            include_futures=not args.no_futures,
            include_rankings=not args.no_rankings,
            include_pointbypoint=not args.no_pointbypoint,
            include_slam_pointbypoint=not args.no_slam_pointbypoint,
            include_match_charting=not args.no_match_charting,
            force_refresh=args.force_refresh,
            delay_between_requests=args.delay
        )
            
        # Recopilar todos los datos
        results = collector.collect_all_data()
            
        # Contar total de partidos
        total_matches = sum(len(df) for tour_dict in [results['main_matches'], 
                                                    results['challenger_matches'], 
                                                    results['futures_matches']] 
                                    for df in tour_dict.values())
            
        logger.info(f"Proceso completado con éxito. Total de partidos: {total_matches:,}")
            
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
        main()