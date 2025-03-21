"""
data_collector.py

Script mejorado para recopilar y procesar datos de partidos de tenis con múltiples fuentes
y extracción de estadísticas detalladas.
"""

import pandas as pd
import numpy as np
import logging
import os
import requests
import traceback
import time
import json
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent
from pathlib import Path
import random
from io import StringIO

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

# Lista de user agents de respaldo
FALLBACK_USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0'
]

def get_user_agent() -> str:
    """
    Obtiene un user agent aleatorio, intentando primero con fake-useragent
    y usando la lista de respaldo si falla.
    
    Returns:
        str: User agent a utilizar
    """
    try:
        ua = UserAgent()
        return ua.random
    except Exception as e:
        logger.warning(f"Error al obtener user agent con fake-useragent: {str(e)}")
        logger.info("Usando user agent de respaldo")
        return random.choice(FALLBACK_USER_AGENTS)

class TennisDataCollector:
    """
    Clase base para recopilar y procesar datos de partidos de tenis.
    """
    
    def __init__(self, start_year=2000, end_year=None, user_agent=None, force_refresh=False):
        """
        Inicializa el colector de datos.
        
        Args:
            start_year: Año de inicio para la recopilación de datos
            end_year: Año final para la recopilación de datos
            user_agent: User Agent para peticiones HTTP
            force_refresh: Si es True, fuerza la recopilación de datos incluso si existen en caché
        """
        self.start_year = start_year
        self.end_year = end_year or datetime.now().year
        self.user_agent = user_agent or get_user_agent()
        self.headers = {'User-Agent': self.user_agent}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.force_refresh = force_refresh
        
        # Configurar directorios
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir = self.data_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Inicializado colector de datos para años {self.start_year}-{self.end_year}")
        if self.force_refresh:
            logger.info("Modo force_refresh activado: se recopilarán todos los datos de nuevo")
    
    def collect_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Recopila datos de partidos de tenis desde 2000 hasta el año actual.
        
        Args:
            start_year: Año de inicio para la recopilación de datos
            end_year: Año final para la recopilación de datos
            
        Returns:
            DataFrame con datos de partidos
        """
        try:
            logger.info(f"Iniciando recopilación de datos desde {start_year} hasta {end_year}")
            
            # Crear directorio de caché si no existe
            cache_dir = Path('data/cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            all_matches = []
            total_matches = 0
            
            # Procesar cada año
            for year in range(start_year, end_year + 1):
                try:
                    logger.info(f"Procesando año {year}...")
                    
                    # Verificar caché
                    cache_file = cache_dir / f"atp_matches_{year}.csv"
                    
                    if cache_file.exists() and not self.force_refresh:
                        logger.info(f"Usando datos en caché para el año {year}")
                        year_matches = pd.read_csv(cache_file)
                    else:
                        if self.force_refresh:
                            logger.info(f"Forzando recopilación de datos para el año {year}")
                        else:
                            logger.info(f"Recopilando datos para el año {year}")
                            
                        # Obtener datos del año
                        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
                        response = self.session.get(url)
                        response.raise_for_status()
                        
                        # Leer CSV directamente
                        content = response.content.decode('utf-8')
                        year_matches = pd.read_csv(StringIO(content))
                        
                        # Guardar en caché
                        year_matches.to_csv(cache_file, index=False)
                        logger.info(f"Datos del año {year} guardados en caché")
                    
                    # Procesar cada partido
                    for _, match in year_matches.iterrows():
                        try:
                            # Crear ID único para el partido
                            match_id = f"{year}_{match['tourney_id']}_{match['match_num']}"
                            
                            # Convertir valores a string si es necesario
                            surface = str(match['surface']).lower() if pd.notna(match['surface']) else 'unknown'
                            tournament_level = str(match['tourney_level']) if pd.notna(match['tourney_level']) else 'unknown'
                            score = str(match['score']) if pd.notna(match['score']) else ''
                            round_value = str(match['round']) if pd.notna(match['round']) else 'unknown'
                            
                            match_data = {
                                'match_id': match_id,
                                'tournament_id': str(match['tourney_id']),
                                'player1_id': str(match['winner_id']),
                                'player2_id': str(match['loser_id']),
                                'player1_name': str(match['winner_name']),
                                'player2_name': str(match['loser_name']),
                                'match_date': pd.to_datetime(str(match['tourney_date']), format='%Y%m%d'),
                                'surface': surface,
                                'tournament_name': str(match['tourney_name']),
                                'tournament_category': tournament_level,
                                'winner_id': str(match['winner_id']),
                                'score': score,
                                'sets_played': int(match['best_of']) if pd.notna(match['best_of']) else 3,
                                'minutes': int(match['minutes']) if pd.notna(match['minutes']) else 0,
                                'round': round_value,
                                'draw_size': int(match['draw_size']) if pd.notna(match['draw_size']) else 0
                            }
                            
                            all_matches.append(match_data)
                            total_matches += 1
                            
                            # Guardar progreso cada 1000 partidos
                            if total_matches % 1000 == 0:
                                logger.info(f"Procesados {total_matches} partidos...")
                                temp_df = pd.DataFrame(all_matches)
                                temp_df.to_csv('data/tennis_data_temp.csv', index=False)
                            
                        except Exception as e:
                            logger.warning(f"Error procesando partido en el año {year}: {str(e)}")
                            continue
                    
                    logger.info(f"Año {year} completado: {len(year_matches)} partidos procesados")
                    
                except Exception as e:
                    logger.warning(f"Error procesando año {year}: {str(e)}")
                    continue
            
            # Crear DataFrame final
            df = pd.DataFrame(all_matches)
            
            # Mostrar resumen final
            logger.info(f"Total de partidos: {len(df)}")
            logger.info(f"Rango de fechas: {df['match_date'].min().strftime('%Y%m%d')} - {df['match_date'].max().strftime('%Y%m%d')}")
            logger.info(f"Tipos de torneos: {df['tournament_category'].unique()}")
            logger.info(f"Total de torneos únicos: {df['tournament_id'].nunique()}")
            logger.info(f"Total de jugadores únicos: {pd.concat([df['player1_id'], df['player2_id']]).nunique()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error en recopilación de datos: {str(e)}")
            return pd.DataFrame()
