
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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ml-service/logs/data_collection.log'),
        logging.StreamHandler()
    ]
)

class TennisDataCollector:
    """
    Clase mejorada para recopilar y procesar datos de partidos de tenis.
    Incluye múltiples fuentes de datos y extracción avanzada de estadísticas.
    """
    
    def __init__(self, start_year=2000, end_year=None, user_agent=None):
        """
        Inicializa el colector de datos con múltiples fuentes.
        
        Args:
            start_year: Año de inicio para la recopilación de datos (default: 2000)
            end_year: Año final para la recopilación de datos (default: año actual)
            user_agent: User Agent para peticiones HTTP (default: Chrome)
        """
        # Configurar años
        self.start_year = start_year
        self.end_year = end_year or datetime.now().year
        
        # Configurar HTTP
        self.user_agent = user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        self.headers = {'User-Agent': self.user_agent}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Configurar rutas de archivos
        self.data_dir = os.path.abspath('data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Añadir fuentes de datos
        self._configure_sources()
        
        # Mapeo de columnas para convertir al formato estándar
        self._configure_mappings()
        
        # Estadísticas adicionales a recopilar
        self.additional_stats = {
            'ace_percentage': True,
            'double_fault_percentage': True,
            'break_points_converted': True,
            'first_serve_percentage': True,
            'service_games_won': True
        }
        
        logging.info(f"Inicializado colector de datos para años {self.start_year}-{self.end_year}")
    
    def _configure_sources(self):
        """Configura múltiples fuentes de datos para tenis."""
        self.sources = {}
        
        # 1. Datos de Jeff Sackmann (ATP y WTA)
        for year in range(self.start_year, self.end_year + 1):
            # ATP (masculino)
            self.sources[f'atp_{year}'] = {
                'url': f'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv',
                'type': 'csv',
                'tour': 'atp'
            }
            
            # WTA (femenino)
            self.sources[f'wta_{year}'] = {
                'url': f'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv',
                'type': 'csv',
                'tour': 'wta'
            }
        
        # 2. Datos de Tennis-Data.co.uk
        current_year = datetime.now().year
        for year in range(self.start_year, self.end_year + 1):
            if year <= current_year:
                self.sources[f'tennisdata_{year}'] = {
                    'url': f'http://www.tennis-data.co.uk/{year}/{year}.zip',
                    'type': 'zip',
                    'tour': 'mixed'
                }
        
        # 3. Ultimate Tennis Statistics (específico para torneos importantes)
        self.sources['uts_grand_slams'] = {
            'url': 'https://www.ultimatetennisstatistics.com/tournamentEvents',
            'type': 'web',
            'tour': 'atp',
            'filter': {'level': 'G'}  # Grand Slams
        }
        
        logging.info(f"Configuradas {len(self.sources)} fuentes de datos")
    
    def _configure_mappings(self):
        """Configura los mapeos de columnas para diferentes fuentes de datos."""
        # Mapeo general para datos de Jeff Sackmann
        self.column_mapping = {
            'winner_name': 'player_1',
            'loser_name': 'player_2',
            'winner_rank': 'ranking_1',
            'loser_rank': 'ranking_2',
            'winner_ht': 'height_1',
            'loser_ht': 'height_2',
            'winner_age': 'age_1',
            'loser_age': 'age_2',
            'winner_hand': 'hand_1',
            'loser_hand': 'hand_2',
            'surface': 'surface',
            'tourney_name': 'tournament',
            'tourney_date': 'match_date',
            'score': 'score',
            'best_of': 'best_of',
            'round': 'round',
            'minutes': 'duration',
            'w_ace': 'aces_1',
            'l_ace': 'aces_2',
            'w_df': 'double_faults_1',
            'l_df': 'double_faults_2',
            'w_svpt': 'service_points_1',
            'l_svpt': 'service_points_2',
            'w_1stIn': 'first_serves_in_1',
            'l_1stIn': 'first_serves_in_2',
            'w_1stWon': 'first_serve_points_won_1',
            'l_1stWon': 'first_serve_points_won_2',
            'w_2ndWon': 'second_serve_points_won_1',
            'l_2ndWon': 'second_serve_points_won_2',
            'w_SvGms': 'service_games_1',
            'l_SvGms': 'service_games_2',
            'w_bpSaved': 'break_points_saved_1',
            'l_bpSaved': 'break_points_saved_2',
            'w_bpFaced': 'break_points_faced_1',
            'l_bpFaced': 'break_points_faced_2'
        }
        
        # Mapeo específico para Tennis-Data.co.uk
        self.tennisdata_mapping = {
            'Winner': 'player_1',
            'Loser': 'player_2',
            'WRank': 'ranking_1',
            'LRank': 'ranking_2',
            'Surface': 'surface',
            'Tournament': 'tournament',
            'Date': 'match_date',
            'Score': 'score',
            'Best of': 'best_of',
            'Round': 'round',
            'WPts': 'ranking_points_1',
            'LPts': 'ranking_points_2'
        }
        
        # Mapeo específico para Ultimate Tennis Statistics
        self.uts_mapping = {
            'winner.name': 'player_1',
            'loser.name': 'player_2',
            'winner.rank': 'ranking_1',
            'loser.rank': 'ranking_2',
            'indoor': 'indoor',
            'surface': 'surface',
            'tournament.name': 'tournament',
            'date': 'match_date',
            'score': 'score',
            'outcome': 'outcome'
        }
    
    def collect_data(self, output_path: str, max_sources: Optional[int] = None, 
                     skip_errors: bool = True, parallel: bool = True) -> Optional[pd.DataFrame]:
        """
        Recopila datos de partidos de tenis de múltiples fuentes configuradas.
        
        Args:
            output_path: Ruta donde guardar los datos procesados
            max_sources: Número máximo de fuentes a procesar (None = todas)
            skip_errors: Si es True, continúa con otras fuentes si una falla
            parallel: Si es True, usa procesamiento en paralelo para fuentes
            
        Returns:
            DataFrame con datos procesados o None si hay error crítico
        """
        try:
            all_data = []
            processed_count = 0
            error_count = 0
            total_sources = len(self.sources)
            
            logging.info(f"Iniciando recopilación de datos desde {total_sources} fuentes")
            
            # Limitar fuentes si se especifica max_sources
            source_items = list(self.sources.items())
            if max_sources is not None:
                # Priorizar años más recientes para ATP y WTA
                source_items = sorted(source_items, key=lambda x: x[0], reverse=True)
                source_items = source_items[:max_sources]
            
            # Función para procesar una fuente
            def process_source(source_name, source_info):
                try:
                    source_type = source_info.get('type', 'csv')
                    logging.info(f"Procesando fuente {source_name} (tipo: {source_type})")
                    
                    if source_type == 'csv':
                        data = self._process_csv_source(source_name, source_info)
                    elif source_type == 'zip':
                        data = self._process_zip_source(source_name, source_info)
                    elif source_type == 'web':
                        data = self._process_web_source(source_name, source_info)
                    else:
                        logging.warning(f"Tipo de fuente desconocido: {source_type}")
                        return None
                    
                    if data is not None and not data.empty:
                        # Añadir identificador de fuente y tour
                        data['source'] = source_name
                        data['tour'] = source_info.get('tour', 'unknown')
                        
                        # Normalizar fechas si es posible
                        if 'match_date' in data.columns:
                            data = self._normalize_dates(data)
                        
                        logging.info(f"Datos de {source_name} procesados: {len(data)} partidos")
                        return data
                    else:
                        logging.warning(f"No se obtuvieron datos de {source_name}")
                        return None
                    
                except Exception as e:
                    logging.error(f"Error procesando {source_name}: {e}")
                    if not skip_errors:
                        raise
                    return None
            
            # Procesar fuentes (en paralelo o secuencial)
            if parallel:
                with ThreadPoolExecutor(max_workers=min(10, len(source_items))) as executor:
                    futures = {executor.submit(process_source, name, info): name 
                               for name, info in source_items}
                    
                    for future in as_completed(futures):
                        source_name = futures[future]
                        try:
                            data = future.result()
                            if data is not None:
                                all_data.append(data)
                                processed_count += 1
                            else:
                                error_count += 1
                        except Exception as e:
                            logging.error(f"Error en fuente {source_name}: {e}")
                            error_count += 1
            else:
                # Procesamiento secuencial
                for source_name, source_info in source_items:
                    data = process_source(source_name, source_info)
                    if data is not None:
                        all_data.append(data)
                        processed_count += 1
                    else:
                        error_count += 1
            
            # Resumen del procesamiento
            logging.info(f"Procesadas {processed_count} fuentes exitosamente, {error_count} con errores")
            
            if not all_data:
                logging.error("No se pudo cargar ninguna fuente de datos")
                return None
            
            # Combinar datos
            logging.info("Combinando datos de todas las fuentes...")
            combined_data = pd.concat(all_data, ignore_index=True)
            raw_count = len(combined_data)
            logging.info(f"Total de partidos recopilados inicialmente: {raw_count}")
            
            # Convertir al formato estándar
            logging.info("Convirtiendo al formato estándar para el modelo...")
            processed_data = self._convert_to_model_format(combined_data)
            
            # Eliminar duplicados
            logging.info("Eliminando partidos duplicados...")
            processed_data = self._remove_duplicates(processed_data)
            
            # Calcular estadísticas derivadas
            processed_data = self._calculate_derived_stats(processed_data)
            
            final_count = len(processed_data)
            logging.info(f"Datos finales: {final_count} partidos ({final_count/raw_count*100:.1f}% de los datos originales)")
            
            # Crear directorio de salida si no existe
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Guardar datos
            processed_data.to_csv(output_path, index=False)
            logging.info(f"Datos guardados exitosamente en {output_path}")
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error crítico en recopilación de datos: {e}")
            traceback.print_exc()
            return None
    
    def _process_csv_source(self, source_name: str, source_info: Dict) -> Optional[pd.DataFrame]:
        """Procesa una fuente de datos CSV."""
        url = source_info['url']
        cache_file = os.path.join(self.cache_dir, f"{source_name}.csv")
        
        # Verificar caché
        if os.path.exists(cache_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 7:
            logging.info(f"Utilizando datos en caché para {source_name}")
            return pd.read_csv(cache_file)
        
        # Descargar datos
        try:
            logging.info(f"Descargando datos de {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Guardar en caché
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            # Cargar datos
            data = pd.read_csv(cache_file)
            
            # Verificar columnas clave
            if 'winner_name' not in data.columns or 'loser_name' not in data.columns:
                logging.warning(f"Formato inesperado en {source_name}: faltan columnas winner_name/loser_name")
            
            return data
            
        except Exception as e:
            logging.error(f"Error descargando/procesando {source_name}: {e}")
            return None
    
    def _process_zip_source(self, source_name: str, source_info: Dict) -> Optional[pd.DataFrame]:
        """Procesa una fuente de datos ZIP."""
        import zipfile
        import io
        
        url = source_info['url']
        cache_file = os.path.join(self.cache_dir, f"{source_name}.zip")
        extracted_dir = os.path.join(self.cache_dir, source_name)
        
        # Verificar caché
        if os.path.exists(extracted_dir) and os.listdir(extracted_dir):
            logging.info(f"Utilizando datos ZIP extraídos para {source_name}")
            # Combinar todos los CSV en el directorio
            dfs = []
            for file in os.listdir(extracted_dir):
                if file.endswith('.csv') or file.endswith('.xls') or file.endswith('.xlsx'):
                    file_path = os.path.join(extracted_dir, file)
                    try:
                        if file.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        else:
                            df = pd.read_excel(file_path)
                        dfs.append(df)
                    except Exception as e:
                        logging.warning(f"Error leyendo {file_path}: {e}")
            
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            
        # Descargar ZIP
        try:
            logging.info(f"Descargando archivo ZIP de {url}")
            response = self.session.get(url, timeout=60)
            
            # Si la respuesta no es exitosa, intentar URL alternativa
            if response.status_code != 200:
                year = source_name.split('_')[1]
                alt_url = f"http://www.tennis-data.co.uk/{year}.zip"
                logging.info(f"Intentando URL alternativa: {alt_url}")
                response = self.session.get(alt_url, timeout=60)
            
            response.raise_for_status()
            
            # Guardar en caché
            with open(cache_file, "wb") as f:
                f.write(response.content)
            
            # Extraer archivos
            os.makedirs(extracted_dir, exist_ok=True)
            with zipfile.ZipFile(cache_file, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            
            # Leer archivos extraídos
            dfs = []
            for file in os.listdir(extracted_dir):
                if file.endswith('.csv') or file.endswith('.xls') or file.endswith('.xlsx'):
                    file_path = os.path.join(extracted_dir, file)
                    try:
                        if file.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        else:
                            df = pd.read_excel(file_path)
                        # Aplicar mapeo específico para Tennis-Data
                        df = self._apply_mapping(df, self.tennisdata_mapping)
                        dfs.append(df)
                    except Exception as e:
                        logging.warning(f"Error leyendo {file_path}: {e}")
            
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            else:
                logging.warning(f"No se encontraron archivos CSV/Excel en el ZIP de {source_name}")
                return None
                
        except Exception as e:
            logging.error(f"Error procesando ZIP {source_name}: {e}")
            return None
    
    def _process_web_source(self, source_name: str, source_info: Dict) -> Optional[pd.DataFrame]:
        """Procesa una fuente de datos web con scraping."""
        # Esta función requeriría implementar web scraping específico
        # para cada sitio web, lo cual está fuera del alcance actual
        logging.warning(f"Fuente web {source_name} aún no implementada")
        
        # En una implementación completa, aquí iría código para:
        # 1. Navegar al sitio web
        # 2. Usar BeautifulSoup o Selenium para extraer datos
        # 3. Convertir los datos extraídos a un DataFrame
        
        return None
    
    def _apply_mapping(self, df: pd.DataFrame, mapping: Dict) -> pd.DataFrame:
        """Aplica un mapeo de columnas a un DataFrame."""
        rename_dict = {}
        
        for orig_col, new_col in mapping.items():
            if orig_col in df.columns:
                rename_dict[orig_col] = new_col
        
        return df.rename(columns=rename_dict)
    
    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza fechas en diferentes formatos."""
        if 'match_date' not in df.columns:
            return df
        
        try:
            # Detectar tipo de fecha
            sample = df['match_date'].iloc[0] if not df.empty else None
            
            if isinstance(sample, (int, float)):
                # Formato YYYYMMDD (común en datos de Jeff Sackmann)
                df['match_date'] = pd.to_datetime(df['match_date'].astype(str), format='%Y%m%d', errors='coerce')
            elif isinstance(sample, str):
                # Intentar varios formatos de fecha
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y.%m.%d']:
                    try:
                        df['match_date'] = pd.to_datetime(df['match_date'], format=fmt, errors='coerce')
                        if not pd.isna(df['match_date']).all():
                            break
                    except:
                        continue
            
            # Si todavía hay fechas nulas, usar pd.to_datetime con inferencia
            null_dates = df['match_date'].isna()
            if null_dates.any():
                df.loc[null_dates, 'match_date'] = pd.to_datetime(
                    df.loc[null_dates, 'match_date'], errors='coerce')
            
            # Asegurarse de que match_date sea datetime
            df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
            
            # Filtrar fechas nulas
            df = df.dropna(subset=['match_date'])
            
            # Añadir columnas derivadas de fecha
            df['year'] = df['match_date'].dt.year
            df['month'] = df['match_date'].dt.month
            df['day'] = df['match_date'].dt.day
            
        except Exception as e:
            logging.warning(f"Error normalizando fechas: {e}")
        
        return df
    
    def _convert_to_model_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte los datos al formato requerido por el modelo de predicción.
        
        Args:
            data: DataFrame con datos originales
            
        Returns:
            DataFrame en formato estándar
        """
        try:
            # Aplicar mapeo de columnas
            mapped_data = self._apply_mapping(data, self.column_mapping)
            
            # Asegurarnos de que winner es siempre 0 (player_1 gana)
            # Esta convención es importante para el modelo
            if 'winner' not in mapped_data.columns:
                mapped_data['winner'] = 0
            
            # Normalizar superficie
            if 'surface' in mapped_data.columns:
                mapped_data['surface'] = mapped_data['surface'].str.lower()
                # Corregir valores comunes
                surface_map = {
                    'hard court': 'hard',
                    'clay court': 'clay',
                    'grass court': 'grass',
                    'carpet court': 'carpet',
                    'h': 'hard',
                    'c': 'clay',
                    'g': 'grass',
                    'indoor': 'hard',
                    'outdoor': 'hard',
                    'hardcourt': 'hard',
                    'claycourt': 'clay',
                    'grasscourt': 'grass'
                }
                mapped_data['surface'] = mapped_data['surface'].replace(surface_map)
            else:
                mapped_data['surface'] = 'unknown'
            
            # Aleatorizar la asignación de jugadores para evitar sesgo
            logging.info("Aleatorizando asignación de jugadores...")
            randomized_data = self._randomize_players(mapped_data)
            
            # Asegurarse de que todas las columnas importantes existan
            required_columns = ['player_1', 'player_2', 'winner', 'surface']
            for col in required_columns:
                if col not in randomized_data.columns:
                    if col == 'winner':
                        randomized_data[col] = randomized_data.apply(
                            lambda x: 0 if x['player_1'] == x.get('original_winner', x['player_1']) else 1, 
                            axis=1)
                    elif col == 'surface':
                        randomized_data[col] = 'hard'  # Valor por defecto
                    else:
                        raise ValueError(f"Columna requerida {col} no encontrada y no se puede derivar")
            
            # Limpiar datos y manejar valores nulos
            clean_data = self._clean_data(randomized_data)
            
            return clean_data
            
        except Exception as e:
            logging.error(f"Error convirtiendo al formato del modelo: {e}")
            traceback.print_exc()
            raise
    
    def _randomize_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aleatoriza la asignación de jugadores para evitar sesgo.
        
        Args:
            df: DataFrame con datos en formato parcialmente mapeado
            
        Returns:
            DataFrame con jugadores aleatorizados
        """
        import random
        randomized = []
        
        for _, row in df.iterrows():
            # Verificar columnas necesarias
            if 'player_1' not in row or 'player_2' not in row:
                continue
            
            # Con aproximadamente 50% de probabilidad, intercambiar jugadores
            if random.random() > 0.5:
                new_row = row.copy()
                
                # Guardar el ganador original para referencia
                new_row['original_winner'] = row['player_1']
                
                # El ganador ahora será el jugador_2
                new_row['winner'] = 1
            else:
                new_row = row.copy()
                
                # Intercambiar jugadores y sus estadísticas
                for field in df.columns:
                    if field.endswith('_1'):
                        base_field = field[:-2]
                        field2 = f"{base_field}_2"
                        if field2 in df.columns:
                            new_row[field], new_row[field2] = row[field2], row[field]
                
                # Intercambiar player_1 y player_2
                new_row['player_1'], new_row['player_2'] = row['player_2'], row['player_1']
                
                # Guardar el ganador original para referencia
                new_row['original_winner'] = row['player_2']
                
                # El ganador seguirá siendo el jugador_1
                new_row['winner'] = 0
            
            randomized.append(new_row)
        
        return pd.DataFrame(randomized)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia los datos y maneja valores nulos.
        
        Args:
            df: DataFrame con datos a limpiar
            
        Returns:
            DataFrame limpio
        """
        clean_df = df.copy()
        
        # Eliminar filas sin jugadores
        clean_df = clean_df.dropna(subset=['player_1', 'player_2'])
        
        # Manejar valores nulos en columnas numéricas
        numeric_columns = ['ranking_1', 'ranking_2', 'height_1', 'height_2', 
                          'age_1', 'age_2', 'duration']
        
        for col in numeric_columns:
            if col in clean_df.columns:
                # Convertir a numérico, forzando errores a NaN
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
                
                # Rellenar NaN con la mediana
                median_value = clean_df[col].median()
                if pd.isna(median_value):
                    # Si la mediana es NaN, usar valores por defecto específicos
                    if 'ranking' in col:
                        default_value = 100
                    elif 'height' in col:
                        default_value = 185
                    elif 'age' in col:
                        default_value = 25
                    elif col == 'duration':
                        default_value = 90
                    else:
                        default_value = 0
                    
                    clean_df[col] = clean_df[col].fillna(default_value)
                else:
                    clean_df[col] = clean_df[col].fillna(median_value)
        
        # Normalizar mano dominante (R=diestro, L=zurdo)
        for hand_col in ['hand_1', 'hand_2']:
            if hand_col in clean_df.columns:
                # Mapear diferentes valores a R/L
                hand_map = {
                    'r': 'R', 
                    'right': 'R',
                    'Right': 'R',
                    'RIGHT': 'R',
                    'l': 'L',
                    'left': 'L',
                    'Left': 'L',
                    'LEFT': 'L',
                    'u': 'U',  # Unknown
                    'unknown': 'U',
                    'Unknown': 'U',
                    '': 'U'
                }
                
                clean_df[hand_col] = clean_df[hand_col].map(hand_map).fillna('U')
        
        # Verificar que la columna winner es numérica
        clean_df['winner'] = clean_df['winner'].astype(int)
        
        # Eliminar columnas temporales de procesamiento
        if 'original_winner' in clean_df.columns:
            clean_df = clean_df.drop('original_winner', axis=1)
        
        return clean_df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina partidos duplicados en los datos.
        
        Args:
            df: DataFrame con posibles duplicados
            
        Returns:
            DataFrame sin duplicados
        """
        # Identificar columnas para detectar duplicados
        dup_cols = ['player_1', 'player_2', 'match_date', 'tournament', 'round']
        
        # Asegurarnos que todas estas columnas existen
        existing_cols = [col for col in dup_cols if col in df.columns]
        
        # Si hay suficientes columnas para identificar duplicados
        if len(existing_cols) >= 3:
            # Ordenar por fecha y calidad de fuente para quedarnos con la mejor versión
            # de cada partido duplicado
            sorted_df = df.sort_values(by=['match_date', 'source'], 
                                      ascending=[True, True])
            
            # Eliminar duplicados, manteniendo la primera aparición (mejor fuente)
            before_count = len(sorted_df)
            no_dups = sorted_df.drop_duplicates(subset=existing_cols, keep='first')
            after_count = len(no_dups)
            
            logging.info(f"Eliminados {before_count - after_count} partidos duplicados")
            
            return no_dups
        else:
            logging.warning("No hay suficientes columnas para detectar duplicados confiablemente")
            return df
    
    def _calculate_derived_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estadísticas derivadas a partir de los datos básicos.
        
        Args:
            df: DataFrame con datos básicos
            
        Returns:
            DataFrame con estadísticas derivadas añadidas
        """
        result_df = df.copy()
        
        try:
            # 1. Calcular estadísticas de servicio si hay datos suficientes
            service_cols = [
                'aces_1', 'aces_2', 'double_faults_1', 'double_faults_2',
                'service_points_1', 'service_points_2', 'first_serves_in_1', 'first_serves_in_2',
                'first_serve_points_won_1', 'first_serve_points_won_2',
                'second_serve_points_won_1', 'second_serve_points_won_2'
            ]
            
            # Verificar si tenemos suficientes columnas
            available_service_cols = [col for col in service_cols if col in result_df.columns]
            
            if len(available_service_cols) >= 6:
                logging.info("Calculando estadísticas de servicio derivadas...")
                
                # Porcentaje de primer servicio
                if all(col in result_df.columns for col in ['first_serves_in_1', 'service_points_1']):
                    result_df['first_serve_pct_1'] = (
                        result_df['first_serves_in_1'] / result_df['service_points_1'] * 100
                    ).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                if all(col in result_df.columns for col in ['first_serves_in_2', 'service_points_2']):
                    result_df['first_serve_pct_2'] = (
                        result_df['first_serves_in_2'] / result_df['service_points_2'] * 100
                    ).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Porcentaje de puntos ganados con primer servicio
                if all(col in result_df.columns for col in ['first_serve_points_won_1', 'first_serves_in_1']):
                    result_df['first_serve_won_pct_1'] = (
                        result_df['first_serve_points_won_1'] / result_df['first_serves_in_1'] * 100
                    ).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                if all(col in result_df.columns for col in ['first_serve_points_won_2', 'first_serves_in_2']):
                    result_df['first_serve_won_pct_2'] = (
                        result_df['first_serve_points_won_2'] / result_df['first_serves_in_2'] * 100
                    ).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Porcentaje de aces
                if all(col in result_df.columns for col in ['aces_1', 'service_points_1']):
                    result_df['ace_pct_1'] = (
                        result_df['aces_1'] / result_df['service_points_1'] * 100
                    ).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                if all(col in result_df.columns for col in ['aces_2', 'service_points_2']):
                    result_df['ace_pct_2'] = (
                        result_df['aces_2'] / result_df['service_points_2'] * 100
                    ).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Porcentaje de dobles faltas
                if all(col in result_df.columns for col in ['double_faults_1', 'service_points_1']):
                    result_df['double_fault_pct_1'] = (
                        result_df['double_faults_1'] / result_df['service_points_1'] * 100
                    ).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                if all(col in result_df.columns for col in ['double_faults_2', 'service_points_2']):
                    result_df['double_fault_pct_2'] = (
                        result_df['double_faults_2'] / result_df['service_points_2'] * 100
                    ).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # 2. Calcular estadísticas H2H históricas
            logging.info("Calculando estadísticas head-to-head...")
            
            # Crear una estructura para almacenar H2H
            h2h_stats = {}
            
            # Ordenar por fecha para procesar cronológicamente
            if 'match_date' in result_df.columns:
                sorted_df = result_df.sort_values('match_date')
            else:
                sorted_df = result_df
            
            # Para cada partido, actualizar estadísticas H2H
            for _, row in sorted_df.iterrows():
                player1 = row['player_1']
                player2 = row['player_2']
                winner = row['winner']
                
                # Crear clave H2H (orden alfabético para consistencia)
                players_key = tuple(sorted([player1, player2]))
                
                # Inicializar si no existe
                if players_key not in h2h_stats:
                    h2h_stats[players_key] = {
                        'matches': 0,
                        'player1_wins': 0,
                        'player2_wins': 0
                    }
                
                # Actualizar estadísticas
                h2h_stats[players_key]['matches'] += 1
                
                if winner == 0:  # player_1 ganó
                    if player1 == players_key[0]:
                        h2h_stats[players_key]['player1_wins'] += 1
                    else:
                        h2h_stats[players_key]['player2_wins'] += 1
                else:  # player_2 ganó
                    if player2 == players_key[0]:
                        h2h_stats[players_key]['player1_wins'] += 1
                    else:
                        h2h_stats[players_key]['player2_wins'] += 1
            
            # 3. Calcular winrates generales por jugador
            logging.info("Calculando winrates generales...")
            
            # Estructura para almacenar estadísticas por jugador
            player_stats = {}
            
            # Calcular victorias y derrotas para cada jugador
            for _, row in result_df.iterrows():
                player1 = row['player_1']
                player2 = row['player_2']
                winner = row['winner']
                
                # Inicializar jugadores si no existen
                for player in [player1, player2]:
                    if player not in player_stats:
                        player_stats[player] = {'wins': 0, 'losses': 0, 'matches': 0}
                
                # Actualizar estadísticas
                if winner == 0:  # player_1 ganó
                    player_stats[player1]['wins'] += 1
                    player_stats[player2]['losses'] += 1
                else:  # player_2 ganó
                    player_stats[player2]['wins'] += 1
                    player_stats[player1]['losses'] += 1
                
                player_stats[player1]['matches'] += 1
                player_stats[player2]['matches'] += 1
            
            # Calcular winrate
            for player, stats in player_stats.items():
                if stats['matches'] > 0:
                    stats['winrate'] = stats['wins'] / stats['matches'] * 100
                else:
                    stats['winrate'] = 50.0  # Valor por defecto para jugadores sin partidos
            
            # 4. Calcular winrates por superficie
            logging.info("Calculando winrates por superficie...")
            
            # Estructura para almacenar estadísticas por jugador y superficie
            surface_stats = {}
            
            # Calcular victorias y derrotas para cada jugador por superficie
            for _, row in result_df.iterrows():
                player1 = row['player_1']
                player2 = row['player_2']
                winner = row['winner']
                surface = row['surface']
                
                # Inicializar jugadores/superficies si no existen
                for player in [player1, player2]:
                    if player not in surface_stats:
                        surface_stats[player] = {}
                    
                    if surface not in surface_stats[player]:
                        surface_stats[player][surface] = {'wins': 0, 'losses': 0, 'matches': 0}
                
                # Actualizar estadísticas
                if winner == 0:  # player_1 ganó
                    surface_stats[player1][surface]['wins'] += 1
                    surface_stats[player2][surface]['losses'] += 1
                else:  # player_2 ganó
                    surface_stats[player2][surface]['wins'] += 1
                    surface_stats[player1][surface]['losses'] += 1
                
                surface_stats[player1][surface]['matches'] += 1
                surface_stats[player2][surface]['matches'] += 1
            
            # Calcular winrate por superficie
            for player, surfaces in surface_stats.items():
                for surface, stats in surfaces.items():
                    if stats['matches'] > 0:
                        stats['winrate'] = stats['wins'] / stats['matches'] * 100
                    else:
                        stats['winrate'] = 50.0  # Valor por defecto
            
            # 5. Añadir estadísticas a los datos
            logging.info("Añadiendo estadísticas calculadas a los datos...")
            
            # Añadir winrates generales
            result_df['winrate_1'] = result_df['player_1'].map(
                lambda p: player_stats.get(p, {}).get('winrate', 50.0))
            
            result_df['winrate_2'] = result_df['player_2'].map(
                lambda p: player_stats.get(p, {}).get('winrate', 50.0))
            
            # Añadir winrates por superficie
            result_df['surface_winrate_1'] = result_df.apply(
                lambda row: surface_stats.get(row['player_1'], {}).get(row['surface'], {}).get('winrate', 50.0),
                axis=1
            )
            
            result_df['surface_winrate_2'] = result_df.apply(
                lambda row: surface_stats.get(row['player_2'], {}).get(row['surface'], {}).get('winrate', 50.0),
                axis=1
            )
            
            # Añadir diferencia de winrates
            result_df['winrate_diff'] = result_df['winrate_1'] - result_df['winrate_2']
            result_df['surface_winrate_diff'] = result_df['surface_winrate_1'] - result_df['surface_winrate_2']
            
            # Añadir estadísticas H2H
            def get_h2h_data(row, stat):
                player1 = row['player_1']
                player2 = row['player_2']
                players_key = tuple(sorted([player1, player2]))
                
                if players_key in h2h_stats:
                    h2h = h2h_stats[players_key]
                    matches = h2h['matches']
                    
                    if player1 == players_key[0]:
                        p1_wins = h2h['player1_wins']
                        p2_wins = h2h['player2_wins']
                    else:
                        p1_wins = h2h['player2_wins']
                        p2_wins = h2h['player1_wins']
                    
                    if stat == 'p1_wins':
                        return p1_wins
                    elif stat == 'p2_wins':
                        return p2_wins
                    elif stat == 'matches':
                        return matches
                    elif stat == 'p1_winrate':
                        return (p1_wins / matches * 100) if matches > 0 else 50.0
                    elif stat == 'p2_winrate':
                        return (p2_wins / matches * 100) if matches > 0 else 50.0
                
                # Valores por defecto
                if stat in ['p1_wins', 'p2_wins', 'matches']:
                    return 0
                else:
                    return 50.0  # Winrate por defecto
            
            result_df['h2h_matches'] = result_df.apply(lambda row: get_h2h_data(row, 'matches'), axis=1)
            result_df['h2h_wins_1'] = result_df.apply(lambda row: get_h2h_data(row, 'p1_wins'), axis=1)
            result_df['h2h_wins_2'] = result_df.apply(lambda row: get_h2h_data(row, 'p2_wins'), axis=1)
            result_df['h2h_winrate_1'] = result_df.apply(lambda row: get_h2h_data(row, 'p1_winrate'), axis=1)
            result_df['h2h_winrate_2'] = result_df.apply(lambda row: get_h2h_data(row, 'p2_winrate'), axis=1)
            result_df['h2h_winrate_diff'] = result_df['h2h_winrate_1'] - result_df['h2h_winrate_2']
            
            logging.info("Estadísticas derivadas calculadas exitosamente")
            
        except Exception as e:
            logging.error(f"Error calculando estadísticas derivadas: {e}")
            traceback.print_exc()
        
        return result_df