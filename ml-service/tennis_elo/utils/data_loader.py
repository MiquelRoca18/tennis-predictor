"""
utils/data_loader.py

Funciones para cargar y procesar datos de tenis:
- Carga de archivos CSV de partidos
- Carga de datos de jugadores
- Extracción de estadísticas de partidos
- Carga de datos punto por punto

Proporciona funciones para acceder a datos desde diferentes fuentes y formatos.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from datetime import datetime

# Configurar logging
logger = logging.getLogger(__name__)

class TennisDataLoader:
    """
    Clase para cargar y procesar datos de tenis desde diferentes fuentes.
    """
    
    def __init__(self, data_dir: str = 'data/processed'):
        """
        Inicializa el cargador con el directorio de datos.
        
        Args:
            data_dir: Directorio base donde se encuentran los datos
        """
        self.data_dir = Path(data_dir)
        self.player_names = {}  # Mapeo de ID a nombre de jugador
        self.stats_cache = {}   # Caché para estadísticas de partidos
        self.pbp_cache = {}     # Caché para datos punto por punto
        
        # Tablas cargadas
        self.atp_players_df = pd.DataFrame()
        self.wta_players_df = pd.DataFrame()
        self.players_df = pd.DataFrame()
        self.match_stats_df = pd.DataFrame()
        self.mcp_matches_df = pd.DataFrame()  # Match Charting Project
        
        # Información sobre archivos punto por punto
        self.pbp_files = []
        self.pbp_indices = {}

    def load_player_data(self, tours: List[str] = ['atp', 'wta']) -> Dict[str, pd.DataFrame]:
        """
        Carga datos de jugadores de los tours especificados.
        
        Args:
            tours: Lista de tours a cargar ('atp' y/o 'wta')
            
        Returns:
            Diccionario con DataFrames cargados
        """
        results = {}
        
        for tour in tours:
            tour_lower = tour.lower()
        
            # Construir patrón de nombre de archivo según parámetros
            if match_type == 'all':
                patterns = [f"{tour_lower}_matches_{year}*.csv" for year in range(year_range[0], year_range[1] + 1)]
                patterns.append(f"{tour_lower}_matches_{year_range[0]}_{year_range[1]}.csv")
                patterns.append(f"{tour_lower}_matches_main_{year_range[0]}_{year_range[1]}.csv")
            else:
                patterns = [
                    f"{tour_lower}_matches_{match_type}_{year_range[0]}_{year_range[1]}.csv",
                    f"{tour_lower}_matches_{match_type}*.csv"
                ]
            
            logger.info(f"Buscando patrones de archivo para {tour_lower}: {patterns}")
            
            # Buscar archivos que coincidan con patrones
            for pattern in patterns:
                search_paths = [
                    self.data_dir / tour_lower / pattern,
                    self.data_dir / pattern
                ]
                
                for search_path in search_paths:
                    logger.info(f"Buscando en: {search_path}")
                    matched_files = list(self.data_dir.glob(str(search_path)))
                    logger.info(f"Archivos encontrados: {matched_files}")

            players_path = self.data_dir / tour_lower / f"{tour_lower}_players.csv"
            
            if players_path.exists():
                try:
                    # Cargar con manejo de errores para columnas con tipos mixtos
                    df = pd.read_csv(players_path, low_memory=False)
                    
                    # Guardar DataFrame en atributo
                    if tour_lower == 'atp':
                        self.atp_players_df = df
                        results['atp'] = df
                    elif tour_lower == 'wta':
                        self.wta_players_df = df
                        results['wta'] = df
                    
                    logger.info(f"Datos de jugadores {tour.upper()} cargados: {len(df)} jugadores")
                    
                    # Actualizar diccionario de nombres
                    self._update_player_names_from_df(df)
                except Exception as e:
                    logger.error(f"Error cargando jugadores {tour}: {str(e)}")
            else:
                logger.warning(f"Archivo de jugadores {tour} no encontrado en {players_path}")
        
        # Crear DataFrame combinado si se cargaron ambos tours
        if not self.atp_players_df.empty and not self.wta_players_df.empty:
            try:
                # Encontrar columnas comunes
                common_cols = list(set(self.atp_players_df.columns) & set(self.wta_players_df.columns))
                
                if common_cols:
                    self.players_df = pd.concat([
                        self.atp_players_df[common_cols],
                        self.wta_players_df[common_cols]
                    ])
                    results['combined'] = self.players_df
                    logger.info(f"DataFrame combinado creado: {len(self.players_df)} jugadores totales")
            except Exception as e:
                logger.error(f"Error creando DataFrame combinado: {str(e)}")
        elif not self.atp_players_df.empty:
            self.players_df = self.atp_players_df
            results['combined'] = self.players_df
        elif not self.wta_players_df.empty:
            self.players_df = self.wta_players_df
            results['combined'] = self.players_df
            
        return results
    
    def _update_player_names_from_df(self, df: pd.DataFrame) -> None:
        """
        Actualiza el diccionario de nombres de jugadores desde un DataFrame.
        
        Args:
            df: DataFrame con datos de jugadores
        """
        if df.empty or 'player_id' not in df.columns:
            return
            
        try:
            # Si tiene columnas de nombre y apellido separadas
            if 'name_first' in df.columns and 'name_last' in df.columns:
                for _, row in df.iterrows():
                    player_id = str(row['player_id'])
                    self.player_names[player_id] = f"{row['name_first']} {row['name_last']}"
            # Si tiene columna de nombre completo
            elif 'name' in df.columns:
                for _, row in df.iterrows():
                    player_id = str(row['player_id'])
                    self.player_names[player_id] = row['name']
            # Intentar otras posibles columnas
            elif 'full_name' in df.columns:
                for _, row in df.iterrows():
                    player_id = str(row['player_id'])
                    self.player_names[player_id] = row['full_name']
            else:
                logger.warning("No se encontraron columnas de nombres en el DataFrame")
        except Exception as e:
            logger.error(f"Error actualizando nombres de jugadores: {str(e)}")

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

    def load_matches_data(self, tours: List[str] = ['atp', 'wta'], 
                        year_range: Tuple[int, int] = (2000, 2025),
                        match_type: str = 'main') -> pd.DataFrame:
        """
        Carga datos de partidos para los tours especificados.
        
        Args:
            tours: Lista de tours a cargar ('atp' y/o 'wta')
            year_range: Rango de años (inicio, fin)
            match_type: Tipo de partidos ('main', 'qual', 'futures', 'all')
            
        Returns:
            DataFrame combinado con todos los partidos
        """
        all_matches = []
        
        for tour in tours:
            tour_lower = tour.lower()
            
            # Construir patrón de nombre de archivo según parámetros
            if match_type == 'all':
                patterns = [f"{tour_lower}_matches_{year}*.csv" for year in range(year_range[0], year_range[1] + 1)]
                patterns.append(f"{tour_lower}_matches_{year_range[0]}_{year_range[1]}.csv")
                patterns.append(f"{tour_lower}_matches_main_{year_range[0]}_{year_range[1]}.csv")
            else:
                patterns = [
                    f"{tour_lower}_matches_{match_type}_{year_range[0]}_{year_range[1]}.csv",
                    f"{tour_lower}_matches_{match_type}*.csv"
                ]
            
            # Buscar archivos que coincidan con patrones
            for pattern in patterns:
                search_paths = [
                    self.data_dir / tour_lower / pattern,
                    self.data_dir / pattern
                ]
                
                for search_path in search_paths:
                    matched_files = list(self.data_dir.glob(str(search_path)))
                    
                    for file_path in matched_files:
                        try:
                            logger.info(f"Cargando partidos desde {file_path}")
                            matches_df = pd.read_csv(file_path, low_memory=False)
                            
                            # Validar que es un DataFrame de partidos
                            if self._validate_matches_dataframe(matches_df):
                                # Asegurar que tiene columna de tour
                                if 'tour' not in matches_df.columns:
                                    matches_df['tour'] = tour_lower
                                
                                # Convertir fechas si es necesario
                                self._process_dates(matches_df)
                                
                                # Filtrar por rango de años si hay columna de fecha
                                if any(col in matches_df.columns for col in ['match_date', 'tourney_date', 'date']):
                                    date_col = next(col for col in ['match_date', 'tourney_date', 'date'] 
                                                  if col in matches_df.columns)
                                    
                                    # Asegurar que es datetime
                                    if not pd.api.types.is_datetime64_any_dtype(matches_df[date_col]):
                                        matches_df[date_col] = pd.to_datetime(matches_df[date_col], errors='coerce')
                                    
                                    # Filtrar por años
                                    year_mask = (matches_df[date_col].dt.year >= year_range[0]) & \
                                              (matches_df[date_col].dt.year <= year_range[1])
                                    matches_df = matches_df[year_mask]
                                
                                all_matches.append(matches_df)
                                logger.info(f"Cargados {len(matches_df)} partidos de {file_path.name}")
                            else:
                                logger.warning(f"Archivo {file_path} no contiene un DataFrame de partidos válido")
                        except Exception as e:
                            logger.error(f"Error cargando partidos desde {file_path}: {str(e)}")
        
        # Combinar todos los DataFrames
        if not all_matches:
            logger.warning("No se encontraron archivos de partidos que coincidan con los criterios")
            return pd.DataFrame()
        
        try:
            # Si hay varios DataFrames, encontrar columnas comunes
            if len(all_matches) > 1:
                # Encontrar columnas comunes entre todos los DataFrames
                common_cols = set(all_matches[0].columns)
                for df in all_matches[1:]:
                    common_cols &= set(df.columns)
                
                # Si hay columnas comunes, usar solo esas para combinar
                if common_cols:
                    combined_df = pd.concat([df[list(common_cols)] for df in all_matches], ignore_index=True)
                    logger.info(f"Combinados {len(combined_df)} partidos usando {len(common_cols)} columnas comunes")
                else:
                    # Si no hay columnas comunes, usar primer DataFrame
                    combined_df = all_matches[0]
                    logger.warning("No hay columnas comunes entre DataFrames de partidos")
            else:
                combined_df = all_matches[0]
                
            # Eliminar duplicados si los hay
            if 'match_id' in combined_df.columns:
                before_count = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=['match_id'])
                after_count = len(combined_df)
                
                if before_count > after_count:
                    logger.info(f"Eliminados {before_count - after_count} partidos duplicados")
            
            return combined_df
        except Exception as e:
            logger.error(f"Error combinando DataFrames de partidos: {str(e)}")
            # Devolver el primer DataFrame si hay error en la combinación
            return all_matches[0] if all_matches else pd.DataFrame()
    
    def _validate_matches_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Valida que un DataFrame contenga datos de partidos de tenis.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            True si es un DataFrame de partidos válido, False en caso contrario
        """
        if df.empty:
            return False
            
        # Columnas mínimas requeridas (al menos 2 de 3)
        required_columns_sets = [
            ['winner_id', 'loser_id'],
            ['winner_name', 'loser_name'],
            ['player1_id', 'player2_id']
        ]
        
        # Verificar si al menos un conjunto de columnas requeridas está presente
        valid_structure = False
        for col_set in required_columns_sets:
            if all(col in df.columns for col in col_set):
                valid_structure = True
                break
        
        if not valid_structure:
            return False
        
        # Verificar que hay suficientes filas
        if len(df) < 10:
            return False
            
        # Intentar verificar que tiene al menos una columna de fecha
        date_columns = ['match_date', 'tourney_date', 'date']
        has_date_col = any(col in df.columns for col in date_columns)
        
        # No es absolutamente necesario tener fecha, pero es preferible
        if not has_date_col:
            logger.warning("DataFrame de partidos no tiene columna de fecha")
            
        return True
    
    def _process_dates(self, df: pd.DataFrame) -> None:
        """
        Procesa y estandariza columnas de fecha en un DataFrame.
        
        Args:
            df: DataFrame a procesar
        """
        # Columnas de fecha comunes
        date_columns = ['match_date', 'tourney_date', 'date']
        
        for col in date_columns:
            if col in df.columns:
                try:
                    # Convertir a string primero para manejar tipos mixtos
                    df[col] = df[col].astype(str)
                    
                    # Intentar diferentes formatos comunes
                    # Formato yyyymmdd típico de datos ATP/WTA
                    if df[col].str.len().max() == 8 and df[col].str.isdigit().all():
                        df[col] = pd.to_datetime(df[col], format='%Y%m%d')
                    else:
                        # Intentar parseo general
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    # Si se creó la fecha correctamente, intentar crear match_date si no existe
                    if col != 'match_date' and 'match_date' not in df.columns:
                        df['match_date'] = df[col]
                        logger.debug(f"Creada columna 'match_date' basada en '{col}'")
                except Exception as e:
                    logger.warning(f"Error procesando columna de fecha '{col}': {str(e)}")

    def load_match_statistics(self, match_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Carga estadísticas detalladas de partidos.
        
        Args:
            match_ids: Lista opcional de IDs de partidos específicos a cargar
            
        Returns:
            DataFrame con estadísticas de partidos
        """
        if not self.match_stats_df.empty and match_ids is None:
            # Ya tenemos las estadísticas cargadas
            return self.match_stats_df
        
        # Buscar archivos de estadísticas
        stats_paths = [
            self.data_dir / 'atp' / 'match_stats.csv',
            self.data_dir / 'wta' / 'match_stats.csv',
            self.data_dir / 'match_stats.csv',
            self.data_dir / 'match_stats' / 'atp_match_stats.csv',
            self.data_dir / 'match_stats' / 'wta_match_stats.csv'
        ]
        
        all_stats_dfs = []
        
        for stats_path in stats_paths:
            if stats_path.exists():
                try:
                    stats_df = pd.read_csv(stats_path, low_memory=False)
                    
                    # Si hay IDs específicos, filtrar
                    if match_ids:
                        if 'match_id' in stats_df.columns:
                            # Convertir a string para comparación
                            stats_df['match_id'] = stats_df['match_id'].astype(str)
                            match_ids = [str(mid) for mid in match_ids]
                            
                            # Filtrar por match_ids
                            stats_df = stats_df[stats_df['match_id'].isin(match_ids)]
                    
                    all_stats_dfs.append(stats_df)
                    logger.info(f"Estadísticas cargadas desde {stats_path}: {len(stats_df)} registros")
                except Exception as e:
                    logger.warning(f"Error cargando estadísticas desde {stats_path}: {str(e)}")
        
        # Si no encontramos archivos específicos, intentar extraer de los archivos principales
        if not all_stats_dfs:
            # Buscar en archivos de partidos principales
            stats_columns = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
                          'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon']
            
            matches_paths = [
                self.data_dir / 'atp' / 'atp_matches_main_2000_2024.csv',
                self.data_dir / 'wta' / 'wta_matches_main_2000_2024.csv'
            ]
            
            for path in matches_paths:
                if path.exists():
                    try:
                        matches_df = pd.read_csv(path, low_memory=False)
                        
                        # Verificar si tiene columnas de estadísticas
                        has_stats = any(col in matches_df.columns for col in stats_columns)
                        
                        if has_stats:
                            # Si hay IDs específicos, filtrar
                            if match_ids and 'match_id' in matches_df.columns:
                                matches_df['match_id'] = matches_df['match_id'].astype(str)
                                match_ids = [str(mid) for mid in match_ids]
                                matches_df = matches_df[matches_df['match_id'].isin(match_ids)]
                            
                            all_stats_dfs.append(matches_df)
                            logger.info(f"Estadísticas extraídas de {path}: {len(matches_df)} registros")
                    except Exception as e:
                        logger.warning(f"Error extrayendo estadísticas de {path}: {str(e)}")
        
        # Combinar DataFrames
        if all_stats_dfs:
            try:
                # Si hay varios DataFrames, combinar
                if len(all_stats_dfs) > 1:
                    # Encontrar columnas comunes
                    common_cols = set(all_stats_dfs[0].columns)
                    for df in all_stats_dfs[1:]:
                        common_cols &= set(df.columns)
                    
                    # Si hay columnas comunes
                    if common_cols:
                        combined_stats = pd.concat([df[list(common_cols)] for df in all_stats_dfs])
                        logger.info(f"Combinadas estadísticas: {len(combined_stats)} registros")
                    else:
                        # Si no hay columnas comunes, solo usar el primer DataFrame
                        combined_stats = all_stats_dfs[0]
                        logger.warning("No hay columnas comunes entre DataFrames de estadísticas")
                else:
                    combined_stats = all_stats_dfs[0]
                
                # Eliminar duplicados
                if 'match_id' in combined_stats.columns:
                    before_count = len(combined_stats)
                    combined_stats = combined_stats.drop_duplicates(subset=['match_id'])
                    after_count = len(combined_stats)
                    
                    if before_count > after_count:
                        logger.info(f"Eliminados {before_count - after_count} registros duplicados")
                
                # Guardar en atributo si se cargó todo
                if match_ids is None:
                    self.match_stats_df = combined_stats
                
                return combined_stats
            except Exception as e:
                logger.error(f"Error combinando estadísticas: {str(e)}")
                
                # Devolver el primer DataFrame en caso de error
                return all_stats_dfs[0] if all_stats_dfs else pd.DataFrame()
        
        # Si no encontramos estadísticas, devolver DataFrame vacío
        logger.warning("No se encontraron estadísticas de partidos")
        return pd.DataFrame()

    def get_match_statistics(self, match_id: str) -> Dict:
        """
        Obtiene estadísticas detalladas de un partido específico.
        
        Args:
            match_id: ID del partido
            
        Returns:
            Diccionario con estadísticas del partido
        """
        # Verificar si ya está en caché
        if match_id in self.stats_cache:
            return self.stats_cache[match_id]
        
        # Si tenemos el DataFrame de estadísticas cargado
        if not self.match_stats_df.empty and 'match_id' in self.match_stats_df.columns:
            # Convertir a string para la comparación
            match_id_str = str(match_id)
            
            # Buscar el partido
            match_stats = self.match_stats_df[self.match_stats_df['match_id'].astype(str) == match_id_str]
            
            if not match_stats.empty:
                # Extraer estadísticas relevantes
                stats = {}
                row = match_stats.iloc[0].to_dict()
                
                # Extraer estadísticas del ganador
                winner_stats = {}
                loser_stats = {}
                
                # Estadísticas comunes
                stat_cols = [
                    'ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 
                    'SvGms', 'bpSaved', 'bpFaced'
                ]
                
                for col in stat_cols:
                    w_col = f'w_{col}'
                    l_col = f'l_{col}'
                    
                    if w_col in row:
                        winner_stats[col] = row[w_col]
                    if l_col in row:
                        loser_stats[col] = row[l_col]
                
                # Construir resultado
                stats['winner'] = winner_stats
                stats['loser'] = loser_stats
                stats['match_id'] = match_id
                
                # Guardar en caché
                self.stats_cache[match_id] = stats
                
                return stats
        
        # Si no está en el DataFrame cargado, cargar específicamente
        stats_df = self.load_match_statistics([match_id])
        
        if not stats_df.empty:
            # Volver a intentar con los datos recién cargados
            return self.get_match_statistics(match_id)
        
        # Si no se encuentra, devolver diccionario vacío
        return {'winner': {}, 'loser': {}, 'match_id': match_id}

    def find_point_by_point_files(self) -> List[Path]:
        """
        Busca archivos de datos punto por punto.
        
        Returns:
            Lista de rutas de archivos encontrados
        """
        if self.pbp_files:
            return self.pbp_files
            
        # Buscar datos punto por punto en varias ubicaciones posibles
        pbp_paths = [
            self.data_dir / 'pointbypoint',
            self.data_dir / 'slam_pointbypoint'
        ]
        
        files = []
        
        for pbp_path in pbp_paths:
            if pbp_path.exists() and pbp_path.is_dir():
                # Buscar archivos CSV en este directorio
                csv_files = list(pbp_path.glob("*.csv"))
                
                # Filtrar archivos relevantes de puntos
                pbp_files = [f for f in csv_files if 'point' in f.name.lower()]
                
                # Añadir al listado
                files.extend(pbp_files)
                
                logger.info(f"Encontrados {len(pbp_files)} archivos punto por punto en {pbp_path}")
        
        # Cargar índices si existen para localizar datos más rápido
        for pbp_path in pbp_paths:
            index_file = pbp_path / 'index.json'
            if index_file.exists():
                try:
                    with open(index_file, 'r') as f:
                        pbp_index = json.load(f)
                        
                        if not self.pbp_indices:
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
        
        self.pbp_files = files
        return files

    def load_point_by_point_data(self, match_id: Optional[str] = None, 
                             player_id: Optional[str] = None,
                             tournament: Optional[str] = None) -> pd.DataFrame:
        """
        Carga datos punto por punto para partidos específicos.
        
        Args:
            match_id: ID del partido (opcional)
            player_id: ID del jugador (opcional)
            tournament: Nombre del torneo (opcional)
            
        Returns:
            DataFrame con datos punto por punto
        """
        # Si está en caché
        cache_key = f"{match_id}_{player_id}_{tournament}"
        if cache_key in self.pbp_cache:
            return self.pbp_cache[cache_key]
        
        # Asegurar que tenemos archivos punto por punto
        if not self.pbp_files:
            self.find_point_by_point_files()
            
        if not self.pbp_files:
            logger.warning("No se encontraron archivos punto por punto")
            return pd.DataFrame()
        
        # Si tenemos índices y match_id
        if match_id and self.pbp_indices:
            match_id_str = str(match_id)
            
            if match_id_str in self.pbp_indices:
                file_info = self.pbp_indices[match_id_str]
                
                if isinstance(file_info, dict) and 'file' in file_info:
                    # Construir ruta basada en información del índice
                    file_path = None
                    
                    if isinstance(file_info['file'], str):
                        # Buscar en varias ubicaciones posibles
                        potential_paths = [
                            self.data_dir / file_info['file'],
                            self.data_dir / 'pointbypoint' / file_info['file'],
                            self.data_dir / 'slam_pointbypoint' / file_info['file']
                        ]
                        
                        for path in potential_paths:
                            if path.exists():
                                file_path = path
                                break
                    
                    if file_path:
                        try:
                            # Cargar archivo
                            df = pd.read_csv(file_path)
                            
                            # Filtrar por match_id si hay una columna para ello
                            if 'match_id' in df.columns:
                                df = df[df['match_id'].astype(str) == match_id_str]
                            
                            # Guardar en caché
                            self.pbp_cache[cache_key] = df
                            
                            return df
                        except Exception as e:
                            logger.error(f"Error cargando datos punto por punto desde {file_path}: {str(e)}")
        
        # Si no tenemos índices o el match_id no está en ellos, buscar en todos los archivos
        point_data = []
        
        # Limitar la búsqueda si hay demasiados archivos
        search_files = self.pbp_files[:30] if len(self.pbp_files) > 30 else self.pbp_files
        
        for file_path in search_files:
            try:
                # Verificar si el nombre del archivo podría contener la información buscada
                file_relevant = True
                
                if match_id and str(match_id) not in file_path.stem:
                    # Si el archivo está organizado por torneo y no por partido, aún podría ser relevante
                    if not tournament or tournament.lower() not in file_path.stem.lower():
                        file_relevant = False
                
                if player_id and ('player_id' in df.columns or 'winner_id' in df.columns or 'loser_id' in df.columns):
                        player_id_str = str(player_id)
                        player_filter = False
                        
                        if 'player_id' in df.columns:
                            player_filter |= (df['player_id'].astype(str) == player_id_str)
                        if 'winner_id' in df.columns:
                            player_filter |= (df['winner_id'].astype(str) == player_id_str)
                        if 'loser_id' in df.columns:
                            player_filter |= (df['loser_id'].astype(str) == player_id_str)
                            
                        df = df[player_filter]
                    
                if tournament and 'tournament' in df.columns:
                        df = df[df['tournament'].str.lower() == tournament.lower()]
                elif tournament and 'tourney_name' in df.columns:
                        df = df[df['tourney_name'].str.lower() == tournament.lower()]
                    
                # Si queda algo después de filtrar, agregar a resultados
                if not df.empty:
                    point_data.append(df)
                    logger.info(f"Encontrados {len(df)} puntos relevantes en {file_path.name}")
                        
                    # Si encontramos exactamente lo que buscamos (match_id), terminar búsqueda
                    if match_id and 'match_id' in df.columns and len(df[df['match_id'].astype(str) == str(match_id)]) > 0:
                        break
            except Exception as e:
                logger.warning(f"Error procesando archivo punto por punto {file_path}: {str(e)}")
        
        # Combinar resultados
        if point_data:
            try:
                # Si hay varios DataFrames, combinar
                if len(point_data) > 1:
                    # Verificar si tienen columnas compatibles
                    common_cols = set(point_data[0].columns)
                    for df in point_data[1:]:
                        common_cols &= set(df.columns)
                    
                    if common_cols:
                        result_df = pd.concat([df[list(common_cols)] for df in point_data], ignore_index=True)
                    else:
                        # Si no hay columnas comunes, usar solo el primero
                        result_df = point_data[0]
                        logger.warning("No hay columnas comunes entre datos punto por punto")
                else:
                    result_df = point_data[0]
                
                # Guardar en caché
                self.pbp_cache[cache_key] = result_df
                
                return result_df
            except Exception as e:
                logger.error(f"Error combinando datos punto por punto: {str(e)}")
                return point_data[0] if point_data else pd.DataFrame()
        
        logger.warning(f"No se encontraron datos punto por punto para los criterios especificados")
        return pd.DataFrame()

    def load_match_charting_data(self) -> pd.DataFrame:
        """
        Carga datos del Match Charting Project.
        
        Returns:
            DataFrame con datos del Match Charting Project
        """
        if not self.mcp_matches_df.empty:
            return self.mcp_matches_df
        
        # Posibles ubicaciones
        mcp_paths = [
            self.data_dir / 'match_charting' / 'all_matches_combined.csv',
            self.data_dir / 'match_charting' / 'matches.csv',
            self.data_dir / 'match_charting.csv'
        ]
        
        for path in mcp_paths:
            if path.exists():
                try:
                    mcp_df = pd.read_csv(path, low_memory=False)
                    self.mcp_matches_df = mcp_df
                    logger.info(f"Datos de Match Charting Project cargados: {len(mcp_df)} partidos")
                    return mcp_df
                except Exception as e:
                    logger.warning(f"Error cargando datos de Match Charting desde {path}: {str(e)}")
        
        logger.warning("No se encontraron datos del Match Charting Project")
        return pd.DataFrame()
    
    def create_combined_dataset(self, tours: List[str] = ['atp', 'wta'], 
                           year_range: Tuple[int, int] = (2000, 2025),
                           include_stats: bool = True) -> pd.DataFrame:
        """
        Crea un dataset combinado con partidos y estadísticas.
        
        Args:
            tours: Lista de tours a incluir
            year_range: Rango de años a incluir
            include_stats: Si debe incluir estadísticas detalladas
            
        Returns:
            DataFrame combinado con partidos y estadísticas
        """
        # Cargar partidos
        matches_df = self.load_matches_data(tours, year_range)
        
        if matches_df.empty:
            logger.warning("No se encontraron datos de partidos para crear dataset combinado")
            return pd.DataFrame()
        
        # Normalizar columnas de fecha
        self._process_dates(matches_df)
        
        # Si no se requieren estadísticas, devolver solo los partidos
        if not include_stats:
            return matches_df
        
        # Cargar estadísticas si se requieren
        stats_df = None
        
        if 'match_id' in matches_df.columns:
            # Cargar estadísticas para estos partidos específicos
            match_ids = matches_df['match_id'].astype(str).unique().tolist()
            stats_df = self.load_match_statistics(match_ids)
        else:
            # Cargar todas las estadísticas disponibles
            stats_df = self.load_match_statistics()
        
        # Si no hay estadísticas, devolver solo los partidos
        if stats_df.empty:
            logger.warning("No se encontraron estadísticas para combinar con los partidos")
            return matches_df
        
        # Combinar partidos con estadísticas
        try:
            # Verificar si tenemos match_id para hacer join
            if 'match_id' in matches_df.columns and 'match_id' in stats_df.columns:
                # Convertir a string para asegurar compatibilidad
                matches_df['match_id'] = matches_df['match_id'].astype(str)
                stats_df['match_id'] = stats_df['match_id'].astype(str)
                
                # Hacer merge
                combined_df = pd.merge(matches_df, stats_df, on='match_id', how='left')
                logger.info(f"Dataset combinado creado: {len(combined_df)} partidos con estadísticas")
                
                return combined_df
            else:
                # Si no tenemos match_id, intentar combinar por jugadores y fecha
                logger.warning("No hay columna match_id común, intentando combinar por jugadores y fecha")
                
                # Verificar si tenemos las columnas necesarias
                player_cols = ['winner_id', 'loser_id']
                date_col = 'match_date' if 'match_date' in matches_df.columns else None
                
                if all(col in matches_df.columns for col in player_cols) and all(col in stats_df.columns for col in player_cols) and date_col:
                    # Convertir IDs a string
                    for col in player_cols:
                        matches_df[col] = matches_df[col].astype(str)
                        stats_df[col] = stats_df[col].astype(str)
                    
                    # Asegurar que fecha es datetime
                    if date_col in stats_df.columns:
                        matches_df[date_col] = pd.to_datetime(matches_df[date_col])
                        stats_df[date_col] = pd.to_datetime(stats_df[date_col])
                        
                        # Hacer merge por jugadores y fecha
                        combined_df = pd.merge(matches_df, stats_df, on=player_cols + [date_col], how='left')
                    else:
                        # Hacer merge solo por jugadores
                        combined_df = pd.merge(matches_df, stats_df, on=player_cols, how='left')
                    
                    logger.info(f"Dataset combinado creado: {len(combined_df)} partidos")
                    return combined_df
                else:
                    logger.warning("No se pueden combinar partidos y estadísticas, faltan columnas necesarias")
                    return matches_df
        except Exception as e:
            logger.error(f"Error combinando partidos y estadísticas: {str(e)}")
            return matches_df