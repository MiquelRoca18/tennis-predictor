"""
io/persistence.py

Módulo para la persistencia de datos del sistema ELO de tenis.
Proporciona funciones para guardar y cargar ratings, historiales y
configuraciones del sistema.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from pathlib import Path

# Configurar logging
logger = logging.getLogger(__name__)

class EloDataManager:
    """
    Clase para gestionar la persistencia de datos del sistema ELO de tenis.
    Proporciona métodos para guardar y cargar datos en diferentes formatos,
    manejar la serialización y validar la integridad de los datos.
    """
    
    def __init__(self, data_dir: str = 'data/processed/elo'):
        """
        Inicializa el gestor de datos ELO.
        
        Args:
            data_dir: Directorio base para guardar/cargar datos
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Definir subdirectorios para diferentes tipos de datos
        self.ratings_dir = self.data_dir / 'ratings'
        self.history_dir = self.data_dir / 'history'
        self.models_dir = self.data_dir / 'models'
        
        # Crear subdirectorios
        self.ratings_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def save_ratings(self, 
                    player_ratings: Dict[str, float],
                    player_ratings_by_surface: Dict[str, Dict[str, float]],
                    player_match_count: Dict[str, int],
                    player_match_count_by_surface: Dict[str, Dict[str, int]],
                    player_uncertainty: Optional[Dict[str, float]] = None,
                    player_form: Optional[Dict[str, Any]] = None,
                    h2h_records: Optional[Dict] = None,
                    output_dir: Optional[str] = None,
                    timestamp: Optional[str] = None,
                    tournament_level: str = 'all') -> Dict[str, str]:
        """
        Guarda los ratings ELO de jugadores y datos relacionados.
        
        Args:
            player_ratings: Diccionario de ratings ELO generales
            player_ratings_by_surface: Diccionario de ratings por superficie
            player_match_count: Contadores de partidos por jugador
            player_match_count_by_surface: Contadores por jugador y superficie
            player_uncertainty: Incertidumbre del rating por jugador (opcional)
            player_form: Forma reciente de jugadores (opcional)
            h2h_records: Historial head-to-head entre jugadores (opcional)
            output_dir: Directorio de salida personalizado (opcional)
            timestamp: Marca de tiempo para los archivos (opcional)
            tournament_level: Nivel de torneo usado para generar estos ratings (por defecto 'all')
            
        Returns:
            Diccionario con rutas de los archivos guardados
        """
        # Determinar directorio de salida
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.ratings_dir
        
        # Crear timestamp si no se proporciona
        if not timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Rutas de archivos a guardar
        file_paths = {}
        
        # Incluir nivel de torneo en los nombres de archivo
        level_suffix = f"_{tournament_level}" if tournament_level != 'all' else ""
        
        # 1. Guardar ratings generales
        general_path = output_path / f'elo_ratings_general{level_suffix}_{timestamp}.json'
        try:
            with open(general_path, 'w', encoding='utf-8') as f:
                json.dump(player_ratings, f, indent=2, default=self._json_serializer)
            file_paths['general_ratings'] = str(general_path)
            logger.info(f"Ratings generales guardados en: {general_path}")
        except Exception as e:
            logger.error(f"Error guardando ratings generales: {str(e)}")
        
        # 2. Guardar ratings por superficie
        surface_path = output_path / f'elo_ratings_by_surface{level_suffix}_{timestamp}.json'
        try:
            with open(surface_path, 'w', encoding='utf-8') as f:
                json.dump(player_ratings_by_surface, f, indent=2, default=self._json_serializer)
            file_paths['surface_ratings'] = str(surface_path)
            logger.info(f"Ratings por superficie guardados en: {surface_path}")
        except Exception as e:
            logger.error(f"Error guardando ratings por superficie: {str(e)}")
        
        # 3. Guardar contadores de partidos
        counts_path = output_path / f'player_match_counts{level_suffix}_{timestamp}.json'
        try:
            with open(counts_path, 'w', encoding='utf-8') as f:
                json.dump(player_match_count, f, indent=2, default=self._json_serializer)
            file_paths['match_counts'] = str(counts_path)
            logger.info(f"Contadores de partidos guardados en: {counts_path}")
        except Exception as e:
            logger.error(f"Error guardando contadores de partidos: {str(e)}")
        
        # 4. Guardar contadores por superficie
        counts_surface_path = output_path / f'player_match_counts_by_surface{level_suffix}_{timestamp}.json'
        try:
            with open(counts_surface_path, 'w', encoding='utf-8') as f:
                json.dump(player_match_count_by_surface, f, indent=2, default=self._json_serializer)
            file_paths['surface_match_counts'] = str(counts_surface_path)
            logger.info(f"Contadores por superficie guardados en: {counts_surface_path}")
        except Exception as e:
            logger.error(f"Error guardando contadores por superficie: {str(e)}")
        
        # 5. Guardar incertidumbres si están disponibles
        if player_uncertainty:
            uncertainty_path = output_path / f'player_rating_uncertainty{level_suffix}_{timestamp}.json'
            try:
                with open(uncertainty_path, 'w', encoding='utf-8') as f:
                    json.dump(player_uncertainty, f, indent=2, default=self._json_serializer)
                file_paths['uncertainty'] = str(uncertainty_path)
                logger.info(f"Incertidumbres guardadas en: {uncertainty_path}")
            except Exception as e:
                logger.error(f"Error guardando incertidumbres: {str(e)}")
        
        # 6. Guardar forma reciente si está disponible
        if player_form:
            form_path = output_path / f'player_form{level_suffix}_{timestamp}.json'
            try:
                with open(form_path, 'w', encoding='utf-8') as f:
                    json.dump(player_form, f, indent=2, default=self._json_serializer)
                file_paths['form'] = str(form_path)
                logger.info(f"Forma reciente guardada en: {form_path}")
            except Exception as e:
                logger.error(f"Error guardando forma reciente: {str(e)}")
        
        # 7. Guardar historial head-to-head si está disponible
        if h2h_records:
            h2h_path = output_path / f'head_to_head_records{level_suffix}_{timestamp}.json'
            try:
                # Convertir defaultdict anidado a dict regular para JSON
                h2h_dict = {}
                for player_id, opponents in h2h_records.items():
                    if hasattr(opponents, 'items'):  # Asegurar que es un diccionario
                        h2h_dict[player_id] = {}
                        for opp_id, record in opponents.items():
                            if hasattr(record, 'items'):  # Asegurar que es un diccionario
                                h2h_dict[player_id][opp_id] = dict(record)
                
                with open(h2h_path, 'w', encoding='utf-8') as f:
                    json.dump(h2h_dict, f, indent=2, default=self._json_serializer)
                file_paths['h2h_records'] = str(h2h_path)
                logger.info(f"Historial head-to-head guardado en: {h2h_path}")
            except Exception as e:
                logger.error(f"Error guardando historial head-to-head: {str(e)}")
        
        # 8. Guardar archivo de índice con referencias a todos los archivos
        index_path = output_path / f'ratings_index{level_suffix}_{timestamp}.json'
        try:
            index_data = {
                'timestamp': timestamp,
                'tournament_level': tournament_level,
                'files': file_paths,
                'total_players': len(player_ratings),
                'total_matches': sum(player_match_count.values()),
                'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
            
            # También guardar un archivo de índice "latest" para referencia rápida
            # Incluir nivel de torneo en el nombre del archivo latest
            latest_index_path = output_path / f'ratings_latest_index{level_suffix}.json'
            with open(latest_index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
            
            logger.info(f"Índice de archivos guardado en: {index_path}")
        except Exception as e:
            logger.error(f"Error guardando índice de archivos: {str(e)}")
        
        return file_paths
    
    def load_ratings(self, 
                   input_dir: Optional[str] = None,
                   timestamp: Optional[str] = None,
                   use_latest: bool = True) -> Dict[str, Any]:
        """
        Carga ratings ELO y datos relacionados.
        
        Args:
            input_dir: Directorio de entrada personalizado (opcional)
            timestamp: Marca de tiempo específica para cargar (opcional)
            use_latest: Si debe usar los datos más recientes cuando no se especifica timestamp
            
        Returns:
            Diccionario con todos los datos cargados
        """
        # Determinar directorio de entrada
        if input_dir:
            input_path = Path(input_dir)
        else:
            input_path = self.ratings_dir
        
        # Verificar que el directorio existe
        if not input_path.exists():
            logger.error(f"Directorio de entrada no existe: {input_path}")
            return {}
        
        # Inicializar diccionario para datos cargados
        loaded_data = {
            'player_ratings': {},
            'player_ratings_by_surface': {
                'hard': {},
                'clay': {},
                'grass': {},
                'carpet': {}
            },
            'player_match_count': {},
            'player_match_count_by_surface': {
                'hard': {},
                'clay': {},
                'grass': {},
                'carpet': {}
            },
            'player_uncertainty': {},
            'player_form': {},
            'h2h_records': {},
            'loaded_files': []
        }
        
        # Determinar qué archivos cargar
        if timestamp:
            # Buscar archivos con el timestamp específico
            index_path = input_path / f'ratings_index_{timestamp}.json'
            if not index_path.exists():
                logger.error(f"No se encontró el índice para el timestamp {timestamp}")
                return loaded_data
            
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                # Extraer rutas de archivos del índice
                file_paths = index_data.get('files', {})
            except Exception as e:
                logger.error(f"Error cargando índice: {str(e)}")
                return loaded_data
        elif use_latest:
            # Buscar el archivo de índice más reciente
            latest_index_path = input_path / 'ratings_latest_index.json'
            if latest_index_path.exists():
                try:
                    with open(latest_index_path, 'r', encoding='utf-8') as f:
                        index_data = json.load(f)
                    
                    # Extraer rutas de archivos del índice
                    file_paths = index_data.get('files', {})
                except Exception as e:
                    logger.error(f"Error cargando índice más reciente: {str(e)}")
                    
                    # Intento alternativo: buscar archivos manualmente
                    file_paths = self._find_latest_rating_files(input_path)
            else:
                # Buscar archivos manualmente
                file_paths = self._find_latest_rating_files(input_path)
        else:
            # Buscar archivos manualmente sin preferencia por los más recientes
            file_paths = {}
            
            general_files = list(input_path.glob('elo_ratings_general_*.json'))
            if general_files:
                file_paths['general_ratings'] = str(general_files[0])
            
            surface_files = list(input_path.glob('elo_ratings_by_surface_*.json'))
            if surface_files:
                file_paths['surface_ratings'] = str(surface_files[0])
            
            count_files = list(input_path.glob('player_match_counts_*.json'))
            if count_files:
                file_paths['match_counts'] = str(count_files[0])
            
            surface_count_files = list(input_path.glob('player_match_counts_by_surface_*.json'))
            if surface_count_files:
                file_paths['surface_match_counts'] = str(surface_count_files[0])
        
        # Cargar cada tipo de archivo
        # 1. Ratings generales
        if 'general_ratings' in file_paths:
            try:
                file_path = file_paths['general_ratings']
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_data['player_ratings'] = json.load(f)
                    
                    loaded_data['loaded_files'].append(file_path)
                    logger.info(f"Ratings generales cargados desde {file_path}")
            except Exception as e:
                logger.error(f"Error cargando ratings generales: {str(e)}")
        
        # 2. Ratings por superficie
        if 'surface_ratings' in file_paths:
            try:
                file_path = file_paths['surface_ratings']
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_data['player_ratings_by_surface'] = json.load(f)
                    
                    loaded_data['loaded_files'].append(file_path)
                    logger.info(f"Ratings por superficie cargados desde {file_path}")
            except Exception as e:
                logger.error(f"Error cargando ratings por superficie: {str(e)}")
        
        # 3. Contadores de partidos
        if 'match_counts' in file_paths:
            try:
                file_path = file_paths['match_counts']
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_data['player_match_count'] = json.load(f)
                    
                    loaded_data['loaded_files'].append(file_path)
                    logger.info(f"Contadores de partidos cargados desde {file_path}")
            except Exception as e:
                logger.error(f"Error cargando contadores de partidos: {str(e)}")
        
        # 4. Contadores por superficie
        if 'surface_match_counts' in file_paths:
            try:
                file_path = file_paths['surface_match_counts']
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_data['player_match_count_by_surface'] = json.load(f)
                    
                    loaded_data['loaded_files'].append(file_path)
                    logger.info(f"Contadores por superficie cargados desde {file_path}")
            except Exception as e:
                logger.error(f"Error cargando contadores por superficie: {str(e)}")
        
        # 5. Incertidumbres
        if 'uncertainty' in file_paths:
            try:
                file_path = file_paths['uncertainty']
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_data['player_uncertainty'] = json.load(f)
                    
                    loaded_data['loaded_files'].append(file_path)
                    logger.info(f"Incertidumbres cargadas desde {file_path}")
            except Exception as e:
                logger.error(f"Error cargando incertidumbres: {str(e)}")
        
        # 6. Forma reciente
        if 'form' in file_paths:
            try:
                file_path = file_paths['form']
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_data['player_form'] = json.load(f)
                    
                    loaded_data['loaded_files'].append(file_path)
                    logger.info(f"Forma reciente cargada desde {file_path}")
            except Exception as e:
                logger.error(f"Error cargando forma reciente: {str(e)}")
        
        # 7. Historial head-to-head
        if 'h2h_records' in file_paths:
            try:
                file_path = file_paths['h2h_records']
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        loaded_data['h2h_records'] = json.load(f)
                    
                    loaded_data['loaded_files'].append(file_path)
                    logger.info(f"Historial head-to-head cargado desde {file_path}")
            except Exception as e:
                logger.error(f"Error cargando historial head-to-head: {str(e)}")
        
        # Verificar si hay datos faltantes y tratar de completarlos
        if not loaded_data['player_ratings']:
            logger.warning("No se cargaron ratings generales, los datos pueden estar incompletos")
        
        if not loaded_data['player_match_count_by_surface'] and loaded_data['player_ratings_by_surface']:
            # Intentar reconstruir contadores por superficie
            logger.warning("Reconstruyendo contadores por superficie...")
            for surface, ratings in loaded_data['player_ratings_by_surface'].items():
                if surface not in loaded_data['player_match_count_by_surface']:
                    loaded_data['player_match_count_by_surface'][surface] = {}
                
                for player_id in ratings:
                    if player_id not in loaded_data['player_match_count_by_surface'][surface]:
                        # Estimar contadores con valor mínimo
                        loaded_data['player_match_count_by_surface'][surface][player_id] = 5
        
        # Verificar integridad general de los datos cargados
        if not loaded_data['loaded_files']:
            logger.error("No se cargó ningún archivo de ratings")
        else:
            logger.info(f"Datos cargados exitosamente desde {len(loaded_data['loaded_files'])} archivos")
        
        return loaded_data
    
    def save_rating_history(self, rating_history: List[Dict[str, Any]],
                          output_file: Optional[str] = None) -> str:
        """
        Guarda el historial de ratings ELO.
        
        Args:
            rating_history: Lista de registros históricos de ratings
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        # Determinar nombre del archivo de salida
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'elo_rating_history_{timestamp}.csv'
        
        output_path = self.history_dir / output_file
        
        try:
            # Convertir a DataFrame si es una lista
            if isinstance(rating_history, list):
                df = pd.DataFrame(rating_history)
            else:
                df = rating_history
            
            # Guardar como CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Historial de ratings guardado en: {output_path}")
            
            # También guardar una versión simplificada para referencias rápidas
            try:
                # Seleccionar columnas clave para la versión resumida
                key_columns = [
                    'date', 'winner_id', 'loser_id', 
                    'winner_rating_before', 'loser_rating_before',
                    'winner_rating_after', 'loser_rating_after',
                    'elo_change_winner', 'elo_change_loser',
                    'expected_win_prob', 'surface', 'tourney_level',
                    'score', 'round'
                ]
                
                # Filtrar sólo las columnas que existen en el DataFrame
                available_columns = [col for col in key_columns if col in df.columns]
                
                if available_columns:
                    summary_df = df[available_columns]
                    summary_path = self.history_dir / output_file.replace('.csv', '_summary.csv')
                    summary_df.to_csv(summary_path, index=False)
                    logger.info(f"Historial resumido guardado en: {summary_path}")
            except Exception as e:
                logger.warning(f"Error guardando historial resumido: {str(e)}")
            
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error guardando historial de ratings: {str(e)}")
            raise
    
    def load_rating_history(self, input_file: Optional[str] = None,
                          use_latest: bool = True) -> pd.DataFrame:
        """
        Carga el historial de ratings ELO.
        
        Args:
            input_file: Nombre del archivo a cargar (opcional)
            use_latest: Si debe cargar el archivo más reciente cuando no se especifica input_file
            
        Returns:
            DataFrame con el historial cargado
        """
        # Determinar archivo a cargar
        if input_file:
            # Verificar si es ruta absoluta o relativa al directorio de historiales
            if os.path.isabs(input_file):
                input_path = Path(input_file)
            else:
                input_path = self.history_dir / input_file
        elif use_latest:
            # Buscar el archivo más reciente
            history_files = list(self.history_dir.glob('elo_rating_history_*.csv'))
            if not history_files:
                logger.error("No se encontraron archivos de historial")
                return pd.DataFrame()
            
            # Ordenar por fecha de modificación (más reciente primero)
            input_path = sorted(history_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        else:
            logger.error("No se especificó archivo de entrada")
            return pd.DataFrame()
        
        # Verificar que el archivo existe
        if not input_path.exists():
            logger.error(f"Archivo de historial no existe: {input_path}")
            return pd.DataFrame()
        
        try:
            # Cargar como DataFrame
            df = pd.read_csv(input_path)
            
            # Convertir columnas de fecha a datetime
            date_columns = ['date', 'match_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            logger.info(f"Historial de ratings cargado desde {input_path}: {len(df)} registros")
            return df
        
        except Exception as e:
            logger.error(f"Error cargando historial de ratings: {str(e)}")
            return pd.DataFrame()
    
    def save_player_match_history(self, player_match_history: Dict[str, List[Dict]],
                                output_file: Optional[str] = None) -> str:
        """
        Guarda el historial de partidos por jugador.
        
        Args:
            player_match_history: Diccionario con historial de partidos por jugador
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        # Determinar nombre del archivo de salida
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'player_match_history_{timestamp}.json'
        
        output_path = self.history_dir / output_file
        
        try:
            # Convertir fechas y otros tipos no serializables
            serializable_history = {}
            
            for player_id, matches in player_match_history.items():
                serializable_history[player_id] = []
                
                for match in matches:
                    match_copy = match.copy()
                    
                    # Convertir fechas a string
                    if 'date' in match_copy and isinstance(match_copy['date'], (datetime, pd.Timestamp)):
                        match_copy['date'] = match_copy['date'].strftime('%Y-%m-%d')
                    
                    # Convertir valores numpy a tipos nativos
                    for key, value in match_copy.items():
                        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                            match_copy[key] = int(value)
                        elif isinstance(value, (np.float64, np.float32, np.float16)):
                            match_copy[key] = float(value)
                    
                    serializable_history[player_id].append(match_copy)
            
            # Guardar como JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2)
            
            logger.info(f"Historial de partidos por jugador guardado en: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error guardando historial de partidos: {str(e)}")
            raise
    
    def load_player_match_history(self, input_file: Optional[str] = None,
                               use_latest: bool = True) -> Dict[str, List[Dict]]:
        """
        Carga el historial de partidos por jugador.
        
        Args:
            input_file: Nombre del archivo a cargar (opcional)
            use_latest: Si debe cargar el archivo más reciente cuando no se especifica input_file
            
        Returns:
            Diccionario con historial de partidos por jugador
        """
        # Determinar archivo a cargar
        if input_file:
            # Verificar si es ruta absoluta o relativa al directorio de historiales
            if os.path.isabs(input_file):
                input_path = Path(input_file)
            else:
                input_path = self.history_dir / input_file
        elif use_latest:
            # Buscar el archivo más reciente
            history_files = list(self.history_dir.glob('player_match_history_*.json'))
            if not history_files:
                logger.error("No se encontraron archivos de historial de partidos")
                return {}
            
            # Ordenar por fecha de modificación (más reciente primero)
            input_path = sorted(history_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        else:
            logger.error("No se especificó archivo de entrada")
            return {}
        
        # Verificar que el archivo existe
        if not input_path.exists():
            logger.error(f"Archivo de historial de partidos no existe: {input_path}")
            return {}
        
        try:
            # Cargar desde JSON
            with open(input_path, 'r', encoding='utf-8') as f:
                history_dict = json.load(f)
            
            # Convertir fechas de string a datetime
            for player_id, matches in history_dict.items():
                for match in matches:
                    if 'date' in match and isinstance(match['date'], str):
                        try:
                            match['date'] = datetime.strptime(match['date'], '%Y-%m-%d')
                        except ValueError:
                            # Mantener como string si no se puede convertir
                            pass
            
            logger.info(f"Historial de partidos cargado desde {input_path}: {len(history_dict)} jugadores")
            return history_dict
        
        except Exception as e:
            logger.error(f"Error cargando historial de partidos: {str(e)}")
            return {}
    
    def save_model(self, model, model_name: str,
                 model_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Guarda un modelo de predicción ELO.
        
        Args:
            model: Objeto del modelo a guardar
            model_name: Nombre del modelo
            model_info: Información adicional sobre el modelo (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        import pickle
        
        # Crear nombre de archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = model_name.replace(' ', '_').lower()
        output_file = f'{safe_name}_{timestamp}.pkl'
        output_path = self.models_dir / output_file
        
        try:
            # Guardar modelo con pickle
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Si hay información adicional, guardarla en un archivo JSON separado
            if model_info:
                info_path = self.models_dir / f'{safe_name}_{timestamp}_info.json'
                
                # Añadir timestamp y metadatos
                model_info['timestamp'] = timestamp
                model_info['model_name'] = model_name
                model_info['model_file'] = str(output_file)
                model_info['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, indent=2, default=self._json_serializer)
                
                logger.info(f"Información del modelo guardada en: {info_path}")
            
            # Actualizar índice de modelos
            self._update_model_index(model_name, output_file, model_info)
            
            logger.info(f"Modelo guardado en: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            raise
    
    def load_model(self, model_name: Optional[str] = None,
                 model_file: Optional[str] = None,
                 use_latest: bool = True) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Carga un modelo de predicción ELO.
        
        Args:
            model_name: Nombre del modelo a cargar (opcional)
            model_file: Nombre específico del archivo del modelo (opcional)
            use_latest: Si debe cargar la versión más reciente del modelo cuando se especifica model_name
            
        Returns:
            Tupla (modelo, información del modelo)
        """
        import pickle
        
        # Determinar archivo a cargar
        if model_file:
            # Verificar si es ruta absoluta o relativa al directorio de modelos
            if os.path.isabs(model_file):
                input_path = Path(model_file)
            else:
                input_path = self.models_dir / model_file
                
            # Buscar archivo de información asociado
            file_stem = input_path.stem
            info_path = input_path.with_name(f"{file_stem}_info.json")
        
        elif model_name:
            # Buscar en el índice de modelos
            model_files = self._find_model_files(model_name)
            
            if not model_files:
                logger.error(f"No se encontraron modelos con nombre '{model_name}'")
                return None, None
            
            if use_latest:
                # Ordenar por fecha de modificación (más reciente primero)
                input_path = sorted(model_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
            else:
                # Usar el primer modelo encontrado
                input_path = model_files[0]
            
            # Buscar archivo de información asociado
            file_stem = input_path.stem
            info_path = input_path.with_name(f"{file_stem}_info.json")
        
        else:
            # Si no se especifica nombre ni archivo, cargar el modelo más reciente
            model_files = list(self.models_dir.glob('*.pkl'))
            
            if not model_files:
                logger.error("No se encontraron modelos")
                return None, None
            
            # Ordenar por fecha de modificación (más reciente primero)
            input_path = sorted(model_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
            
            # Buscar archivo de información asociado
            file_stem = input_path.stem
            info_path = input_path.with_name(f"{file_stem}_info.json")
        
        # Verificar que el archivo existe
        if not input_path.exists():
            logger.error(f"Archivo de modelo no existe: {input_path}")
            return None, None
        
        try:
            # Cargar modelo con pickle
            with open(input_path, 'rb') as f:
                model = pickle.load(f)
            
            # Cargar información si está disponible
            model_info = None
            if info_path.exists():
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
                except Exception as e:
                    logger.warning(f"Error cargando información del modelo: {str(e)}")
            
            logger.info(f"Modelo cargado desde {input_path}")
            return model, model_info
        
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            return None, None
    
    def save_processed_matches(self, processed_df: pd.DataFrame,
                             output_file: Optional[str] = None) -> str:
        """
        Guarda un DataFrame con partidos procesados.
        
        Args:
            processed_df: DataFrame con partidos procesados
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        # Determinar nombre del archivo de salida
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'processed_matches_{timestamp}.csv'
        
        output_path = self.data_dir / output_file
        
        try:
            # Guardar como CSV
            processed_df.to_csv(output_path, index=False)
            logger.info(f"Partidos procesados guardados en: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error guardando partidos procesados: {str(e)}")
            raise
    
    def load_processed_matches(self, input_file: Optional[str] = None,
                            use_latest: bool = True) -> pd.DataFrame:
        """
        Carga un DataFrame con partidos procesados.
        
        Args:
            input_file: Nombre del archivo a cargar (opcional)
            use_latest: Si debe cargar el archivo más reciente cuando no se especifica input_file
            
        Returns:
            DataFrame con partidos procesados
        """
        # Determinar archivo a cargar
        if input_file:
            # Verificar si es ruta absoluta o relativa al directorio de datos
            if os.path.isabs(input_file):
                input_path = Path(input_file)
            else:
                input_path = self.data_dir / input_file
        elif use_latest:
            # Buscar el archivo más reciente
            files = list(self.data_dir.glob('processed_matches_*.csv'))
            if not files:
                logger.error("No se encontraron archivos de partidos procesados")
                return pd.DataFrame()
            
            # Ordenar por fecha de modificación (más reciente primero)
            input_path = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        else:
            logger.error("No se especificó archivo de entrada")
            return pd.DataFrame()
        
        # Verificar que el archivo existe
        if not input_path.exists():
            logger.error(f"Archivo de partidos procesados no existe: {input_path}")
            return pd.DataFrame()
        
        try:
            # Cargar como DataFrame
            df = pd.read_csv(input_path, low_memory=False)
            
            # Convertir columnas de fecha a datetime
            date_columns = ['match_date', 'tourney_date', 'date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            logger.info(f"Partidos procesados cargados desde {input_path}: {len(df)} registros")
            return df
        
        except Exception as e:
            logger.error(f"Error cargando partidos procesados: {str(e)}")
            return pd.DataFrame()
    
    def save_system_state(self, system_state: Dict[str, Any],
                        output_file: Optional[str] = None) -> str:
        """
        Guarda el estado completo del sistema ELO.
        
        Args:
            system_state: Diccionario con el estado del sistema
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        # Determinar nombre del archivo de salida
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'elo_system_state_{timestamp}.pkl'
        
        output_path = self.data_dir / output_file
        
        try:
            import pickle
            
            # Guardar estado completo con pickle
            with open(output_path, 'wb') as f:
                pickle.dump(system_state, f)
            
            # También guardar un registro JSON para facilitar la inspección
            json_path = output_path.with_suffix('.json')
            
            # Extraer y serializar solo componentes JSON-serializables
            serializable_state = {
                'timestamp': timestamp,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'components': list(system_state.keys()),
                'player_count': len(system_state.get('player_ratings', {})),
                'version': system_state.get('version', '1.0')
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_state, f, indent=2, default=self._json_serializer)
            
            logger.info(f"Estado del sistema guardado en: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error guardando estado del sistema: {str(e)}")
            raise
    
    def load_system_state(self, input_file: Optional[str] = None,
                        use_latest: bool = True) -> Dict[str, Any]:
        """
        Carga el estado completo del sistema ELO.
        
        Args:
            input_file: Nombre del archivo a cargar (opcional)
            use_latest: Si debe cargar el archivo más reciente cuando no se especifica input_file
            
        Returns:
            Diccionario con el estado del sistema
        """
        # Determinar archivo a cargar
        if input_file:
            # Verificar si es ruta absoluta o relativa al directorio de datos
            if os.path.isabs(input_file):
                input_path = Path(input_file)
            else:
                input_path = self.data_dir / input_file
        elif use_latest:
            # Buscar el archivo más reciente
            files = list(self.data_dir.glob('elo_system_state_*.pkl'))
            if not files:
                logger.error("No se encontraron archivos de estado del sistema")
                return {}
            
            # Ordenar por fecha de modificación (más reciente primero)
            input_path = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        else:
            logger.error("No se especificó archivo de entrada")
            return {}
        
        # Verificar que el archivo existe
        if not input_path.exists():
            logger.error(f"Archivo de estado del sistema no existe: {input_path}")
            return {}
        
        try:
            import pickle
            
            # Cargar estado completo con pickle
            with open(input_path, 'rb') as f:
                system_state = pickle.load(f)
            
            logger.info(f"Estado del sistema cargado desde {input_path}")
            
            # Verificar componentes clave
            required_components = [
                'player_ratings', 
                'player_ratings_by_surface',
                'player_match_count', 
                'player_match_count_by_surface'
            ]
            
            missing_components = [comp for comp in required_components if comp not in system_state]
            if missing_components:
                logger.warning(f"Faltan componentes en el estado cargado: {missing_components}")
            
            return system_state
        
        except Exception as e:
            logger.error(f"Error cargando estado del sistema: {str(e)}")
            return {}
    
    def _find_latest_rating_files(self, directory: Path) -> Dict[str, str]:
        """
        Busca los archivos de ratings más recientes en un directorio.
        
        Args:
            directory: Directorio donde buscar
            
        Returns:
            Diccionario con rutas de archivos encontrados
        """
        file_patterns = {
            'general_ratings': 'elo_ratings_general_*.json',
            'surface_ratings': 'elo_ratings_by_surface_*.json',
            'match_counts': 'player_match_counts_*.json',
            'surface_match_counts': 'player_match_counts_by_surface_*.json',
            'uncertainty': 'player_rating_uncertainty_*.json',
            'form': 'player_form_*.json',
            'h2h_records': 'head_to_head_records_*.json'
        }
        
        result = {}
        
        for key, pattern in file_patterns.items():
            files = list(directory.glob(pattern))
            if files:
                # Ordenar por fecha de modificación (más reciente primero)
                latest_file = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
                result[key] = str(latest_file)
        
        return result
    
    def _find_model_files(self, model_name: str) -> List[Path]:
        """
        Busca archivos de modelo por nombre.
        
        Args:
            model_name: Nombre del modelo a buscar
            
        Returns:
            Lista de rutas de archivos encontrados
        """
        # Preparar nombre para búsqueda
        safe_name = model_name.replace(' ', '_').lower()
        
        # Buscar archivos que coincidan con el patrón
        return list(self.models_dir.glob(f'{safe_name}_*.pkl'))
    
    def _update_model_index(self, model_name: str, model_file: str, model_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Actualiza el índice de modelos.
        
        Args:
            model_name: Nombre del modelo
            model_file: Nombre del archivo del modelo
            model_info: Información del modelo (opcional)
        """
        index_path = self.models_dir / 'models_index.json'
        
        # Cargar índice existente o crear uno nuevo
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    models_index = json.load(f)
            except Exception:
                models_index = {'models': []}
        else:
            models_index = {'models': []}
        
        # Preparar entrada para el índice
        entry = {
            'name': model_name,
            'file': model_file,
            'added_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Añadir información adicional si está disponible
        if model_info:
            entry['info'] = {
                k: v for k, v in model_info.items() 
                if isinstance(v, (str, int, float, bool, list, dict))
            }
        
        # Añadir al índice
        models_index['models'].append(entry)
        
        # Actualizar el índice
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(models_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Error actualizando índice de modelos: {str(e)}")
    
    def _json_serializer(self, obj: Any) -> Any:
        """
        Serializa tipos especiales para JSON.
        
        Args:
            obj: Objeto a serializar
            
        Returns:
            Objeto serializable
        """
        if isinstance(obj, (datetime, np.datetime64, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        return str(obj)


# Funciones de utilidad para operaciones comunes

def create_backup(source_dir: str, backup_dir: Optional[str] = None) -> str:
    """
    Crea una copia de seguridad de los datos ELO.
    
    Args:
        source_dir: Directorio con los datos originales
        backup_dir: Directorio donde guardar la copia (opcional)
        
    Returns:
        Ruta del directorio de copia
    """
    import shutil
    from datetime import datetime
    
    # Crear nombre del directorio de copia si no se especifica
    if not backup_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_path = Path(source_dir)
        backup_dir = str(source_path.parent / f"{source_path.name}_backup_{timestamp}")
    
    # Crear copia de seguridad
    try:
        shutil.copytree(source_dir, backup_dir)
        logger.info(f"Copia de seguridad creada en: {backup_dir}")
        return backup_dir
    except Exception as e:
        logger.error(f"Error creando copia de seguridad: {str(e)}")
        raise


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Carga configuración desde un archivo JSON.
    
    Args:
        config_file: Ruta al archivo de configuración
        
    Returns:
        Diccionario con la configuración
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Configuración cargada desde: {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error cargando configuración: {str(e)}")
        return {}


def save_config(config: Dict[str, Any], config_file: str) -> bool:
    """
    Guarda configuración en un archivo JSON.
    
    Args:
        config: Diccionario con la configuración
        config_file: Ruta al archivo de configuración
        
    Returns:
        True si se guardó correctamente, False en caso contrario
    """
    try:
        # Asegurar que el directorio existe
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar configuración
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuración guardada en: {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error guardando configuración: {str(e)}")
        return False