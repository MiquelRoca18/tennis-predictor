import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# Configurar logging
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Clase para cargar y preparar datos de tenis desde archivos CSV.
    
    Esta clase proporciona métodos para cargar partidos, jugadores y rankings
    desde los directorios de datos existentes, combinarlos correctamente y
    filtrarlos según diversos criterios para preparar conjuntos de datos
    de entrenamiento para modelos de machine learning.
    """
    
    def __init__(self, base_dir: str = '.'):
        """
        Inicializa el DataLoader con el directorio base de datos.
        
        Args:
            base_dir: Directorio base donde se encuentran las carpetas atp, wta, etc.
        """
        self.base_dir = base_dir
        self.atp_dir = os.path.join(base_dir, 'atp')
        self.wta_dir = os.path.join(base_dir, 'wta')
        self.elo_dir = os.path.join(base_dir, 'elo')
        self.match_charting_dir = os.path.join(base_dir, 'match_charting')
        self.pointbypoint_dir = os.path.join(base_dir, 'pointbypoint')
        self.slam_pointbypoint_dir = os.path.join(base_dir, 'slam_pointbypoint')
        self.data_cache: Dict[str, Union[pd.DataFrame, dict]] = {}
        
    def _get_csv_files(self, directory: str, prefix: str = '') -> List[str]:
        """
        Obtiene todos los archivos CSV con un prefijo determinado en un directorio.
        
        Args:
            directory: Directorio donde buscar
            prefix: Prefijo de los archivos a buscar
            
        Returns:
            Lista de rutas completas a los archivos CSV
        """
        if not os.path.exists(directory):
            logger.warning(f"El directorio {directory} no existe")
            return []
        
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.startswith(prefix) and f.endswith('.csv')]
        return sorted(files)
    
    def load_matches(self, tour: str = 'atp', level: str = 'main', years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Carga los datos de partidos del tour especificado, nivel y años.
        
        Args:
            tour: 'atp' para hombres, 'wta' para mujeres
            level: 'main', 'challenger', 'futures' o 'all'
            years: Lista de años a cargar. Si es None, carga todos los años disponibles
            
        Returns:
            DataFrame con todos los partidos
        """
        cache_key = f"{tour}_matches_{level}_{str(years)}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        tour_dir = self.atp_dir if tour.lower() == 'atp' else self.wta_dir
        
        # Si se solicita 'all', cargar el archivo combinado en el directorio raíz
        if level == 'all':
            if tour.lower() == 'atp':
                file_path = os.path.join(self.base_dir, 'all_matches_2000_2024.csv')
            else:
                # Para WTA puede ser diferente, adaptar según sea necesario
                file_path = os.path.join(self.base_dir, 'all_matches_2000_2024.csv')
                
            if not os.path.exists(file_path):
                logger.warning(f"No se encontró el archivo combinado para {tour}")
                return pd.DataFrame()
            
            try:
                matches_df = pd.read_csv(file_path, encoding='utf-8')
                # Filtrar por tour si es necesario
                if 'tour' in matches_df.columns:
                    matches_df = matches_df[matches_df['tour'].str.lower() == tour.lower()]
                
                # Filtrar por años si se especifica
                if years and 'year' in matches_df.columns:
                    matches_df = matches_df[matches_df['year'].isin(years)]
                elif years and 'date' in matches_df.columns:
                    matches_df['year'] = pd.to_datetime(matches_df['date']).dt.year
                    matches_df = matches_df[matches_df['year'].isin(years)]
                
                # Convertir fechas a datetime
                if 'date' in matches_df.columns:
                    matches_df['date'] = pd.to_datetime(matches_df['date'])
                
                self.data_cache[cache_key] = matches_df
                logger.info(f"Cargados {len(matches_df)} partidos para {tour} (todos los niveles)")
                return matches_df
            except Exception as e:
                logger.error(f"Error al cargar {file_path}: {str(e)}")
                return pd.DataFrame()
        
        # Buscar archivos específicos para el nivel solicitado
        file_pattern = f"{tour}_matches_{level}_2000_2024.csv"
        match_files = [os.path.join(tour_dir, f) for f in os.listdir(tour_dir) 
                        if f == file_pattern]
        
        if not match_files:
            logger.warning(f"No se encontró el archivo {file_pattern}")
            return pd.DataFrame()
        
        # Cargar los datos
        try:
            matches_df = pd.read_csv(match_files[0], encoding='utf-8')
            
            # Filtrar por años si se especifica
            if years:
                if 'year' in matches_df.columns:
                    matches_df = matches_df[matches_df['year'].isin(years)]
                elif 'date' in matches_df.columns:
                    matches_df['year'] = pd.to_datetime(matches_df['date']).dt.year
                    matches_df = matches_df[matches_df['year'].isin(years)]
            
            # Convertir fechas a datetime
            if 'date' in matches_df.columns:
                matches_df['date'] = pd.to_datetime(matches_df['date'])
            
            # Guardar en caché
            self.data_cache[cache_key] = matches_df
            
            logger.info(f"Cargados {len(matches_df)} partidos para {tour} ({level})")
            return matches_df
        except Exception as e:
            logger.error(f"Error al cargar {match_files[0]}: {str(e)}")
            return pd.DataFrame()
    
    def load_players(self, tour: str = 'atp') -> pd.DataFrame:
        """
        Carga los datos de jugadores del tour especificado.
        
        Args:
            tour: 'atp' para hombres, 'wta' para mujeres
            
        Returns:
            DataFrame con información de jugadores
        """
        cache_key = f"{tour}_players"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        tour_dir = self.atp_dir if tour.lower() == 'atp' else self.wta_dir
        player_file = os.path.join(tour_dir, 'players.csv')
        
        if not os.path.exists(player_file):
            logger.warning(f"Archivo de jugadores no encontrado: {player_file}")
            return pd.DataFrame()
        
        try:
            players_df = pd.read_csv(player_file, encoding='utf-8')
            logger.info(f"Cargados {len(players_df)} jugadores para {tour}")
            self.data_cache[cache_key] = players_df
            return players_df
        except Exception as e:
            logger.error(f"Error al cargar {player_file}: {str(e)}")
            return pd.DataFrame()
    
    def load_rankings(self, tour: str = 'atp', years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Carga los datos de rankings del tour especificado y años.
        
        Args:
            tour: 'atp' para hombres, 'wta' para mujeres
            years: Lista de años a cargar. Si es None, carga todos los años disponibles
            
        Returns:
            DataFrame con todos los rankings
        """
        cache_key = f"{tour}_rankings_{str(years)}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        tour_dir = self.atp_dir if tour.lower() == 'atp' else self.wta_dir
        ranking_files = self._get_csv_files(tour_dir, prefix='rankings_')
        
        # Filtrar por años si se especifica
        if years:
            ranking_files = [f for f in ranking_files if any(str(year) in os.path.basename(f) for year in years)]
        
        if not ranking_files:
            logger.warning(f"No se encontraron archivos de rankings para {tour} en los años {years}")
            return pd.DataFrame()
        
        # Cargar y concatenar todos los archivos
        dfs = []
        for file in ranking_files:
            try:
                df = pd.read_csv(file, encoding='utf-8')
                year = os.path.basename(file).split('_')[1].split('.')[0]
                df['year'] = year
                dfs.append(df)
                logger.info(f"Cargado archivo {file} con {len(df)} rankings")
            except Exception as e:
                logger.error(f"Error al cargar {file}: {str(e)}")
        
        if not dfs:
            return pd.DataFrame()
        
        # Concatenar todos los DataFrames
        rankings_df = pd.concat(dfs, ignore_index=True)
        
        # Convertir fechas a datetime
        if 'ranking_date' in rankings_df.columns:
            rankings_df['ranking_date'] = pd.to_datetime(rankings_df['ranking_date'])
        
        # Guardar en caché
        self.data_cache[cache_key] = rankings_df
        
        logger.info(f"Cargados {len(rankings_df)} registros de ranking para {tour}")
        return rankings_df
    
    def load_match_stats(self, tour: str = 'atp', stats_type: str = 'overview', years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Carga estadísticas detalladas de partidos del Match Charting Project.
        
        Args:
            tour: 'atp' (o 'm' para hombres), 'wta' (o 'w' para mujeres)
            stats_type: Tipo de estadísticas a cargar ('Overview', 'ServeBasics', etc.)
            years: Lista de años a filtrar. Si es None, carga todos los años disponibles
            
        Returns:
            DataFrame con estadísticas de partidos
        """
        cache_key = f"{tour}_match_stats_{stats_type}_{str(years)}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Convertir tour al formato del MCP (m/w)
        tour_code = 'm' if tour.lower() in ['atp', 'm'] else 'w'
        
        # Determinar el archivo correcto basado en el tipo de estadística
        file_pattern = f"tennis_MatchChartingProject_charting_{tour_code}_stats_{stats_type}.csv"
        stats_file = os.path.join(self.match_charting_dir, file_pattern)
        
        if not os.path.exists(stats_file):
            logger.warning(f"No se encontró el archivo de estadísticas {file_pattern}")
            
            # Intentar con archivos combinados si no se encuentra el específico
            combined_file = os.path.join(self.match_charting_dir, "all_stats_combined.csv")
            if os.path.exists(combined_file):
                try:
                    stats_df = pd.read_csv(combined_file, encoding='utf-8')
                    # Filtrar por tour y tipo de estadística si es posible
                    if 'tour' in stats_df.columns:
                        stats_df = stats_df[stats_df['tour'].str.lower() == tour.lower()]
                    if 'stats_type' in stats_df.columns:
                        stats_df = stats_df[stats_df['stats_type'].str.lower() == stats_type.lower()]
                    
                    # Filtrar por años si se especifica
                    if years and 'year' in stats_df.columns:
                        stats_df = stats_df[stats_df['year'].isin(years)]
                    
                    self.data_cache[cache_key] = stats_df
                    logger.info(f"Cargadas {len(stats_df)} estadísticas de partidos para {tour} (tipo: {stats_type})")
                    return stats_df
                except Exception as e:
                    logger.error(f"Error al cargar estadísticas combinadas: {str(e)}")
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
        
        try:
            stats_df = pd.read_csv(stats_file, encoding='utf-8')
            
            # Filtrar por años si se especifica
            if years and 'year' in stats_df.columns:
                stats_df = stats_df[stats_df['year'].isin(years)]
            elif years and 'date' in stats_df.columns:
                stats_df['year'] = pd.to_datetime(stats_df['date']).dt.year
                stats_df = stats_df[stats_df['year'].isin(years)]
            
            # Guardar en caché
            self.data_cache[cache_key] = stats_df
            
            logger.info(f"Cargadas {len(stats_df)} estadísticas de partidos para {tour} (tipo: {stats_type})")
            return stats_df
        except Exception as e:
            logger.error(f"Error al cargar {stats_file}: {str(e)}")
            return pd.DataFrame()
    
    def merge_player_data(self, matches_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combina datos de partidos con información de jugadores.
        
        Args:
            matches_df: DataFrame con partidos
            players_df: DataFrame con información de jugadores
            
        Returns:
            DataFrame combinado con información adicional de jugadores
        """
        if matches_df.empty or players_df.empty:
            return matches_df
        
        # Crear copias para no modificar los originales
        matches = matches_df.copy()
        players = players_df.copy()
        
        # Unir con datos del jugador 1
        merged_df = pd.merge(
            matches,
            players,
            left_on='player1_id',
            right_on='player_id',
            how='left',
            suffixes=('', '_p1')
        )
        
        # Unir con datos del jugador 2
        merged_df = pd.merge(
            merged_df,
            players,
            left_on='player2_id',
            right_on='player_id',
            how='left',
            suffixes=('_p1', '_p2')
        )
        
        logger.info(f"Datos de partidos unidos con información de jugadores")
        return merged_df
    
    def add_rankings_to_matches(self, matches_df: pd.DataFrame, rankings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade la información de ranking más cercana a la fecha del partido.
        
        Args:
            matches_df: DataFrame con partidos
            rankings_df: DataFrame con rankings
            
        Returns:
            DataFrame de partidos con información de ranking añadida
        """
        if matches_df.empty or rankings_df.empty:
            return matches_df
        
        # Verificar que las columnas necesarias existen
        required_cols = ['date', 'player1_id', 'player2_id']
        if not all(col in matches_df.columns for col in required_cols):
            logger.error(f"Faltan columnas requeridas en matches_df: {required_cols}")
            return matches_df
        
        # Crear copias para no modificar los originales
        matches = matches_df.copy()
        rankings = rankings_df.copy()
        
        if 'ranking_date' not in rankings.columns:
            logger.error("La columna 'ranking_date' no está en el DataFrame de rankings")
            return matches
        
        # Asegurarse que las fechas son datetime
        if not pd.api.types.is_datetime64_dtype(matches['date']):
            matches['date'] = pd.to_datetime(matches['date'])
        
        if not pd.api.types.is_datetime64_dtype(rankings['ranking_date']):
            rankings['ranking_date'] = pd.to_datetime(rankings['ranking_date'])
        
        # Función para obtener el ranking más cercano anterior a la fecha del partido
        def get_closest_ranking(player_id, match_date):
            player_rankings = rankings[rankings['player_id'] == player_id]
            prior_rankings = player_rankings[player_rankings['ranking_date'] <= match_date]
            
            if prior_rankings.empty:
                return None, None
            
            closest_ranking = prior_rankings.loc[prior_rankings['ranking_date'].idxmax()]
            return closest_ranking['ranking'], closest_ranking['ranking_points']
        
        # Aplicar la función a cada fila para ambos jugadores
        ranking_data = []
        for _, row in matches.iterrows():
            p1_rank, p1_points = get_closest_ranking(row['player1_id'], row['date'])
            p2_rank, p2_points = get_closest_ranking(row['player2_id'], row['date'])
            ranking_data.append((p1_rank, p1_points, p2_rank, p2_points))
        
        # Añadir los rankings al DataFrame de partidos
        ranking_df = pd.DataFrame(
            ranking_data, 
            columns=['player1_rank', 'player1_points', 'player2_rank', 'player2_points']
        )
        matches = pd.concat([matches, ranking_df], axis=1)
        
        logger.info(f"Datos de ranking añadidos a {len(matches)} partidos")
        return matches
    
    def load_elo_ratings(self, rating_type: str = 'general', level: str = 'main') -> dict:
        """
        Carga las calificaciones ELO desde los archivos JSON.
        
        Args:
            rating_type: 'general' o nombre de superficie ('clay', 'hard', 'grass', 'carpet')
            level: 'main', 'challenger', 'futures'
            
        Returns:
            Diccionario con las calificaciones ELO
        """
        cache_key = f"elo_{rating_type}_{level}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Determinar el archivo correcto basado en el tipo y nivel
        if rating_type == 'general':
            file_pattern = f"elo_ratings_general_{level}.json"
            if level == 'main':
                file_pattern = "elo_ratings_general_20250406_081013.json"
            elif level == 'challenger':
                file_pattern = "elo_ratings_general_challenger_20250405_182402.json"
            elif level == 'futures':
                file_pattern = "elo_ratings_general_futures_20250406_045938.json"
        else:
            file_pattern = f"elo_ratings_by_surface_{level}.json"
            if level == 'main':
                file_pattern = "elo_ratings_by_surface_20250406_081013.json"
            elif level == 'challenger':
                file_pattern = "elo_ratings_by_surface_challenger_20250405_182402.json"
            elif level == 'futures':
                file_pattern = "elo_ratings_by_surface_futures_20250406_045938.json"
        
        ratings_file = os.path.join(self.elo_dir, 'ratings', file_pattern)
        
        if not os.path.exists(ratings_file):
            # Intentar con el archivo de índice más reciente
            if rating_type == 'general':
                if level == 'main':
                    file_pattern = "ratings_latest_index.json"
                elif level == 'challenger':
                    file_pattern = "ratings_latest_index_challenger.json"
                elif level == 'futures':
                    file_pattern = "ratings_latest_index_futures.json"
            
            ratings_file = os.path.join(self.elo_dir, 'ratings', file_pattern)
            
            if not os.path.exists(ratings_file):
                logger.warning(f"No se encontró archivo de ELO para {rating_type} ({level})")
                return {}
        
        try:
            with open(ratings_file, 'r', encoding='utf-8') as f:
                elo_data = json.load(f)
            
            # Si es el índice, cargar el archivo específico más reciente
            if 'latest_ratings_file' in elo_data:
                ratings_file = os.path.join(self.elo_dir, 'ratings', elo_data['latest_ratings_file'])
                with open(ratings_file, 'r', encoding='utf-8') as f:
                    elo_data = json.load(f)
            
            # Procesar según el tipo
            if rating_type == 'general':
                # El archivo contiene directamente las calificaciones generales
                ratings = elo_data
            else:
                # Para calificaciones por superficie, extraer solo la superficie solicitada
                if rating_type in elo_data:
                    ratings = elo_data[rating_type]
                else:
                    logger.warning(f"Superficie {rating_type} no encontrada en el archivo ELO")
                    return {}
            
            self.data_cache[cache_key] = ratings
            logger.info(f"Cargadas calificaciones ELO para {rating_type} ({level})")
            return ratings
        except Exception as e:
            logger.error(f"Error al cargar ELO desde {ratings_file}: {str(e)}")
            return {}

    def load_pointbypoint_data(self, tour: str = 'atp', level: str = 'main', source: str = 'regular') -> pd.DataFrame:
        """
        Carga datos punto por punto de partidos.
        
        Args:
            tour: 'atp', 'wta', 'ch' (challenger), 'fu' (futures), 'itf'
            level: 'main', 'qual' (qualifying)
            source: 'regular' o 'slam' para datos de Grand Slam
            
        Returns:
            DataFrame con datos punto por punto
        """
        cache_key = f"pbp_{tour}_{level}_{source}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        if source == 'slam':
            # Cargar datos de Grand Slam punto por punto
            # Buscar todos los archivos de puntos
            point_files = [f for f in os.listdir(self.slam_pointbypoint_dir) 
                        if f.endswith('_points.csv') and not any(x in f for x in ['doubles', 'mixed'])]
            
            if not point_files:
                logger.warning("No se encontraron archivos de puntos de Grand Slam")
                return pd.DataFrame()
            
            dfs = []
            for file in point_files:
                try:
                    file_path = os.path.join(self.slam_pointbypoint_dir, file)
                    df = pd.read_csv(file_path, encoding='utf-8')
                    # Extraer información del slam del nombre del archivo
                    year, slam = file.split('_')[0:2]
                    df['year'] = year
                    df['tournament'] = slam
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error al cargar {file}: {str(e)}")
            
            if not dfs:
                return pd.DataFrame()
            
            pointbypoint_df = pd.concat(dfs, ignore_index=True)
            
        else:
            # Cargar datos regulares punto por punto
            file_pattern = f"pbp_matches_{tour}_{level}_current.csv"
            if tour in ['atp', 'wta', 'ch', 'fu', 'itf']:
                pbp_file = os.path.join(self.pointbypoint_dir, file_pattern)
                archive_file = os.path.join(self.pointbypoint_dir, f"pbp_matches_{tour}_{level}_archive.csv")
                
                if not os.path.exists(pbp_file) and not os.path.exists(archive_file):
                    logger.warning(f"No se encontraron archivos punto por punto para {tour} ({level})")
                    return pd.DataFrame()
                
                dfs = []
                if os.path.exists(pbp_file):
                    try:
                        df = pd.read_csv(pbp_file, encoding='utf-8')
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error al cargar {pbp_file}: {str(e)}")
                
                if os.path.exists(archive_file):
                    try:
                        df = pd.read_csv(archive_file, encoding='utf-8')
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error al cargar {archive_file}: {str(e)}")
                
                if not dfs:
                    return pd.DataFrame()
                
                pointbypoint_df = pd.concat(dfs, ignore_index=True)
            else:
                logger.warning(f"Tour no reconocido: {tour}")
                return pd.DataFrame()
        
        # Convertir fechas a datetime si es necesario
        if 'date' in pointbypoint_df.columns:
            pointbypoint_df['date'] = pd.to_datetime(pointbypoint_df['date'])
        
        self.data_cache[cache_key] = pointbypoint_df
        logger.info(f"Cargados {len(pointbypoint_df)} registros punto por punto para {tour} ({level})")
        return pointbypoint_df

    def prepare_training_data(
        self, 
        tour: str = 'atp',
        level: str = 'main',
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        surfaces: Optional[List[str]] = None,
        min_rank: Optional[int] = None,
        include_stats: bool = True,
        include_elo: bool = True,
        include_pbp: bool = False
    ) -> pd.DataFrame:
        """
        Prepara un conjunto de datos completo para entrenamiento.
        
        Args:
            tour: 'atp' para hombres, 'wta' para mujeres
            level: 'main', 'challenger', 'futures' o 'all'
            start_date: Fecha de inicio para filtrar partidos
            end_date: Fecha final para filtrar partidos
            surfaces: Lista de superficies a incluir
            min_rank: Ranking mínimo de jugadores a considerar
            include_stats: Si se deben incluir estadísticas detalladas de partidos
            include_elo: Si se deben incluir calificaciones ELO
            include_pbp: Si se deben incluir datos punto por punto
            
        Returns:
            DataFrame preparado para entrenamiento
        """
        # Determinar los años a cargar basados en las fechas
        years = None
        if start_date:
            start_date = pd.to_datetime(start_date)
            if end_date:
                end_date = pd.to_datetime(end_date)
                years = list(range(start_date.year, end_date.year + 1))
            else:
                # Si no hay end_date, cargar desde start_date hasta ahora
                years = list(range(start_date.year, datetime.now().year + 1))
        
        # Cargar los datos necesarios
        matches_df = self.load_matches(tour, level, years)
        players_df = self.load_players(tour)
        rankings_df = self.load_rankings(tour, years)
        
        if matches_df.empty:
            logger.warning(f"No se encontraron partidos para {tour} ({level})")
            return pd.DataFrame()
        
        # Filtrar por fecha si se especifica
        if start_date:
            matches_df = matches_df[matches_df['date'] >= start_date]
        if end_date:
            matches_df = matches_df[matches_df['date'] <= end_date]
        
        # Filtrar por superficie si se especifica
        if surfaces:
            matches_df = matches_df[matches_df['surface'].isin(surfaces)]
    
    def split_train_test(
        self, 
        data: pd.DataFrame, 
        test_size: float = 0.2,
        temporal: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Args:
            data: DataFrame con los datos completos
            test_size: Proporción de datos para prueba
            temporal: Si es True, la división es temporal (los últimos datos para prueba)
            
        Returns:
            Tupla con (datos_entrenamiento, datos_prueba)
        """
        if data.empty:
            logger.warning("No hay datos para dividir")
            return pd.DataFrame(), pd.DataFrame()
        
        if temporal and 'date' in data.columns:
            # División temporal - los últimos datos para prueba
            data = data.sort_values('date')
            split_idx = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()
        else:
            # División aleatoria
            np.random.seed(42)  # Para reproducibilidad
            mask = np.random.rand(len(data)) < (1 - test_size)
            train_data = data[mask].copy()
            test_data = data[~mask].copy()
        
        logger.info(f"Datos divididos en {len(train_data)} para entrenamiento y {len(test_data)} para prueba")
        return train_data, test_data
    
    def get_features_and_target(
        self, 
        data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = 'player1_won'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extrae matrices de características y vector objetivo de los datos.
        
        Args:
            data: DataFrame con los datos
            feature_cols: Lista de columnas a usar como características
            target_col: Nombre de la columna objetivo
            
        Returns:
            Tupla con (X, y) para entrenamiento
        """
        if data.empty:
            logger.warning("No hay datos para extraer características")
            return pd.DataFrame(), pd.Series()
        
        if target_col not in data.columns:
            logger.error(f"La columna objetivo '{target_col}' no está en los datos")
            return pd.DataFrame(), pd.Series()
        
        y = data[target_col]
        
        if feature_cols:
            # Verificar que todas las columnas existen
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Columnas no encontradas: {missing_cols}")
                feature_cols = [col for col in feature_cols if col in data.columns]
            
            X = data[feature_cols].copy()
        else:
            # Excluir columnas que no son características
            exclude_cols = [
                target_col, 'match_id', 'tournament_id', 'date', 
                'player1_id', 'player2_id', 'winner_id'
            ]
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            X = data[feature_cols].copy()
        
        # Manejar valores faltantes con una estrategia simple
        X = X.fillna(X.mean())
        
        logger.info(f"Extraídas {X.shape[1]} características para {X.shape[0]} instancias")
        return X, y