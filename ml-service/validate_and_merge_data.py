import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Tuple
from elo_system import TennisEloSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TennisDataValidator:
    def __init__(self):
        self.elo_system = TennisEloSystem()
        
    def temporal_split(self, data: pd.DataFrame, 
                      train_end_date: str = '2020-12-31',
                      val_end_date: str = '2021-12-31') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba basados en fechas.
        
        Args:
            data: DataFrame con datos de partidos
            train_end_date: Fecha final para datos de entrenamiento
            val_end_date: Fecha final para datos de validación
            
        Returns:
            Tupla con (train_data, val_data, test_data)
        """
        logger.info("Realizando división temporal de datos...")
        
        # Convertir fechas a datetime
        data['match_date'] = pd.to_datetime(data['match_date'])
        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)
        
        # Dividir datos
        train_data = data[data['match_date'] <= train_end].copy()
        val_data = data[(data['match_date'] > train_end) & 
                       (data['match_date'] <= val_end)].copy()
        test_data = data[data['match_date'] > val_end].copy()
        
        logger.info(f"División temporal completada:")
        logger.info(f"- Entrenamiento: {len(train_data)} partidos")
        logger.info(f"- Validación: {len(val_data)} partidos")
        logger.info(f"- Prueba: {len(test_data)} partidos")
        
        return train_data, val_data, test_data
    
    def calculate_h2h_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula características head-to-head para cada partido.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            DataFrame con características head-to-head añadidas
        """
        logger.info("Calculando características head-to-head...")
        
        # Crear copia del DataFrame
        df = data.copy()
        
        # Inicializar columnas head-to-head
        df['h2h_matches'] = 0
        df['h2h_wins'] = 0
        df['h2h_win_rate'] = 0.0
        df['h2h_surface_matches'] = 0
        df['h2h_surface_wins'] = 0
        df['h2h_surface_win_rate'] = 0.0
        
        # Ordenar por fecha para procesar cronológicamente
        df = df.sort_values('match_date')
        
        # Diccionario para almacenar historial head-to-head
        h2h_history = {}
        
        # Procesar cada partido
        for idx, match in df.iterrows():
            player1_id = str(match['player1_id'])
            player2_id = str(match['player2_id'])
            surface = match['surface']
            
            # Crear clave única para el par de jugadores
            h2h_key = tuple(sorted([player1_id, player2_id]))
            
            # Inicializar historial si no existe
            if h2h_key not in h2h_history:
                h2h_history[h2h_key] = {
                    'total_matches': 0,
                    'total_wins': 0,
                    'surface_matches': {},
                    'surface_wins': {}
                }
            
            # Actualizar estadísticas generales
            h2h_history[h2h_key]['total_matches'] += 1
            if match['winner_id'] == player1_id:
                h2h_history[h2h_key]['total_wins'] += 1
            
            # Actualizar estadísticas por superficie
            if surface not in h2h_history[h2h_key]['surface_matches']:
                h2h_history[h2h_key]['surface_matches'][surface] = 0
                h2h_history[h2h_key]['surface_wins'][surface] = 0
            
            h2h_history[h2h_key]['surface_matches'][surface] += 1
            if match['winner_id'] == player1_id:
                h2h_history[h2h_key]['surface_wins'][surface] += 1
            
            # Guardar características para el partido actual
            df.at[idx, 'h2h_matches'] = h2h_history[h2h_key]['total_matches'] - 1  # Excluir partido actual
            df.at[idx, 'h2h_wins'] = h2h_history[h2h_key]['total_wins'] - (1 if match['winner_id'] == player1_id else 0)
            df.at[idx, 'h2h_win_rate'] = (df.at[idx, 'h2h_wins'] / df.at[idx, 'h2h_matches'] 
                                        if df.at[idx, 'h2h_matches'] > 0 else 0.5)
            
            df.at[idx, 'h2h_surface_matches'] = h2h_history[h2h_key]['surface_matches'][surface] - 1
            df.at[idx, 'h2h_surface_wins'] = h2h_history[h2h_key]['surface_wins'][surface] - (1 if match['winner_id'] == player1_id else 0)
            df.at[idx, 'h2h_surface_win_rate'] = (df.at[idx, 'h2h_surface_wins'] / df.at[idx, 'h2h_surface_matches'] 
                                                if df.at[idx, 'h2h_surface_matches'] > 0 else 0.5)
        
        logger.info("Características head-to-head calculadas exitosamente")
        return df
    
    def calculate_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula características temporales para cada partido.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            DataFrame con características temporales añadidas
        """
        logger.info("Calculando características temporales...")
        
        # Crear copia del DataFrame
        df = data.copy()
        
        # Ordenar por fecha
        df = df.sort_values('match_date')
        
        # Inicializar columnas temporales
        df['days_since_last_match'] = 0
        df['recent_form'] = 0.0
        df['recent_matches'] = 0
        df['recent_wins'] = 0
        df['recent_win_rate'] = 0.0
        
        # Diccionario para almacenar historial de jugadores
        player_history = {}
        
        # Procesar cada partido
        for idx, match in df.iterrows():
            player1_id = str(match['player1_id'])
            player2_id = str(match['player2_id'])
            match_date = pd.to_datetime(match['match_date'])
            
            # Procesar para cada jugador
            for player_id in [player1_id, player2_id]:
                if player_id not in player_history:
                    player_history[player_id] = {
                        'last_match_date': None,
                        'recent_matches': [],
                        'recent_wins': []
                    }
                
                # Calcular días desde último partido
                if player_history[player_id]['last_match_date'] is not None:
                    days_since_last = (match_date - player_history[player_id]['last_match_date']).days
                else:
                    days_since_last = 30  # Valor por defecto para primer partido
                
                # Calcular forma reciente (últimos 3 meses)
                recent_matches = [m for m in player_history[player_id]['recent_matches'] 
                                if (match_date - m['date']).days <= 90]
                recent_wins = [m for m in player_history[player_id]['recent_wins'] 
                             if (match_date - m['date']).days <= 90]
                
                # Guardar características
                if player_id == player1_id:
                    df.at[idx, 'days_since_last_match'] = days_since_last
                    df.at[idx, 'recent_matches'] = len(recent_matches)
                    df.at[idx, 'recent_wins'] = len(recent_wins)
                    df.at[idx, 'recent_win_rate'] = (len(recent_wins) / len(recent_matches) 
                                                   if recent_matches else 0.5)
                    df.at[idx, 'recent_form'] = (df.at[idx, 'recent_win_rate'] * 2 - 1)  # Normalizar a [-1, 1]
                
                # Actualizar historial
                player_history[player_id]['last_match_date'] = match_date
                player_history[player_id]['recent_matches'].append({'date': match_date})
                if match['winner_id'] == player_id:
                    player_history[player_id]['recent_wins'].append({'date': match_date})
        
        logger.info("Características temporales calculadas exitosamente")
        return df
    
    def calculate_surface_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula características específicas por superficie.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            DataFrame con características por superficie añadidas
        """
        logger.info("Calculando características por superficie...")
        
        # Crear copia del DataFrame
        df = data.copy()
        
        # Ordenar por fecha
        df = df.sort_values('match_date')
        
        # Inicializar columnas por superficie
        for surface in ['hard', 'clay', 'grass', 'carpet']:
            df[f'{surface}_matches'] = 0
            df[f'{surface}_wins'] = 0
            df[f'{surface}_win_rate'] = 0.0
        
        # Diccionario para almacenar historial por superficie
        surface_history = {}
        
        # Procesar cada partido
        for idx, match in df.iterrows():
            player1_id = str(match['player1_id'])
            player2_id = str(match['player2_id'])
            surface = match['surface']
            
            # Procesar para cada jugador
            for player_id in [player1_id, player2_id]:
                if player_id not in surface_history:
                    surface_history[player_id] = {
                        s: {'matches': 0, 'wins': 0} for s in ['hard', 'clay', 'grass', 'carpet']
                    }
                
                # Actualizar estadísticas por superficie
                surface_history[player_id][surface]['matches'] += 1
                if match['winner_id'] == player_id:
                    surface_history[player_id][surface]['wins'] += 1
                
                # Guardar características para player1
                if player_id == player1_id:
                    for s in ['hard', 'clay', 'grass', 'carpet']:
                        stats = surface_history[player_id][s]
                        df.at[idx, f'{s}_matches'] = stats['matches']
                        df.at[idx, f'{s}_wins'] = stats['wins']
                        df.at[idx, f'{s}_win_rate'] = (stats['wins'] / stats['matches'] 
                                                     if stats['matches'] > 0 else 0.5)
        
        logger.info("Características por superficie calculadas exitosamente")
        return df
    
    def validate_and_merge_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Valida y enriquece los datos con todas las características.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            DataFrame enriquecido con todas las características
        """
        logger.info("Iniciando validación y enriquecimiento de datos...")
        
        # Validar datos básicos
        data = self._validate_basic_data(data)
        
        # Calcular características ELO
        data = self.elo_system.calculate_elo_features(data)
        
        # Calcular características head-to-head
        data = self.calculate_h2h_features(data)
        
        # Calcular características temporales
        data = self.calculate_temporal_features(data)
        
        # Calcular características por superficie
        data = self.calculate_surface_features(data)
        
        logger.info("Validación y enriquecimiento de datos completado")
        return data
    
    def _validate_basic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Valida y limpia datos básicos.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            DataFrame validado y limpio
        """
        logger.info("Validando datos básicos...")
        
        # Crear copia del DataFrame
        df = data.copy()
        
        # Convertir fechas
        df['match_date'] = pd.to_datetime(df['match_date'])
        
        # Eliminar partidos futuros
        current_date = pd.Timestamp.now()
        df = df[df['match_date'] <= current_date]
        
        # Eliminar partidos con valores nulos críticos
        critical_columns = ['player1_id', 'player2_id', 'winner_id', 'surface']
        df = df.dropna(subset=critical_columns)
        
        # Validar superficies
        valid_surfaces = ['hard', 'clay', 'grass', 'carpet']
        df = df[df['surface'].isin(valid_surfaces)]
        
        # Validar IDs de jugadores
        df = df[df['player1_id'] != df['player2_id']]
        
        logger.info(f"Datos básicos validados: {len(df)} partidos válidos")
        return df

def validate_data(df):
    """Valida la calidad de los datos."""
    logger.info("\nValidando datos...")
    
    # Verificar fechas
    df['match_date'] = pd.to_datetime(df['match_date'])
    current_date = datetime.now()
    future_matches = df[df['match_date'] > current_date]
    
    if len(future_matches) > 0:
        logger.warning(f"Se encontraron {len(future_matches)} partidos futuros que serán excluidos")
        df = df[df['match_date'] <= current_date]
    
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("\nColumnas con valores nulos:")
        for col, count in null_counts[null_counts > 0].items():
            logger.warning(f"{col}: {count} valores nulos")
    
    # Verificar valores inválidos solo para columnas que existen
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in df.columns:
            invalid_count = (df[col] < 0).sum()
            if invalid_count > 0:
                logger.warning(f"{col}: {invalid_count} valores inválidos")
    
    # Verificar consistencia en los datos solo para columnas que existen
    if all(col in df.columns for col in ['winner_1stWon', 'winner_1stIn', 'winner_svpt', 'winner_2ndWon']):
        inconsistent_matches = df[
            (df['winner_1stWon'] > df['winner_1stIn']) |
            (df['winner_2ndWon'] > (df['winner_svpt'] - df['winner_1stIn']))
        ]
        if len(inconsistent_matches) > 0:
            logger.warning(f"Se encontraron {len(inconsistent_matches)} partidos con estadísticas inconsistentes")
    
    if all(col in df.columns for col in ['loser_1stWon', 'loser_1stIn', 'loser_svpt', 'loser_2ndWon']):
        inconsistent_matches = df[
            (df['loser_1stWon'] > df['loser_1stIn']) |
            (df['loser_2ndWon'] > (df['loser_svpt'] - df['loser_1stIn']))
        ]
        if len(inconsistent_matches) > 0:
            logger.warning(f"Se encontraron {len(inconsistent_matches)} partidos con estadísticas inconsistentes")
    
    return df

def merge_and_update_data():
    """Combina y actualiza los datos de diferentes fuentes."""
    logger.info("Iniciando proceso de validación y actualización de datos...")
    
    # Rutas de los archivos
    current_data_path = 'ml-service/data/tennis_matches.csv'
    new_data_path = 'data/tennis_data_temp.csv'
    output_path = 'ml-service/data/tennis_matches_updated.csv'
    
    # Cargar datos actuales
    logger.info("Cargando datos actuales...")
    current_data = pd.read_csv(current_data_path)
    logger.info(f"Datos actuales: {len(current_data)} partidos")
    
    # Cargar nuevos datos
    logger.info("Cargando nuevos datos...")
    new_data = pd.read_csv(new_data_path)
    logger.info(f"Nuevos datos: {len(new_data)} partidos")
    
    # Mostrar columnas disponibles en cada conjunto de datos
    logger.info("\nColumnas en datos actuales:")
    logger.info(current_data.columns.tolist())
    logger.info("\nColumnas en nuevos datos:")
    logger.info(new_data.columns.tolist())
    
    # Validar ambos conjuntos de datos
    current_data = validate_data(current_data)
    new_data = validate_data(new_data)
    
    # Combinar datos
    logger.info("\nCombinando datos...")
    combined_data = pd.concat([current_data, new_data], ignore_index=True)
    
    # Eliminar duplicados basados en match_id
    combined_data = combined_data.drop_duplicates(subset=['match_id'], keep='last')
    
    # Ordenar por fecha
    combined_data = combined_data.sort_values('match_date')
    
    # Guardar datos actualizados
    logger.info(f"\nGuardando {len(combined_data)} partidos en {output_path}")
    combined_data.to_csv(output_path, index=False)
    
    # Resumen de cambios
    logger.info("\nResumen de cambios:")
    logger.info(f"- Partidos originales: {len(current_data)}")
    logger.info(f"- Nuevos partidos: {len(new_data)}")
    logger.info(f"- Partidos finales: {len(combined_data)}")
    logger.info(f"- Partidos añadidos: {len(combined_data) - len(current_data)}")
    
    # Verificar distribución temporal
    logger.info("\nDistribución temporal de los datos:")
    combined_data['year'] = combined_data['match_date'].dt.year
    year_counts = combined_data['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        logger.info(f"{year}: {count} partidos")

if __name__ == "__main__":
    merge_and_update_data() 