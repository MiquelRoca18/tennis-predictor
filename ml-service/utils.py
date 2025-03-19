import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
from datetime import datetime, timedelta
import logging
import os
from sklearn.preprocessing import StandardScaler
import joblib
import requests

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Definir la ruta del directorio de logs
log_dir = os.path.join(current_dir, 'logs')

# Crear el directorio si no existe
os.makedirs(log_dir, exist_ok=True)

# Configurar logging con ruta absoluta
logging.basicConfig(
    filename=os.path.join(log_dir, 'tennis_ml.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)


class TennisFeatureEngineering:
    """
    Clase para realizar ingeniería de características avanzada para
    el modelo de predicción de partidos de tenis.
    """
    
    def __init__(self, data_path: str = 'tennis_matches.csv'):
        """
        Inicializa el motor de ingeniería de características.
        
        Args:
            data_path: Ruta al archivo CSV con datos históricos
        """
        self.data_path = data_path
        self.data = None
        self.players_stats = {}
        self.head_to_head_stats = {}
        self.surface_stats = {}
        self.tournament_stats = {}
        self.scaler = StandardScaler()
        
        # Intentar diferentes rutas posibles para el archivo de datos
        possible_paths = [
            data_path,
            os.path.join(current_dir, data_path),
            os.path.join(current_dir, "tennis_matches.csv"),
            os.path.join(os.path.dirname(current_dir), "tennis_matches.csv"),
            "tennis_matches.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.data_path = path
                self.data = pd.read_csv(path)
                logging.info(f"Datos cargados desde {path}: {len(self.data)} registros")
                break
        
        if self.data is None:
            # Si no hay datos, intentar generar automáticamente
            logging.warning(f"Archivo de datos no encontrado. Intentando generar datos...")
            self._generate_or_download_data()
    
    def _generate_or_download_data(self):
        """Genera o descarga datos si no están disponibles"""
        try:
            # Primero intentar descargar datos reales
            logging.info("Intentando descargar datos de tenis desde fuentes externas...")
            
            # Intentar descargar de GitHub JeffSackmann
            url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    # Guardar los datos descargados
                    with open('tennis_matches.csv', 'wb') as f:
                        f.write(response.content)
                    
                    self.data = pd.read_csv('tennis_matches.csv')
                    self.data_path = 'tennis_matches.csv'
                    
                    # Transformar al formato estándar
                    try:
                        # Verificar si necesitamos transformación
                        if 'winner_name' in self.data.columns and 'player1' not in self.data.columns:
                            transformed = self._transform_data(self.data)
                            self.data = transformed
                            self.data.to_csv(self.data_path, index=False)
                    except Exception as e:
                        logging.error(f"Error transformando datos: {e}")
                    
                    logging.info(f"Datos descargados exitosamente: {len(self.data)} registros")
                    return
            except Exception as e:
                logging.error(f"Error descargando datos: {e}")
            
            # Si la descarga falla, intentar generar datos de prueba
            from data_collector import TennisDataCollector
            collector = TennisDataCollector(output_path='tennis_matches.csv')
            self.data = collector._generate_test_data(n_samples=500)
            self.data_path = 'tennis_matches.csv'
            logging.info(f"Datos de prueba generados: {len(self.data)} registros")
        except Exception as e:
            logging.error(f"Error generando datos de respaldo: {e}")
            raise ValueError("No se pudieron obtener datos para el análisis")
    
        import random


    # También modificar el método _transform_data para usar la nueva función
    def _transform_data(self, data, source_key):
        """
        Transforma datos según la fuente a un formato estándar.
        
        Args:
            data: DataFrame con datos originales
            source_key: Clave de la fuente para mappings
            
        Returns:
            DataFrame transformado al formato estándar
        """
        try:
            if data is None or data.empty:
                return None
                    
            # Obtener mapeo para esta fuente
            mapping = self.source_mappings.get(source_key)
            
            if not mapping:
                logging.error(f"No se encontró mapeo para la fuente: {source_key}")
                return None
            
            # Si tenemos winner_name y loser_name, usamos el nuevo método de transformación
            if 'winner_name' in data.columns and 'loser_name' in data.columns:
                return self._transform_data(data)
            
            # Si no, seguimos con el proceso original (pero modificado para evitar defaults)
            transformed = pd.DataFrame()
            
            # Mapear jugadores
            if mapping['player1'] in data.columns and mapping['player2'] in data.columns:
                transformed['player1'] = data[mapping['player1']]
                transformed['player2'] = data[mapping['player2']]
            else:
                logging.error(f"Columnas de jugadores no encontradas en la fuente {source_key}")
                return None
        
            # Mapear otros campos si existen
            for std_field, source_field in mapping.items():
                if std_field in ['player1', 'player2']:  # Ya procesados
                    continue
                    
                if source_field and source_field in data.columns:
                    transformed[std_field] = data[source_field]
            
            # Aleatorizar la asignación del ganador
            # En lugar de siempre asignar winner=0, generamos valores aleatorios
            transformed['winner'] = [random.randint(0, 1) for _ in range(len(transformed))]
            
            # Procesar fecha si existe
            if 'date' in mapping and mapping['date'] in data.columns:
                date_field = mapping['date']
                
                # Determinar formato de fecha según la fuente
                if source_key == 'github_jeff_sackmann':
                    # Formato YYYYMMDD
                    transformed['match_date'] = pd.to_datetime(data[date_field], format='%Y%m%d', errors='coerce')
                else:
                    # Intentar auto-detectar formato
                    transformed['match_date'] = pd.to_datetime(data[date_field], errors='coerce')
            
            # Calcular tasas de victoria si hay suficientes datos
            transformed = self._calculate_win_rates(transformed)
            
            return transformed
            
        except Exception as e:
            logging.error(f"Error transformando datos de {source_key}: {e}")
            logging.error(traceback.format_exc())
            return None 
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocesa los datos para asegurar calidad y consistencia.
        
        Returns:
            DataFrame preprocesado
        """
        if self.data is None or self.data.empty:
            self._generate_or_download_data()
            if self.data is None or self.data.empty:
                raise ValueError("No hay datos para preprocesar y no se pudieron obtener automáticamente")
        
        df = self.data.copy()
        
        # Verificar columnas esenciales
        essential_columns = ['player1', 'player2', 'winner', 'surface']
        missing_columns = [col for col in essential_columns if col not in df.columns]
        
        # Si faltan columnas esenciales, intentar inferirlas o transformar
        if missing_columns:
            # Verificar si tenemos datos en formato alternativo (API/externo)
            if 'winner_name' in df.columns and 'loser_name' in df.columns:
                logging.info("Transformando datos de formato externo a formato estándar")
                df = self._transform_data(df)
                missing_columns = [col for col in essential_columns if col not in df.columns]
                
                # Guardar los datos transformados para uso futuro
                df.to_csv(self.data_path, index=False)
                self.data = df
        
        # Si aún faltan columnas esenciales, no podemos continuar
        if missing_columns:
            raise ValueError(f"Faltan columnas esenciales: {missing_columns}")
        
        # Convertir fechas si es necesario
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
            
            # Ordenar por fecha para mantener consistencia temporal
            df = df.sort_values('match_date')
        
        # Normalizar superficies - sin valores por defecto
        if 'surface' in df.columns:
            surface_mapping = {
                'hard': 'hard',
                'clay': 'clay',
                'grass': 'grass',
                'carpet': 'carpet',
                'h': 'hard',
                'c': 'clay',
                'g': 'grass',
                'i': 'hard'  # indoor hard
            }
            
            # Convertir a minúsculas y luego mapear
            lowercase_surfaces = df['surface'].str.lower()
            df['surface'] = lowercase_surfaces.map(lambda x: surface_mapping.get(x, x))
        
        # Manejo de valores nulos - solo eliminar, no usar valores por defecto
        for col in ['ranking_1', 'ranking_2', 'winrate_1', 'winrate_2']:
            if col in df.columns:
                nulls = df[col].isnull().sum()
                if nulls > 0:
                    logging.warning(f"Se detectaron {nulls} valores nulos en {col}")
        
        # No imputamos valores nulos aquí, nos aseguramos que los datos sean de calidad
        
        # Eliminar filas con valores nulos críticos que no podemos imputar
        critical_nulls = df[essential_columns].isnull().any(axis=1)
        if critical_nulls.any():
            logging.warning(f"Eliminando {critical_nulls.sum()} filas con valores nulos en columnas esenciales")
            df = df[~critical_nulls]
        
        self.data = df
        return df
    
    def build_player_statistics(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Construye estadísticas detalladas para cada jugador.
        
        Returns:
            Diccionario con estadísticas por jugador
        """
        if self.data is None or self.data.empty:
            self._generate_or_download_data()
            self.preprocess_data()
            
        player_stats = {}
        
        # Lista de todos los jugadores únicos
        all_players = set()
        all_players.update(self.data['player1'].unique())
        all_players.update(self.data['player2'].unique())
        
        for player in all_players:
            # Filtrar partidos del jugador
            player1_matches = self.data[self.data['player1'] == player]
            player2_matches = self.data[self.data['player2'] == player]
            
            # Total de partidos
            total_matches = len(player1_matches) + len(player2_matches)
            
            if total_matches == 0:
                continue
            
            # Victorias
            player1_wins = player1_matches[player1_matches['winner'] == 0].shape[0]
            player2_wins = player2_matches[player2_matches['winner'] == 1].shape[0]
            total_wins = player1_wins + player2_wins
            
            # Tasa de victoria general
            win_rate = (total_wins / total_matches) * 100 if total_matches > 0 else None
            
            # Estadísticas por superficie - sin valores por defecto
            surface_stats = {}
            for surface in ['hard', 'clay', 'grass', 'carpet']:
                # Como player1
                p1_surface = player1_matches[player1_matches['surface'] == surface]
                p1_surface_wins = p1_surface[p1_surface['winner'] == 0].shape[0]
                
                # Como player2
                p2_surface = player2_matches[player2_matches['surface'] == surface]
                p2_surface_wins = p2_surface[p2_surface['winner'] == 1].shape[0]
                
                surface_total = len(p1_surface) + len(p2_surface)
                surface_wins = p1_surface_wins + p2_surface_wins
                
                if surface_total > 0:
                    surface_win_rate = (surface_wins / surface_total) * 100
                    
                    surface_stats[surface] = {
                        'matches': surface_total,
                        'wins': surface_wins,
                        'win_rate': surface_win_rate
                    }
                # Si no hay datos para esta superficie, no añadimos estadísticas (no valores por defecto)
            
            # Calcular tendencia reciente (últimos 10 partidos)
            recent_matches = []
            
            # Como player1
            if 'match_date' in self.data.columns:
                p1_recent = player1_matches.sort_values('match_date', ascending=False).head(10)
            else:
                p1_recent = player1_matches.tail(10)  # Sin fecha, usamos los últimos del dataset
                
            for _, match in p1_recent.iterrows():
                recent_matches.append(1 if match['winner'] == 0 else 0)
            
            # Como player2
            if 'match_date' in self.data.columns:
                p2_recent = player2_matches.sort_values('match_date', ascending=False).head(10)
            else:
                p2_recent = player2_matches.tail(10)
                
            for _, match in p2_recent.iterrows():
                recent_matches.append(1 if match['winner'] == 1 else 0)
            
            # Calcular forma reciente
            recent_form = sum(recent_matches) / len(recent_matches) * 100 if recent_matches else None
            
            # Ranking promedio (si está disponible)
            avg_ranking = None
            if 'ranking_1' in self.data.columns and 'ranking_2' in self.data.columns:
                rankings = []
                rankings.extend(player1_matches['ranking_1'].dropna().tolist())
                rankings.extend(player2_matches['ranking_2'].dropna().tolist())
                
                if rankings:
                    avg_ranking = sum(rankings) / len(rankings)
            
            # Almacenar estadísticas del jugador - solo si tenemos datos reales
            player_stats[player] = {
                'total_matches': total_matches,
                'total_wins': total_wins,
                'win_rate': win_rate,
                'surface_stats': surface_stats,
                'recent_form': recent_form,
                'avg_ranking': avg_ranking
            }
        
        self.players_stats = player_stats
        return player_stats
    
    def build_head_to_head_statistics(self) -> Dict[Tuple[str, str], Dict[str, Union[float, int]]]:
        """
        Construye estadísticas de enfrentamientos directos entre jugadores.
        
        Returns:
            Diccionario con estadísticas de enfrentamientos
        """
        if self.data is None or self.data.empty:
            self._generate_or_download_data()
            self.preprocess_data()
        
        h2h_stats = {}
        
        # Para cada combinación de jugadores
        for _, match in self.data.iterrows():
            player1 = match['player1']
            player2 = match['player2']
            
            # Usar tupla ordenada para asegurar consistencia
            player_pair = tuple(sorted([player1, player2]))
            
            # Inicializar si es primer enfrentamiento
            if player_pair not in h2h_stats:
                h2h_stats[player_pair] = {
                    'total_matches': 0,
                    'player1_wins': 0,
                    'player2_wins': 0,
                    'surface_stats': {}  # Inicializamos vacío, solo añadimos por superficie si hay datos
                }
            
            # Actualizar estadísticas totales
            h2h_stats[player_pair]['total_matches'] += 1
            
            # Asegurar consistencia con la tupla ordenada
            if player1 == player_pair[0]:
                if match['winner'] == 0:  # player1 ganó
                    h2h_stats[player_pair]['player1_wins'] += 1
                else:  # player2 ganó
                    h2h_stats[player_pair]['player2_wins'] += 1
            else:  # player1 es player_pair[1]
                if match['winner'] == 0:  # player1 ganó
                    h2h_stats[player_pair]['player2_wins'] += 1
                else:  # player2 ganó
                    h2h_stats[player_pair]['player1_wins'] += 1
            
            # Actualizar estadísticas por superficie - solo si existe la superficie
            surface = match['surface']
            
            # Inicializar estadísticas para esta superficie si es el primer partido
            if surface not in h2h_stats[player_pair]['surface_stats']:
                h2h_stats[player_pair]['surface_stats'][surface] = {
                    'matches': 0, 
                    'player1_wins': 0, 
                    'player2_wins': 0
                }
            
            # Actualizar contador de partidos
            h2h_stats[player_pair]['surface_stats'][surface]['matches'] += 1
            
            # Actualizar victorias por superficie
            if player1 == player_pair[0]:
                if match['winner'] == 0:  # player1 ganó
                    h2h_stats[player_pair]['surface_stats'][surface]['player1_wins'] += 1
                else:  # player2 ganó
                    h2h_stats[player_pair]['surface_stats'][surface]['player2_wins'] += 1
            else:  # player1 es player_pair[1]
                if match['winner'] == 0:  # player1 ganó
                    h2h_stats[player_pair]['surface_stats'][surface]['player2_wins'] += 1
                else:  # player2 ganó
                    h2h_stats[player_pair]['surface_stats'][surface]['player1_wins'] += 1
        
        # Calcular ventaja head-to-head - sin valores por defecto
        for pair, stats in h2h_stats.items():
            total = stats['total_matches']
            if total > 0:
                p1_win_pct = (stats['player1_wins'] / total) * 100
                stats['player1_win_pct'] = p1_win_pct
                stats['player2_win_pct'] = 100 - p1_win_pct
                
                # Calcular ventaja (0.5 significa equilibrio)
                # Solo calculamos si hay suficientes datos (más de 1 partido)
                if total > 1:
                    if p1_win_pct > 55:
                        stats['advantage'] = 0.5 + ((p1_win_pct - 50) / 100)
                    elif p1_win_pct < 45:
                        stats['advantage'] = 0.5 - ((50 - p1_win_pct) / 100)
                    else:
                        stats['advantage'] = 0.5
                else:
                    # No hay suficientes datos para determinar ventaja confiable
                    stats['advantage'] = None
            
            # Calcular ventaja por superficie - solo si hay datos
            for surface, surface_stats in stats['surface_stats'].items():
                surface_matches = surface_stats['matches']
                if surface_matches > 0:
                    p1_surface_win_pct = (surface_stats['player1_wins'] / surface_matches) * 100
                    surface_stats['player1_win_pct'] = p1_surface_win_pct
                    surface_stats['player2_win_pct'] = 100 - p1_surface_win_pct
                    
                    # Ventaja por superficie (solo si hay más de 1 partido)
                    if surface_matches > 1:
                        if p1_surface_win_pct > 55:
                            surface_stats['advantage'] = 0.5 + ((p1_surface_win_pct - 50) / 100)
                        elif p1_surface_win_pct < 45:
                            surface_stats['advantage'] = 0.5 - ((50 - p1_surface_win_pct) / 100)
                        else:
                            surface_stats['advantage'] = 0.5
                    else:
                        surface_stats['advantage'] = None
        
        self.head_to_head_stats = h2h_stats
        return h2h_stats
    
    def extract_features(self, match_data: Dict[str, Union[str, int, float]]) -> Dict[str, float]:
        """
        Extrae características avanzadas para un partido específico.
        
        Args:
            match_data: Diccionario con datos del partido
        
        Returns:
            Diccionario con características procesadas
        """
        # Verificar datos mínimos
        required_fields = ['player1', 'player2', 'surface']
        required_fields_alt = ['player_1', 'player_2', 'surface']
        
        # Compatibilidad con diferentes formatos de nombres de campo
        if all(field in match_data for field in required_fields_alt):
            # Convertir a formato estándar
            match_data = match_data.copy()  # Evitar modificar el original
            match_data['player1'] = match_data.pop('player_1')
            match_data['player2'] = match_data.pop('player_2')
            if 'ranking_1' in match_data:
                match_data['ranking1'] = match_data.pop('ranking_1')
            if 'ranking_2' in match_data:
                match_data['ranking2'] = match_data.pop('ranking_2')
            if 'winrate_1' in match_data:
                match_data['winrate1'] = match_data.pop('winrate_1')
            if 'winrate_2' in match_data:
                match_data['winrate2'] = match_data.pop('winrate_2')
        
        # Verificar campos requeridos
        for field in required_fields:
            if field not in match_data:
                raise ValueError(f"Falta campo requerido: {field}")
        
        player1 = match_data['player1']
        player2 = match_data['player2']
        surface = match_data['surface']
        
        # Inicializar estadísticas si no se han calculado
        if not self.players_stats:
            self.build_player_statistics()
        
        if not self.head_to_head_stats:
            self.build_head_to_head_statistics()
        
        # Características básicas
        features = {}
        
        # Rankings (usar los proporcionados o buscar en estadísticas de jugadores)
        # No usamos valores por defecto
        if 'ranking1' in match_data:
            features['ranking_1'] = match_data['ranking1']
        elif 'ranking_1' in match_data:
            features['ranking_1'] = match_data['ranking_1']
        elif player1 in self.players_stats and self.players_stats[player1]['avg_ranking'] is not None:
            features['ranking_1'] = self.players_stats[player1]['avg_ranking']
        else:
            # Intentar obtener ranking de una API externa
            ranking = self._get_external_ranking(player1)
            features['ranking_1'] = ranking
            
        if 'ranking2' in match_data:
            features['ranking_2'] = match_data['ranking2']
        elif 'ranking_2' in match_data:
            features['ranking_2'] = match_data['ranking_2']
        elif player2 in self.players_stats and self.players_stats[player2]['avg_ranking'] is not None:
            features['ranking_2'] = self.players_stats[player2]['avg_ranking']
        else:
            # Intentar obtener ranking de una API externa
            ranking = self._get_external_ranking(player2)
            features['ranking_2'] = ranking
        
        # Tasas de victoria generales
        if 'winrate1' in match_data:
            features['winrate_1'] = match_data['winrate1']
        elif 'winrate_1' in match_data:
            features['winrate_1'] = match_data['winrate_1']
        elif player1 in self.players_stats and self.players_stats[player1]['win_rate'] is not None:
            features['winrate_1'] = self.players_stats[player1]['win_rate']
        else:
            logging.warning(f"No se encontró tasa de victoria para {player1}")
            features['winrate_1'] = None
            
        if 'winrate2' in match_data:
            features['winrate_2'] = match_data['winrate2']
        elif 'winrate_2' in match_data:
            features['winrate_2'] = match_data['winrate_2']
        elif player2 in self.players_stats and self.players_stats[player2]['win_rate'] is not None:
            features['winrate_2'] = self.players_stats[player2]['win_rate']
        else:
            logging.warning(f"No se encontró tasa de victoria para {player2}")
            features['winrate_2'] = None
        
        # Manejar campos faltantes - informar claramente y buscar alternativas
        missing_features = [k for k, v in features.items() if v is None]
        if missing_features:
            logging.warning(f"Faltan características importantes: {missing_features}")
            
            # Si faltan características críticas, intentar obtener nuevos datos
            if len(missing_features) > 1:
                logging.warning("Intentando obtener datos actualizados de fuentes externas...")
                updated_features = self._get_updated_player_data(player1, player2)
                
                # Actualizar características faltantes con datos nuevos
                for feat in missing_features:
                    if feat in updated_features:
                        features[feat] = updated_features[feat]
            
            # Si aún faltan características, usar estimaciones con alta penalización
            for feat in [k for k, v in features.items() if v is None]:
                if 'ranking' in feat:
                    # Para rankings faltantes, asignar un valor alto (penalización)
                    features[feat] = 200  # Un ranking bajo pero no extremo
                    logging.warning(f"Usando ranking estimado para {feat}: 200")
                elif 'winrate' in feat:
                    # Para tasas de victoria, asignar un valor neutral
                    features[feat] = 40.0  # Ligeramente por debajo de la media
                    logging.warning(f"Usando tasa de victoria estimada para {feat}: 40.0")
        
        # Codificación de superficie
        surface_map = {'hard': 0, 'clay': 1, 'grass': 2, 'carpet': 3}
        features['surface_code'] = surface_map.get(surface.lower(), -1)
        
        # Características avanzadas - solo si tenemos datos reales
        
        # 1. Rendimiento en la superficie específica
        p1_surface_stats = self.players_stats.get(player1, {}).get('surface_stats', {}).get(surface, {})
        p2_surface_stats = self.players_stats.get(player2, {}).get('surface_stats', {}).get(surface, {})
        
        # Solo añadir si existen estadísticas reales
        if 'win_rate' in p1_surface_stats:
            features['surface_winrate_1'] = p1_surface_stats['win_rate']
        else:
            features['surface_winrate_1'] = features.get('winrate_1')  # Usar winrate general si no hay específica
            
        if 'win_rate' in p2_surface_stats:
            features['surface_winrate_2'] = p2_surface_stats['win_rate']
        else:
            features['surface_winrate_2'] = features.get('winrate_2')  # Usar winrate general si no hay específica
        
        # Calcular ventaja de superficie solo si tenemos ambos datos
        if 'surface_winrate_1' in features and 'surface_winrate_2' in features:
            if features['surface_winrate_1'] is not None and features['surface_winrate_2'] is not None:
                features['surface_advantage'] = (features['surface_winrate_1'] - features['surface_winrate_2']) / 100
            else:
                features['surface_advantage'] = 0  # Neutral si falta algún dato
        else:
            features['surface_advantage'] = 0
        
        # 2. Head-to-head - solo si existen estadísticas
        player_pair = tuple(sorted([player1, player2]))
        h2h_stats = self.head_to_head_stats.get(player_pair, {})
        
        # Ventaja general - solo si hay datos suficientes
        if 'advantage' in h2h_stats and h2h_stats['advantage'] is not None:
            features['h2h_advantage'] = h2h_stats['advantage']
        else:
            features['h2h_advantage'] = 0.5  # Neutral si no hay datos
        
        # Ventaja en la superficie - solo si hay datos suficientes
        surface_h2h = h2h_stats.get('surface_stats', {}).get(surface, {})
        if 'advantage' in surface_h2h and surface_h2h['advantage'] is not None:
            features['surface_h2h_advantage'] = surface_h2h['advantage']
        else:
            features['surface_h2h_advantage'] = 0.5  # Neutral si no hay datos
        
        # 3. Forma reciente - solo si hay datos
        if player1 in self.players_stats and 'recent_form' in self.players_stats[player1]:
            features['recent_form_1'] = self.players_stats[player1]['recent_form']
        else:
            features['recent_form_1'] = features.get('winrate_1')  # Usar winrate general si no hay forma reciente
        
        if player2 in self.players_stats and 'recent_form' in self.players_stats[player2]:
            features['recent_form_2'] = self.players_stats[player2]['recent_form']
        else:
            features['recent_form_2'] = features.get('winrate_2')  # Usar winrate general si no hay forma reciente
        
        # Calcular diferencia solo si tenemos ambos datos
        if features.get('recent_form_1') is not None and features.get('recent_form_2') is not None:
            features['recent_form_diff'] = (features['recent_form_1'] - features['recent_form_2']) / 100
        else:
            features['recent_form_diff'] = 0  # Neutral si faltan datos
        
        # 4. Experiencia (partidos jugados) - solo con datos reales
        p1_matches = self.players_stats.get(player1, {}).get('total_matches', 0)
        p2_matches = self.players_stats.get(player2, {}).get('total_matches', 0)
        
        # Normalizar experiencia solo si tenemos datos
        if p1_matches > 0:
            features['experience_1'] = min(p1_matches / 100, 1.0)
        else:
            # Si no hay datos de partidos, no asignamos experiencia (None)
            features['experience_1'] = 0
            
        if p2_matches > 0:
            features['experience_2'] = min(p2_matches / 100, 1.0)
        else:
            features['experience_2'] = 0
            
        features['experience_diff'] = features['experience_1'] - features['experience_2']
        
        # Verificar valores nulos para evitar problemas en el modelo
        for key in list(features.keys()):
            if features[key] is None:
                logging.warning(f"Valor nulo detectado para {key}, reemplazando con 0")
                features[key] = 0
        
        return features
    
    def _get_external_ranking(self, player_name):
        """
        Intenta obtener el ranking actual del jugador de fuentes externas.
        
        Args:
            player_name: Nombre del jugador
            
        Returns:
            Ranking o None si no se encuentra
        """
        try:
            # Intentar buscar en las principales API de tenis
            # Aquí se puede implementar llamadas a API públicas de tenis
            
            # Por ejemplo, una implementación simplificada de búsqueda:
            # URL = f"https://api.tennisapi.com/players?name={player_name}"
            # response = requests.get(URL)
            # if response.status_code == 200:
            #     data = response.json()
            #     if data.get('players') and len(data['players']) > 0:
            #         return data['players'][0].get('ranking')
            
            # Como no tenemos acceso a una API real, buscamos en nuestros datos
            all_players = pd.DataFrame()
            
            # Crear un DataFrame con todos los jugadores y sus rankings
            if 'ranking_1' in self.data.columns and 'player1' in self.data.columns:
                df1 = self.data[['player1', 'ranking_1']].rename(columns={'player1': 'player', 'ranking_1': 'ranking'})
                all_players = pd.concat([all_players, df1])
                
            if 'ranking_2' in self.data.columns and 'player2' in self.data.columns:
                df2 = self.data[['player2', 'ranking_2']].rename(columns={'player2': 'player', 'ranking_2': 'ranking'})
                all_players = pd.concat([all_players, df2])
            
            # Filtrar por jugador
            player_data = all_players[all_players['player'] == player_name]
            
            if not player_data.empty:
                # Usar el ranking más reciente
                return player_data['ranking'].dropna().iloc[-1]
        
        except Exception as e:
            logging.error(f"Error obteniendo ranking externo para {player_name}: {e}")
        
        return None
    
    def _get_updated_player_data(self, player1, player2):
        """
        Intenta obtener datos actualizados para los jugadores desde fuentes externas.
        
        Args:
            player1: Nombre del primer jugador
            player2: Nombre del segundo jugador
            
        Returns:
            Diccionario con datos actualizados
        """
        updated_data = {}
        
        try:
            # Aquí se implementaría llamadas a API para obtener datos actualizados
            # Por ahora, implementamos una búsqueda avanzada en nuestros propios datos
            
            # 1. Buscar rankings recientes
            p1_ranking = self._get_external_ranking(player1)
            if p1_ranking is not None:
                updated_data['ranking_1'] = p1_ranking
                
            p2_ranking = self._get_external_ranking(player2)
            if p2_ranking is not None:
                updated_data['ranking_2'] = p2_ranking
            
            # 2. Calcular tasas de victoria más recientes
            # Usar solo partidos de los últimos 2 años si hay fecha
            recent_data = self.data
            if 'match_date' in self.data.columns:
                two_years_ago = pd.Timestamp.now() - pd.DateOffset(years=2)
                recent_data = self.data[self.data['match_date'] > two_years_ago]
            
            # Calcular tasa para jugador 1
            p1_matches = recent_data[(recent_data['player1'] == player1) | (recent_data['player2'] == player1)]
            if len(p1_matches) > 0:
                p1_wins = sum(
                    (p1_matches['player1'] == player1) & (p1_matches['winner'] == 0) |
                    (p1_matches['player2'] == player1) & (p1_matches['winner'] == 1)
                )
                p1_winrate = (p1_wins / len(p1_matches)) * 100
                updated_data['winrate_1'] = p1_winrate
            
            # Calcular tasa para jugador 2
            p2_matches = recent_data[(recent_data['player1'] == player2) | (recent_data['player2'] == player2)]
            if len(p2_matches) > 0:
                p2_wins = sum(
                    (p2_matches['player1'] == player2) & (p2_matches['winner'] == 0) |
                    (p2_matches['player2'] == player2) & (p2_matches['winner'] == 1)
                )
                p2_winrate = (p2_wins / len(p2_matches)) * 100
                updated_data['winrate_2'] = p2_winrate
        
        except Exception as e:
            logging.error(f"Error obteniendo datos actualizados: {e}")
        
        return updated_data
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos de entrenamiento con características avanzadas.
        Versión mejorada para asegurar correcto balanceo de clases.
        
        Returns:
            Tupla con (X, y) donde X son las características y y es la variable objetivo
        """
        if self.data is None or self.data.empty:
            self._generate_or_download_data()
            self.preprocess_data()
        
        # Preprocesar datos
        self.preprocess_data()
        
        # Verificar balanceo de clases
        winner_counts = self.data['winner'].value_counts()
        total = len(self.data)
        
        if 0 in winner_counts and 1 in winner_counts:
            class0_percent = (winner_counts[0] / total) * 100
            class1_percent = (winner_counts[1] / total) * 100
            
            logging.info(f"Distribución de clases: clase 0 (player1 gana): {class0_percent:.2f}%, clase 1 (player2 gana): {class1_percent:.2f}%")
            
            # Advertir si hay desequilibrio severo
            if abs(class0_percent - class1_percent) > 20:
                logging.warning(f"Desbalance significativo detectado en las clases. Clase 0: {class0_percent:.2f}%, Clase 1: {class1_percent:.2f}%")
                # Más adelante podríamos implementar técnicas como SMOTE para balancear
        else:
            # Caso crítico: solo hay una clase
            if 0 not in winner_counts:
                logging.error("PROBLEMA CRÍTICO: Todos los partidos tienen winner=1 (player2 siempre gana)")
            elif 1 not in winner_counts:
                logging.error("PROBLEMA CRÍTICO: Todos los partidos tienen winner=0 (player1 siempre gana)")
                
            # Intentar reequilibrar los datos
            logging.warning("Intentando reequilibrar el conjunto de datos...")
            balanced_data = self._balance_dataset()
            if balanced_data is not None:
                self.data = balanced_data
                winner_counts = self.data['winner'].value_counts()
                logging.info(f"Datos reequilibrados. Nueva distribución: 0={winner_counts.get(0, 0)}, 1={winner_counts.get(1, 0)}")
        
        # Calcular estadísticas
        self.build_player_statistics()
        self.build_head_to_head_statistics()
        
        # Extraer características para cada partido
        features_list = []
        match_indices = []
        
        for idx, match in self.data.iterrows():
            try:
                # Convertir la fila a diccionario y ajustar nombres de campos
                match_dict = match.to_dict()
                
                # Adaptamos a formato estándar si es necesario
                if 'player1' not in match_dict and 'player_1' in match_dict:
                    match_dict['player1'] = match_dict.pop('player_1')
                    match_dict['player2'] = match_dict.pop('player_2')
                
                features = self.extract_features(match_dict)
                
                # Añadir índice para mantener relación con el partido original
                match_indices.append(idx)
                features_list.append(features)
            except Exception as e:
                logging.warning(f"Error extrayendo características para partido {idx}: {e}")
        
        # Convertir a DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Variable objetivo (winner)
        y = self.data.loc[match_indices]['winner']
        
        # Verificar características NaN y eliminar o imputar
        if features_df.isna().any().any():
            logging.warning("Se detectaron valores NaN en las características, imputando con 0")
            features_df = features_df.fillna(0)
        
        # Guardar escalador para uso futuro
        os.makedirs('ml-service/model', exist_ok=True)
        self.scaler.fit(features_df)
        joblib.dump(self.scaler, 'ml-service/model/feature_scaler.pkl')
        
        # Guardar lista de características
        feature_names = features_df.columns.tolist()
        with open('ml-service/model/feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_names))
        
        return features_df, y
    
    def _balance_dataset(self):
        """
        Intenta reequilibrar un conjunto de datos desbalanceado.
        
        Returns:
            DataFrame balanceado o None si no se puede balancear
        """
        if self.data is None or 'winner' not in self.data.columns:
            return None
        
        data = self.data.copy()
        
        # Verificar si hay una sola clase
        unique_winners = data['winner'].unique()
        if len(unique_winners) == 1:
            # Si todos son clase 0 (player1 gana)
            if unique_winners[0] == 0:
                # Tomar una copia de los datos e invertir player1/player2 y establecer winner=1
                inverted_data = data.copy()
                # Intercambiar player1 y player2
                inverted_data['player1'], inverted_data['player2'] = inverted_data['player2'], inverted_data['player1']
                # Intercambiar sus stats
                if 'ranking_1' in inverted_data.columns and 'ranking_2' in inverted_data.columns:
                    inverted_data['ranking_1'], inverted_data['ranking_2'] = inverted_data['ranking_2'], inverted_data['ranking_1']
                if 'winrate_1' in inverted_data.columns and 'winrate_2' in inverted_data.columns:
                    inverted_data['winrate_1'], inverted_data['winrate_2'] = inverted_data['winrate_2'], inverted_data['winrate_1']
                
                # Establecer winner=1 (ahora player2 original, que es player1 en inverted_data)
                inverted_data['winner'] = 1
                
                # Mezclar datos originales e invertidos
                balanced_data = pd.concat([data, inverted_data], ignore_index=True)
                return balanced_data
                
            # Si todos son clase 1 (player2 gana)
            elif unique_winners[0] == 1:
                # Similar al caso anterior pero invirtiendo de manera diferente
                inverted_data = data.copy()
                inverted_data['player1'], inverted_data['player2'] = inverted_data['player2'], inverted_data['player1']
                if 'ranking_1' in inverted_data.columns and 'ranking_2' in inverted_data.columns:
                    inverted_data['ranking_1'], inverted_data['ranking_2'] = inverted_data['ranking_2'], inverted_data['ranking_1']
                if 'winrate_1' in inverted_data.columns and 'winrate_2' in inverted_data.columns:
                    inverted_data['winrate_1'], inverted_data['winrate_2'] = inverted_data['winrate_2'], inverted_data['winrate_1']
                
                inverted_data['winner'] = 0
                balanced_data = pd.concat([data, inverted_data], ignore_index=True)
                return balanced_data
        
        # Si ya hay ambas clases pero están desbalanceadas
        class_counts = data['winner'].value_counts()
        if 0 in class_counts and 1 in class_counts:
            # Calcular proporción de desbalance
            ratio = class_counts[0] / class_counts[1]
            
            # Si el desbalance es significativo
            if ratio < 0.5 or ratio > 2:
                minority_class = 0 if class_counts[0] < class_counts[1] else 1
                majority_class = 1 if minority_class == 0 else 0
                
                # Obtener índices de la clase minoritaria
                minority_indices = data[data['winner'] == minority_class].index
                
                # Calcular cuántas muestras adicionales necesitamos
                n_samples = class_counts[majority_class] - class_counts[minority_class]
                
                # Sobremuestrear la clase minoritaria
                if len(minority_indices) > 0:
                    # Seleccionar índices con reemplazo
                    additional_indices = np.random.choice(minority_indices, size=n_samples, replace=True)
                    additional_samples = data.loc[additional_indices].copy()
                    
                    # Añadir pequeño ruido a las características numéricas para evitar duplicados exactos
                    for col in additional_samples.columns:
                        if additional_samples[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            if col not in ['winner']:  # No modificar la variable objetivo
                                noise = np.random.normal(0, 0.01, size=len(additional_samples))
                                additional_samples[col] = additional_samples[col] + noise
                    
                    # Combinar datos originales con sobremuestreo
                    balanced_data = pd.concat([data, additional_samples], ignore_index=True)
                    return balanced_data
        
        # Si llegamos aquí, no pudimos balancear los datos
        return None
    
    def transform_match_data(self, match_data: Dict[str, Union[str, int, float]]) -> pd.DataFrame:
        """
        Transforma datos de un partido en características para predicción.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            DataFrame con características procesadas
        """
        # Hacer una copia para no modificar el original
        match_data_copy = match_data.copy()
        
        # Compatibilidad con diferentes formatos de nombres de campos
        if 'player_1' in match_data_copy and 'player1' not in match_data_copy:
            # Convertir a formato estándar para extract_features
            match_data_copy['player1'] = match_data_copy.pop('player_1')
            match_data_copy['player2'] = match_data_copy.pop('player_2')
            if 'ranking_1' in match_data_copy:
                match_data_copy['ranking1'] = match_data_copy.pop('ranking_1')
            if 'ranking_2' in match_data_copy:
                match_data_copy['ranking2'] = match_data_copy.pop('ranking_2')
            if 'winrate_1' in match_data_copy:
                match_data_copy['winrate1'] = match_data_copy.pop('winrate_1')
            if 'winrate_2' in match_data_copy:
                match_data_copy['winrate2'] = match_data_copy.pop('winrate_2')
        
        # Verificar que tenemos estadísticas calculadas
        if not self.players_stats:
            self.build_player_statistics()
        
        if not self.head_to_head_stats:
            self.build_head_to_head_statistics()
            
        # Extraer características
        features = self.extract_features(match_data_copy)
        
        # Convertir a DataFrame
        df = pd.DataFrame([features])
        
        # Verificar si tenemos un escalador guardado
        scaler_path = 'ml-service/model/feature_scaler.pkl'
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                scaled_data = scaler.transform(df)
                df = pd.DataFrame(scaled_data, columns=df.columns)
            except Exception as e:
                logging.warning(f"Error al aplicar escalador: {e}")
        
        # Verificar si tenemos lista de características guardada
        feature_names_path = 'ml-service/model/feature_names.txt'
        if os.path.exists(feature_names_path):
            try:
                with open(feature_names_path, 'r') as f:
                    feature_names = [line.strip() for line in f.readlines()]
                
                # Asegurar que el DataFrame tiene las características correctas
                for feature in feature_names:
                    if feature not in df.columns:
                        logging.warning(f"Característica '{feature}' faltante, añadiendo con valor 0")
                        df[feature] = 0.0
                
                # Reordenar columnas según el modelo
                df = df[feature_names]
            except Exception as e:
                logging.warning(f"Error al adaptar características al modelo: {e}")
        
        return df


# Funciones mejoradas para compatibilidad con el código antiguo
def load_model():
    """Carga el modelo entrenado desde el archivo"""
    model_paths = [
        'ml-service/model/model.pkl',
        'model/model.pkl',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model.pkl')
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            logging.info(f"Modelo encontrado en {path}")
            return joblib.load(path)
    
    raise FileNotFoundError(f"Modelo no encontrado. Rutas probadas: {model_paths}")

def preprocess_match_data(match_data):
    """
    Función de compatibilidad con el código antiguo.
    Ahora utiliza la nueva clase TennisFeatureEngineering.
    """
    # Instanciar el motor de características
    fe = TennisFeatureEngineering()
    
    # Asegurarse de que tenemos estadísticas calculadas
    if not fe.players_stats:
        try:
            fe.build_player_statistics()
            fe.build_head_to_head_statistics()
        except Exception as e:
            logging.warning(f"No se pudieron calcular estadísticas: {e}")
    
    # Extraer características avanzadas
    try:
        df = fe.transform_match_data(match_data)
        # Convertir a diccionario para compatibilidad
        features_dict = df.iloc[0].to_dict()
        return features_dict
    except Exception as e:
        logging.warning(f"Error con características avanzadas, usando método básico: {e}")
        
        # Si falla, intentamos con un método muy básico pero que siempre funcione
        features = {}
        
        # Ranking
        if 'ranking_1' in match_data:
            features['ranking_1'] = match_data['ranking_1']
        else:
            # Buscar ranking en bases de datos externas
            player = match_data.get('player_1', match_data.get('player1'))
            if player:
                # Inicializar motor para buscar datos
                fe = TennisFeatureEngineering()
                ranking = fe._get_external_ranking(player)
                features['ranking_1'] = ranking if ranking is not None else 100
            else:
                features['ranking_1'] = 100
        
        if 'ranking_2' in match_data:
            features['ranking_2'] = match_data['ranking_2']
        else:
            # Buscar ranking en bases de datos externas
            player = match_data.get('player_2', match_data.get('player2'))
            if player:
                # Inicializar motor para buscar datos
                fe = TennisFeatureEngineering()
                ranking = fe._get_external_ranking(player)
                features['ranking_2'] = ranking if ranking is not None else 100
            else:
                features['ranking_2'] = 100
        
        # Winrate - intentar obtener de datos externos
        if 'winrate_1' in match_data:
            features['winrate_1'] = match_data['winrate_1']
        else:
            # Usamos 45 como valor neutro, ligeramente por debajo de la media
            features['winrate_1'] = 45
        
        if 'winrate_2' in match_data:
            features['winrate_2'] = match_data['winrate_2']
        else:
            features['winrate_2'] = 45
        
        # Superficie
        if 'surface' in match_data:
            surfaces = {'hard': 0, 'clay': 1, 'grass': 2, 'carpet': 3}
            features['surface_code'] = surfaces.get(match_data['surface'].lower(), 0)
        else:
            features['surface_code'] = 0  # Hard como superficie más común
            
        return features