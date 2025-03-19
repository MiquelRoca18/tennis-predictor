#!/usr/bin/env python3
"""
Script mejorado para recopilar datos históricos de partidos de tenis
de diversas fuentes y almacenarlos para entrenamiento.
Eliminación de valores por defecto y mejora en la obtención de datos de fuentes externas.
"""

import requests
import pandas as pd
import numpy as np
import os
import logging
import time
from datetime import datetime, timedelta
import concurrent.futures
import traceback
from tqdm import tqdm

# Configurar logging
log_dir = 'ml-service/logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'data_collection.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

class TennisDataCollector:
    """
    Recopila datos históricos de partidos de tenis de diversas fuentes.
    Implementación mejorada para priorizar datos externos y eliminar valores por defecto.
    """
    
    def __init__(self, output_path='tennis_matches.csv'):
        """
        Inicializa el recolector de datos.
        
        Args:
            output_path: Ruta donde se guardará el CSV de datos
        """
        self.output_path = output_path
        self.data = None
        
        # Fuentes de datos ampliadas
        self.sources = {
            'github_jeff_sackmann': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master',
            'github_jeff_sackmann_wta': 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master',
            'ultimatetennisstatistics': 'https://www.ultimatetennisstatistics.com/matchesTable',
            # Añadir más fuentes aquí según necesites
        }
        
        # Estructura de campos para cada fuente
        self.source_mappings = {
            'github_jeff_sackmann': {
                'player1': 'winner_name',
                'player2': 'loser_name',
                'ranking_1': 'winner_rank',
                'ranking_2': 'loser_rank',
                'surface': 'surface',
                'date': 'tourney_date',
                'tournament': 'tourney_name',
                'winner_column': None  # El ganador es siempre player1
            },
            # Añadir mapeos para otras fuentes aquí
        }
        
        # Límites de intentos y reintentos
        self.max_retries = 3
        self.retry_delay = 2  # segundos
    
    def collect_data(self, start_year=2015, end_year=None, tours=None):
        """
        Recopila datos históricos de múltiples fuentes.
        
        Args:
            start_year: Año de inicio para la recopilación
            end_year: Año final (por defecto, año actual)
            tours: Lista de tours a incluir ('atp', 'wta')
            
        Returns:
            DataFrame con datos recopilados
        """
        if end_year is None:
            end_year = datetime.now().year
        
        if tours is None:
            tours = ['atp', 'wta']  # Incluir ambos tours por defecto
            
        logging.info(f"Iniciando recopilación de datos de {start_year} a {end_year} para tours: {', '.join(tours)}")
        print(f"Recopilando datos de tenis desde {start_year} hasta {end_year}...")
        
        all_data = []
        years = range(start_year, end_year + 1)
        
        total_tasks = len(years) * len(tours)
        print(f"Se procesarán {total_tasks} combinaciones de año/tour")
        
        # Usar paralelización para acelerar la recopilación
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            # Crear tareas para cada año y tour
            for year in years:
                for tour in tours:
                    futures.append(
                        executor.submit(self._collect_year_data, year, tour)
                    )
            
            # Procesar resultados con barra de progreso
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures), 
                              desc="Recopilando datos"):
                try:
                    year_data = future.result()
                    if year_data is not None and not year_data.empty:
                        all_data.append(year_data)
                except Exception as e:
                    logging.error(f"Error recopilando datos: {e}")
                    logging.error(traceback.format_exc())
        
        # Combinar todos los datos
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            
            # Validar y limpiar datos
            self._validate_and_clean_data()
            
            # Guardar datos
            os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
            self.data.to_csv(self.output_path, index=False)
            
            logging.info(f"Recopilación completada: {len(self.data)} partidos guardados")
            print(f"Recopilación completada. {len(self.data)} partidos guardados en {self.output_path}")
            
            # Mostrar estadísticas
            self._print_statistics()
            
            return self.data
        else:
            logging.warning("No se pudieron recopilar datos. Generando datos de prueba.")
            print("No se pudieron recopilar datos reales. Generando datos sintéticos...")
            self._generate_test_data(n_samples=500)
            return self.data
    
    def _validate_and_clean_data(self):
        """Valida y limpia los datos recopilados"""
        if self.data is None or self.data.empty:
            return
            
        original_count = len(self.data)
        
        # 1. Verificar columnas esenciales
        essential_columns = ['player1', 'player2', 'surface', 'winner']
        missing_columns = [col for col in essential_columns if col not in self.data.columns]
        
        if missing_columns:
            # Intentar inferir columnas faltantes
            logging.warning(f"Faltan columnas esenciales: {missing_columns}")
            
            if 'winner' in missing_columns and 'player1' in self.data.columns:
                # Asumimos que player1 es el ganador (formato Jeff Sackmann)
                logging.info("Inferiendo la columna 'winner' (0 = player1, 1 = player2)")
                self.data['winner'] = 0  # Por convención, 0 significa que ganó player1
            
            # Verificar nuevamente columnas faltantes
            missing_columns = [col for col in essential_columns if col not in self.data.columns]
            if missing_columns:
                logging.error(f"No se pueden inferir columnas faltantes: {missing_columns}")
                raise ValueError(f"Datos incompletos. Faltan columnas: {missing_columns}")
        
        # 2. Limpiar valores nulos
        null_counts = self.data.isnull().sum()
        if null_counts.sum() > 0:
            logging.warning(f"Se encontraron valores nulos en los datos:\n{null_counts[null_counts > 0]}")
            
            # Eliminar filas con valores nulos en columnas esenciales
            self.data = self.data.dropna(subset=essential_columns)
            
            # Para otras columnas, podemos manejarlas de manera diferente
            # Rankings: usamos NaN en lugar de valores por defecto
            # Fechas: convertimos a datetime y manejamos formatos
            if 'match_date' in self.data.columns:
                self.data['match_date'] = pd.to_datetime(self.data['match_date'], errors='coerce')
        
        # 3. Normalizar superficies
        if 'surface' in self.data.columns:
            surfaces_before = self.data['surface'].value_counts()
            
            # Normalizar a minúsculas
            self.data['surface'] = self.data['surface'].str.lower()
            
            # Mapear variantes a nombres estándar
            surface_mapping = {
                'hard': 'hard', 'h': 'hard', 'indoor': 'hard', 'indoor hard': 'hard', 'outdoor hard': 'hard',
                'clay': 'clay', 'c': 'clay', 'red clay': 'clay', 'green clay': 'clay',
                'grass': 'grass', 'g': 'grass',
                'carpet': 'carpet', 'indoor carpet': 'carpet'
            }
            
            # Aplicar mapeo solo donde coincida, mantener original si no hay coincidencia
            self.data['surface'] = self.data['surface'].map(lambda x: surface_mapping.get(x, x))
            
            surfaces_after = self.data['surface'].value_counts()
            logging.info(f"Normalización de superficies: {surfaces_before.to_dict()} -> {surfaces_after.to_dict()}")
        
        # 4. Verificar valores duplicados
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            logging.warning(f"Se encontraron {duplicates} registros duplicados, eliminando...")
            self.data = self.data.drop_duplicates()
        
        # 5. Verificar consistencia de ganadores
        if 'winner' in self.data.columns:
            invalid_winners = self.data[~self.data['winner'].isin([0, 1])]
            if not invalid_winners.empty:
                logging.warning(f"Se encontraron {len(invalid_winners)} ganadores inválidos, corrigiendo...")
                # Corregir valores inválidos (asumiendo player1 como ganador)
                self.data.loc[~self.data['winner'].isin([0, 1]), 'winner'] = 0
        
        # Resumen de cambios
        final_count = len(self.data)
        if final_count < original_count:
            logging.info(f"Limpieza de datos: {original_count} -> {final_count} registros")
    
    def _print_statistics(self):
        """Muestra estadísticas de los datos recopilados"""
        if self.data is None or self.data.empty:
            print("No hay datos para mostrar estadísticas")
            return
            
        print("\n--- ESTADÍSTICAS DE LOS DATOS RECOPILADOS ---")
        
        # Número total de partidos
        print(f"Total de partidos: {len(self.data)}")
        
        # Distribución por superficie
        if 'surface' in self.data.columns:
            print("\nDistribución por superficie:")
            surface_counts = self.data['surface'].value_counts()
            for surface, count in surface_counts.items():
                print(f"  {surface}: {count} partidos ({count/len(self.data)*100:.1f}%)")
        
        # Distribución por año
        if 'match_date' in self.data.columns:
            self.data['year'] = pd.to_datetime(self.data['match_date']).dt.year
            year_counts = self.data['year'].value_counts().sort_index()
            
            print("\nDistribución por año:")
            for year, count in year_counts.items():
                print(f"  {year}: {count} partidos")
        
        # Jugadores más frecuentes
        print("\nJugadores más frecuentes:")
        all_players = pd.Series(list(self.data['player1']) + list(self.data['player2']))
        top_players = all_players.value_counts().head(10)
        
        for player, count in top_players.items():
            print(f"  {player}: {count} apariciones")
        
        # Distribución de ganadores
        if 'winner' in self.data.columns:
            print("\nDistribución de ganadores:")
            winner_counts = self.data['winner'].value_counts()
            p1_wins = winner_counts.get(0, 0)
            p2_wins = winner_counts.get(1, 0)
            
            print(f"  Player1 (ganador): {p1_wins} partidos ({p1_wins/len(self.data)*100:.1f}%)")
            print(f"  Player2 (ganador): {p2_wins} partidos ({p2_wins/len(self.data)*100:.1f}%)")
        
        print("\n--- FIN DE ESTADÍSTICAS ---")

    def _collect_year_data(self, year, tour):
        """
        Recopila datos para un año y tour específicos usando múltiples fuentes.
        
        Args:
            year: Año a recopilar
            tour: Tour (atp, wta)
            
        Returns:
            DataFrame con datos del año
        """
        # Determinar qué funciones de recopilación usar según el tour
        if tour.lower() == 'atp':
            return self._collect_atp_data(year)
        elif tour.lower() == 'wta':
            return self._collect_wta_data(year)
        else:
            logging.warning(f"Tour no soportado: {tour}")
            return None
    
    def _collect_atp_data(self, year):
        """
        Recopila datos de ATP para un año específico desde múltiples fuentes.
        
        Args:
            year: Año a recopilar
            
        Returns:
            DataFrame con datos ATP del año
        """
        # Intentamos primero la fuente de GitHub
        df_github = self._collect_from_github(year, 'atp')
        
        # Si no obtenemos datos de GitHub, intentamos otras fuentes
        if df_github is not None and not df_github.empty:
            logging.info(f"Datos ATP {year} recopilados de GitHub: {len(df_github)} partidos")
            return df_github
        
        # Intentar otras fuentes si GitHub falla
        # 1. UltimateTennisStatistics
        df_uts = self._collect_from_ultimate_tennis(year, 'atp')
        if df_uts is not None and not df_uts.empty:
            logging.info(f"Datos ATP {year} recopilados de UltimateTennisStatistics: {len(df_uts)} partidos")
            return df_uts
        
        # 2. Otros APIs (aquí podrías implementar más fuentes)
        
        # Si todas las fuentes fallan, registramos el error
        logging.error(f"No se pudieron obtener datos ATP para {year} de ninguna fuente")
        return None
    
    def _collect_wta_data(self, year):
        """
        Recopila datos de WTA para un año específico desde múltiples fuentes.
        
        Args:
            year: Año a recopilar
            
        Returns:
            DataFrame con datos WTA del año
        """
        # Implementar la recopilación de datos WTA similar a ATP
        # Intentamos primero la fuente de GitHub
        df_github = self._collect_from_github(year, 'wta')
        
        if df_github is not None and not df_github.empty:
            logging.info(f"Datos WTA {year} recopilados de GitHub: {len(df_github)} partidos")
            return df_github
        
        # Intentar otras fuentes si es necesario
        
        logging.warning(f"No se pudieron obtener datos WTA para {year}")
        return None
    
    def _collect_from_github(self, year, tour):
        """
        Recopila datos de la fuente de GitHub de Jeff Sackmann.
        
        Args:
            year: Año a recopilar
            tour: Tour (atp o wta)
            
        Returns:
            DataFrame con datos transformados
        """
        source_key = 'github_jeff_sackmann' if tour == 'atp' else 'github_jeff_sackmann_wta'
        base_url = self.sources.get(source_key)
        
        if not base_url:
            logging.error(f"Fuente no configurada: {source_key}")
            return None
        
        url = f"{base_url}/{tour}_matches_{year}.csv"
        
        # Intentar descargar con reintentos
        for attempt in range(self.max_retries):
            try:
                logging.info(f"Descargando datos de {url} (intento {attempt+1}/{self.max_retries})")
                
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Guardar temporalmente y cargar con pandas para mejor manejo de errores
                    temp_file = f"temp_{tour}_{year}.csv"
                    with open(temp_file, 'wb') as f:
                        f.write(response.content)
                    
                    data = pd.read_csv(temp_file)
                    os.remove(temp_file)  # Eliminar archivo temporal
                    
                    # Transformar al formato estándar
                    transformed_data = self._transform_data(data, source_key)
                    
                    return transformed_data
                elif response.status_code == 404:
                    logging.warning(f"Datos para {tour} {year} no encontrados (404)")
                    return None
                else:
                    logging.warning(f"Error descargando datos: Código HTTP {response.status_code}")
            
            except Exception as e:
                logging.error(f"Error recopilando datos {tour} {year}: {e}")
                logging.error(traceback.format_exc())
                
                if attempt < self.max_retries - 1:
                    logging.info(f"Reintentando en {self.retry_delay} segundos...")
                    time.sleep(self.retry_delay)
        
        return None
    
    def _collect_from_ultimate_tennis(self, year, tour):
        """
        Recopila datos de UltimateTennisStatistics.
        
        Args:
            year: Año a recopilar
            tour: Tour (atp o wta)
            
        Returns:
            DataFrame con datos transformados
        """
        # Implementación para UltimateTennisStatistics (si necesitas usarla)
        # Este es solo un placeholder, necesitarías implementar la lógica real
        logging.info(f"Intentando obtener datos de UltimateTennisStatistics para {tour} {year}")
        
        # Esta fuente probablemente requiera web scraping
        return None
    
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
            
            # Crear nuevo DataFrame con columnas estándar
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
            
            # Determinar ganador
            if mapping['winner_column'] and mapping['winner_column'] in data.columns:
                # Si hay columna específica de ganador, usarla
                winner_field = mapping['winner_column']
                if winner_field in data.columns:
                    # Mapear valor de la columna ganador al formato estándar (0/1)
                    # Esto dependerá de cada fuente
                    pass
            else:
                # Por defecto para JeffSackmann, el ganador es player1
                transformed['winner'] = 0
            
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
    
    def _calculate_win_rates(self, data):
        """
        Calcula tasas de victoria para cada jugador basado en los datos.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            DataFrame con tasas de victoria añadidas
        """
        if data is None or data.empty or 'player1' not in data.columns:
            return data
            
        try:
            # Crear DataFrame de resultado
            result = data.copy()
            
            # Recopilación de estadísticas de jugadores
            player_stats = {}
            
            # Para cada jugador, recopilar victorias y derrotas
            all_players = set(data['player1'].tolist() + data['player2'].tolist())
            
            for player in all_players:
                # Partidos como player1
                p1_matches = data[data['player1'] == player]
                p1_wins = p1_matches[p1_matches['winner'] == 0].shape[0]
                
                # Partidos como player2
                p2_matches = data[data['player2'] == player]
                p2_wins = p2_matches[p2_matches['winner'] == 1].shape[0]
                
                # Total de partidos y victorias
                total_matches = len(p1_matches) + len(p2_matches)
                total_wins = p1_wins + p2_wins
                
                if total_matches > 0:
                    win_rate = (total_wins / total_matches) * 100
                    player_stats[player] = win_rate
            
            # Aplicar tasas de victoria a los datos
            result['winrate_1'] = result['player1'].map(player_stats)
            result['winrate_2'] = result['player2'].map(player_stats)
            
            return result
            
        except Exception as e:
            logging.error(f"Error calculando tasas de victoria: {e}")
            return data
    
    def _generate_test_data(self, n_samples=200):
        """
        Genera datos de prueba realistas cuando no se pueden recopilar datos reales.
        
        Args:
            n_samples: Número de partidos a generar
            
        Returns:
            DataFrame con datos generados
        """
        logging.info(f"Generando {n_samples} partidos de prueba")
        print(f"Generando {n_samples} partidos de prueba...")
        
        # Lista de jugadores actuales para datos más realistas
        players = [
            'Alcaraz', 'Djokovic', 'Sinner', 'Medvedev', 'Zverev', 'Rublev', 'Ruud', 'Hurkacz', 
            'Fritz', 'De Minaur', 'Paul', 'Tsitsipas', 'Rune', 'Shelton', 'Dimitrov', 'Khachanov', 
            'Norrie', 'Tiafoe', 'Musetti', 'Auger-Aliassime', 'Jarry', 'Baez', 'Humbert', 'Cerundolo', 
            'Bublik', 'Nadal', 'Navone', 'Etcheverry', 'Nakashima', 'Griekspoor', 'Lehecka', 'Machac',
            'Korda', 'Zhang', 'Michelsen', 'Draper', 'Arnaldi', 'Mannarino', 'Evans', 'Tabilo', 
            'Fils', 'Thompson', 'Nishioka', 'Darderi', 'Mensik', 'Giron', 'Eubanks', 'Shang', 'Rinderknech', 'Thiem'
        ]
        
        # Jugadores WTA para más diversidad
        wta_players = [
            'Swiatek', 'Sabalenka', 'Gauff', 'Rybakina', 'Pegula', 'Jabeur', 'Zheng', 'Vondrousova',
            'Sakkari', 'Krejcikova', 'Kasatkina', 'Haddad Maia', 'Ostapenko', 'Azarenka', 'Keys',
            'Samsonova', 'Kudermetova', 'Alexandrova', 'Mertens', 'Svitolina', 'Collins', 'Navarro'
        ]
        
        # Combinar listas de jugadores
        players = players + wta_players
        
        # Superficies y torneos
        surfaces = ['hard', 'clay', 'grass', 'carpet']
        surface_weights = [0.6, 0.3, 0.08, 0.02]  # Distribución más realista
        
        tournaments = [
            'Australian Open', 'Roland Garros', 'Wimbledon', 'US Open',
            'ATP Finals', 'Indian Wells Masters', 'Miami Open', 'Madrid Masters', 
            'Rome Masters', 'Canada Masters', 'Cincinnati Masters', 'Paris Masters',
            'Monte Carlo Masters', 'Shanghai Masters', 'Barcelona Open', 'Queens Club',
            'Halle Open', 'Vienna Open', 'Dubai Tennis Championships', 'Rio Open'
        ]
        
        # Generar datos más realistas con sesgo
        data = []
        
        # Pre-calcular estadísticas de jugadores para mayor realismo
        player_stats = {}
        for player in players:
            # Asignar ranking basado en el orden de la lista (aproximado)
            ranking = players.index(player) + 1
            
            # Asignar tasa de victoria base
            base_winrate = max(30, 80 - (ranking * 0.5) + np.random.normal(0, 5))
            
            # Especialistas en superficies
            surface_specialty = np.random.choice(surfaces)
            surface_winrates = {
                'hard': base_winrate,
                'clay': base_winrate,
                'grass': base_winrate,
                'carpet': base_winrate
            }
            
            # Aumentar tasa en su superficie preferida
            surface_winrates[surface_specialty] += 10
            
            player_stats[player] = {
                'ranking': ranking,
                'base_winrate': base_winrate,
                'surface_specialty': surface_specialty,
                'surface_winrates': surface_winrates
            }
        
        # Generar partidos
        for _ in range(n_samples):
            # Seleccionar jugadores aleatorios (asegurar que son diferentes)
            p1_idx = np.random.randint(0, len(players))
            p2_idx = np.random.randint(0, len(players))
            while p2_idx == p1_idx:
                p2_idx = np.random.randint(0, len(players))
            
            player1 = players[p1_idx]
            player2 = players[p2_idx]
            
            # Obtener rankings y winrates de las estadísticas pre-calculadas
            ranking_1 = player_stats[player1]['ranking']
            ranking_2 = player_stats[player2]['ranking']
            
            # Seleccionar superficie con pesos realistas
            surface = np.random.choice(surfaces, p=surface_weights)
            
            # Obtener winrates específicas para esta superficie
            winrate_1 = player_stats[player1]['surface_winrates'][surface]
            winrate_2 = player_stats[player2]['surface_winrates'][surface]
            
            # Seleccionar torneo
            tournament = np.random.choice(tournaments)
            
            # Generar fecha aleatoria en los últimos 5 años
            days_back = np.random.randint(0, 365*5)
            match_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Calcular probabilidad de victoria basada en ranking, winrate y superficie
            p1_score = winrate_1 / (ranking_1 + 1)  # +1 para evitar división por cero
            p2_score = winrate_2 / (ranking_2 + 1)
            
            # Añadir algo de aleatoriedad
            p1_score += np.random.normal(0, 0.1)
            p2_score += np.random.normal(0, 0.1)
            
            # Determinar ganador (0 = player1, 1 = player2)
            winner = 0 if p1_score > p2_score else 1
            
            # Añadir efectos especiales para ciertos jugadores conocidos
            if surface == 'clay' and player1 == 'Nadal':
                winner = 0  # Nadal casi siempre gana en tierra
            elif surface == 'clay' and player2 == 'Nadal':
                winner = 1
            elif surface == 'grass' and player1 == 'Djokovic':
                # Aumentar probabilidad de Djokovic en hierba
                if np.random.random() < 0.8:
                    winner = 0
            elif surface == 'grass' and player2 == 'Djokovic':
                if np.random.random() < 0.8:
                    winner = 1
            
            # Crear registro
            data.append({
                'player1': player1,
                'player2': player2,
                'ranking_1': ranking_1,
                'ranking_2': ranking_2,
                'winrate_1': winrate_1,
                'winrate_2': winrate_2,
                'surface': surface,
                'tournament': tournament,
                'match_date': match_date,
                'winner': winner
            })
        
        # Crear DataFrame
        self.data = pd.DataFrame(data)
        
        # Guardar datos generados
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
        self.data.to_csv(self.output_path, index=False)
        logging.info(f"Datos de prueba generados y guardados en {self.output_path}")
        
        return self.data

def main():
    """Función principal para ejecutar la recopilación de datos."""
    print("Iniciando recopilación de datos históricos de tenis...")
    
    collector = TennisDataCollector()
    
    # Parámetros por defecto
    start_year = 2015
    end_year = datetime.now().year
    tours = ['atp', 'wta']  # Incluir ambos tours
    
    # Recopilar datos
    data = collector.collect_data(start_year=start_year, end_year=end_year, tours=tours)
    
    print(f"Recopilación completada: {len(data)} partidos")
    print(f"Datos guardados en: {collector.output_path}")
    
if __name__ == "__main__":
    main()