#!/usr/bin/env python3
"""
Script para recopilar y procesar datos de partidos de tenis desde el año 2000 hasta la actualidad.
"""

import pandas as pd
import numpy as np
import logging
import os
import requests
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

class TennisDataCollector:
    """
    Clase para recopilar y procesar datos de partidos de tenis.
    """
    
    def __init__(self):
        """Inicializa el colector de datos con fuentes desde el año 2000 hasta el presente."""
        # Generar URLs para años 2000 hasta el año actual
        self.sources = {}
        current_year = datetime.now().year
        
        # Añadir datos ATP (masculinos)
        for year in range(2000, current_year + 1):
            self.sources[f'atp_{year}'] = f'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv'
        
        # Añadir datos WTA (femeninos)
        for year in range(2000, current_year + 1):
            self.sources[f'wta_{year}'] = f'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv'
        
        # Mapeo de columnas para convertir al formato necesario
        self.column_mapping = {
            'winner_name': 'player_1',
            'loser_name': 'player_2',
            'winner_rank': 'ranking_1',
            'loser_rank': 'ranking_2',
            'surface': 'surface',
            'tourney_name': 'tournament',
            'tourney_date': 'match_date'
        }
        
        logging.info(f"Configurado para recopilar datos desde 2000 hasta {current_year}")
    
    def collect_data(self, output_path: str, max_years: Optional[int] = None, 
                     skip_errors: bool = True) -> Optional[pd.DataFrame]:
        """
        Recopila datos de partidos de tenis de múltiples fuentes.
        
        Args:
            output_path: Ruta donde guardar los datos procesados
            max_years: Número máximo de años a recopilar (None = todos)
            skip_errors: Si es True, continúa con otras fuentes si una falla
            
        Returns:
            DataFrame con datos procesados o None si hay error
        """
        try:
            all_data = []
            processed_count = 0
            total_sources = len(self.sources)
            
            # Limitar fuentes si se especifica max_years
            source_items = list(self.sources.items())
            if max_years is not None:
                # Priorizar años más recientes
                source_items = sorted(source_items, key=lambda x: x[0], reverse=True)
                source_items = source_items[:max_years * 2]  # *2 porque tenemos ATP y WTA
            
            for i, (source_name, url) in enumerate(source_items):
                logging.info(f"Descargando datos de {source_name} ({i+1}/{len(source_items)})")
                
                try:
                    # Verificar si la URL existe antes de intentar descargarla
                    response = requests.head(url)
                    if response.status_code != 200:
                        logging.warning(f"La URL {url} no está disponible (código {response.status_code})")
                        continue
                    
                    # Descargar datos
                    response = requests.get(url)
                    
                    # Guardar en archivo temporal
                    temp_file = f"temp_{source_name}.csv"
                    with open(temp_file, "w") as f:
                        f.write(response.text)
                    
                    # Cargar datos
                    data = pd.read_csv(temp_file)
                    os.remove(temp_file)  # Eliminar archivo temporal
                    
                    # Verificar que tenga las columnas necesarias
                    required_columns = ['winner_name', 'loser_name']
                    if not all(col in data.columns for col in required_columns):
                        missing = [col for col in required_columns if col not in data.columns]
                        logging.warning(f"Datos de {source_name} no tienen columnas requeridas: {missing}")
                        continue
                    
                    # Añadir columnas faltantes si es necesario
                    for col in ['winner_rank', 'loser_rank', 'surface']:
                        if col not in data.columns:
                            data[col] = None
                    
                    # Añadir info de fuente y año
                    data['source'] = source_name
                    year = int(source_name.split('_')[-1])
                    data['year'] = year
                    
                    all_data.append(data)
                    processed_count += 1
                    logging.info(f"Datos de {source_name} cargados: {len(data)} partidos")
                    
                except requests.exceptions.RequestException as e:
                    logging.error(f"Error de conexión con {source_name}: {e}")
                    if not skip_errors:
                        raise
                    continue
                except Exception as e:
                    logging.error(f"Error procesando {source_name}: {e}")
                    if not skip_errors:
                        raise
                    continue
            
            logging.info(f"Procesadas {processed_count} de {total_sources} fuentes")
            
            if not all_data:
                logging.error("No se pudo cargar ninguna fuente de datos")
                return None
            
            # Combinar datos
            logging.info("Combinando datos...")
            combined_data = pd.concat(all_data, ignore_index=True)
            logging.info(f"Total de partidos recopilados inicialmente: {len(combined_data)}")
            
            # Convertir al formato necesario para el modelo
            logging.info("Convirtiendo al formato del modelo...")
            processed_data = self._convert_to_model_format(combined_data)
            
            # Crear directorio de salida si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Guardar datos
            logging.info(f"Guardando {len(processed_data)} partidos en {output_path}...")
            processed_data.to_csv(output_path, index=False)
            logging.info(f"Datos guardados exitosamente")
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error en recopilación de datos: {e}")
            traceback.print_exc()
            return None
    
    def _convert_to_model_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte los datos al formato requerido por el modelo.
        
        Args:
            data: DataFrame con datos originales
            
        Returns:
            DataFrame en formato para entrenamiento
        """
        try:
            # Crear copia de trabajo
            df = data.copy()
            
            # Verificar columna de fecha y convertir si es necesario
            if 'tourney_date' in df.columns:
                # Las fechas en formato YYYYMMDD necesitan conversión
                if df['tourney_date'].dtype == 'int64' or df['tourney_date'].dtype == 'float64':
                    df['tourney_date'] = df['tourney_date'].astype(str)
                    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
                else:
                    df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
            else:
                # Si no existe la columna, crear con fecha actual
                df['tourney_date'] = pd.Timestamp.now()
            
            # Renombrar columnas según mapeo y seleccionar solo las necesarias
            columns_to_keep = []
            rename_dict = {}
            
            for orig_col, new_col in self.column_mapping.items():
                if orig_col in df.columns:
                    rename_dict[orig_col] = new_col
                    columns_to_keep.append(orig_col)
            
            # Solo quedarnos con columnas que existen
            df_subset = df[columns_to_keep].rename(columns=rename_dict)
            
            # Añadir columna winner (0 = player_1 gana, 1 = player_2 gana)
            # En los datos de Jeff Sackmann, player_1 siempre es el ganador
            df_subset['winner'] = 0
            
            # Calcular estadísticas adicionales
            logging.info("Calculando estadísticas de jugadores...")
            player_stats = self._calculate_player_stats(df)
            
            # Aleatorizar la asignación de jugadores para evitar sesgo
            logging.info("Aleatorizando asignación de jugadores...")
            randomized_df = self._randomize_players(df_subset)
            
            # Añadir tasas de victoria si es posible calcularlas
            if player_stats is not None:
                logging.info("Añadiendo tasas de victoria...")
                randomized_df = self._add_win_rates(randomized_df, player_stats)
            
            # Asegurarse de que todas las columnas necesarias existan
            required_columns = ['player_1', 'player_2', 'ranking_1', 'ranking_2', 'surface', 'winner']
            for col in required_columns:
                if col not in randomized_df.columns:
                    if col in ['ranking_1', 'ranking_2']:
                        randomized_df[col] = 100  # Valor predeterminado para rankings
                    elif col == 'surface':
                        randomized_df[col] = 'hard'  # Superficie predeterminada
                    else:
                        raise ValueError(f"Columna requerida faltante: {col}")
            
            # Eliminar filas con valores faltantes en columnas clave
            randomized_df = randomized_df.dropna(subset=['player_1', 'player_2', 'winner'])
            
            # Convertir tipos de datos
            if 'ranking_1' in randomized_df.columns:
                randomized_df['ranking_1'] = pd.to_numeric(randomized_df['ranking_1'], errors='coerce').fillna(100)
            if 'ranking_2' in randomized_df.columns:
                randomized_df['ranking_2'] = pd.to_numeric(randomized_df['ranking_2'], errors='coerce').fillna(100)
            
            # Asegurarse de que winner es 0 o 1
            randomized_df['winner'] = randomized_df['winner'].astype(int)
            
            return randomized_df
            
        except Exception as e:
            logging.error(f"Error convirtiendo datos al formato del modelo: {e}")
            traceback.print_exc()
            raise
    
    def _randomize_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aleatoriza la asignación de jugadores para evitar sesgo.
        En los datos originales, player_1 siempre es el ganador.
        """
        try:
            import random
            randomized = []
            
            for _, row in df.iterrows():
                # Con 50% de probabilidad, intercambiar jugadores
                if random.random() > 0.5:
                    new_row = {
                        'player_1': row['player_1'],
                        'player_2': row['player_2'],
                        'ranking_1': row.get('ranking_1'),
                        'ranking_2': row.get('ranking_2'),
                        'surface': row.get('surface', 'unknown'),
                        'tournament': row.get('tournament'),
                        'match_date': row.get('match_date'),
                        'winner': 0  # player_1 gana
                    }
                else:
                    new_row = {
                        'player_1': row['player_2'],  # Invertido
                        'player_2': row['player_1'],  # Invertido
                        'ranking_1': row.get('ranking_2'),  # Invertido
                        'ranking_2': row.get('ranking_1'),  # Invertido
                        'surface': row.get('surface', 'unknown'),
                        'tournament': row.get('tournament'),
                        'match_date': row.get('match_date'),
                        'winner': 1  # player_2 gana (que era player_1 originalmente)
                    }
                
                randomized.append(new_row)
            
            return pd.DataFrame(randomized)
            
        except Exception as e:
            logging.error(f"Error aleatorizando jugadores: {e}")
            traceback.print_exc()
            # En caso de error, devolver el DataFrame original
            return df
    
    def _calculate_player_stats(self, data: pd.DataFrame) -> Optional[Dict[str, Dict]]:
        """
        Calcula estadísticas generales por jugador.
        
        Args:
            data: DataFrame con datos originales
            
        Returns:
            Diccionario con estadísticas por jugador
        """
        try:
            stats = {}
            # Verificar columnas necesarias
            if not all(col in data.columns for col in ['winner_name', 'loser_name']):
                return None
            
            # Extraer todos los jugadores
            players = set(data['winner_name'].tolist() + data['loser_name'].tolist())
            total_players = len(players)
            
            logging.info(f"Calculando estadísticas para {total_players} jugadores...")
            
            for i, player in enumerate(players):
                if i % 1000 == 0 and i > 0:
                    logging.info(f"Procesados {i}/{total_players} jugadores")
                
                # Partidos ganados
                wins = data[data['winner_name'] == player].shape[0]
                # Partidos perdidos
                losses = data[data['loser_name'] == player].shape[0]
                # Total partidos
                total = wins + losses
                # Tasa de victoria
                winrate = (wins / total * 100) if total > 0 else 50
                
                # Estadísticas por superficie
                surface_stats = {}
                if 'surface' in data.columns:
                    for surface in data['surface'].dropna().unique():
                        surface_wins = data[(data['winner_name'] == player) & 
                                           (data['surface'] == surface)].shape[0]
                        
                        surface_losses = data[(data['loser_name'] == player) & 
                                             (data['surface'] == surface)].shape[0]
                        
                        surface_total = surface_wins + surface_losses
                        surface_winrate = (surface_wins / surface_total * 100) if surface_total > 0 else 50
                        
                        surface_stats[surface] = {
                            'matches': surface_total,
                            'wins': surface_wins,
                            'losses': surface_losses,
                            'winrate': surface_winrate
                        }
                
                # Obtener último ranking conocido
                last_ranking = None
                if 'winner_rank' in data.columns and 'loser_rank' in data.columns and 'tourney_date' in data.columns:
                    # Partidos como ganador, ordenados por fecha
                    wins_df = data[data['winner_name'] == player].sort_values('tourney_date', ascending=False)
                    if not wins_df.empty and pd.notna(wins_df.iloc[0]['winner_rank']):
                        last_ranking = wins_df.iloc[0]['winner_rank']
                    else:
                        # Partidos como perdedor, ordenados por fecha
                        losses_df = data[data['loser_name'] == player].sort_values('tourney_date', ascending=False)
                        if not losses_df.empty and pd.notna(losses_df.iloc[0]['loser_rank']):
                            last_ranking = losses_df.iloc[0]['loser_rank']
                
                # Guardar estadísticas
                stats[player] = {
                    'matches': total,
                    'wins': wins,
                    'losses': losses,
                    'winrate': winrate,
                    'last_ranking': last_ranking,
                    'surface_stats': surface_stats
                }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error calculando estadísticas de jugadores: {e}")
            traceback.print_exc()
            return None
    
    def _add_win_rates(self, df: pd.DataFrame, player_stats: Dict[str, Dict]) -> pd.DataFrame:
        """
        Añade tasas de victoria al DataFrame.
        
        Args:
            df: DataFrame con datos en formato del modelo
            player_stats: Diccionario con estadísticas por jugador
            
        Returns:
            DataFrame con tasas de victoria añadidas
        """
        try:
            result = df.copy()
            
            # Añadir winrate_1
            result['winrate_1'] = result['player_1'].apply(
                lambda p: player_stats.get(p, {}).get('winrate', 50))
            
            # Añadir winrate_2
            result['winrate_2'] = result['player_2'].apply(
                lambda p: player_stats.get(p, {}).get('winrate', 50))
            
            # Añadir tasas específicas por superficie
            if 'surface' in result.columns:
                # Función para obtener winrate por superficie
                def get_surface_winrate(player, surface):
                    if not player or not surface:
                        return 50
                    player_data = player_stats.get(player, {})
                    surface_data = player_data.get('surface_stats', {}).get(surface, {})
                    return surface_data.get('winrate', player_data.get('winrate', 50))
                
                # Añadir columnas
                result['surface_winrate_1'] = result.apply(
                    lambda row: get_surface_winrate(row['player_1'], row['surface']), axis=1)
                result['surface_winrate_2'] = result.apply(
                    lambda row: get_surface_winrate(row['player_2'], row['surface']), axis=1)
                
                # Añadir diferencia
                result['surface_winrate_diff'] = result['surface_winrate_1'] - result['surface_winrate_2']
            
            # Añadir diferencia general de winrate
            result['winrate_diff'] = result['winrate_1'] - result['winrate_2']
            
            return result
            
        except Exception as e:
            logging.error(f"Error añadiendo tasas de victoria: {e}")
            traceback.print_exc()
            # En caso de error, devolver el DataFrame original
            return df

def main():
    """Función principal para recopilar datos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recopilar datos de partidos de tenis')
    parser.add_argument('--output', default='data/tennis_matches.csv',
                      help='Ruta donde guardar los datos procesados')
    parser.add_argument('--years', type=int, default=None,
                      help='Número máximo de años a recopilar (None = todos)')
    parser.add_argument('--sample', type=int, default=None,
                      help='Limitar a una muestra aleatoria de N partidos (None = todos)')
    parser.add_argument('--skip-errors', action='store_true', default=True,
                      help='Continuar si hay errores en una fuente de datos')
    
    args = parser.parse_args()
    
    collector = TennisDataCollector()
    data = collector.collect_data(args.output, args.years, args.skip_errors)
    
    if data is not None:
        # Opcionalmente tomar muestra si se especifica
        if args.sample is not None and args.sample < len(data):
            data = data.sample(n=args.sample, random_state=42)
            # Guardar muestra
            sample_path = args.output.replace('.csv', f'_sample_{args.sample}.csv')
            data.to_csv(sample_path, index=False)
            print(f"Muestra de {args.sample} partidos guardada en {sample_path}")
        
        # Mostrar información sobre los datos recopilados
        print("\n===== RESUMEN DE DATOS RECOPILADOS =====")
        print(f"Total de partidos: {len(data)}")
        
        # Verificar columna de fecha si existe
        if 'match_date' in data.columns:
            try:
                min_date = data['match_date'].min()
                max_date = data['match_date'].max()
                print(f"Rango de fechas: {min_date} a {max_date}")
            except:
                print("No se pudo determinar el rango de fechas")
        
        # Distribución de superficies
        if 'surface' in data.columns:
            surface_counts = data['surface'].value_counts()
            print("\nDistribución de superficies:")
            for surface, count in surface_counts.items():
                print(f"  - {surface}: {count} partidos ({count/len(data)*100:.1f}%)")
        
        # Distribución de ganadores
        winner_counts = data['winner'].value_counts()
        print("\nDistribución de ganadores:")
        print(f"  - Jugador 1 (winner=0): {winner_counts.get(0, 0)} partidos ({winner_counts.get(0, 0)/len(data)*100:.1f}%)")
        print(f"  - Jugador 2 (winner=1): {winner_counts.get(1, 0)} partidos ({winner_counts.get(1, 0)/len(data)*100:.1f}%)")
        
        # Distribución por año
        if 'year' in data.columns:
            year_counts = data['year'].value_counts().sort_index()
            print("\nDistribución por año:")
            for year, count in year_counts.items():
                print(f"  - {year}: {count} partidos")
        
        # Jugadores más frecuentes
        if 'player_1' in data.columns and 'player_2' in data.columns:
            all_players = pd.Series(data['player_1'].tolist() + data['player_2'].tolist())
            top_players = all_players.value_counts().head(10)
            print("\nJugadores más frecuentes:")
            for player, count in top_players.items():
                print(f"  - {player}: {count} partidos")
        
        print(f"\n¡Datos guardados en {args.output} y listos para entrenamiento!")
    else:
        print("Error en la recopilación de datos. Revisa los logs para más detalles.")

if __name__ == '__main__':
    main()