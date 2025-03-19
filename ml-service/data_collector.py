#!/usr/bin/env python3
"""
Script para recopilar y procesar datos de partidos de tenis.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class TennisDataCollector:
    """
    Clase para recopilar y procesar datos de partidos de tenis.
    """
    
    def __init__(self):
        """Inicializa el colector de datos."""
        self.sources = {
            'atp': 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv',
            'wta': 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2024.csv'
        }
        
        self.source_mappings = {
            'atp': {
                'winner_name': 'winner_name',
                'loser_name': 'loser_name',
                'winner_rank': 'winner_rank',
                'loser_rank': 'loser_rank',
                'winner_age': 'winner_age',
                'loser_age': 'loser_age',
                'winner_ht': 'winner_ht',
                'loser_ht': 'loser_ht',
                'surface': 'surface',
                'tournament': 'tournament',
                'match_date': 'tourney_date'
            },
            'wta': {
                'winner_name': 'winner_name',
                'loser_name': 'loser_name',
                'winner_rank': 'winner_rank',
                'loser_rank': 'loser_rank',
                'winner_age': 'winner_age',
                'loser_age': 'loser_age',
                'winner_ht': 'winner_ht',
                'loser_ht': 'loser_ht',
                'surface': 'surface',
                'tournament': 'tournament',
                'match_date': 'tourney_date'
            }
        }
    
    def collect_data(self, output_path: str) -> Optional[pd.DataFrame]:
        """
        Recopila datos de partidos de tenis de múltiples fuentes.
        
        Args:
            output_path: Ruta donde guardar los datos procesados
            
        Returns:
            DataFrame con datos procesados o None si hay error
        """
        try:
            all_data = []
            
            for tour, url in self.sources.items():
                logging.info(f"Recopilando datos de {tour.upper()}")
                
                # Cargar datos
                data = pd.read_csv(url)
                
                # Renombrar columnas según mapeo
                data = data.rename(columns=self.source_mappings[tour])
                
                # Añadir columna de tour
                data['tour'] = tour
                
                all_data.append(data)
            
            # Combinar datos
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Procesar datos
            processed_data = self._process_data(combined_data)
            
            # Guardar datos
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            processed_data.to_csv(output_path, index=False)
            logging.info(f"Datos guardados en {output_path}")
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error recopilando datos: {e}")
            return None
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa los datos de partidos.
        
        Args:
            data: DataFrame con datos crudos
            
        Returns:
            DataFrame con datos procesados
        """
        try:
            # Convertir fecha
            data['match_date'] = pd.to_datetime(data['match_date'])
            
            # Calcular estadísticas por jugador
            player_stats = self._calculate_player_stats(data)
            
            # Calcular estadísticas head-to-head
            h2h_stats = self._calculate_h2h_stats(data)
            
            # Calcular estadísticas recientes
            recent_stats = self._calculate_recent_stats(data)
            
            # Combinar estadísticas
            processed_data = data.copy()
            processed_data = processed_data.merge(player_stats, on='winner_name', suffixes=('', '_winner'))
            processed_data = processed_data.merge(player_stats, on='loser_name', suffixes=('', '_loser'))
            processed_data = processed_data.merge(h2h_stats, on=['winner_name', 'loser_name'], suffixes=('', '_h2h'))
            processed_data = processed_data.merge(recent_stats, on='winner_name', suffixes=('', '_winner_recent'))
            processed_data = processed_data.merge(recent_stats, on='loser_name', suffixes=('', '_loser_recent'))
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error procesando datos: {e}")
            return data
    
    def _calculate_player_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula estadísticas por jugador."""
        stats = data.groupby('winner_name').agg({
            'winner_rank': 'mean',
            'winner_age': 'mean',
            'winner_ht': 'mean'
        }).reset_index()
        
        stats.columns = ['player_name', 'avg_rank', 'avg_age', 'avg_height']
        return stats
    
    def _calculate_h2h_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula estadísticas head-to-head."""
        h2h = []
        for _, row in data.iterrows():
            player1, player2 = row['winner_name'], row['loser_name']
            
            # Filtrar partidos anteriores entre estos jugadores
            previous_matches = data[
                ((data['winner_name'] == player1) & (data['loser_name'] == player2)) |
                ((data['winner_name'] == player2) & (data['loser_name'] == player1))
            ]
            
            h2h.append({
                'winner_name': player1,
                'loser_name': player2,
                'winner_h2h_wins': len(previous_matches[previous_matches['winner_name'] == player1]),
                'loser_h2h_wins': len(previous_matches[previous_matches['winner_name'] == player2])
            })
        
        return pd.DataFrame(h2h)
    
    def _calculate_recent_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula estadísticas recientes por jugador."""
        recent = []
        window = timedelta(days=90)  # 3 meses
        
        for player in data['winner_name'].unique():
            player_matches = data[
                (data['winner_name'] == player) |
                (data['loser_name'] == player)
            ].sort_values('match_date', ascending=False)
            
            recent_matches = player_matches[
                player_matches['match_date'] > player_matches['match_date'].max() - window
            ]
            
            recent.append({
                'player_name': player,
                'recent_matches': len(recent_matches),
                'recent_wins': len(recent_matches[recent_matches['winner_name'] == player]),
                'days_rest': (player_matches['match_date'].max() - recent_matches['match_date'].min()).days
            })
        
        return pd.DataFrame(recent)

def main():
    """Función principal para recopilar datos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recopilar datos de partidos de tenis')
    parser.add_argument('--output', default='ml-service/data/tennis_matches.csv',
                      help='Ruta donde guardar los datos procesados')
    
    args = parser.parse_args()
    
    collector = TennisDataCollector()
    data = collector.collect_data(args.output)
    
    if data is not None:
        logging.info(f"Total de partidos recopilados: {len(data)}")
        logging.info(f"Rango de fechas: {data['match_date'].min()} a {data['match_date'].max()}")
        logging.info(f"Distribución de superficies:\n{data['surface'].value_counts()}")
    else:
        logging.error("Error en la recopilación de datos")

if __name__ == '__main__':
    main()