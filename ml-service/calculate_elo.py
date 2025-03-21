#!/usr/bin/env python3
"""
calculate_elo.py

Script mejorado para calcular las puntuaciones ELO para todos los jugadores de tenis.
Soporta múltiples fuentes de datos y exportación de resultados en varios formatos.
"""

import os
import sys
import logging
import argparse
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Añadir directorio padre al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ml-service/logs/elo_calculation.log'),
        logging.StreamHandler()
    ]
)

# Importar sistema ELO mejorado
try:
    from improved_elo_system import TennisELOSystem
except ImportError:
    # Intentar otra ruta de importación
    try:
        from ml_service.improved_elo_system import TennisELOSystem
    except ImportError:
        logging.error("No se pudo importar TennisELOSystem. Verificar rutas de importación.")
        sys.exit(1)

def get_db_connection(args):
    """
    Establece conexión con la base de datos PostgreSQL.
    
    Args:
        args: Argumentos de línea de comandos con credenciales
        
    Returns:
        Conexión a PostgreSQL o None si hay error
    """
    try:
        if not args.db_name or not args.db_user:
            logging.warning("Faltan credenciales para conectar a la base de datos")
            return None
        
        conn = psycopg2.connect(
            host=args.db_host,
            port=args.db_port,
            dbname=args.db_name,
            user=args.db_user,
            password=args.db_password or ""
        )
        
        logging.info(f"Conexión establecida a la base de datos {args.db_name}@{args.db_host}")
        return conn
        
    except Exception as e:
        logging.error(f"Error conectando a la base de datos: {e}")
        return None

def process_analysis(elo_system, args):
    """
    Realiza análisis utilizando el sistema ELO.
    
    Args:
        elo_system: Instancia de TennisELOSystem
        args: Argumentos de línea de comandos
    """
    # Obtener ranking ELO
    if args.top_players:
        surface = args.surface or None
        top_players = elo_system.get_elo_ranking(
            surface=surface, 
            min_matches=args.min_matches or 10,
            limit=args.top_players
        )
        
        if not top_players.empty:
            surface_str = f" en {surface}" if surface else ""
            print(f"\n=== TOP {args.top_players} JUGADORES POR ELO{surface_str} ===")
            
            # Mostrar en formato tabla
            if 'player' in top_players.columns:
                player_col = 'player'
            elif 'name' in top_players.columns:
                player_col = 'name'
            else:
                player_col = top_players.columns[1]  # Segunda columna como fallback
            
            # Formatear para mostrar
            display_cols = [player_col, 'elo', 'matches']
            if 'atp_ranking' in top_players.columns:
                display_cols.append('atp_ranking')
            
            pd.set_option('display.max_rows', args.top_players + 5)
            pd.set_option('display.width', 120)
            
            display_df = top_players[display_cols].copy()
            # Redondear valores de ELO
            if 'elo' in display_df.columns:
                display_df['elo'] = display_df['elo'].round(1)
            
            print(display_df)
            
            if args.output_csv:
                # Guardar ranking completo a CSV
                csv_path = args.output_csv
                top_players.to_csv(csv_path, index=False)
                print(f"\nRanking completo guardado en: {csv_path}")
    
    # Graficar historial ELO si se solicita
    if args.plot_players:
        player_names = args.plot_players.split(',')
        output_path = args.plot_output or 'elo_history.png'
        
        print(f"\nGenerando gráfico de historial ELO para: {', '.join(player_names)}")
        elo_system.plot_elo_history(
            player_names=player_names,
            output_path=output_path,
            surface=args.surface,
            start_date=args.start_date,
            end_date=args.end_date,
            show_plot=False
        )
        print(f"Gráfico guardado en: {output_path}")
    
    # Comparar jugadores si se solicita
    if args.compare_players:
        player_pairs = args.compare_players.split(',')
        
        if len(player_pairs) >= 2:
            player1 = player_pairs[0].strip()
            player2 = player_pairs[1].strip()
            
            # Predecir resultado
            surface = args.surface or 'hard'
            prediction = elo_system.predict_match_outcome(player1, player2, surface)
            
            print(f"\n=== PREDICCIÓN BASADA EN ELO ===")
            print(f"Partido: {player1} vs {player2} en superficie {surface}")
            print(f"Probabilidad de victoria de {player1}: {prediction['p1_probability']*100:.1f}%")
            print(f"Probabilidad de victoria de {player2}: {prediction['p2_probability']*100:.1f}%")
            print("\nRatings ELO:")
            print(f"{player1}: {prediction['p1_elo']:.1f} (general), {prediction['p1_surface_elo']:.1f} ({surface})")
            print(f"{player2}: {prediction['p2_elo']:.1f} (general), {prediction['p2_surface_elo']:.1f} ({surface})")
            
            # Probabilidades basadas en diferencia de ELO
            elo_diff = abs(prediction['p1_effective_elo'] - prediction['p2_effective_elo'])
            favorite = player1 if prediction['p1_effective_elo'] > prediction['p2_effective_elo'] else player2
            
            print(f"\nDiferencia de ELO efectivo: {elo_diff:.1f} puntos a favor de {favorite}")
            if elo_diff < 50:
                print("Partido muy equilibrado (diferencia < 50 puntos)")
            elif elo_diff < 100:
                print("Ligera ventaja (diferencia entre 50-100 puntos)")
            elif elo_diff < 200:
                print("Ventaja clara (diferencia entre 100-200 puntos)")
            else:
                print("Fuerte favorito (diferencia > 200 puntos)")

def main():
    """Función principal para calcular puntuaciones ELO"""
    parser = argparse.ArgumentParser(description='Calculador avanzado de puntuaciones ELO para tenis')
    
    # Parámetros de entrada de datos
    parser.add_argument('--data', type=str, help='Ruta al archivo CSV con datos históricos')
    parser.add_argument('--initialize', action='store_true', help='Inicializar ELO para todos los jugadores')
    parser.add_argument('--recalculate', action='store_true', help='Recalcular ELO para todos los partidos históricos')
    
    # Parámetros de conexión a base de datos
    parser.add_argument('--db-host', default='localhost', help='Host de la base de datos')
    parser.add_argument('--db-port', type=int, default=5432, help='Puerto de la base de datos')
    parser.add_argument('--db-name', help='Nombre de la base de datos')
    parser.add_argument('--db-user', help='Usuario de la base de datos')
    parser.add_argument('--db-password', help='Contraseña de la base de datos')
    
    # Parámetros de exportación y análisis
    parser.add_argument('--export-json', type=str, help='Exportar datos ELO a archivo JSON')
    parser.add_argument('--export-csv', type=str, help='Exportar todos los ratings ELO a CSV')
    parser.add_argument('--output-csv', type=str, help='Guardar resultados en archivo CSV')
    parser.add_argument('--top-players', type=int, help='Mostrar top N jugadores por ELO')
    parser.add_argument('--min-matches', type=int, default=10, help='Mínimo de partidos para aparecer en el ranking')
    parser.add_argument('--surface', type=str, choices=['hard', 'clay', 'grass', 'carpet'],
                      help='Superficie específica para análisis')
    
    # Parámetros para gráficos y comparaciones
    parser.add_argument('--plot-players', type=str, help='Lista separada por comas de jugadores para graficar historial ELO')
    parser.add_argument('--plot-output', type=str, help='Ruta para guardar el gráfico')
    parser.add_argument('--start-date', type=str, help='Fecha de inicio para gráficos (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Fecha final para gráficos (YYYY-MM-DD)')
    parser.add_argument('--compare-players', type=str, 
                      help='Comparar dos jugadores separados por coma (ej: "Nadal,Federer")')
    
    # Parámetros avanzados
    parser.add_argument('--k-factor', type=float, help='Factor K base para todos los cálculos')
    parser.add_argument('--initial-elo', type=float, default=1500, help='Rating ELO inicial para nuevos jugadores')
    
    args = parser.parse_args()
    
    try:
        # Establecer tiempo de inicio
        start_time = datetime.now()
        
        # Conectar a la base de datos si es necesario
        db_conn = None
        if args.db_name and args.db_user:
            db_conn = get_db_connection(args)
        
        # Crear sistema ELO
        elo_system = TennisELOSystem(db_connection=db_conn, data_path=args.data)
        
        # Actualizar configuración si se especifican parámetros
        if args.initial_elo:
            elo_system.initial_elo = args.initial_elo
            logging.info(f"ELO inicial configurado a {args.initial_elo}")
        
        if args.k_factor:
            # Actualizar todos los factores K
            for surface in elo_system.k_factors:
                elo_system.k_factors[surface] = args.k_factor
            logging.info(f"Factor K base configurado a {args.k_factor} para todas las superficies")
        
        # Inicializar ELO para todos los jugadores si se solicita
        if args.initialize:
            print("Inicializando puntuaciones ELO para todos los jugadores...")
            if elo_system._initialize_all_elo():
                print("Inicialización ELO completada exitosamente.")
            else:
                print("Error en la inicialización ELO.")
        
        # Recalcular ELO para todos los partidos históricos
        if args.recalculate:
            print("Recalculando puntuaciones ELO desde partidos históricos...")
            
            if elo_system.recalculate_all_elo():
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                print(f"Recálculo ELO completado exitosamente en {duration:.2f} segundos.")
                
                # Mostrar algunos jugadores top por ELO
                surface = args.surface or None
                top_players = elo_system.get_elo_ranking(surface=surface, limit=10)
                
                if not top_players.empty:
                    surface_str = f" en {surface}" if surface else ""
                    print(f"\nTop 10 jugadores por ELO{surface_str}:")
                    
                    # Determinar columna de jugador
                    if 'player' in top_players.columns:
                        player_col = 'player'
                    elif 'name' in top_players.columns:
                        player_col = 'name'
                    else:
                        player_col = top_players.columns[1]
                    
                    # Mostrar información relevante
                    display_cols = [player_col, 'elo']
                    if 'matches' in top_players.columns:
                        display_cols.append('matches')
                    if 'atp_ranking' in top_players.columns:
                        display_cols.append('atp_ranking')
                    
                    print(top_players[display_cols].to_string(index=False))
            else:
                print("Error en el recálculo ELO.")
        
        # Exportar datos a JSON si se solicita
        if args.export_json:
            print(f"Exportando datos ELO a {args.export_json}...")
            if elo_system.export_elo_data(args.export_json):
                print(f"Datos ELO exportados exitosamente a {args.export_json}")
            else:
                print(f"Error exportando datos a {args.export_json}")
        
        # Exportar ratings a CSV si se solicita
        if args.export_csv:
            print(f"Exportando ratings ELO a {args.export_csv}...")
            
            if hasattr(elo_system, '_save_elo_to_csv'):
                if elo_system._save_elo_to_csv(args.export_csv):
                    print(f"Ratings ELO exportados exitosamente a {args.export_csv}")
                else:
                    print(f"Error exportando ratings a {args.export_csv}")
            else:
                # Crear manualmente si el método no existe
                if db_conn:
                    # Obtener de base de datos
                    cursor = db_conn.cursor()
                    cursor.execute("""
                        SELECT name, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet, 
                               elo_last_update, matches_played
                        FROM players
                        ORDER BY elo_rating DESC
                    """)
                    
                    columns = [desc[0] for desc in cursor.description]
                    players_data = cursor.fetchall()
                    
                    df = pd.DataFrame(players_data, columns=columns)
                    df.to_csv(args.export_csv, index=False)
                    
                    print(f"Ratings ELO exportados exitosamente a {args.export_csv}")
                    cursor.close()
                elif hasattr(elo_system, 'players_elo'):
                    # Obtener de memoria
                    players_data = []
                    for player, elo_data in elo_system.players_elo.items():
                        row = {'player': player}
                        row.update(elo_data)
                        players_data.append(row)
                    
                    df = pd.DataFrame(players_data)
                    df.to_csv(args.export_csv, index=False)
                    
                    print(f"Ratings ELO exportados exitosamente a {args.export_csv}")
                else:
                    print("No se pudo exportar a CSV: método no disponible")
        
        # Realizar análisis adicionales
        process_analysis(elo_system, args)
        
        # Cerrar conexión a base de datos
        if db_conn:
            db_conn.close()
            logging.info("Conexión a base de datos cerrada")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()