#!/usr/bin/env python3
"""
Script para calcular las puntuaciones ELO para todos los jugadores.
"""

import os
import sys
import logging
import argparse
import psycopg2
import pandas as pd
from datetime import datetime
import traceback

# Añadir directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar nuestro sistema ELO
from elo_system import TennisELOSystem

def main():
    """Función principal para inicializar y calcular puntuaciones ELO."""
    parser = argparse.ArgumentParser(description='Calcular puntuaciones ELO para jugadores de tenis')
    parser.add_argument('--host', default='localhost', help='Host de la base de datos')
    parser.add_argument('--port', type=int, default=5432, help='Puerto de la base de datos')
    parser.add_argument('--dbname', required=True, help='Nombre de la base de datos')
    parser.add_argument('--user', required=True, help='Usuario de la base de datos')
    parser.add_argument('--password', help='Contraseña de la base de datos')
    parser.add_argument('--initialize', action='store_true', help='Inicializar ELO para todos los jugadores')
    
    args = parser.parse_args()
    
    try:
        # Conectar a la base de datos
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password
        )
        
        print("Conexión a base de datos establecida.")
        
        # Crear instancia del sistema ELO
        elo_system = TennisELOSystem(conn)
        
        if args.initialize:
            print("Inicializando puntuaciones ELO para todos los jugadores...")
            if elo_system._initialize_all_player_elo():
                print("Inicialización ELO completada.")
            else:
                print("Error en la inicialización ELO.")
        
        # Recalcular ELO para todos los partidos históricos
        print("Recalculando puntuaciones ELO desde partidos históricos...")
        start_time = datetime.now()
        
        if elo_system.recalculate_all_elo():
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"Recálculo ELO completado en {duration:.2f} segundos.")
            
            # Mostrar algunos jugadores top por ELO
            top_players = elo_system.get_elo_ranking(limit=10)
            print("\nTop 10 jugadores por ELO general:")
            print(top_players[['name', 'elo', 'atp_ranking']])
            
            # Mostrar algunos jugadores top por superficie (clay)
            top_clay = elo_system.get_elo_ranking(surface='clay', limit=10)
            print("\nTop 10 jugadores por ELO en tierra batida:")
            print(top_clay[['name', 'elo', 'atp_ranking']])
        else:
            print("Error en el recálculo ELO.")
        
        # Cerrar conexión
        conn.close()
        print("Conexión a base de datos cerrada.")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()