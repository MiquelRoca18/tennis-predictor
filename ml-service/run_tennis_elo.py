#!/usr/bin/env python3
import os
import sys

# Añadir el directorio actual al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importar los argumentos de línea de comandos
import argparse

# Parsear argumentos
parser = argparse.ArgumentParser(description='Tennis ELO Rating System')
parser.add_argument('--action', required=True, help='Action to perform')
parser.add_argument('--data-dir', help='Directory containing input data files')
parser.add_argument('--output-dir', help='Directory for output files')
parser.add_argument('--start-date', help='Start date for analysis (YYYY-MM-DD)')
parser.add_argument('--end-date', help='End date for analysis (YYYY-MM-DD)')
parser.add_argument('--tournament-level', help='Tournament level to analyze')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

args = parser.parse_args()

# Ejecutar el módulo principal
from tennis_elo.tennis_elo_main import main

# Llamar a la función principal
sys.exit(main())