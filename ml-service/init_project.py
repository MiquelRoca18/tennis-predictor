#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
import logging
from typing import List, Dict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectInitializer:
    def __init__(self):
        self.root_dir = Path(__file__).parent.absolute()
        self.required_dirs = [
            'data',
            'logs',
            'model',
            'model/backups',
            'evaluation_results',
            'config',
            'tests',
            'tests/integration',
            'tests/unit',
            'sql'
        ]
        self.config_files = {
            'model_config.json': {
                'model_version': '1.0.0',
                'update_frequency': '24h',
                'performance_threshold': 0.65,
                'training_params': {
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
            },
            'api_config.json': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'timeout': 60
            }
        }

    def create_directory_structure(self) -> None:
        """Crear la estructura de directorios necesaria."""
        try:
            for directory in self.required_dirs:
                dir_path = self.root_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                # Crear .gitkeep
                (dir_path / '.gitkeep').touch()
                # Crear README.md básico
                with open(dir_path / 'README.md', 'w') as f:
                    f.write(f"# {directory.capitalize()}\n\nEste directorio contiene {directory}.")
            
            logger.info("✅ Estructura de directorios creada correctamente")
        except Exception as e:
            logger.error(f"❌ Error creando directorios: {str(e)}")
            raise

    def setup_virtual_environment(self) -> None:
        """Configurar el entorno virtual y las dependencias."""
        try:
            venv_path = self.root_dir / 'venv'
            if not venv_path.exists():
                subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
                logger.info("✅ Entorno virtual creado")

            # Instalar dependencias
            pip_path = str(venv_path / ('Scripts' if sys.platform == 'win32' else 'bin') / 'pip')
            subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
            logger.info("✅ Dependencias instaladas correctamente")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error configurando el entorno virtual: {str(e)}")
            raise

    def setup_config_files(self) -> None:
        """Configurar archivos de configuración."""
        try:
            config_dir = self.root_dir / 'config'
            
            # Crear archivos de configuración
            for filename, config in self.config_files.items():
                with open(config_dir / filename, 'w') as f:
                    json.dump(config, f, indent=4)
            
            # Copiar .env.example si existe
            env_example = self.root_dir / '.env.example'
            env_file = self.root_dir / '.env'
            if env_example.exists() and not env_file.exists():
                shutil.copy(env_example, env_file)
                logger.info("✅ Archivo .env creado (por favor, edítalo con tus configuraciones)")
            
            logger.info("✅ Archivos de configuración creados correctamente")
        except Exception as e:
            logger.error(f"❌ Error configurando archivos: {str(e)}")
            raise

    def setup_database(self) -> None:
        """Configurar la base de datos."""
        try:
            sql_dir = self.root_dir / 'sql'
            schema_file = sql_dir / 'db_schema.sql'
            initial_data = sql_dir / 'initial_data.sql'

            # Crear archivos SQL si no existen
            if not schema_file.exists():
                with open(schema_file, 'w') as f:
                    f.write("""-- Esquema de la base de datos
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    country VARCHAR(3),
    birth_date DATE
);

CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    tournament_id INTEGER NOT NULL,
    player1_id INTEGER REFERENCES players(id),
    player2_id INTEGER REFERENCES players(id),
    match_date TIMESTAMP NOT NULL,
    surface VARCHAR(20),
    winner_id INTEGER REFERENCES players(id)
);
""")

            if not initial_data.exists():
                with open(initial_data, 'w') as f:
                    f.write("-- Datos iniciales para la base de datos\n")

            logger.info("""
🔧 Para inicializar la base de datos:
1. Crea una base de datos PostgreSQL
2. Configura las variables DB_* en el archivo .env
3. Ejecuta:
   psql -U tu_usuario -d tu_base_de_datos -f sql/db_schema.sql
   psql -U tu_usuario -d tu_base_de_datos -f sql/initial_data.sql
""")
        except Exception as e:
            logger.error(f"❌ Error configurando la base de datos: {str(e)}")
            raise

    def run(self) -> None:
        """Ejecutar todo el proceso de inicialización."""
        logger.info("🎾 Iniciando configuración del proyecto Tennis Predictor...")
        
        try:
            self.create_directory_structure()
            self.setup_virtual_environment()
            self.setup_config_files()
            self.setup_database()
            
            logger.info("""
✨ Configuración inicial completada!

Pasos siguientes:
1. Edita el archivo .env con tus configuraciones
2. Configura la base de datos siguiendo las instrucciones anteriores
3. Activa el entorno virtual:
   - Linux/Mac: source venv/bin/activate
   - Windows: .\\venv\\Scripts\\activate
4. Ejecuta los tests: pytest
5. Inicia el servicio: python api.py

Para más información, consulta el README.md
""")
        except Exception as e:
            logger.error(f"❌ Error durante la inicialización: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    initializer = ProjectInitializer()
    initializer.run() 