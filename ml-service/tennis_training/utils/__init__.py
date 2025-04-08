"""
Módulo de utilidades para el sistema de entrenamiento de modelos de predicción de tenis.

Este paquete contiene clases y funciones auxiliares para cargar datos,
serializar modelos, gestionar memoria y otras utilidades comunes.
"""

import logging
from .logging_config import setup_logging, get_logger, set_log_level

# Configurar logging para todo el módulo
setup_logging(default_level=logging.INFO)

# Importar clases principales para facilitar su uso
from .data_loader import DataLoader
from .serialization import ModelSerializer
from .memory_manager import (
    MemoryManager, DiskOffloader, monitor_memory, 
    chunked_processing, get_dataframe_size, check_system_memory
)

__all__ = [
    'DataLoader', 
    'ModelSerializer', 
    'MemoryManager', 
    'DiskOffloader',
    'monitor_memory', 
    'chunked_processing',
    'get_dataframe_size',
    'check_system_memory',
    'setup_logging',
    'get_logger',
    'set_log_level'
]