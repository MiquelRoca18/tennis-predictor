"""
utils/__init__.py

Inicialización del paquete de utilidades para el sistema ELO de tenis.
Proporciona acceso a los diferentes módulos de la carpeta utils:
- data_loader: Funciones para cargar datos
- normalizers: Funciones para normalizar datos
- validators: Funciones para validar datos
- error_tracking: Sistema de rastreo de errores
"""

# Importar módulos principales
from .data_loader import TennisDataLoader
from .normalizers import TennisDataNormalizer
from .validators import TennisDataValidator
from .error_tracking import (
    trace_errors, 
    apply_tracers, 
    save_error_report, 
    clear_error_history, 
    get_error_statistics
)

# Versión del paquete
__version__ = '1.0.0'

# Exponer clases principales para acceso directo
__all__ = [
    'TennisDataLoader',
    'TennisDataNormalizer',
    'TennisDataValidator',
    'trace_errors',
    'apply_tracers',
    'save_error_report',
    'clear_error_history',
    'get_error_statistics'
]