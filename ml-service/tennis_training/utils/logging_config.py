"""
Módulo para configuración avanzada de logging en el sistema de entrenamiento.

Este módulo permite configurar el logging de forma flexible, mediante:
- Configuración desde archivos JSON/YAML
- Configuración programática
- Diferentes niveles de detalle para diferentes módulos
"""

import os
import json
import yaml
import logging
import logging.config
from typing import Dict, Any, Optional, Union


def setup_logging(
    default_level: int = logging.INFO,
    config_file: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    env_key: str = 'LOG_CFG'
) -> None:
    """
    Configura el sistema de logging con opciones flexibles.
    
    Args:
        default_level: Nivel de logging predeterminado si no hay configuración
        config_file: Ruta al archivo de configuración JSON/YAML
        config_dict: Diccionario de configuración directo
        env_key: Variable de entorno que puede contener la ruta al archivo de configuración
    """
    config_path = os.getenv(env_key, config_file)
    
    if config_dict:
        # Usar configuración proporcionada como diccionario
        logging.config.dictConfig(config_dict)
        logging.info(f"Logging configurado desde diccionario")
        return
    
    if config_path and os.path.exists(config_path):
        # Intentar cargar configuración desde archivo
        try:
            ext = os.path.splitext(config_path)[1].lower()
            if ext in ('.yml', '.yaml'):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logging.config.dictConfig(config)
            else:  # Asumir JSON para cualquier otra extensión
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logging.config.dictConfig(config)
            
            logging.info(f"Logging configurado desde archivo: {config_path}")
            return
        except Exception as e:
            print(f"Error al cargar configuración de logging desde {config_path}: {str(e)}")
            print("Usando configuración básica...")
    
    # Configuración básica si no hay archivo o hay error
    logging.basicConfig(
        level=default_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Logging configurado con configuración básica")


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Obtiene un logger configurado para un módulo específico.
    
    Args:
        name: Nombre del logger/módulo
        level: Nivel de logging específico para este logger
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def set_log_level(level: Union[int, str], module: Optional[str] = None) -> None:
    """
    Cambia el nivel de logging durante la ejecución.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                o constantes de logging (10, 20, 30, 40, 50)
        module: Módulo específico para cambiar nivel (None para el root logger)
    """
    # Convertir string a nivel numérico si es necesario
    if isinstance(level, str):
        level = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }.get(level.upper(), logging.INFO)
    
    if module:
        logger = logging.getLogger(module)
        logger.setLevel(level)
        logger.info(f"Nivel de logging cambiado a {logging.getLevelName(level)} para {module}")
    else:
        logging.getLogger().setLevel(level)
        logging.info(f"Nivel de logging global cambiado a {logging.getLevelName(level)}")


# Configuración predefinida para desarrollo
DEFAULT_DEV_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': 'tennis_ml.log',
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 5,
            'encoding': 'utf8',
        },
    },
    'loggers': {
        'ml_training': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
        'ml_training.data': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
        'ml_training.models': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
}

# Configuración predefinida para producción
DEFAULT_PROD_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'default',
            'filename': 'tennis_ml.log',
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 5,
            'encoding': 'utf8',
        },
    },
    'loggers': {
        'ml_training': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False,
        },
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console', 'file'],
    },
}


def setup_dev_logging(log_file: Optional[str] = 'tennis_ml_dev.log') -> None:
    """Configura logging para entorno de desarrollo"""
    config = DEFAULT_DEV_CONFIG.copy()
    if log_file:
        config['handlers']['file']['filename'] = log_file
    setup_logging(config_dict=config)


def setup_prod_logging(log_file: Optional[str] = 'tennis_ml_prod.log') -> None:
    """Configura logging para entorno de producción"""
    config = DEFAULT_PROD_CONFIG.copy()
    if log_file:
        config['handlers']['file']['filename'] = log_file
    setup_logging(config_dict=config)