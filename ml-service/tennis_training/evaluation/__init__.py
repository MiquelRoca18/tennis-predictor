"""
evaluation/__init__.py

Este módulo inicializa el paquete de evaluación para el sistema de predicción de partidos de tenis.
Proporciona acceso a herramientas para calcular métricas, analizar resultados, comparar modelos,
evaluar el rendimiento temporal y detectar sesgos en las predicciones.
"""

# Importar clases y funciones principales para exposición directa
from .metrics import TennisMetrics
from .results_analyzer import ResultsAnalyzer
from .model_comparator import ModelComparator
from .temporal_analysis import TemporalAnalyzer
from .bias_detector import BiasDetector

# Definir qué módulos/clases se importarán con "from evaluation import *"
__all__ = [
    'TennisMetrics',
    'ResultsAnalyzer',
    'ModelComparator',
    'TemporalAnalyzer',
    'BiasDetector'
]

# Información sobre el paquete
__version__ = '1.0.0'
__author__ = 'Tennis ELO Team'


def get_evaluation_components():
    """
    Devuelve un diccionario con todas las clases de evaluación disponibles.
    
    Returns:
        dict: Diccionario con nombres de componentes y sus clases correspondientes
    """
    return {
        'metrics': TennisMetrics,
        'analyzer': ResultsAnalyzer,
        'comparator': ModelComparator,
        'temporal': TemporalAnalyzer,
        'bias': BiasDetector
    }


def create_evaluation_pipeline(output_dir=None):
    """
    Crea una instancia de cada componente de evaluación configurada con el mismo directorio de salida.
    
    Args:
        output_dir: Directorio donde se guardarán los resultados de evaluación
        
    Returns:
        dict: Diccionario con instancias de cada componente de evaluación
    """
    return {
        'metrics': TennisMetrics(),
        'analyzer': ResultsAnalyzer(output_dir),
        'comparator': ModelComparator(output_dir),
        'temporal': TemporalAnalyzer(output_dir),
        'bias': BiasDetector(output_dir)
    }