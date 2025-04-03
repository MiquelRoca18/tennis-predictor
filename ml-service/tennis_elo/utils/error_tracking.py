"""
utils/error_tracking.py

Sistema de rastreo y gestión de errores para el sistema ELO de tenis.
Proporciona funciones para seguimiento detallado de excepciones, 
generación de informes y decoradores para funciones críticas.
"""

import functools
import traceback
import sys
import logging
import os
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional, Union

# Configurar logging
logger = logging.getLogger(__name__)

# Variables globales para tracking de errores
error_locations = []
function_errors = {}

def trace_errors(func: Callable) -> Callable:
    """
    Decorador para rastrear errores en funciones.
    Registra información detallada sobre las excepciones y proporciona valores
    de retorno seguros en caso de error.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada con manejo de errores
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Obtener información sobre el error
            tb = traceback.extract_tb(sys.exc_info()[2])
            filename, line, func_name, text = tb[-1]
            
            # Registrar el error
            error_msg = f"ERROR en {func.__name__}: {str(e)} en línea {line} - '{text}'"
            logger.error(error_msg)
            
            # Guardar información detallada
            error_info = {
                'function': func.__name__,
                'exception': str(e),
                'exception_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'line': line,
                'text': text,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            error_locations.append(error_info)
            
            # Incrementar contador para esta función
            if func.__name__ not in function_errors:
                function_errors[func.__name__] = []
            function_errors[func.__name__].append(error_info)
            
            # Valores de retorno seguros según el nombre de la función
            return _get_safe_return_value(func.__name__, args)
    
    return wrapper

def _get_safe_return_value(func_name: str, args: tuple) -> Any:
    """
    Proporciona un valor de retorno seguro según el tipo de función cuando ocurre un error.
    
    Args:
        func_name: Nombre de la función que falló
        args: Argumentos pasados a la función
        
    Returns:
        Valor seguro apropiado según el tipo de función
    """
    # Funciones que devuelven una tupla de dos flotantes
    if func_name == "update_ratings":
        return 0.0, 0.0
    
    # Funciones que devuelven un valor flotante
    elif func_name in ["get_player_rating", "get_player_form", "get_combined_surface_rating", 
                     "get_h2h_advantage", "calculate_expected_win_probability", 
                     "_get_match_importance_factor", "_get_margin_multiplier",
                     "_get_dynamic_k_factor", "_get_match_stats_factor"]:
        return 1.0
    
    # Funciones que procesan DataFrames
    elif func_name in ["process_matches_dataframe"]:
        return args[1]  # Devolver el DataFrame original
    
    # Valor por defecto para otras funciones
    else:
        return None

def apply_tracers(class_obj: Any, methods_to_trace: List[str]) -> None:
    """
    Aplica decoradores de rastreo a los métodos de una clase.
    
    Args:
        class_obj: Clase cuyos métodos se decorarán
        methods_to_trace: Lista de nombres de métodos a decorar
    """
    for method_name in methods_to_trace:
        if hasattr(class_obj, method_name):
            original_method = getattr(class_obj, method_name)
            wrapped_method = trace_errors(original_method)
            setattr(class_obj, method_name, wrapped_method)

def save_error_report(output_dir: Optional[str] = None) -> str:
    """
    Guarda un informe detallado de los errores encontrados durante la ejecución.
    
    Args:
        output_dir: Directorio donde guardar el informe (opcional)
        
    Returns:
        Ruta del archivo de informe generado
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Crear directorio si se especifica
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, f"error_report_{timestamp}.txt")
    else:
        report_file = f"error_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("=== INFORME DE ERRORES EN TENNIS ELO PROCESSOR ===\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if not function_errors:
            f.write("¡No se encontraron errores!\n")
            return report_file
            
        f.write("=== RESUMEN DE ERRORES POR FUNCIÓN ===\n")
        for func_name, errors in sorted(function_errors.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"{func_name}: {len(errors)} errores\n")
        
        f.write("\n=== ERRORES DETALLADOS ===\n\n")
        for i, error in enumerate(error_locations, 1):
            f.write(f"ERROR #{i}\n")
            f.write(f"Función: {error['function']}\n")
            f.write(f"Tipo de excepción: {error['exception_type']}\n")
            f.write(f"Mensaje: {error['exception']}\n")
            f.write(f"Línea: {error['line']}\n")
            f.write(f"Código: {error['text']}\n")
            f.write(f"Timestamp: {error['timestamp']}\n")
            f.write("\nTraceback:\n")
            f.write(error['traceback'])
            f.write("\n" + "-"*50 + "\n\n")
    
    logger.info(f"Informe de errores guardado en {report_file}")
    return report_file

def clear_error_history() -> None:
    """
    Limpia el historial de errores registrados.
    Útil para resetear el seguimiento al iniciar una nueva operación.
    """
    global error_locations, function_errors
    error_locations = []
    function_errors = {}
    logger.debug("Historial de errores limpiado")

def get_error_statistics() -> Dict:
    """
    Proporciona estadísticas resumidas sobre los errores registrados.
    
    Returns:
        Diccionario con estadísticas de errores
    """
    stats = {
        'total_errors': len(error_locations),
        'errors_by_function': {},
        'errors_by_type': {},
        'most_common_functions': [],
        'most_common_exceptions': []
    }
    
    # Contar errores por función
    for func_name, errors in function_errors.items():
        stats['errors_by_function'][func_name] = len(errors)
    
    # Contar errores por tipo de excepción
    exception_counts = {}
    for error in error_locations:
        exc_type = error['exception_type']
        if exc_type not in exception_counts:
            exception_counts[exc_type] = 0
        exception_counts[exc_type] += 1
    
    stats['errors_by_type'] = exception_counts
    
    # Obtener funciones con más errores
    sorted_functions = sorted(stats['errors_by_function'].items(), key=lambda x: x[1], reverse=True)
    stats['most_common_functions'] = sorted_functions[:5] if len(sorted_functions) > 5 else sorted_functions
    
    # Obtener excepciones más comunes
    sorted_exceptions = sorted(exception_counts.items(), key=lambda x: x[1], reverse=True)
    stats['most_common_exceptions'] = sorted_exceptions[:5] if len(sorted_exceptions) > 5 else sorted_exceptions
    
    return stats