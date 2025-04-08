"""
Módulo para gestión optimizada de memoria con datasets grandes de tenis.

Este módulo proporciona clases y funciones para:
- Monitorear el uso de memoria
- Cargar datos en porciones (chunking)
- Implementar estrategias de descarga selectiva
- Serialización/deserialización eficiente
"""

import os
import gc
import pickle
import psutil
import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Generator
import datetime
import numpy as np
import pandas as pd
from functools import wraps

# Configurar logger
logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Clase para gestionar eficientemente la memoria durante el procesamiento
    de grandes conjuntos de datos de tenis.
    """
    
    def __init__(
        self, 
        memory_threshold_pct: float = 80.0, 
        auto_collect: bool = True,
        chunk_size: int = 10000
    ):
        """
        Inicializa el MemoryManager.
        
        Args:
            memory_threshold_pct: Porcentaje de memoria usado que dispara advertencias
            auto_collect: Si se debe realizar recolección de basura automáticamente
            chunk_size: Tamaño predeterminado de chunks para procesamiento
        """
        self.memory_threshold_pct = memory_threshold_pct
        self.auto_collect = auto_collect
        self.chunk_size = chunk_size
        self.cached_dataframes: Dict[str, pd.DataFrame] = {}
        self.cached_dataframes_metadata: Dict[str, Dict[str, Any]] = {}
        self.memory_usage_history: List[Tuple[datetime.datetime, float]] = []
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """
        Obtiene el uso actual de memoria.
        
        Returns:
            Tupla de (memoria_usada_MB, porcentaje_usado)
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        memory_used_mb = mem_info.rss / 1024 / 1024  # En MB
        
        # Obtener porcentaje de uso del sistema
        system_pct = psutil.virtual_memory().percent
        
        # Registrar en historial
        self.memory_usage_history.append((datetime.datetime.now(), memory_used_mb))
        if len(self.memory_usage_history) > 1000:  # Limitar historial
            self.memory_usage_history.pop(0)
        
        return memory_used_mb, system_pct
    
    def check_memory_threshold(self, log_warning: bool = True) -> bool:
        """
        Verifica si el uso de memoria supera el umbral establecido.
        
        Args:
            log_warning: Si se debe registrar una advertencia
            
        Returns:
            True si se supera el umbral, False en caso contrario
        """
        _, memory_pct = self.get_memory_usage()
        threshold_exceeded = memory_pct > self.memory_threshold_pct
        
        if threshold_exceeded and log_warning:
            logger.warning(
                f"¡Uso de memoria ({memory_pct:.1f}%) supera umbral ({self.memory_threshold_pct:.1f}%)! "
                f"Considere liberar memoria o usar procesamiento en chunks."
            )
            if self.auto_collect:
                self.collect_garbage()
        
        return threshold_exceeded
    
    def collect_garbage(self) -> float:
        """
        Fuerza la recolección de basura para liberar memoria.
        
        Returns:
            Memoria liberada en MB (aproximación)
        """
        mem_before, _ = self.get_memory_usage()
        
        # Forzar recolección de basura
        gc.collect()
        
        mem_after, _ = self.get_memory_usage()
        mem_freed = mem_before - mem_after
        
        logger.info(f"Recolección de basura realizada. Memoria liberada: {mem_freed:.2f} MB")
        return mem_freed
    
    def cache_dataframe(
        self, 
        df: pd.DataFrame, 
        key: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Almacena un DataFrame en caché con gestión de memoria.
        
        Args:
            df: DataFrame a almacenar
            key: Clave única para identificar el DataFrame
            metadata: Información adicional sobre el DataFrame
        """
        # Verificar uso de memoria antes de cachear
        if self.check_memory_threshold(log_warning=False):
            logger.warning(
                f"Almacenando DataFrame ({key}) en caché con uso de memoria elevado. "
                f"Considere liberar memoria primero."
            )
        
        # Estimar tamaño del DataFrame
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Almacenar DataFrame y metadatos
        self.cached_dataframes[key] = df
        self.cached_dataframes_metadata[key] = {
            'rows': len(df),
            'columns': list(df.columns),
            'size_mb': df_size_mb,
            'cached_at': datetime.datetime.now().isoformat(),
            'last_accessed': datetime.datetime.now().isoformat()
        }
        
        if metadata:
            self.cached_dataframes_metadata[key].update(metadata)
        
        logger.info(f"DataFrame '{key}' almacenado en caché ({df_size_mb:.2f} MB, {len(df)} filas)")
    
    def get_cached_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        Recupera un DataFrame de la caché.
        
        Args:
            key: Clave del DataFrame
            
        Returns:
            DataFrame almacenado o None si no existe
        """
        if key in self.cached_dataframes:
            # Actualizar timestamp de último acceso
            if key in self.cached_dataframes_metadata:
                self.cached_dataframes_metadata[key]['last_accessed'] = datetime.datetime.now().isoformat()
            
            logger.debug(f"DataFrame '{key}' recuperado de caché")
            return self.cached_dataframes[key]
        else:
            logger.warning(f"DataFrame '{key}' no encontrado en caché")
            return None
    
    def clear_cache(
        self, 
        keys: Optional[List[str]] = None,
        older_than: Optional[datetime.timedelta] = None
    ) -> int:
        """
        Limpia la caché de DataFrames.
        
        Args:
            keys: Lista de claves a eliminar (None para eliminar todas)
            older_than: Eliminar entradas más antiguas que este timedelta
            
        Returns:
            Número de entradas eliminadas
        """
        if keys is None and older_than is None:
            # Limpiar toda la caché
            count = len(self.cached_dataframes)
            self.cached_dataframes.clear()
            self.cached_dataframes_metadata.clear()
            logger.info(f"Caché completa limpiada ({count} entradas eliminadas)")
            return count
        
        count = 0
        
        # Obtener lista de claves a eliminar
        to_remove = []
        
        if keys:
            to_remove.extend(keys)
        
        if older_than:
            now = datetime.datetime.now()
            threshold = now - older_than
            
            for key, metadata in self.cached_dataframes_metadata.items():
                if key in to_remove:
                    continue
                    
                last_accessed = datetime.datetime.fromisoformat(metadata['last_accessed'])
                if last_accessed < threshold:
                    to_remove.append(key)
        
        # Eliminar entradas
        for key in to_remove:
            if key in self.cached_dataframes:
                del self.cached_dataframes[key]
                if key in self.cached_dataframes_metadata:
                    del self.cached_dataframes_metadata[key]
                count += 1
        
        if count > 0:
            logger.info(f"Caché parcialmente limpiada ({count} entradas eliminadas)")
            # Forzar recolección de basura
            if self.auto_collect:
                self.collect_garbage()
        
        return count
    
    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimiza la memoria utilizada por un DataFrame.
        
        Args:
            df: DataFrame a optimizar
            
        Returns:
            DataFrame optimizado
        """
        before_size = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Intentar convertir tipos numéricos a más eficientes
        for col in df.select_dtypes(include=['int64']).columns:
            # Verificar rango de valores
            col_min, col_max = df[col].min(), df[col].max()
            
            # Elegir tipo más eficiente
            if col_min >= 0:
                if col_max < 256:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65536:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967296:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > -128 and col_max < 128:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32768:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483648:
                    df[col] = df[col].astype(np.int32)
        
        # Optimizar flotantes
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        
        # Convertir strings categóricos
        categorical_threshold = 0.5  # Proporción de valores únicos/total bajo la cual convertir a categórico
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            if num_unique / len(df) < categorical_threshold:
                df[col] = df[col].astype('category')
        
        after_size = df.memory_usage(deep=True).sum() / 1024 / 1024
        savings = before_size - after_size
        savings_pct = (savings / before_size) * 100 if before_size > 0 else 0
        
        logger.info(
            f"Optimización de memoria: {before_size:.2f}MB → {after_size:.2f}MB "
            f"(ahorro: {savings:.2f}MB, {savings_pct:.1f}%)"
        )
        
        return df
    
    def read_csv_in_chunks(
        self, 
        filepath: str, 
        chunk_size: Optional[int] = None, 
        optimize: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Lee un archivo CSV en chunks y los combina eficientemente.
        
        Args:
            filepath: Ruta al archivo CSV
            chunk_size: Tamaño de cada chunk a leer
            optimize: Si se debe optimizar la memoria tras la carga
            **kwargs: Argumentos adicionales para pd.read_csv
            
        Returns:
            DataFrame completo
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        logger.info(f"Leyendo '{filepath}' en chunks de {chunk_size} filas")
        
        chunks = []
        total_rows = 0
        
        # Leer en chunks
        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size, **kwargs)):
            total_rows += len(chunk)
            chunks.append(chunk)
            
            logger.debug(f"Leído chunk {i+1} ({len(chunk)} filas, total: {total_rows})")
            
            # Verificar memoria periódicamente
            if (i + 1) % 10 == 0:
                self.check_memory_threshold()
        
        # Combinar chunks
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Lectura completa: {total_rows} filas")
        
        # Optimizar memoria si se solicita
        if optimize:
            df = self.optimize_memory(df)
        
        return df
    
    def process_dataframe_in_chunks(
        self,
        df: pd.DataFrame,
        process_func: Callable[[pd.DataFrame], Any],
        chunk_size: Optional[int] = None,
        combine_func: Optional[Callable[[List[Any]], Any]] = None
    ) -> Any:
        """
        Procesa un DataFrame en chunks para evitar problemas de memoria.
        
        Args:
            df: DataFrame a procesar
            process_func: Función para procesar cada chunk
            chunk_size: Tamaño de cada chunk
            combine_func: Función para combinar resultados (None usa append/extend/concat)
            
        Returns:
            Resultado combinado del procesamiento
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        total_rows = len(df)
        chunk_count = (total_rows + chunk_size - 1) // chunk_size  # Techo de la división
        
        logger.info(f"Procesando DataFrame de {total_rows} filas en {chunk_count} chunks")
        
        # Lista para almacenar resultados
        results = []
        
        # Procesar en chunks
        for i in range(chunk_count):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            
            chunk = df.iloc[start_idx:end_idx]
            logger.debug(f"Procesando chunk {i+1}/{chunk_count} ({len(chunk)} filas)")
            
            # Procesar chunk
            chunk_result = process_func(chunk)
            results.append(chunk_result)
            
            # Verificar memoria periódicamente
            if (i + 1) % 5 == 0:
                self.check_memory_threshold()
        
        # Combinar resultados
        if combine_func:
            return combine_func(results)
        else:
            # Intentar combinar de forma inteligente según el tipo
            if results and isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            elif results and isinstance(results[0], list):
                combined = []
                for r in results:
                    combined.extend(r)
                return combined
            elif results and isinstance(results[0], dict):
                combined = {}
                for r in results:
                    combined.update(r)
                return combined
            else:
                return results
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas del uso de memoria.
        
        Returns:
            Diccionario con estadísticas de memoria
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        # Información del sistema
        system_mem = psutil.virtual_memory()
        
        # Estadísticas de la caché
        cache_size_mb = sum(m.get('size_mb', 0) for m in self.cached_dataframes_metadata.values())
        
        # Construir resultado
        stats = {
            'timestamp': datetime.datetime.now().isoformat(),
            'process': {
                'rss_mb': mem_info.rss / 1024 / 1024,
                'vms_mb': mem_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent()
            },
            'system': {
                'total_mb': system_mem.total / 1024 / 1024,
                'available_mb': system_mem.available / 1024 / 1024,
                'used_mb': system_mem.used / 1024 / 1024,
                'percent': system_mem.percent
            },
            'cache': {
                'dataframes_count': len(self.cached_dataframes),
                'total_size_mb': cache_size_mb,
                'items': {k: v for k, v in self.cached_dataframes_metadata.items()}
            },
            'history': {
                'points': len(self.memory_usage_history),
                'start': self.memory_usage_history[0][0].isoformat() if self.memory_usage_history else None,
                'end': self.memory_usage_history[-1][0].isoformat() if self.memory_usage_history else None,
                'min_mb': min(h[1] for h in self.memory_usage_history) if self.memory_usage_history else 0,
                'max_mb': max(h[1] for h in self.memory_usage_history) if self.memory_usage_history else 0,
                'avg_mb': sum(h[1] for h in self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0
            }
        }
        
        return stats


# Clases para diferentes estrategias de serialización
class DiskOffloader:
    """
    Clase para descargar y cargar DataFrames a/desde disco para ahorrar memoria.
    """
    
    def __init__(self, temp_dir: str = '.temp_dataframes'):
        """
        Inicializa el DiskOffloader.
        
        Args:
            temp_dir: Directorio temporal para almacenar DataFrames
        """
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        self.offloaded_metadata: Dict[str, Dict[str, Any]] = {}
    
    def offload_to_disk(self, df: pd.DataFrame, key: str) -> str:
        """
        Guarda un DataFrame a disco y libera memoria.
        
        Args:
            df: DataFrame a guardar
            key: Identificador único
            
        Returns:
            Ruta del archivo guardado
        """
        # Crear nombre de archivo único
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{key}_{timestamp}.pkl"
        filepath = os.path.join(self.temp_dir, filename)
        
        # Guardar metadatos
        self.offloaded_metadata[key] = {
            'filepath': filepath,
            'rows': len(df),
            'columns': list(df.columns),
            'size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'offloaded_at': datetime.datetime.now().isoformat()
        }
        
        # Guardar a disco
        df.to_pickle(filepath)
        
        logger.info(f"DataFrame '{key}' guardado a disco: {filepath}")
        return filepath
    
    def load_from_disk(self, key: str) -> Optional[pd.DataFrame]:
        """
        Carga un DataFrame previamente guardado en disco.
        
        Args:
            key: Identificador del DataFrame
            
        Returns:
            DataFrame cargado o None si no existe
        """
        if key not in self.offloaded_metadata:
            logger.warning(f"No hay un DataFrame guardado con la clave '{key}'")
            return None
        
        filepath = self.offloaded_metadata[key]['filepath']
        if not os.path.exists(filepath):
            logger.error(f"Archivo no encontrado: {filepath}")
            return None
        
        # Cargar desde disco
        df = pd.read_pickle(filepath)
        logger.info(f"DataFrame '{key}' cargado desde disco: {filepath}")
        return df
    
    def delete_from_disk(self, key: str) -> bool:
        """
        Elimina un DataFrame guardado en disco.
        
        Args:
            key: Identificador del DataFrame
            
        Returns:
            True si se eliminó correctamente, False si hubo error
        """
        if key not in self.offloaded_metadata:
            logger.warning(f"No hay un DataFrame guardado con la clave '{key}'")
            return False
        
        filepath = self.offloaded_metadata[key]['filepath']
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Archivo eliminado: {filepath}")
            except Exception as e:
                logger.error(f"Error al eliminar {filepath}: {str(e)}")
                return False
        
        # Eliminar metadatos
        del self.offloaded_metadata[key]
        return True
    
    def clean_temp_dir(self) -> int:
        """
        Limpia archivos temporales.
        
        Returns:
            Número de archivos eliminados
        """
        files = os.listdir(self.temp_dir)
        count = 0
        
        for file in files:
            filepath = os.path.join(self.temp_dir, file)
            if os.path.isfile(filepath) and file.endswith('.pkl'):
                try:
                    os.remove(filepath)
                    count += 1
                except Exception as e:
                    logger.error(f"Error al eliminar {filepath}: {str(e)}")
        
        # Actualizar metadatos
        self.offloaded_metadata = {
            k: v for k, v in self.offloaded_metadata.items()
            if os.path.exists(v['filepath'])
        }
        
        logger.info(f"Limpiados {count} archivos temporales")
        return count


# Decoradores para uso simplificado
def monitor_memory(func):
    """
    Decorador que monitorea el uso de memoria antes y después de la función.
    
    Args:
        func: Función a decorar
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        memory_manager = MemoryManager()
        mem_before, pct_before = memory_manager.get_memory_usage()
        
        logger.debug(f"Ejecutando {func.__name__}: Memoria inicial {mem_before:.2f} MB ({pct_before:.1f}%)")
        
        try:
            result = func(*args, **kwargs)
            
            mem_after, pct_after = memory_manager.get_memory_usage()
            mem_diff = mem_after - mem_before
            
            logger.debug(
                f"Finalizado {func.__name__}: Memoria final {mem_after:.2f} MB ({pct_after:.1f}%), "
                f"Diferencia: {mem_diff:.2f} MB"
            )
            
            return result
        except Exception as e:
            mem_after, pct_after = memory_manager.get_memory_usage()
            logger.error(
                f"Error en {func.__name__}: {str(e)}. "
                f"Memoria final {mem_after:.2f} MB ({pct_after:.1f}%)"
            )
            raise
    
    return wrapper


def chunked_processing(chunk_size=10000):
    """
    Decorador para aplicar procesamiento en chunks a una función que recibe un DataFrame.
    
    Args:
        chunk_size: Tamaño de cada chunk
    """
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            memory_manager = MemoryManager(chunk_size=chunk_size)
            
            # Verificar si el objeto es un DataFrame
            if not isinstance(df, pd.DataFrame):
                return func(df, *args, **kwargs)
            
            # Decidir si usar chunks
            if len(df) <= chunk_size:
                return func(df, *args, **kwargs)
            
            # Crear función que aplica la función original a cada chunk
            def process_chunk(chunk):
                return func(chunk, *args, **kwargs)
            
            return memory_manager.process_dataframe_in_chunks(
                df, process_chunk, chunk_size=chunk_size
            )
        
        return wrapper
    
    return decorator


# Funciones de utilidad
def get_dataframe_size(df: pd.DataFrame) -> float:
    """
    Obtiene el tamaño en memoria de un DataFrame en MB.
    
    Args:
        df: DataFrame
        
    Returns:
        Tamaño en MB
    """
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def check_system_memory() -> Dict[str, float]:
    """
    Verifica la memoria disponible en el sistema.
    
    Returns:
        Diccionario con información de memoria
    """
    vm = psutil.virtual_memory()
    return {
        'total_gb': vm.total / 1024 / 1024 / 1024,
        'available_gb': vm.available / 1024 / 1024 / 1024,
        'used_gb': vm.used / 1024 / 1024 / 1024,
        'percent': vm.percent
    }


def print_memory_usage(message: str = "Uso de memoria actual"):
    """
    Imprime información sobre el uso de memoria actual.
    
    Args:
        message: Mensaje a mostrar
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()
    
    print(f"{message}:")
    print(f"  Proceso: {mem_info.rss / 1024 / 1024:.2f} MB (RSS)")
    print(f"  Sistema: {system_mem.percent:.1f}% usado ({system_mem.available / 1024 / 1024 / 1024:.2f} GB disponible)")