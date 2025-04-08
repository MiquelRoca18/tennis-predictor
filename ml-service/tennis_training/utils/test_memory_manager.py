"""
Pruebas unitarias para el módulo memory_manager.

Este archivo contiene pruebas para verificar la funcionalidad
del gestor de memoria para grandes conjuntos de datos de tenis.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Asegurar que el módulo está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar el módulo a probar
from utils.memory_manager import (
    MemoryManager, DiskOffloader, monitor_memory, 
    chunked_processing, get_dataframe_size, check_system_memory
)


class TestMemoryManager(unittest.TestCase):
    """Clase de pruebas para MemoryManager."""
    
    def setUp(self):
        """Configuración previa a cada prueba."""
        # Crear directorio temporal para las pruebas
        self.test_dir = tempfile.mkdtemp()
        
        # Inicializar el MemoryManager
        self.memory_manager = MemoryManager(memory_threshold_pct=90.0)
        
        # Crear datos de prueba
        self._create_test_data()
    
    def tearDown(self):
        """Limpieza posterior a cada prueba."""
        # Eliminar directorio temporal
        shutil.rmtree(self.test_dir)
    
    def _create_test_data(self):
        """Crea datos de prueba para las pruebas."""
        # Crear un DataFrame pequeño para pruebas
        self.small_df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.rand(1000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 1000)
        })
        
        # Crear un archivo CSV temporal para pruebas
        self.csv_path = os.path.join(self.test_dir, 'test_data.csv')
        self.small_df.to_csv(self.csv_path, index=False)
    
    def test_get_memory_usage(self):
        """Prueba obtener uso de memoria."""
        memory_mb, memory_pct = self.memory_manager.get_memory_usage()
        
        # Verificar que los valores son razonables
        self.assertGreater(memory_mb, 0)
        self.assertGreater(memory_pct, 0)
        self.assertLess(memory_pct, 100)
        
        # Verificar que se registró en el historial
        self.assertEqual(len(self.memory_manager.memory_usage_history), 1)
    
    def test_cache_and_retrieve_dataframe(self):
        """Prueba almacenar y recuperar un DataFrame en caché."""
        # Almacenar en caché
        self.memory_manager.cache_dataframe(
            self.small_df,
            'test_df',
            metadata={'description': 'DataFrame de prueba'}
        )
        
        # Verificar que está en la caché
        self.assertIn('test_df', self.memory_manager.cached_dataframes)
        self.assertIn('test_df', self.memory_manager.cached_dataframes_metadata)
        
        # Recuperar de la caché
        retrieved_df = self.memory_manager.get_cached_dataframe('test_df')
        
        # Verificar que es el mismo DataFrame
        pd.testing.assert_frame_equal(retrieved_df, self.small_df)
        
        # Verificar que se actualizó el timestamp de último acceso
        self.assertIn('last_accessed', self.memory_manager.cached_dataframes_metadata['test_df'])
    
    def test_clear_cache(self):
        """Prueba limpiar la caché."""
        # Almacenar varios DataFrames
        self.memory_manager.cache_dataframe(self.small_df, 'df1')
        self.memory_manager.cache_dataframe(self.small_df, 'df2')
        self.memory_manager.cache_dataframe(self.small_df, 'df3')
        
        # Verificar que están en la caché
        self.assertEqual(len(self.memory_manager.cached_dataframes), 3)
        
        # Limpiar caché específica
        count = self.memory_manager.clear_cache(keys=['df1', 'df2'])
        
        # Verificar que se eliminaron solo esos
        self.assertEqual(count, 2)
        self.assertEqual(len(self.memory_manager.cached_dataframes), 1)
        self.assertIn('df3', self.memory_manager.cached_dataframes)
        
        # Limpiar toda la caché
        count = self.memory_manager.clear_cache()
        
        # Verificar que se eliminó todo
        self.assertEqual(count, 1)
        self.assertEqual(len(self.memory_manager.cached_dataframes), 0)
    
    def test_optimize_memory(self):
        """Prueba optimización de memoria de un DataFrame."""
        # Crear un DataFrame con diferentes tipos de datos
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),  # int64
            'float_col': np.random.rand(1000),  # float64
            'category_col': np.random.choice(['A', 'B', 'C'], 1000)  # object
        })
        
        # Optimizar
        optimized_df = self.memory_manager.optimize_memory(df)
        
        # Verificar que los tipos han cambiado
        self.assertNotEqual(df.dtypes['int_col'], optimized_df.dtypes['int_col'])
        self.assertNotEqual(df.dtypes['float_col'], optimized_df.dtypes['float_col'])
        self.assertEqual(optimized_df.dtypes['category_col'].name, 'category')
        
        # Verificar que los datos son iguales
        np.testing.assert_array_equal(df['int_col'].values, optimized_df['int_col'].values)
        np.testing.assert_array_almost_equal(df['float_col'].values, optimized_df['float_col'].values, decimal=5)
        np.testing.assert_array_equal(df['category_col'].values, optimized_df['category_col'].values)
    
    def test_read_csv_in_chunks(self):
        """Prueba leer CSV en chunks."""
        # Leer el CSV en chunks
        df = self.memory_manager.read_csv_in_chunks(
            self.csv_path,
            chunk_size=200,
            optimize=True
        )
        
        # Verificar que se leyó correctamente
        self.assertEqual(len(df), 1000)
        self.assertIn('id', df.columns)
        self.assertIn('value', df.columns)
        self.assertIn('category', df.columns)
    
    def test_process_dataframe_in_chunks(self):
        """Prueba procesar un DataFrame en chunks."""
        # Función de procesamiento simple
        def process_func(chunk):
            return chunk['value'].mean()
        
        # Función para combinar resultados
        def combine_func(results):
            return sum(results) / len(results)
        
        # Procesar en chunks
        result = self.memory_manager.process_dataframe_in_chunks(
            self.small_df,
            process_func,
            chunk_size=200,
            combine_func=combine_func
        )
        
        # Calcular el resultado esperado
        expected = self.small_df['value'].mean()
        
        # Verificar resultado
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_get_memory_usage_stats(self):
        """Prueba obtener estadísticas detalladas de memoria."""
        # Almacenar un DataFrame en caché
        self.memory_manager.cache_dataframe(self.small_df, 'stats_test')
        
        # Obtener estadísticas
        stats = self.memory_manager.get_memory_usage_stats()
        
        # Verificar estructura
        self.assertIn('process', stats)
        self.assertIn('system', stats)
        self.assertIn('cache', stats)
        self.assertIn('history', stats)
        
        # Verificar datos de caché
        self.assertEqual(stats['cache']['dataframes_count'], 1)
        self.assertIn('stats_test', stats['cache']['items'])
    
    @patch('psutil.virtual_memory')
    def test_check_memory_threshold(self, mock_virtual_memory):
        """Prueba verificar umbral de memoria."""
        # Simular uso de memoria por debajo del umbral
        mock_vm = MagicMock()
        mock_vm.percent = 70.0
        mock_virtual_memory.return_value = mock_vm
        
        # Verificar umbral (no debería superarse)
        result = self.memory_manager.check_memory_threshold()
        self.assertFalse(result)
        
        # Simular uso de memoria por encima del umbral
        mock_vm.percent = 95.0
        
        # Verificar umbral (debería superarse)
        result = self.memory_manager.check_memory_threshold()
        self.assertTrue(result)


class TestDiskOffloader(unittest.TestCase):
    """Clase de pruebas para DiskOffloader."""
    
    def setUp(self):
        """Configuración previa a cada prueba."""
        # Crear directorio temporal para las pruebas
        self.test_dir = tempfile.mkdtemp()
        
        # Inicializar el DiskOffloader
        self.offloader = DiskOffloader(temp_dir=os.path.join(self.test_dir, 'temp_df'))
        
        # Crear datos de prueba
        self.test_df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.rand(1000)
        })
    
    def tearDown(self):
        """Limpieza posterior a cada prueba."""
        # Eliminar directorio temporal
        shutil.rmtree(self.test_dir)
    
    def test_offload_and_load_dataframe(self):
        """Prueba descargar y cargar un DataFrame a/desde disco."""
        # Descargar a disco
        filepath = self.offloader.offload_to_disk(self.test_df, 'test_df')
        
        # Verificar que el archivo existe
        self.assertTrue(os.path.exists(filepath))
        
        # Verificar que se guardaron los metadatos
        self.assertIn('test_df', self.offloader.offloaded_metadata)
        self.assertEqual(self.offloader.offloaded_metadata['test_df']['rows'], 1000)
        
        # Cargar desde disco
        loaded_df = self.offloader.load_from_disk('test_df')
        
        # Verificar que es el mismo DataFrame
        pd.testing.assert_frame_equal(loaded_df, self.test_df)
    
    def test_delete_from_disk(self):
        """Prueba eliminar un DataFrame descargado."""
        # Descargar a disco
        filepath = self.offloader.offload_to_disk(self.test_df, 'delete_df')
        
        # Verificar que existe
        self.assertTrue(os.path.exists(filepath))
        
        # Eliminar
        result = self.offloader.delete_from_disk('delete_df')
        
        # Verificar que se eliminó correctamente
        self.assertTrue(result)
        self.assertFalse(os.path.exists(filepath))
        self.assertNotIn('delete_df', self.offloader.offloaded_metadata)
    
    def test_clean_temp_dir(self):
        """Prueba limpiar directorio temporal."""
        # Descargar varios DataFrames
        self.offloader.offload_to_disk(self.test_df, 'df1')
        self.offloader.offload_to_disk(self.test_df, 'df2')
        self.offloader.offload_to_disk(self.test_df, 'df3')
        
        # Limpiar directorio
        count = self.offloader.clean_temp_dir()
        
        # Verificar que se eliminaron todos
        self.assertEqual(count, 3)
        self.assertEqual(len(self.offloader.offloaded_metadata), 0)
        self.assertEqual(len(os.listdir(self.offloader.temp_dir)), 0)


class TestDecorators(unittest.TestCase):
    """Clase de pruebas para decoradores de gestión de memoria."""
    
    def setUp(self):
        """Configuración previa a cada prueba."""
        # Crear un DataFrame de prueba
        self.test_df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.rand(1000)
        })
    
    def test_monitor_memory_decorator(self):
        """Prueba el decorador monitor_memory."""
        # Función decorada
        @monitor_memory
        def test_function(df):
            time.sleep(0.1)  # Pequeña pausa
            return df.mean()
        
        # Ejecutar función
        result = test_function(self.test_df)
        
        # Verificar que devuelve el resultado esperado
        self.assertAlmostEqual(result['value'], self.test_df['value'].mean())
    
    def test_chunked_processing_decorator(self):
        """Prueba el decorador chunked_processing."""
        # Función decorada para procesar en chunks
        @chunked_processing(chunk_size=200)
        def sum_values(df):
            return df['value'].sum()
        
        # Ejecutar función
        result = sum_values(self.test_df)
        
        # Verificar resultado
        self.assertAlmostEqual(result, self.test_df['value'].sum())
    
    def test_get_dataframe_size(self):
        """Prueba función get_dataframe_size."""
        # Obtener tamaño
        size_mb = get_dataframe_size(self.test_df)
        
        # Verificar que es un valor positivo
        self.assertGreater(size_mb, 0)
    
    def test_check_system_memory(self):
        """Prueba función check_system_memory."""
        # Obtener información de memoria
        memory_info = check_system_memory()
        
        # Verificar estructura
        self.assertIn('total_gb', memory_info)
        self.assertIn('available_gb', memory_info)
        self.assertIn('used_gb', memory_info)
        self.assertIn('percent', memory_info)
        
        # Verificar valores
        self.assertGreater(memory_info['total_gb'], 0)
        self.assertLess(memory_info['percent'], 100)


if __name__ == '__main__':
    unittest.main()