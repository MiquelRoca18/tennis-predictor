"""
Pruebas unitarias para el módulo serialization.

Este archivo contiene pruebas para verificar la funcionalidad
del serializador de modelos para el sistema de predicción de tenis.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import pickle
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Asegurar que el módulo está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar el módulo a probar
from utils.serialization import ModelSerializer

# Intentar importar librerías de ML opcionales
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# Clase mock para cuando no está disponible alguna biblioteca
class MockModel:
    """Modelo simulado para pruebas."""
    
    def __init__(self, model_type='sklearn', **kwargs):
        self.model_type = model_type
        self.params = kwargs
        self.__module__ = f"{model_type}.mock_module"
    
    def predict(self, X):
        """Simulación de predicción."""
        return np.ones(len(X))
    
    def save_model(self, filepath):
        """Simulación de guardado para XGBoost."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


class TestModelSerializer(unittest.TestCase):
    """Clase de pruebas para ModelSerializer."""
    
    def setUp(self):
        """Configuración previa a cada prueba."""
        # Crear directorio temporal para las pruebas
        self.test_dir = tempfile.mkdtemp()
        
        # Inicializar el ModelSerializer
        self.serializer = ModelSerializer(base_dir=self.test_dir)
        
        # Crear modelos de prueba
        self._create_test_models()
    
    def tearDown(self):
        """Limpieza posterior a cada prueba."""
        # Eliminar directorio temporal
        shutil.rmtree(self.test_dir)
    
    def _create_test_models(self):
        """Crea modelos de prueba para las pruebas."""
        # Modelos simulados para diferentes tipos
        self.mock_sklearn = MockModel(model_type='sklearn')
        self.mock_xgboost = MockModel(model_type='xgboost')
        self.mock_tensorflow = MockModel(model_type='tensorflow')
        self.mock_ensemble = MockModel(model_type='ensemble')
        
        # Crear modelos reales si las bibliotecas están disponibles
        if SKLEARN_AVAILABLE:
            self.sklearn_model = RandomForestClassifier(n_estimators=10, random_state=42)
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            self.sklearn_model.fit(X, y)
        else:
            self.sklearn_model = self.mock_sklearn
        
        # XGBoost si está disponible
        if XGBOOST_AVAILABLE:
            self.xgboost_model = xgb.XGBClassifier(n_estimators=10, random_state=42)
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            self.xgboost_model.fit(X, y)
        else:
            self.xgboost_model = self.mock_xgboost
    
    def test_initialization(self):
        """Prueba la inicialización del serializador."""
        # Verificar directorios creados
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'sklearn')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'xgboost')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'tensorflow')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'ensemble')))
        
        # Verificar archivo de índice
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'model_index.json')))
    
    def test_get_model_type(self):
        """Prueba la detección de tipo de modelo."""
        # Probar con diferentes tipos de modelos
        self.assertEqual(self.serializer._get_model_type(self.mock_sklearn), 'sklearn')
        self.assertEqual(self.serializer._get_model_type(self.mock_xgboost), 'xgboost')
        self.assertEqual(self.serializer._get_model_type(self.mock_tensorflow), 'tensorflow')
        self.assertEqual(self.serializer._get_model_type(self.mock_ensemble), 'ensemble')
    
    def test_save_and_load_sklearn(self):
        """Prueba guardar y cargar un modelo scikit-learn."""
        # Guardar modelo
        model_path = self.serializer.save(
            self.sklearn_model, 
            'test_sklearn',
            metadata={'description': 'Modelo de prueba'}
        )
        
        # Verificar que el archivo existe
        self.assertTrue(os.path.exists(model_path))
        
        # Cargar modelo
        loaded_model = self.serializer.load('test_sklearn', with_metadata=False)
        
        # Verificar que es el mismo tipo
        self.assertEqual(type(loaded_model), type(self.sklearn_model))
        
        # Cargar con metadatos
        loaded_model, metadata = self.serializer.load('test_sklearn', with_metadata=True)
        
        # Verificar metadatos
        self.assertEqual(metadata['description'], 'Modelo de prueba')
    
    def test_save_with_version(self):
        """Prueba guardar un modelo con versión específica."""
        # Versión personalizada
        version = '20220101_test'
        
        # Guardar modelo con versión
        model_path = self.serializer.save(
            self.sklearn_model, 
            'test_version',
            version=version
        )
        
        # Verificar que se utilizó la versión correcta
        self.assertIn(version, model_path)
        
        # Cargar modelo específico
        loaded_model = self.serializer.load('test_version', version=version)
        
        # Verificar que es el mismo tipo
        self.assertEqual(type(loaded_model), type(self.sklearn_model))
    
    def test_list_models(self):
        """Prueba listar modelos guardados."""
        # Guardar varios modelos
        self.serializer.save(self.sklearn_model, 'model1')
        self.serializer.save(self.sklearn_model, 'model2')
        self.serializer.save(self.sklearn_model, 'model3')
        
        # Listar modelos
        models = self.serializer.list_models()
        
        # Verificar que los modelos están en la lista
        self.assertIn('model1', models)
        self.assertIn('model2', models)
        self.assertIn('model3', models)
    
    def test_get_model_info(self):
        """Prueba obtener información de un modelo."""
        # Guardar modelo con metadatos
        self.serializer.save(
            self.sklearn_model, 
            'info_test',
            metadata={
                'accuracy': 0.85,
                'f1_score': 0.82,
                'hyperparameters': {'n_estimators': 10}
            }
        )
        
        # Obtener info
        info = self.serializer.get_model_info('info_test')
        
        # Verificar contenido
        self.assertEqual(info['model_type'], 'sklearn')
        self.assertEqual(info['metadata']['accuracy'], 0.85)
        self.assertEqual(info['metadata']['hyperparameters']['n_estimators'], 10)
    
    def test_delete_model(self):
        """Prueba eliminar un modelo."""
        # Guardar modelo
        self.serializer.save(self.sklearn_model, 'delete_test')
        
        # Verificar que existe
        self.assertTrue('delete_test' in self.serializer.list_models())
        
        # Eliminar
        result = self.serializer.delete_model('delete_test')
        
        # Verificar que se eliminó correctamente
        self.assertTrue(result)
        self.assertFalse('delete_test' in self.serializer.list_models())
    
    def test_delete_specific_version(self):
        """Prueba eliminar una versión específica de un modelo."""
        # Guardar dos versiones
        self.serializer.save(self.sklearn_model, 'multi_version', version='v1')
        self.serializer.save(self.sklearn_model, 'multi_version', version='v2')
        
        # Verificar que existen dos versiones
        self.assertEqual(len(self.serializer.list_models()['multi_version']), 2)
        
        # Eliminar una versión
        result = self.serializer.delete_model('multi_version', version='v1')
        
        # Verificar que se eliminó esa versión
        self.assertTrue(result)
        self.assertEqual(len(self.serializer.list_models()['multi_version']), 1)
        self.assertEqual(self.serializer.list_models()['multi_version'][0], 'v2')
    
    def test_compare_models(self):
        """Prueba comparar dos modelos."""
        # Guardar dos modelos con métricas diferentes
        self.serializer.save(
            self.sklearn_model, 
            'model_a',
            metadata={
                'metrics': {'accuracy': 0.85, 'f1_score': 0.82},
                'hyperparameters': {'n_estimators': 10}
            }
        )
        
        self.serializer.save(
            self.sklearn_model, 
            'model_b',
            metadata={
                'metrics': {'accuracy': 0.88, 'f1_score': 0.84},
                'hyperparameters': {'n_estimators': 20}
            }
        )
        
        # Comparar
        comparison = self.serializer.compare_models('model_a', 'model_b')
        
        # Verificar resultados de la comparación
        self.assertIn('metrics_comparison', comparison)
        self.assertIn('params_comparison', comparison)
        
        # Verificar métricas específicas
        metrics = comparison['metrics_comparison']
        self.assertIn('accuracy', metrics)
        self.assertEqual(metrics['accuracy']['model_a'], 0.85)
        self.assertEqual(metrics['accuracy']['model_b'], 0.88)
        self.assertAlmostEqual(metrics['accuracy']['difference'], -0.03)
        
        # Verificar parámetros
        params = comparison['params_comparison']
        self.assertIn('n_estimators', params)
        self.assertEqual(params['n_estimators']['model_a'], 10)
        self.assertEqual(params['n_estimators']['model_b'], 20)
        self.assertTrue(params['n_estimators']['is_different'])
    
    def test_get_best_model(self):
        """Prueba encontrar el mejor modelo según una métrica."""
        # Guardar varios modelos con diferentes métricas
        self.serializer.save(
            self.sklearn_model, 
            'model_1',
            metadata={'metrics': {'accuracy': 0.82, 'loss': 0.25}}
        )
        
        self.serializer.save(
            self.sklearn_model, 
            'model_2',
            metadata={'metrics': {'accuracy': 0.85, 'loss': 0.22}}
        )
        
        self.serializer.save(
            self.sklearn_model, 
            'model_3',
            metadata={'metrics': {'accuracy': 0.81, 'loss': 0.28}}
        )
        
        # Encontrar mejor modelo por accuracy (mayor es mejor)
        best_name, best_version = self.serializer.get_best_model(
            metric_name='accuracy',
            higher_is_better=True
        )
        
        # Verificar que es el esperado
        self.assertEqual(best_name, 'model_2')
        
        # Encontrar mejor modelo por loss (menor es mejor)
        best_name, best_version = self.serializer.get_best_model(
            metric_name='loss',
            higher_is_better=False
        )
        
        # Verificar que es el esperado
        self.assertEqual(best_name, 'model_2')
    
    def test_export_import_model(self):
        """Prueba exportar e importar un modelo."""
        # Guardar modelo original
        original_metadata = {
            'metrics': {'accuracy': 0.88, 'f1_score': 0.85},
            'hyperparameters': {'n_estimators': 20},
            'description': 'Modelo para exportar'
        }
        
        self.serializer.save(
            self.sklearn_model, 
            'export_test',
            metadata=original_metadata
        )
        
        # Exportar a un nuevo directorio
        export_dir = os.path.join(self.test_dir, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        exported_path = self.serializer.export_model(
            'export_test',
            export_dir,
            include_metadata=True
        )
        
        # Verificar que existe el archivo exportado
        self.assertTrue(os.path.exists(exported_path))
        
        # Metadatos exportados
        metadata_path = f"{exported_path}.metadata.json"
        self.assertTrue(os.path.exists(metadata_path))
        
        # Importar con un nuevo nombre
        new_version = self.serializer.import_model(
            exported_path,
            'imported_model',
            metadata_path=metadata_path
        )
        
        # Verificar que se importó correctamente
        self.assertIn('imported_model', self.serializer.list_models())
        
        # Cargar y verificar metadatos
        _, metadata = self.serializer.load('imported_model', with_metadata=True)
        self.assertEqual(metadata['description'], 'Modelo para exportar')
        self.assertEqual(metadata['metrics']['accuracy'], 0.88)


if __name__ == '__main__':
    unittest.main()