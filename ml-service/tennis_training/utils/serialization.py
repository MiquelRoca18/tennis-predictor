import os
import json
import pickle
import logging
import datetime
import hashlib
from typing import Any, Dict, Optional, Union, List, Tuple

import pandas as pd
import numpy as np

# Para modelos específicos
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

# Configurar logging
logger = logging.getLogger(__name__)

class ModelSerializer:
    """
    Clase para serializar y deserializar modelos de machine learning,
    junto con metadatos asociados como configuración, métricas y fechas.
    
    Soporta modelos de:
    - scikit-learn
    - XGBoost
    - TensorFlow/Keras
    """
    
    def __init__(self, base_dir: str = 'ml_training/models'):
        """
        Inicializa el ModelSerializer con el directorio base para modelos.
        
        Args:
            base_dir: Directorio base donde se guardarán los modelos
        """
        self.base_dir = base_dir
        
        # Crear directorios si no existen
        os.makedirs(base_dir, exist_ok=True)
        self.sklearn_dir = os.path.join(base_dir, 'sklearn')
        os.makedirs(self.sklearn_dir, exist_ok=True)
        self.xgboost_dir = os.path.join(base_dir, 'xgboost')
        os.makedirs(self.xgboost_dir, exist_ok=True)
        self.tensorflow_dir = os.path.join(base_dir, 'tensorflow')
        os.makedirs(self.tensorflow_dir, exist_ok=True)
        self.ensemble_dir = os.path.join(base_dir, 'ensemble')
        os.makedirs(self.ensemble_dir, exist_ok=True)
        
        # Crear archivo de índice si no existe
        self.index_file = os.path.join(base_dir, 'model_index.json')
        if not os.path.exists(self.index_file):
            self._initialize_index()
    
    def _initialize_index(self):
        """Inicializa el archivo de índice de modelos."""
        index = {
            'models': {},
            'latest_versions': {},
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Inicializado índice de modelos en {self.index_file}")
    
    def _load_index(self) -> dict:
        """
        Carga el índice de modelos.
        
        Returns:
            Diccionario con información del índice
        """
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"No se pudo cargar el índice, inicializando uno nuevo")
            self._initialize_index()
            with open(self.index_file, 'r') as f:
                return json.load(f)
    
    def _save_index(self, index: dict):
        """
        Guarda el índice de modelos.
        
        Args:
            index: Diccionario con información del índice
        """
        index['last_updated'] = datetime.datetime.now().isoformat()
        
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _get_model_directory(self, model_type: str) -> str:
        """
        Obtiene el directorio correspondiente al tipo de modelo.
        
        Args:
            model_type: Tipo de modelo ('sklearn', 'xgboost', 'tensorflow', 'ensemble')
            
        Returns:
            Ruta al directorio del tipo de modelo
        """
        if model_type == 'sklearn':
            return self.sklearn_dir
        elif model_type == 'xgboost':
            return self.xgboost_dir
        elif model_type == 'tensorflow':
            return self.tensorflow_dir
        elif model_type == 'ensemble':
            return self.ensemble_dir
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    def _get_model_hash(self, model: Any) -> str:
        """
        Genera un hash único para el modelo.
        
        Args:
            model: Objeto del modelo
            
        Returns:
            String con el hash
        """
        # Serializar el modelo para calcular el hash
        temp_file = os.path.join(self.base_dir, 'temp_model.pkl')
        with open(temp_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Calcular el hash
        with open(temp_file, 'rb') as f:
            model_bytes = f.read()
            model_hash = hashlib.md5(model_bytes).hexdigest()
        
        # Eliminar archivo temporal
        os.remove(temp_file)
        
        return model_hash
    
    def _get_model_type(self, model: Any) -> str:
        """
        Determina el tipo de modelo.
        
        Args:
            model: Objeto del modelo
            
        Returns:
            String con el tipo de modelo
        """
        model_module = model.__module__
        
        if model_module.startswith('sklearn'):
            return 'sklearn'
        elif xgb and model_module.startswith('xgboost'):
            return 'xgboost'
        elif tf and model_module.startswith('tensorflow'):
            return 'tensorflow'
        elif model_module.startswith('__main__') and hasattr(model, 'models'):
            # Asumir que es un ensemble si tiene un atributo 'models'
            return 'ensemble'
        else:
            # Caso especial para ensembles personalizados
            if hasattr(model, 'model_type') and model.model_type == 'ensemble':
                return 'ensemble'
            
            # Si no se puede determinar, usar el módulo base como tipo
            base_module = model_module.split('.')[0]
            logger.warning(f"Tipo de modelo no reconocido, usando módulo base: {base_module}")
            return base_module
    
    def save(
        self, 
        model: Any, 
        model_name: str, 
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        overwrite: bool = False
    ) -> str:
        """
        Guarda un modelo con sus metadatos.
        
        Args:
            model: Objeto del modelo a guardar
            model_name: Nombre del modelo (identificador para el usuario)
            metadata: Diccionario con metadatos adicionales
            version: Versión del modelo (si es None, se genera automáticamente)
            overwrite: Si es True, sobrescribe la versión existente
            
        Returns:
            Ruta al archivo del modelo guardado
        """
        # Determinar el tipo de modelo
        model_type = self._get_model_type(model)
        model_dir = self._get_model_directory(model_type)
        
        # Generar directorio específico para este modelo
        model_name_dir = os.path.join(model_dir, model_name)
        os.makedirs(model_name_dir, exist_ok=True)
        
        # Preparar metadatos
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'model_type': model_type,
            'saved_at': datetime.datetime.now().isoformat(),
            'model_hash': self._get_model_hash(model)
        })
        
        # Generar o usar versión proporcionada
        if version is None:
            # Formato: YYYYMMDD_HHMMSS
            version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Comprobar si la versión existe y manejar conflictos
        version_dir = os.path.join(model_name_dir, version)
        if os.path.exists(version_dir) and not overwrite:
            logger.warning(f"La versión {version} ya existe para {model_name}")
            version = f"{version}_alt"
            version_dir = os.path.join(model_name_dir, version)
        
        os.makedirs(version_dir, exist_ok=True)
        
        # Guardar el modelo según su tipo
        model_path = None
        if model_type == 'sklearn' or model_type == 'ensemble':
            model_path = os.path.join(version_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        elif model_type == 'xgboost':
            model_path = os.path.join(version_dir, 'model.xgb')
            model.save_model(model_path)
        
        elif model_type == 'tensorflow':
            model_path = os.path.join(version_dir, 'model')
            model.save(model_path)
        
        else:
            # Para tipos desconocidos, intentar pickle
            model_path = os.path.join(version_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Guardar metadatos
        metadata_path = os.path.join(version_dir, 'metadata.json')
        
        # Convertir valores numpy a tipos Python nativos
        cleaned_metadata = self._clean_metadata_for_json(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(cleaned_metadata, f, indent=2)
        
        # Actualizar índice
        index = self._load_index()
        
        if model_name not in index['models']:
            index['models'][model_name] = {}
        
        index['models'][model_name][version] = {
            'path': os.path.relpath(version_dir, self.base_dir),
            'model_type': model_type,
            'saved_at': metadata['saved_at']
        }
        
        # Actualizar última versión
        index['latest_versions'][model_name] = version
        
        self._save_index(index)
        
        logger.info(f"Modelo {model_name} (v{version}) guardado en {model_path}")
        return model_path
    
    def _clean_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Limpia los metadatos para que sean serializables en JSON.
        
        Args:
            metadata: Diccionario con metadatos
            
        Returns:
            Diccionario limpio serializable
        """
        cleaned = {}
        
        for key, value in metadata.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                cleaned[key] = float(value)
            elif isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                cleaned[key] = [
                    self._clean_value_for_json(item) for item in value
                ]
            elif isinstance(value, dict):
                cleaned[key] = self._clean_metadata_for_json(value)
            else:
                # Intentar serializar, o convertir a string si falla
                try:
                    json.dumps({key: value})
                    cleaned[key] = value
                except (TypeError, OverflowError):
                    cleaned[key] = str(value)
        
        return cleaned
    
    def _clean_value_for_json(self, value: Any) -> Any:
        """
        Limpia un valor individual para que sea serializable en JSON.
        
        Args:
            value: Valor a limpiar
            
        Returns:
            Valor limpio serializable
        """
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return [self._clean_value_for_json(item) for item in value]
        elif isinstance(value, dict):
            return self._clean_metadata_for_json(value)
        else:
            # Intentar serializar, o convertir a string si falla
            try:
                json.dumps(value)
                return value
            except (TypeError, OverflowError):
                return str(value)
    
    def load(
        self, 
        model_name: str, 
        version: Optional[str] = None, 
        with_metadata: bool = True
    ) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        """
        Carga un modelo y opcionalmente sus metadatos.
        
        Args:
            model_name: Nombre del modelo a cargar
            version: Versión específica a cargar (None para la última)
            with_metadata: Si es True, devuelve también los metadatos
            
        Returns:
            Si with_metadata es False, devuelve el modelo.
            Si with_metadata es True, devuelve una tupla (modelo, metadatos).
        """
        # Cargar índice
        index = self._load_index()
        
        # Verificar que el modelo existe
        if model_name not in index['models']:
            logger.error(f"Modelo {model_name} no encontrado")
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        # Determinar la versión a cargar
        if version is None:
            if model_name in index['latest_versions']:
                version = index['latest_versions'][model_name]
            else:
                # Si no hay versión predeterminada, tomar la más reciente
                versions = list(index['models'][model_name].keys())
                versions.sort(reverse=True)  # Asumir formato que ordena correctamente (ej. YYYYMMDD)
                version = versions[0]
        
        # Verificar que la versión existe
        if version not in index['models'][model_name]:
            logger.error(f"Versión {version} no encontrada para modelo {model_name}")
            raise ValueError(f"Versión {version} no encontrada para modelo {model_name}")
        
        # Obtener información del modelo
        model_info = index['models'][model_name][version]
        model_path = os.path.join(self.base_dir, model_info['path'])
        model_type = model_info['model_type']
        
        # Cargar el modelo según su tipo
        model = None
        if model_type == 'sklearn' or model_type == 'ensemble':
            model_file = os.path.join(model_path, 'model.pkl')
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        
        elif model_type == 'xgboost':
            if xgb is None:
                logger.error("No se pudo cargar modelo XGBoost: biblioteca no disponible")
                raise ImportError("XGBoost no está instalado")
            
            model_file = os.path.join(model_path, 'model.xgb')
            model = xgb.Booster()
            model.load_model(model_file)
        
        elif model_type == 'tensorflow':
            if tf is None:
                logger.error("No se pudo cargar modelo TensorFlow: biblioteca no disponible")
                raise ImportError("TensorFlow no está instalado")
            
            model_file = os.path.join(model_path, 'model')
            model = tf.keras.models.load_model(model_file)
        
        else:
            # Para tipos desconocidos, intentar pickle
            model_file = os.path.join(model_path, 'model.pkl')
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        
        logger.info(f"Modelo {model_name} (v{version}) cargado desde {model_path}")
        
        # Devolver solo el modelo o modelo con metadatos
        if with_metadata:
            metadata_file = os.path.join(model_path, 'metadata.json')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return model, metadata
        else:
            return model
    
    def list_models(self) -> Dict[str, List[str]]:
        """
        Obtiene una lista de todos los modelos guardados y sus versiones.
        
        Returns:
            Diccionario con nombres de modelos como claves y listas de versiones como valores
        """
        index = self._load_index()
        
        result = {}
        for model_name, versions in index['models'].items():
            # Ordenar versiones por fecha de guardado (más reciente primero)
            sorted_versions = sorted(
                versions.keys(),
                key=lambda v: versions[v]['saved_at'],
                reverse=True
            )
            result[model_name] = sorted_versions
        
        return result
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene información sobre un modelo específico.
        
        Args:
            model_name: Nombre del modelo
            version: Versión específica (None para la última)
            
        Returns:
            Diccionario con información del modelo
        """
        index = self._load_index()
        
        if model_name not in index['models']:
            logger.error(f"Modelo {model_name} no encontrado")
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        if version is None:
            if model_name in index['latest_versions']:
                version = index['latest_versions'][model_name]
            else:
                versions = list(index['models'][model_name].keys())
                versions.sort(reverse=True)
                version = versions[0]
        
        if version not in index['models'][model_name]:
            logger.error(f"Versión {version} no encontrada para modelo {model_name}")
            raise ValueError(f"Versión {version} no encontrada para modelo {model_name}")
        
        model_info = index['models'][model_name][version]
        
        # Añadir información de metadatos
        model_path = os.path.join(self.base_dir, model_info['path'])
        metadata_file = os.path.join(model_path, 'metadata.json')
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Combinar información básica con metadatos
            result = {**model_info, 'metadata': metadata}
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"No se pudieron cargar metadatos para {model_name} (v{version})")
            result = model_info
        
        return result
    
    def export_model(
        self, 
        model_name: str, 
        output_dir: str, 
        version: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Exporta un modelo a un directorio externo.
        
        Args:
            model_name: Nombre del modelo a exportar
            output_dir: Directorio de destino
            version: Versión específica (None para la última)
            include_metadata: Si se deben incluir los metadatos
            
        Returns:
            Ruta al archivo o directorio exportado
        """
        import shutil
        
        # Cargar el modelo y sus metadatos
        model, metadata = self.load(model_name, version, with_metadata=True)
        
        # Crear directorio de destino si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Determinar la versión
        if version is None:
            index = self._load_index()
            if model_name in index['latest_versions']:
                version = index['latest_versions'][model_name]
            else:
                versions = list(index['models'][model_name].keys())
                versions.sort(reverse=True)
                version = versions[0]
        
        # Determinar el tipo de modelo
        model_type = metadata.get('model_type', self._get_model_type(model))
        
        # Nombre de archivo de salida
        output_file = os.path.join(output_dir, f"{model_name}_v{version}")
        
        # Exportar según el tipo
        if model_type == 'sklearn' or model_type == 'ensemble':
            output_file += '.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(model, f)
        
        elif model_type == 'xgboost':
            output_file += '.xgb'
            model.save_model(output_file)
        
        elif model_type == 'tensorflow':
            # Para TensorFlow, se necesita un directorio
            output_file = os.path.join(output_dir, f"{model_name}_v{version}")
            model.save(output_file)
        
        else:
            # Para tipos desconocidos, usar pickle
            output_file += '.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Exportar metadatos si se solicita
        if include_metadata:
            metadata_file = f"{output_file}.metadata.json"
            if model_type == 'tensorflow':
                # Para TensorFlow, el archivo de metadatos va dentro del directorio
                metadata_file = os.path.join(output_file, 'metadata.json')
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Modelo {model_name} (v{version}) exportado a {output_file}")
        return output_file
    
    def import_model(
        self, 
        model_path: str, 
        model_name: str,
        model_type: Optional[str] = None,
        metadata_path: Optional[str] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Importa un modelo desde un archivo externo.
        
        Args:
            model_path: Ruta al archivo o directorio del modelo
            model_name: Nombre a asignar al modelo importado
            model_type: Tipo del modelo ('sklearn', 'xgboost', 'tensorflow', 'ensemble')
            metadata_path: Ruta al archivo de metadatos (opcional)
            version: Versión a asignar (opcional)
            
        Returns:
            Versión asignada al modelo importado
        """
        # Inferir tipo de modelo si no se proporciona
        if model_type is None:
            if os.path.isdir(model_path):
                # Probablemente sea un modelo TensorFlow
                model_type = 'tensorflow'
            elif model_path.endswith('.xgb'):
                model_type = 'xgboost'
            elif model_path.endswith('.pkl'):
                # Podría ser sklearn o ensemble, se determinará al cargar
                model_type = 'sklearn'
            else:
                raise ValueError(f"No se pudo determinar el tipo de modelo para {model_path}")
        
        # Cargar el modelo según su tipo
        model = None
        if model_type == 'sklearn' or model_type == 'ensemble':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Verificar si es un ensemble
            if hasattr(model, 'models') or (hasattr(model, 'model_type') and model.model_type == 'ensemble'):
                model_type = 'ensemble'
        
        elif model_type == 'xgboost':
            if xgb is None:
                raise ImportError("XGBoost no está instalado")
            model = xgb.Booster()
            model.load_model(model_path)
        
        elif model_type == 'tensorflow':
            if tf is None:
                raise ImportError("TensorFlow no está instalado")
            model = tf.keras.models.load_model(model_path)
        
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Cargar metadatos si se proporcionan
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Si no hay metadatos pero el modelo es TensorFlow, buscar en el directorio
        if not metadata and model_type == 'tensorflow':
            metadata_in_dir = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_in_dir):
                with open(metadata_in_dir, 'r') as f:
                    metadata = json.load(f)
        
        # Guardar el modelo importado
        saved_path = self.save(model, model_name, metadata, version)
        
        # Extraer la versión del path
        saved_version = os.path.basename(os.path.dirname(saved_path))
        
        logger.info(f"Modelo importado desde {model_path} como {model_name} (v{saved_version})")
        return saved_version
    
    def compare_models(
        self, 
        model_name1: str, 
        model_name2: str, 
        version1: Optional[str] = None, 
        version2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compara dos modelos basándose en sus metadatos.
        
        Args:
            model_name1: Nombre del primer modelo
            model_name2: Nombre del segundo modelo
            version1: Versión del primer modelo (None para la última)
            version2: Versión del segundo modelo (None para la última)
            
        Returns:
            Diccionario con resultados de la comparación
        """
        # Obtener información de ambos modelos
        info1 = self.get_model_info(model_name1, version1)
        info2 = self.get_model_info(model_name2, version2)
        
        # Verificar que ambos tienen metadatos
        if 'metadata' not in info1 or 'metadata' not in info2:
            raise ValueError("No se pueden comparar modelos sin metadatos")
        
        metadata1 = info1['metadata']
        metadata2 = info2['metadata']
        
        # Comparar métricas si están disponibles
        metrics_comparison = {}
        if 'metrics' in metadata1 and 'metrics' in metadata2:
            metrics1 = metadata1['metrics']
            metrics2 = metadata2['metrics']
            
            # Encontrar métricas comunes
            common_metrics = set(metrics1.keys()) & set(metrics2.keys())
            
            for metric in common_metrics:
                value1 = metrics1[metric]
                value2 = metrics2[metric]
                
                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    diff = value1 - value2
                    pct_change = (diff / value2) * 100 if value2 != 0 else float('inf')
                    
                    metrics_comparison[metric] = {
                        f"{model_name1}": value1,
                        f"{model_name2}": value2,
                        'difference': diff,
                        'pct_change': pct_change
                    }
        
        # Comparar hiperparámetros si están disponibles
        params_comparison = {}
        if 'hyperparameters' in metadata1 and 'hyperparameters' in metadata2:
            params1 = metadata1['hyperparameters']
            params2 = metadata2['hyperparameters']
            
            # Encontrar parámetros comunes
            common_params = set(params1.keys()) & set(params2.keys())
            only_in_1 = set(params1.keys()) - set(params2.keys())
            only_in_2 = set(params2.keys()) - set(params1.keys())
            
            for param in common_params:
                value1 = params1[param]
                value2 = params2[param]
                
                params_comparison[param] = {
                    f"{model_name1}": value1,
                    f"{model_name2}": value2,
                    'is_different': value1 != value2
                }
            
            for param in only_in_1:
                params_comparison[param] = {
                    f"{model_name1}": params1[param],
                    f"{model_name2}": None,
                    'is_different': True
                }
            
            for param in only_in_2:
                params_comparison[param] = {
                    f"{model_name1}": None,
                    f"{model_name2}": params2[param],
                    'is_different': True
                }
        
        # Preparar resultado
        result = {
            'models': {
                model_name1: {
                    'version': version1 or info1.get('version', 'latest'),
                    'type': info1.get('model_type', 'unknown'),
                    'saved_at': info1.get('saved_at', 'unknown')
                },
                model_name2: {
                    'version': version2 or info2.get('version', 'latest'),
                    'type': info2.get('model_type', 'unknown'),
                    'saved_at': info2.get('saved_at', 'unknown')
                }
            },
            'metrics_comparison': metrics_comparison,
            'params_comparison': params_comparison,
            'is_same_type': info1.get('model_type') == info2.get('model_type')
        }
        
        return result
    
    def get_best_model(
        self, 
        model_prefix: str = '', 
        metric_name: str = 'accuracy', 
        higher_is_better: bool = True
    ) -> Tuple[str, str]:
        """
        Encuentra el mejor modelo según una métrica específica.
        
        Args:
            model_prefix: Prefijo para filtrar modelos (opcional)
            metric_name: Nombre de la métrica a usar para comparación
            higher_is_better: True si valores más altos son mejores, False en caso contrario
            
        Returns:
            Tupla con (nombre_modelo, versión) del mejor modelo
        """
        # Cargar índice
        index = self._load_index()
        
        best_model = None
        best_version = None
        best_value = float('-inf') if higher_is_better else float('inf')
        
        # Filtrar modelos por prefijo si se proporciona
        model_names = [name for name in index['models'].keys() 
                      if not model_prefix or name.startswith(model_prefix)]
        
        for model_name in model_names:
            for version, info in index['models'][model_name].items():
                # Obtener metadatos
                model_path = os.path.join(self.base_dir, info['path'])
                metadata_file = os.path.join(model_path, 'metadata.json')
                
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Verificar si la métrica está disponible
                    if 'metrics' in metadata and metric_name in metadata['metrics']:
                        metric_value = metadata['metrics'][metric_name]
                        
                        # Comparar con el mejor actual
                        if (higher_is_better and metric_value > best_value) or \
                           (not higher_is_better and metric_value < best_value):
                            best_model = model_name
                            best_version = version
                            best_value = metric_value
                
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    # Ignorar modelos sin metadatos o sin la métrica
                    continue
        
        if best_model is None:
            logger.warning(f"No se encontró ningún modelo con la métrica {metric_name}")
            return None, None
        
        logger.info(f"Mejor modelo según {metric_name}: {best_model} (v{best_version}) con valor {best_value}")
        return best_model, best_version    
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """
        Elimina un modelo o una versión específica.
        
        Args:
            model_name: Nombre del modelo a eliminar
            version: Versión específica (None para eliminar todas las versiones)
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        import shutil
        
        index = self._load_index()
        
        if model_name not in index['models']:
            logger.error(f"Modelo {model_name} no encontrado")
            return False
        
        if version is None:
            # Eliminar todas las versiones del modelo
            for v in list(index['models'][model_name].keys()):
                model_path = os.path.join(self.base_dir, index['models'][model_name][v]['path'])
                try:
                    shutil.rmtree(model_path)
                    logger.info(f"Eliminada versión {v} del modelo {model_name}")
                except Exception as e:
                    logger.error(f"Error al eliminar {model_path}: {str(e)}")
                    return False
            
            # Eliminar del índice
            del index['models'][model_name]
            if model_name in index['latest_versions']:
                del index['latest_versions'][model_name]
            
            # Eliminar directorio del modelo si está vacío
            model_type = None
            for t in ['sklearn', 'xgboost', 'tensorflow', 'ensemble']:
                model_dir = os.path.join(self._get_model_directory(t), model_name)
                if os.path.exists(model_dir):
                    model_type = t
                    if not os.listdir(model_dir):
                        os.rmdir(model_dir)
            
            logger.info(f"Eliminado modelo {model_name} completamente")
        
        else:
            # Eliminar solo la versión especificada
            if version not in index['models'][model_name]:
                logger.error(f"Versión {version} no encontrada para modelo {model_name}")
                return False
            
            model_path = os.path.join(self.base_dir, index['models'][model_name][version]['path'])
            try:
                shutil.rmtree(model_path)
                logger.info(f"Eliminada versión {version} del modelo {model_name}")
            except Exception as e:
                logger.error(f"Error al eliminar {model_path}: {str(e)}")
                return False
            
            # Eliminar del índice
            del index['models'][model_name][version]
            
            # Si era la última versión, actualizar o eliminar la referencia
            if index['latest_versions'].get(model_name) == version:
                if index['models'][model_name]:
                    # Establecer la versión más reciente como la última
                    versions = list(index['models'][model_name].keys())
                    versions.sort(reverse=True)
                    index['latest_versions'][model_name] = versions[0]
                else:
                    # No quedan versiones, eliminar la referencia
                    del index['latest_versions'][model_name]
                    
                    # Y también eliminar el modelo si no quedan versiones
                    del index['models'][model_name]
                    
                    # Eliminar directorio del modelo si está vacío
                    model_type = None
                    for t in ['sklearn', 'xgboost', 'tensorflow', 'ensemble']:
                        model_dir = os.path.join(self._get_model_directory(t), model_name)
                        if os.path.exists(model_dir):
                            model_type = t
                            if not os.listdir(model_dir):
                                os.rmdir(model_dir)
        
        # Guardar cambios en el índice
        self._save_index(index)
        
        return True