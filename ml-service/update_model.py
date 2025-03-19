#!/usr/bin/env python3
"""
Script para actualizar el modelo de predicción de tenis con nuevos datos.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
import os
from datetime import datetime
from utils import TennisFeatureEngineering

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml-service/logs/model_updates.log'),
        logging.StreamHandler()
    ]
)

def update_model(new_data_path: str, model_path: str, output_dir: str = 'ml-service/model') -> bool:
    """
    Actualiza el modelo con nuevos datos.
    
    Args:
        new_data_path: Ruta al archivo CSV con nuevos datos
        model_path: Ruta al modelo actual
        output_dir: Directorio para guardar el modelo actualizado
        
    Returns:
        True si la actualización fue exitosa, False en caso contrario
    """
    try:
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar modelo actual o crear uno nuevo
        try:
            model = joblib.load(model_path)
            logging.info(f"Modelo actual cargado desde {model_path}")
        except:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            logging.info("Creando nuevo modelo")
        
        # Cargar nuevos datos
        new_data = pd.read_csv(new_data_path)
        logging.info(f"Nuevos datos cargados: {len(new_data)} partidos")
        
        # Preparar características
        feature_engineering = TennisFeatureEngineering()
        X = feature_engineering.extract_features(new_data)
        y = new_data['winner']
        
        # Verificar compatibilidad de características
        if hasattr(model, 'feature_names_in_'):
            missing_features = set(model.feature_names_in_) - set(X.columns)
            if missing_features:
                logging.warning(f"Características faltantes en nuevos datos: {missing_features}")
                for feature in missing_features:
                    X[feature] = 0  # Añadir con valor por defecto
        
        # Actualizar modelo
        if isinstance(model, RandomForestClassifier):
            model.warm_start = True
            model.fit(X, y)
        else:
            model.fit(X, y)
        
        # Guardar modelo actualizado
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        updated_model_path = os.path.join(output_dir, f'model_{timestamp}.pkl')
        joblib.dump(model, updated_model_path)
        logging.info(f"Modelo actualizado guardado en {updated_model_path}")
        
        # Guardar metadatos de actualización
        metadata = {
            'timestamp': timestamp,
            'model_type': type(model).__name__,
            'features': X.columns.tolist(),
            'new_data_size': len(new_data),
            'model_params': model.get_params()
        }
        
        metadata_path = os.path.join(output_dir, f'update_metadata_{timestamp}.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=4)
        
        logging.info(f"Metadatos guardados en {metadata_path}")
        
        # Actualizar modelo principal
        main_model_path = os.path.join(output_dir, 'model.pkl')
        joblib.dump(model, main_model_path)
        logging.info(f"Modelo principal actualizado en {main_model_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error actualizando modelo: {e}")
        return False

def main():
    """Función principal para actualizar el modelo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Actualizar modelo de predicción de tenis')
    parser.add_argument('--new-data', required=True,
                      help='Ruta al archivo CSV con nuevos datos')
    parser.add_argument('--model', required=True,
                      help='Ruta al modelo actual')
    parser.add_argument('--output-dir', default='ml-service/model',
                      help='Directorio para guardar el modelo actualizado')
    
    args = parser.parse_args()
    
    success = update_model(args.new_data, args.model, args.output_dir)
    if success:
        logging.info("Actualización del modelo completada exitosamente")
    else:
        logging.error("Error en la actualización del modelo")

if __name__ == '__main__':
    main() 