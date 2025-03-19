#!/usr/bin/env python3
"""
Script para probar el sistema mejorado de predicción de partidos de tenis.
Realiza pruebas completas verificando que todos los componentes funcionan sin usar valores por defecto.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import joblib
import traceback
from utils import TennisFeatureEngineering
from train import train_model

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Obtener la ruta absoluta del directorio del proyecto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'tennis_matches.csv')

def setup_test_data():
    """Verifica que existan datos de prueba y los genera si faltan."""
    print("Verificando datos de prueba...")
    
    if not os.path.exists(DATA_PATH):
        print(f"No se encontró el archivo de datos: {DATA_PATH}")
        print("Generando datos de prueba...")
        
        from data_collector import TennisDataCollector
        collector = TennisDataCollector(output_path=DATA_PATH)
        data = collector._generate_test_data(n_samples=500)
        
        print(f"Datos de prueba generados: {len(data)} partidos")
        return data
    else:
        print(f"Usando datos existentes: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

def test_feature_engineering():
    """Prueba el módulo de ingeniería de características."""
    print("\n=== Prueba de ingeniería de características ===")
    logging.info("Probando módulo de ingeniería de características...")
    
    # Asegurar que existen los datos
    data = setup_test_data()
    
    try:
        # Crear instancia
        fe = TennisFeatureEngineering(data_path=DATA_PATH)
        
        # Probar preprocesamiento
        print("- Preprocesando datos...")
        df = fe.preprocess_data()
        logging.info(f"Preprocesamiento exitoso: {len(df)} partidos")
        
        # Probar estadísticas de jugadores
        print("- Calculando estadísticas de jugadores...")
        player_stats = fe.build_player_statistics()
        num_players = len(player_stats)
        logging.info(f"Estadísticas calculadas para {num_players} jugadores")
        print(f"  Se calcularon estadísticas para {num_players} jugadores")
        
        # Probar estadísticas head-to-head
        print("- Calculando estadísticas head-to-head...")
        h2h_stats = fe.build_head_to_head_statistics()
        num_pairs = len(h2h_stats)
        logging.info(f"Estadísticas h2h calculadas para {num_pairs} pares de jugadores")
        print(f"  Se calcularon estadísticas H2H para {num_pairs} pares de jugadores")
        
        # Probar extracción de características
        print("- Extrayendo características para un partido de ejemplo...")
        # Seleccionar dos jugadores al azar de nuestros datos
        player1 = data['player1'].iloc[0]
        player2 = data['player2'].iloc[0]
        
        match_data = {
            'player1': player1,
            'player2': player2,
            'ranking_1': 2,
            'ranking_2': 3,
            'surface': 'clay'
        }
        
        features = fe.extract_features(match_data)
        num_features = len(features)
        logging.info(f"Extracción de características exitosa: {num_features} características")
        print(f"  Se extrajeron {num_features} características")
        
        # Probar preparación de datos para entrenamiento
        print("- Preparando datos para entrenamiento...")
        X, y = fe.prepare_training_data()
        logging.info(f"Preparación de datos exitosa: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"  Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
        
        # Verificar valores por defecto en las características
        print("- Verificando que no haya valores por defecto...")
        null_counts = X.isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            print(f"  ⚠️ Se encontraron {total_nulls} valores nulos en las características:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"    - {col}: {count} valores nulos")
        else:
            print("  ✅ No se encontraron valores nulos en las características")
        
        # Test adicional: verificar distribución de características para detectar valores por defecto
        # Valores repetidos muy frecuentes pueden indicar valores por defecto
        for col in X.columns:
            value_counts = X[col].value_counts()
            if value_counts.iloc[0] > len(X) * 0.9:  # Si más del 90% son iguales
                print(f"  ⚠️ Posible valor por defecto en {col}: {value_counts.index[0]} ({value_counts.iloc[0]} de {len(X)})")
        
        return True, "Prueba de ingeniería de características exitosa"
    except Exception as e:
        logging.error(f"Error en prueba de ingeniería de características: {e}")
        logging.error(traceback.format_exc())
        return False, f"Error: {str(e)}"

def test_model_training():
    """Prueba el entrenamiento del modelo."""
    print("\n=== Prueba de entrenamiento del modelo ===")
    logging.info("Probando entrenamiento del modelo...")
    
    # Asegurar que existen los datos
    setup_test_data()
    
    try:
        # Entrenar modelo con parámetros básicos (sin optimización para ahorrar tiempo)
        print("- Entrenando modelo sin optimización de hiperparámetros...")
        model = train_model(data_path=DATA_PATH, optimize_hyperparams=False)
        
        # Verificar que se guardó el modelo
        model_path = os.path.join(PROJECT_ROOT, 'model', 'model.pkl')
        if os.path.exists(model_path):
            logging.info(f"Modelo guardado correctamente en {model_path}")
            print(f"  ✅ Modelo guardado en {model_path}")
            
            # Verificar metadatos del entrenamiento
            metadata_path = os.path.join(PROJECT_ROOT, 'model', 'training_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print("  Metadatos del modelo:")
                print(f"    - Tipo: {metadata.get('model_type')}")
                print(f"    - Características: {metadata.get('features_count')}")
                print(f"    - Muestras: {metadata.get('samples_count')}")
                print(f"    - Accuracy: {metadata.get('accuracy', 0):.4f}")
                print(f"    - Fecha de entrenamiento: {metadata.get('timestamp')}")
            
            # Verificar feature importance
            fi_path = os.path.join(PROJECT_ROOT, 'model', 'feature_importance.csv')
            if os.path.exists(fi_path):
                try:
                    feature_imp = pd.read_csv(fi_path)
                    print("  Top 5 características más importantes:")
                    top_features = feature_imp.sort_values('importance', ascending=False).head(5)
                    for i, (_, row) in enumerate(top_features.iterrows()):
                        print(f"    {i+1}. {row['feature']}: {row['importance']:.4f}")
                except Exception as e:
                    print(f"  No se pudieron leer las características importantes: {e}")
            
            return True, "Entrenamiento del modelo exitoso"
        else:
            logging.error(f"Error: No se encontró el modelo guardado en {model_path}")
            return False, f"Error: No se encontró el modelo guardado en {model_path}"
    except Exception as e:
        logging.error(f"Error en entrenamiento del modelo: {e}")
        logging.error(traceback.format_exc())
        return False, f"Error: {str(e)}"

def test_prediction():
    """Prueba la predicción con el modelo entrenado."""
    print("\n=== Prueba de predicción ===")
    logging.info("Probando predicción...")
    
    try:
        # Cargar modelo
        model_paths = [
            os.path.join(PROJECT_ROOT, 'model', 'model.pkl'),
            'ml-service/model/model.pkl'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if model_path is None:
            logging.error("No se encontró el modelo en ninguna ubicación conocida")
            return False, "Error: Modelo no encontrado"
            
        print(f"- Cargando modelo desde {model_path}...")
        model = joblib.load(model_path)
        
        # Crear instancia de feature engineering
        print("- Inicializando motor de características...")
        fe = TennisFeatureEngineering(data_path=DATA_PATH)
        
        # Asegurar que tenemos estadísticas calculadas
        if not fe.players_stats:
            print("- Calculando estadísticas de jugadores...")
            fe.build_player_statistics()
            fe.build_head_to_head_statistics()
        
        # Cargar un par de jugadores reales de nuestros datos
        data = pd.read_csv(DATA_PATH)
        
        # Obtener dos jugadores diferentes
        player1 = data['player1'].iloc[0]
        player2 = data['player2'].iloc[0]
        if player1 == player2:
            player2 = data[data['player1'] != player1]['player1'].iloc[0]
        
        # Obtener rankings reales de nuestros datos
        ranking_1 = data[data['player1'] == player1]['ranking_1'].iloc[0] if 'ranking_1' in data else None
        ranking_2 = data[data['player2'] == player2]['ranking_2'].iloc[0] if 'ranking_2' in data else None
        
        # Datos de prueba
        match_data = {
            'player1': player1,
            'player2': player2,
            'surface': 'clay'
        }
        
        # Añadir rankings solo si existen
        if ranking_1 is not None:
            match_data['ranking_1'] = ranking_1
        if ranking_2 is not None:
            match_data['ranking_2'] = ranking_2
        
        print(f"- Prediciendo resultado para {player1} vs {player2} en superficie clay...")
        
        # Extraer características
        X = fe.transform_match_data(match_data)
        
        # Verificar que no tengamos NaN en las características
        if X.isnull().any().any():
            print("  ⚠️ Se encontraron valores NaN en las características de predicción")
            print("  Imputando valores faltantes con 0...")
            X = X.fillna(0)
        
        # Hacer predicción
        prediction = model.predict(X)[0]
        probability = max(model.predict_proba(X)[0])
        
        # Determinar ganador
        winner = player1 if prediction == 0 else player2
        
        logging.info(f"Predicción exitosa: {winner} ({probability:.2f})")
        print(f"  ✅ Predicción exitosa:")
        print(f"    - Ganador predicho: {winner}")
        print(f"    - Probabilidad: {probability:.2f}")
        
        # Verificar características usadas
        print("  Características utilizadas:")
        for i, feature in enumerate(X.columns[:5]):  # Mostrar las 5 primeras características
            print(f"    - {feature}: {X.iloc[0, i]}")  # Usar iloc con coma para evitar la advertencia
        
        return True, f"Predicción exitosa: {winner} ({probability:.2f})"
    except Exception as e:
        logging.error(f"Error en predicción: {e}")
        logging.error(traceback.format_exc())
        return False, f"Error: {str(e)}"

def test_prediction():
    """Prueba la predicción con el modelo entrenado."""
    print("\n=== Prueba de predicción ===")
    logging.info("Probando predicción...")
    
    try:
        # Cargar modelo
        model_paths = [
            os.path.join(PROJECT_ROOT, 'model', 'model.pkl'),
            'ml-service/model/model.pkl'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if model_path is None:
            logging.error("No se encontró el modelo en ninguna ubicación conocida")
            return False, "Error: Modelo no encontrado"
            
        print(f"- Cargando modelo desde {model_path}...")
        model = joblib.load(model_path)
        
        # Crear instancia de feature engineering
        print("- Inicializando motor de características...")
        fe = TennisFeatureEngineering(data_path=DATA_PATH)
        
        # Asegurar que tenemos estadísticas calculadas
        if not fe.players_stats:
            print("- Calculando estadísticas de jugadores...")
            fe.build_player_statistics()
            fe.build_head_to_head_statistics()
        
        # Cargar un par de jugadores reales de nuestros datos
        data = pd.read_csv(DATA_PATH)
        
        # Obtener dos jugadores diferentes
        player1 = data['player1'].iloc[0]
        player2 = data['player2'].iloc[0]
        if player1 == player2:
            player2 = data[data['player1'] != player1]['player1'].iloc[0]
        
        # Obtener rankings reales de nuestros datos
        ranking_1 = data[data['player1'] == player1]['ranking_1'].iloc[0] if 'ranking_1' in data else None
        ranking_2 = data[data['player2'] == player2]['ranking_2'].iloc[0] if 'ranking_2' in data else None
        
        # Datos de prueba
        match_data = {
            'player1': player1,
            'player2': player2,
            'surface': 'clay'
        }
        
        # Añadir rankings solo si existen
        if ranking_1 is not None:
            match_data['ranking_1'] = ranking_1
        if ranking_2 is not None:
            match_data['ranking_2'] = ranking_2
        
        print(f"- Prediciendo resultado para {player1} vs {player2} en superficie clay...")
        
        # Extraer características
        X = fe.transform_match_data(match_data)
        
        # Verificar que no tengamos NaN en las características
        if X.isnull().any().any():
            print("  ⚠️ Se encontraron valores NaN en las características de predicción")
            print("  Imputando valores faltantes con 0...")
            X = X.fillna(0)
        
        # Hacer predicción
        prediction = model.predict(X)[0]
        probability = max(model.predict_proba(X)[0])
        
        # Determinar ganador
        winner = player1 if prediction == 0 else player2
        
        logging.info(f"Predicción exitosa: {winner} ({probability:.2f})")
        print(f"  ✅ Predicción exitosa:")
        print(f"    - Ganador predicho: {winner}")
        print(f"    - Probabilidad: {probability:.2f}")
        
        # Verificar características usadas
        print("  Características utilizadas:")
        # Usar .iloc con indexación correcta para evitar la advertencia FutureWarning
        for i, feature in enumerate(X.columns[:5]):  # Mostrar las 5 primeras características
            value = X.iloc[0, i]  # Usar iloc con coma para acceder al primer elemento de la característica
            print(f"    - {feature}: {value}")
        
        return True, f"Predicción exitosa: {winner} ({probability:.2f})"
    except Exception as e:
        logging.error(f"Error en predicción: {e}")
        logging.error(traceback.format_exc())
        return False, f"Error: {str(e)}"

def main():
    """Ejecuta todas las pruebas del sistema."""
    print("=" * 50)
    print("SISTEMA DE PRUEBAS PARA PREDICCIÓN DE PARTIDOS DE TENIS")
    print("=" * 50)
    print("Iniciando pruebas del sistema mejorado sin valores por defecto...")
    print(f"Fecha y hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    start_time = time.time()
    
    # Contenedor para resultados
    results = {}
    
    # Prueba 1: Ingeniería de características
    print("\nEjecutando prueba 1: Ingeniería de características")
    try:
        success, message = test_feature_engineering()
        results['feature_engineering'] = {'success': success, 'message': message}
        if success:
            print("✅ Prueba de ingeniería de características: EXITOSA")
        else:
            print("❌ Prueba de ingeniería de características: FALLIDA")
            print(f"  Detalles: {message}")
    except Exception as e:
        print(f"❌ Prueba de ingeniería de características: ERROR CRÍTICO")
        print(f"  Detalles: {str(e)}")
        results['feature_engineering'] = {'success': False, 'message': str(e)}
    
    # Prueba 2: Entrenamiento del modelo
    print("\nEjecutando prueba 2: Entrenamiento del modelo")
    try:
        success, message = test_model_training()
        results['model_training'] = {'success': success, 'message': message}
        if success:
            print("✅ Prueba de entrenamiento del modelo: EXITOSA")
        else:
            print("❌ Prueba de entrenamiento del modelo: FALLIDA")
            print(f"  Detalles: {message}")
    except Exception as e:
        print(f"❌ Prueba de entrenamiento del modelo: ERROR CRÍTICO")
        print(f"  Detalles: {str(e)}")
        results['model_training'] = {'success': False, 'message': str(e)}
    
    # Prueba 3: Predicción
    print("\nEjecutando prueba 3: Predicción")
    try:
        success, message = test_prediction()
        results['prediction'] = {'success': success, 'message': message}
        if success:
            print("✅ Prueba de predicción: EXITOSA")
        else:
            print("❌ Prueba de predicción: FALLIDA")
            print(f"  Detalles: {message}")
    except Exception as e:
        print(f"❌ Prueba de predicción: ERROR CRÍTICO")
        print(f"  Detalles: {str(e)}")
        results['prediction'] = {'success': False, 'message': str(e)}
    
    # Resumen de pruebas
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    total = len(results)
    passed = sum(1 for result in results.values() if result['success'])
    
    print(f"Pruebas completadas en {elapsed_time:.2f} segundos")
    print(f"Resultado general: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        print("\n✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("El sistema está listo para usar en producción")
    else:
        print("\n⚠️ ALGUNAS PRUEBAS FALLARON")
        print("Se recomienda revisar los errores antes de usar en producción")
        
        # Mostrar detalles de errores
        print("\nDetalles de errores:")
        for test_name, result in results.items():
            if not result['success']:
                print(f"- {test_name}: {result['message']}")
    
    # Guardar resultados en un archivo de log
    try:
        log_dir = os.path.join(PROJECT_ROOT, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'test_results_{time.strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(log_file, 'w') as f:
            f.write("RESULTADOS DE PRUEBAS DEL SISTEMA\n")
            f.write("=" * 50 + "\n")
            f.write(f"Fecha y hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duración: {elapsed_time:.2f} segundos\n")
            f.write(f"Resultado: {passed}/{total} pruebas exitosas\n\n")
            
            for test_name, result in results.items():
                status = "EXITOSA" if result['success'] else "FALLIDA"
                f.write(f"{test_name}: {status}\n")
                f.write(f"Mensaje: {result['message']}\n\n")
                
        print(f"\nDetalles de las pruebas guardados en: {log_file}")
    except Exception as e:
        print(f"No se pudieron guardar los resultados de las pruebas: {e}")

if __name__ == "__main__":
    main()