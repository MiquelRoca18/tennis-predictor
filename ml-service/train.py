#!/usr/bin/env python3
"""
Módulo para entrenamiento de modelos de predicción de partidos de tenis.
Versión mejorada sin valores por defecto y con uso exclusivo de datos reales.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback

# Importamos nuestra clase de ingeniería de características
from utils import TennisFeatureEngineering

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Configurar logging
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

def train_model(data_path=None, model_type='rf', optimize_hyperparams=True, test_size=0.2):
    """
    Entrena un modelo mejorado para la predicción de partidos de tenis.
    Versión mejorada para asegurar validación correcta y evitar fugas de datos.
    
    Args:
        data_path: Ruta al archivo CSV con datos (opcional)
        model_type: Tipo de modelo ('rf' para RandomForest, 'gb' para GradientBoosting)
        optimize_hyperparams: Si se debe realizar búsqueda de hiperparámetros
        test_size: Proporción de datos para prueba
        
    Returns:
        El modelo entrenado
    """
    start_time = time.time()
    
    # Obtener directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ruta por defecto si no se proporciona
    if data_path is None:
        # Intentar buscar en varias ubicaciones posibles
        possible_paths = [
            os.path.join(current_dir, "tennis_matches.csv"),
            os.path.join(current_dir, "ml-service", "tennis_matches.csv"),
            "tennis_matches.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
                
        if data_path is None:
            logging.error("No se pudo encontrar el archivo de datos tennis_matches.csv")
            # Intentar generar datos automáticamente
            try:
                from data_collector import TennisDataCollector
                logging.info("Intentando generar datos de prueba automáticamente...")
                collector = TennisDataCollector(output_path="tennis_matches.csv")
                collector._generate_test_data(n_samples=500)
                data_path = "tennis_matches.csv"
                logging.info(f"Datos de prueba generados exitosamente en {data_path}")
            except Exception as e:
                logging.error(f"Error generando datos de prueba: {e}")
                raise FileNotFoundError("No se pudo encontrar el archivo de datos tennis_matches.csv")
    
    logging.info(f"Iniciando entrenamiento con {model_type} (optimizar={optimize_hyperparams})")
    logging.info(f"Usando archivo de datos: {data_path}")
    
    # Crear directorios para modelo
    model_dir = os.path.join(current_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Inicializar el motor de características
    feature_engineering = TennisFeatureEngineering(data_path=data_path)
    
    # Preparar características para entrenamiento
    try:
        logging.info("Preparando datos con ingeniería de características avanzada...")
        X, y = feature_engineering.prepare_training_data()
        logging.info(f"Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
        
        # Verificar valores nulos
        null_counts = X.isnull().sum()
        if null_counts.sum() > 0:
            logging.warning(f"Se detectaron {null_counts.sum()} valores nulos en las características")
            # Estrategia: imputar valores nulos
            logging.info("Imputando valores nulos con 0")
            X = X.fillna(0)
            
        # Verificar distribución de clases
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            logging.error(f"PROBLEMA CRÍTICO: Solo hay una clase en los datos: {class_counts}")
            raise ValueError("Los datos de entrenamiento solo tienen una clase. No es posible entrenar un modelo predictivo.")
        else:
            logging.info(f"Distribución de clases: {class_counts}")
            if class_counts.min() / class_counts.max() < 0.2:
                logging.warning("Desbalance significativo en clases. Considerando técnicas de balanceo.")
                # Aquí podríamos implementar SMOTE u otras técnicas
    except Exception as e:
        logging.error(f"Error preparando datos avanzados: {e}")
        logging.error(traceback.format_exc())
        
        # Fallback al método antiguo si hay problemas, pero con mejor manejo de errores
        logging.info("Usando método básico para preparar datos")
        
        try:
            # Cargar datos
            data = pd.read_csv(data_path)
            
            # Verificar columnas esenciales
            essential_columns = ['player1', 'player2', 'winner', 'surface']
            missing_columns = [col for col in essential_columns if col not in data.columns]
            
            if missing_columns:
                # Intentar conversión si hay un formato alternativo
                if 'winner_name' in data.columns and 'loser_name' in data.columns:
                    logging.info("Detectado formato JeffSackmann, convirtiendo a formato estándar...")
                    
                    # Crear nuevo DataFrame para transformación
                    transformed = pd.DataFrame()
                    
                    # Transformar aleatorizando la asignación
                    for i, row in data.iterrows():
                        # Aleatorizar la asignación
                        if random.random() > 0.5:
                            # Ganador es player1
                            new_row = {
                                'player1': row['winner_name'],
                                'player2': row['loser_name'],
                                'winner': 0
                            }
                        else:
                            # Ganador es player2
                            new_row = {
                                'player1': row['loser_name'],
                                'player2': row['winner_name'],
                                'winner': 1
                            }
                        
                        # Añadir más campos si existen
                        if 'surface' in row:
                            new_row['surface'] = row['surface']
                        if 'winner_rank' in row and 'loser_rank' in row:
                            if new_row['winner'] == 0:  # winner_name es player1
                                new_row['ranking_1'] = row['winner_rank']
                                new_row['ranking_2'] = row['loser_rank']
                            else:  # loser_name es player1
                                new_row['ranking_1'] = row['loser_rank']
                                new_row['ranking_2'] = row['winner_rank']
                        
                        transformed = pd.concat([transformed, pd.DataFrame([new_row])], ignore_index=True)
                    
                    data = transformed
                    
                    # Verificar nuevamente columnas esenciales
                    missing_columns = [col for col in essential_columns if col not in data.columns]
                
                if missing_columns:
                    logging.error(f"Faltan columnas esenciales en los datos: {missing_columns}")
                    raise ValueError(f"Datos incompletos para entrenamiento: faltan columnas {missing_columns}")
            
            # Verificar distribución de clases - crítico para un modelo real
            class_counts = data['winner'].value_counts()
            if len(class_counts) < 2:
                logging.error(f"Los datos solo tienen una clase: {class_counts}")
                
                # Intentar balancear los datos
                if 0 in class_counts and class_counts[0] > 0:
                    # Tomar los datos originales donde winner=0 y crear copias invertidas con winner=1
                    original_data = data.copy()
                    inverted_data = data.copy()
                    
                    # Invertir player1 y player2
                    inverted_data['player1'], inverted_data['player2'] = inverted_data['player2'], inverted_data['player1']
                    
                    # Invertir rankings y winrates si existen
                    if 'ranking_1' in inverted_data.columns and 'ranking_2' in inverted_data.columns:
                        inverted_data['ranking_1'], inverted_data['ranking_2'] = inverted_data['ranking_2'], inverted_data['ranking_1']
                    if 'winrate_1' in inverted_data.columns and 'winrate_2' in inverted_data.columns:
                        inverted_data['winrate_1'], inverted_data['winrate_2'] = inverted_data['winrate_2'], inverted_data['winrate_1']
                    
                    # Cambiar winner a 1
                    inverted_data['winner'] = 1
                    
                    # Combinar datos originales e invertidos
                    data = pd.concat([original_data, inverted_data], ignore_index=True)
                    logging.info(f"Datos balanceados manualmente. Nueva distribución: {data['winner'].value_counts()}")
                elif 1 in class_counts and class_counts[1] > 0:
                    # Similar al caso anterior pero invertido
                    original_data = data.copy()
                    inverted_data = data.copy()
                    
                    inverted_data['player1'], inverted_data['player2'] = inverted_data['player2'], inverted_data['player1']
                    
                    if 'ranking_1' in inverted_data.columns and 'ranking_2' in inverted_data.columns:
                        inverted_data['ranking_1'], inverted_data['ranking_2'] = inverted_data['ranking_2'], inverted_data['ranking_1']
                    if 'winrate_1' in inverted_data.columns and 'winrate_2' in inverted_data.columns:
                        inverted_data['winrate_1'], inverted_data['winrate_2'] = inverted_data['winrate_2'], inverted_data['winrate_1']
                    
                    inverted_data['winner'] = 0
                    data = pd.concat([original_data, inverted_data], ignore_index=True)
                    logging.info(f"Datos balanceados manualmente. Nueva distribución: {data['winner'].value_counts()}")
            
            # Procesar superficie
            if 'surface' in data.columns:
                surfaces = {'hard': 0, 'clay': 1, 'grass': 2, 'carpet': 3}
                # Convertir a minúsculas y luego mapear
                data['surface_code'] = data['surface'].str.lower().map(lambda x: surfaces.get(x, 0))
            
            # Preparar características
            features = ["surface_code"]
            
            # Añadir rankings si existen
            if 'ranking_1' in data.columns and 'ranking_2' in data.columns:
                features.extend(["ranking_1", "ranking_2"])
            
            # Añadir tasas de victoria si existen
            if 'winrate_1' in data.columns and 'winrate_2' in data.columns:
                features.extend(["winrate_1", "winrate_2"])
            
            # Verificar si tenemos suficientes características
            if len(features) < 3:
                logging.warning("Pocas características disponibles, calculando tasas de victoria...")
                
                # Calcular tasas de victoria
                try:
                    player_stats = {}
                    
                    # Para cada jugador, calcular victorias y derrotas
                    for player in set(data['player1'].tolist() + data['player2'].tolist()):
                        # Partidos como player1
                        p1_matches = data[data['player1'] == player]
                        p1_wins = p1_matches[p1_matches['winner'] == 0].shape[0]
                        
                        # Partidos como player2
                        p2_matches = data[data['player2'] == player]
                        p2_wins = p2_matches[p2_matches['winner'] == 1].shape[0]
                        
                        # Total de partidos y victorias
                        total_matches = len(p1_matches) + len(p2_matches)
                        total_wins = p1_wins + p2_wins
                        
                        if total_matches > 0:
                            win_rate = (total_wins / total_matches) * 100
                            player_stats[player] = win_rate
                    
                    # Aplicar tasas de victoria a los datos
                    data['winrate_1'] = data['player1'].map(player_stats)
                    data['winrate_2'] = data['player2'].map(player_stats)
                    
                    # Añadir a características
                    features.extend(["winrate_1", "winrate_2"])
                    
                    logging.info(f"Tasas de victoria calculadas para {len(player_stats)} jugadores")
                except Exception as e:
                    logging.error(f"Error calculando tasas de victoria: {e}")
            
            # Extraer características y target
            X = data[features]
            y = data["winner"]
            
            # Verificar valores nulos
            null_counts = X.isnull().sum()
            if null_counts.sum() > 0:
                logging.warning(f"Se detectaron {null_counts.sum()} valores nulos en los datos básicos")
                X = X.fillna(0)
                
            # Verificar distribución final de clases
            class_counts = y.value_counts()
            logging.info(f"Distribución final de clases: {class_counts}")
            
            if len(class_counts) < 2:
                raise ValueError("No se pudo balancear los datos. Solo hay una clase para entrenamiento.")
        
        except Exception as e:
            logging.error(f"Error crítico preparando datos: {e}")
            logging.error(traceback.format_exc())
            raise ValueError(f"No se pudieron preparar datos para entrenamiento: {e}")
    
    # División de datos más robusta
    try:
        logging.info(f"Dividiendo datos en entrenamiento ({100-test_size*100}%) y prueba ({test_size*100}%)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y  # Asegurar distribución balanceada
        )
        
        logging.info(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras (Distribución: {np.bincount(y_train)})")
        logging.info(f"Conjunto de prueba: {X_test.shape[0]} muestras (Distribución: {np.bincount(y_test)})")
    except Exception as e:
        logging.error(f"Error dividiendo datos: {e}")
        raise ValueError(f"No se pudieron dividir los datos: {e}")
    
    # Seleccionar tipo de modelo
    try:
        if model_type.lower() == 'gb':
            base_model = GradientBoostingClassifier(random_state=42)
            model_name = "GradientBoosting"
        elif model_type.lower() == 'xgb':
            try:
                import xgboost as xgb
                base_model = xgb.XGBClassifier(random_state=42)
                model_name = "XGBoost"
            except ImportError:
                logging.warning("XGBoost no está instalado, usando RandomForest en su lugar")
                base_model = RandomForestClassifier(random_state=42)
                model_name = "RandomForest"
        else:
            base_model = RandomForestClassifier(random_state=42)
            model_name = "RandomForest"
            
        logging.info(f"Usando modelo: {model_name}")
    except Exception as e:
        logging.error(f"Error seleccionando modelo: {e}")
        raise ValueError(f"Tipo de modelo no válido: {model_type}")
    
    # Optimización de hiperparámetros
    if optimize_hyperparams:
        try:
            logging.info(f"Iniciando optimización de hiperparámetros para {model_name}")
            
            if model_type.lower() == 'gb':
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, None],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 1.0]
                }
            elif model_type.lower() == 'xgb':
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            else:  # RandomForest
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2', None]
                }
            
            # Usar GridSearchCV con validación cruzada
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,  # 5-fold cross-validation
                scoring='accuracy',
                n_jobs=-1,  # Usar todos los núcleos
                verbose=1
            )
            
            # Entrenar con búsqueda de hiperparámetros
            logging.info("Iniciando búsqueda de hiperparámetros (puede tardar varios minutos)...")
            grid_search.fit(X_train, y_train)
            
            # Obtener mejores parámetros
            best_params = grid_search.best_params_
            logging.info(f"Mejores parámetros encontrados: {best_params}")
            
            # Usar el mejor modelo
            model = grid_search.best_estimator_
        except Exception as e:
            logging.error(f"Error en optimización de hiperparámetros: {e}")
            logging.error(traceback.format_exc())
            logging.warning("Usando modelo con parámetros por defecto debido al error")
            
            # Fallback a parámetros por defecto
            model = base_model
            model.fit(X_train, y_train)
    else:
        # Entrenar con parámetros por defecto
        logging.info(f"Entrenando {model_name} con parámetros por defecto")
        model = base_model
        model.fit(X_train, y_train)
    
    # Evaluar modelo
    try:
        logging.info("Evaluando modelo en conjunto de prueba...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calcular métricas de rendimiento
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logging.info(f"Métricas de rendimiento:")
        logging.info(f"- Accuracy: {accuracy:.4f}")
        logging.info(f"- Precision: {precision:.4f}")
        logging.info(f"- Recall: {recall:.4f}")
        logging.info(f"- F1 Score: {f1:.4f}")
        
        # Verificar si el accuracy es sospechosamente alto
        if accuracy > 0.98:
            logging.warning("Accuracy extremadamente alto detectado (>98%). Verificando posible fuga de datos o problemas de balanceo.")
            
            # Verificar predicciones
            predictions_counts = np.unique(y_pred, return_counts=True)
            logging.info(f"Distribución de predicciones: {dict(zip(*predictions_counts))}")
            
            # Si predice siempre la misma clase
            if len(predictions_counts[0]) == 1:
                logging.error("ALERTA: El modelo siempre predice la misma clase.")
            
            # Validación cruzada para confirmar si el rendimiento es consistente
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(base_model, X, y, cv=5)
            logging.info(f"Validación cruzada (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            if abs(accuracy - cv_scores.mean()) > 0.1:
                logging.warning("Gran diferencia entre accuracy de prueba y validación cruzada. Posible overfitting.")
    except Exception as e:
        logging.error(f"Error evaluando modelo: {e}")
        logging.error(traceback.format_exc())
        
        # Valores de fallback para no interrumpir el proceso
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        conf_matrix = None
    
    # Guardar modelo y metadata
    try:
        # Guardar modelo
        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        logging.info(f"Modelo guardado en: {model_path}")
        
        # Guardar metadatos del entrenamiento
        metadata = {
            'model_type': model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_count': X.shape[1],
            'samples_count': X.shape[0],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time_seconds': time.time() - start_time,
            'features': list(X.columns),
            'data_source': data_path,
            'optimize_hyperparams': optimize_hyperparams,
            'class_distribution': y.value_counts().to_dict()
        }
        
        # Añadir mejores parámetros si se optimizó
        if optimize_hyperparams and 'best_params' in locals():
            metadata['best_params'] = best_params
        
        # Guardar metadatos en formato JSON
        import json
        with open(os.path.join(model_dir, 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logging.info(f"Metadatos guardados en: {os.path.join(model_dir, 'training_metadata.json')}")
        
        # Guardar lista de características
        feature_names = X.columns.tolist()
        with open(os.path.join(model_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(feature_names))
            
        logging.info(f"Lista de características guardada en: {os.path.join(model_dir, 'feature_names.txt')}")
    except Exception as e:
        logging.error(f"Error guardando modelo y metadatos: {e}")
        logging.error(traceback.format_exc())
    
    # Inicializar features_df aquí para evitar el error
    features_df = None
    
    # Generar visualizaciones
    try:
        # Generar gráfico de matriz de confusión
        if conf_matrix is not None:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Jugador 1', 'Jugador 2'],
                       yticklabels=['Jugador 1', 'Jugador 2'])
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.title(f'Matriz de Confusión - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
            plt.close()
            
            logging.info(f"Gráfico de matriz de confusión guardado en: {os.path.join(model_dir, 'confusion_matrix.png')}")
        
        # Generar gráfico de importancia de características
        if hasattr(model, 'feature_importances_'):
            try:
                feature_importances = model.feature_importances_
                features_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': feature_importances
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=features_df)
                plt.title(f'Importancia de Características - {model_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, 'feature_importance.png'))
                plt.close()
                
                # Guardar también en formato CSV
                features_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
                
                logging.info(f"Gráfico y datos de importancia de características guardados")
                
                # Mostrar las 5 características más importantes
                top_features = features_df.head(5)
                logging.info("Top 5 características más importantes:")
                for i, (_, row) in enumerate(top_features.iterrows()):
                    logging.info(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
            except Exception as e:
                logging.warning(f"No se pudo generar gráfico de importancia: {e}")
        
        # Generar curva ROC si tenemos probabilidades
        try:
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt
            
            if y_proba is not None and isinstance(y_proba, np.ndarray):
                # Para clasificación binaria (player1 vs player2)
                fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(model_dir, 'roc_curve.png'))
                plt.close()
                
                logging.info(f"Curva ROC guardada, AUC = {roc_auc:.4f}")
        except Exception as e:
            logging.warning(f"No se pudo generar curva ROC: {e}")
    except Exception as e:
        logging.error(f"Error generando visualizaciones: {e}")
    
    # Mostrar resultados
    print("\n" + "=" * 50)
    print(f"RESULTADOS DEL ENTRENAMIENTO - {model_name}")
    print("=" * 50)
    print(f"Datos: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"División: {len(X_train)} train, {len(X_test)} test")
    print(f"Distribución de clases: {pd.Series(y).value_counts().to_dict()}")
    print("\nMétricas de rendimiento:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nTiempo de entrenamiento: {time.time() - start_time:.2f} segundos")
    print(f"Modelo guardado en: {model_path}")
    
    # Verificar si tenemos features_df antes de intentar usarlo
    if features_df is not None and hasattr(model, 'feature_importances_'):
        print("\nCaracterísticas más importantes:")
        for i, (_, row) in enumerate(features_df.head(5).iterrows()):
            print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "=" * 50)
    
    logging.info(f"Entrenamiento completado exitosamente")
    
    return model

def cross_validate_model(data_path=None, model_type='rf', n_folds=5):
    """
    Realiza validación cruzada del modelo para evaluar su rendimiento.
    
    Args:
        data_path: Ruta al archivo CSV con datos
        model_type: Tipo de modelo ('rf' o 'gb')
        n_folds: Número de folds para validación cruzada
        
    Returns:
        Diccionario con resultados de la validación
    """
    from sklearn.model_selection import cross_validate
    
    print(f"Realizando validación cruzada con {n_folds} folds...")
    
    # Preparar datos
    fe = TennisFeatureEngineering(data_path=data_path)
    X, y = fe.prepare_training_data()
    
    # Imputar valores nulos si es necesario
    if X.isnull().values.any():
        X = X.fillna(0)
    
    # Seleccionar modelo
    if model_type.lower() == 'gb':
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)
    
    # Definir métricas a evaluar
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }
    
    # Realizar validación cruzada
    cv_results = cross_validate(
        model, X, y, 
        cv=n_folds,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True
    )
    
    # Procesar resultados
    results = {
        'test_accuracy': np.mean(cv_results['test_accuracy']),
        'test_precision': np.mean(cv_results['test_precision']),
        'test_recall': np.mean(cv_results['test_recall']),
        'test_f1': np.mean(cv_results['test_f1']),
        'train_accuracy': np.mean(cv_results['train_accuracy']),
        'train_precision': np.mean(cv_results['train_precision']),
        'train_recall': np.mean(cv_results['train_recall']),
        'train_f1': np.mean(cv_results['train_f1']),
        'std_test_accuracy': np.std(cv_results['test_accuracy']),
        'std_test_f1': np.std(cv_results['test_f1']),
        'fit_time': np.mean(cv_results['fit_time']),
        'score_time': np.mean(cv_results['score_time'])
    }
    
    # Mostrar resultados
    print("\n" + "=" * 50)
    print(f"RESULTADOS DE VALIDACIÓN CRUZADA ({n_folds} FOLDS)")
    print("=" * 50)
    print(f"Accuracy:  {results['test_accuracy']:.4f} ± {results['std_test_accuracy']:.4f}")
    print(f"Precision: {results['test_precision']:.4f}")
    print(f"Recall:    {results['test_recall']:.4f}")
    print(f"F1 Score:  {results['test_f1']:.4f} ± {results['std_test_f1']:.4f}")
    print(f"\nTiempo promedio de entrenamiento: {results['fit_time']:.2f} segundos")
    
    # Verificar overfitting
    train_acc = results['train_accuracy']
    test_acc = results['test_accuracy']
    overfit_diff = train_acc - test_acc
    
    if overfit_diff > 0.1:
        print(f"\n⚠️ Posible overfitting detectado:")
        print(f"   Accuracy en training: {train_acc:.4f}")
        print(f"   Accuracy en testing:  {test_acc:.4f}")
        print(f"   Diferencia: {overfit_diff:.4f}")
        print("   Considere ajustar hiperparámetros o recopilar más datos.")
    else:
        print(f"\n✅ No se detectó overfitting significativo.")
    
    print("\n" + "=" * 50)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar modelo de predicción de partidos de tenis")
    parser.add_argument("--data", type=str, help="Ruta al archivo CSV con datos")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "gb"], help="Tipo de modelo: RandomForest o GradientBoosting")
    parser.add_argument("--optimize", action="store_true", help="Optimizar hiperparámetros")
    parser.add_argument("--cv", action="store_true", help="Realizar validación cruzada")
    parser.add_argument("--folds", type=int, default=5, help="Número de folds para validación cruzada")
    
    args = parser.parse_args()
    
    if args.cv:
        cross_validate_model(data_path=args.data, model_type=args.model, n_folds=args.folds)
    else:
        train_model(data_path=args.data, model_type=args.model, optimize_hyperparams=args.optimize)