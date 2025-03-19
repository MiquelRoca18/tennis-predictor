#!/usr/bin/env python3
"""
Script para evaluar exhaustivamente el modelo de predicción de partidos de tenis.
Realiza validación cruzada, análisis de curvas ROC, y evaluación con datos de prueba externos.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import argparse
from datetime import datetime
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score,
    auc
)

# Importar utilidades
try:
    from utils import TennisFeatureEngineering
except ImportError:
    print("Error: No se pudo importar TennisFeatureEngineering. Verificando rutas...")
    # Intentar añadir directorio actual al path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    sys.path.append(os.path.dirname(current_dir))
    try:
        from utils import TennisFeatureEngineering
        print("Módulo utils importado correctamente después de ajustar path.")
    except ImportError:
        print("Error crítico: No se pudo importar el módulo utils.")
        print("Asegúrate de que utils.py está en el mismo directorio o en el PYTHONPATH.")
        sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml-service/logs/test_model.log'),
        logging.StreamHandler()
    ]
)

def load_model(model_path=None):
    """
    Carga el modelo entrenado.
    
    Args:
        model_path: Ruta al archivo del modelo (opcional)
        
    Returns:
        Modelo cargado
    """
    # Si no se especifica ruta, buscar en ubicaciones predeterminadas
    if model_path is None:
        possible_paths = [
            'model/model.pkl',
            'ml-service/model/model.pkl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model.pkl')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError("No se encontró el modelo en ninguna ubicación conocida")
    
    print(f"Cargando modelo desde: {model_path}")
    return joblib.load(model_path)

def load_test_data(data_path, test_size=0.2, random_state=42):
    """
    Carga y prepara datos de prueba.
    
    Args:
        data_path: Ruta al archivo de datos
        test_size: Proporción para datos de prueba
        random_state: Semilla aleatoria
        
    Returns:
        X_test, y_test: Características y etiquetas de prueba
    """
    from sklearn.model_selection import train_test_split
    
    fe = TennisFeatureEngineering(data_path=data_path)
    
    try:
        # Preparar características avanzadas
        X, y = fe.prepare_training_data()
        
        # Dividir en entrenamiento y prueba
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        print(f"Datos de prueba cargados: {X_test.shape[0]} muestras, {X_test.shape[1]} características")
        print(f"Distribución de clases en prueba: {pd.Series(y_test).value_counts().to_dict()}")
        
        return X_test, y_test
    except Exception as e:
        logging.error(f"Error cargando datos de prueba: {e}")
        raise

def load_external_test_data(data_path):
    """
    Carga y prepara datos de prueba externos (no utilizados en entrenamiento).
    
    Args:
        data_path: Ruta al archivo de datos externos
        
    Returns:
        X_test, y_test: Características y etiquetas de prueba
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el archivo de datos externos: {data_path}")
            
        fe = TennisFeatureEngineering(data_path=data_path)
        
        # Verificar si los datos tienen la clase objetivo
        data = pd.read_csv(data_path)
        if 'winner' not in data.columns:
            print("Los datos externos no contienen la columna 'winner'.")
            print("Solo se realizará predicción sin evaluación.")
            
            # Preparar características sin etiquetas
            X = fe.prepare_features_only()
            return X, None
        
        # Preparar características y etiquetas
        X, y = fe.prepare_training_data()
        
        print(f"Datos externos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"Distribución de clases: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y
    except Exception as e:
        logging.error(f"Error cargando datos externos: {e}")
        raise

def cross_validate_model(model, X, y, n_folds=5):
    """
    Realiza validación cruzada del modelo.
    
    Args:
        model: Modelo a evaluar
        X: Características
        y: Etiquetas
        n_folds: Número de folds
        
    Returns:
        Diccionario con resultados
    """
    print(f"\n=== VALIDACIÓN CRUZADA ({n_folds}-fold) ===")
    
    # Crear validación cruzada estratificada
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Evaluar métricas
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted')
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall_weighted')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    
    # Mostrar resultados
    print(f"Accuracy:  {accuracy.mean():.4f} ± {accuracy.std():.4f}")
    print(f"Precision: {precision.mean():.4f} ± {precision.std():.4f}")
    print(f"Recall:    {recall.mean():.4f} ± {recall.std():.4f}")
    print(f"F1 Score:  {f1.mean():.4f} ± {f1.std():.4f}")
    
    # Detectar posible overfitting
    # Realizar una predicción en todo el conjunto para comparar
    train_pred = model.predict(X)
    train_acc = accuracy_score(y, train_pred)
    
    if train_acc - accuracy.mean() > 0.1:
        print("\n⚠️ Posible overfitting detectado:")
        print(f"  - Accuracy en datos de entrenamiento: {train_acc:.4f}")
        print(f"  - Accuracy en validación cruzada: {accuracy.mean():.4f}")
        print(f"  - Diferencia: {train_acc - accuracy.mean():.4f}")
    
    return {
        'accuracy': accuracy.mean(),
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': f1.mean(),
        'accuracy_std': accuracy.std(),
        'precision_std': precision.std(),
        'recall_std': recall.std(),
        'f1_std': f1.std()
    }

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Etiquetas de prueba
        
    Returns:
        dict: Métricas de evaluación
    """
    try:
        # Realizar predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': auc(y_test, y_pred_proba)
        }
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error evaluando modelo: {e}")
        return None

def analyze_model(model, X):
    """
    Analiza el modelo y sus predicciones.
    
    Args:
        model: Modelo a analizar
        X: Características
        
    Returns:
        None
    """
    print("\n=== ANÁLISIS DEL MODELO ===")
    
    # Verificar tipo de modelo
    model_type = type(model).__name__
    print(f"Tipo de modelo: {model_type}")
    
    # Analizar importancia de características
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nImportancia de Características (Top 10):")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
        # Visualizar importancia
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Importancia de Características')
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png')
        plt.close()
        
        # Analizar agrupaciones de características
        # Agrupar por prefijos comunes
        feature_groups = {}
        for feature in X.columns:
            # Extraer prefijo (e.g., 'ranking_', 'winrate_', etc.)
            parts = feature.split('_')
            if len(parts) > 1:
                prefix = parts[0]
                if prefix not in feature_groups:
                    feature_groups[prefix] = []
                feature_groups[prefix].append(feature)
            else:
                if 'other' not in feature_groups:
                    feature_groups['other'] = []
                feature_groups['other'].append(feature)
        
        # Calcular importancia por grupo
        group_importance = {}
        for group, features in feature_groups.items():
            indices = [list(X.columns).index(f) for f in features]
            total_importance = sum(importances[i] for i in indices)
            group_importance[group] = total_importance
        
        # Ordenar y mostrar
        group_imp_df = pd.DataFrame({
            'group': list(group_importance.keys()),
            'importance': list(group_importance.values())
        }).sort_values('importance', ascending=False)
        
        print("\nImportancia por Grupo de Características:")
        for _, row in group_imp_df.iterrows():
            print(f"  {row['group']}: {row['importance']:.4f}")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=group_imp_df, x='importance', y='group')
        plt.title('Importancia por Grupo de Características')
        plt.tight_layout()
        plt.savefig('feature_group_importance.png')
        plt.close()
    
    # Si es un modelo de árbol, analizar profundidad
    if hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'tree_'):
        depths = [estimator.tree_.max_depth for estimator in model.estimators_]
        avg_depth = sum(depths) / len(depths)
        
        print(f"\nProfundidad promedio de árboles: {avg_depth:.2f}")
        print(f"Profundidad mínima: {min(depths)}")
        print(f"Profundidad máxima: {max(depths)}")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(depths, bins=20)
        plt.title('Distribución de Profundidad de Árboles')
        plt.xlabel('Profundidad')
        plt.ylabel('Frecuencia')
        plt.savefig('tree_depth_distribution.png')
        plt.close()
    
    # Analizar hiperparámetros
    print("\nHiperparámetros del modelo:")
    for param, value in model.get_params().items():
        # Filtrar parámetros relevantes
        if not param.startswith('_') and not isinstance(value, (list, dict, set)):
            print(f"  {param}: {value}")

def predict_matches(model, test_matches, output_file='predictions.csv'):
    """
    Predice resultados para partidos específicos.
    
    Args:
        model: Modelo entrenado
        test_matches: Lista de diccionarios con datos de partidos
        output_file: Archivo para guardar resultados
        
    Returns:
        DataFrame con predicciones
    """
    print("\n=== PREDICCIÓN DE PARTIDOS ===")
    
    fe = TennisFeatureEngineering()
    
    # Inicializar listas para resultados
    predictions = []
    
    for i, match in enumerate(test_matches):
        print(f"\nPartido {i+1}: {match.get('player1', match.get('player_1'))} vs {match.get('player2', match.get('player_2'))}")
        
        try:
            # Transformar datos del partido
            X = fe.transform_match_data(match)
            
            # Realizar predicción
            pred_winner = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Determinar jugador ganador
            player1 = match.get('player1', match.get('player_1'))
            player2 = match.get('player2', match.get('player_2'))
            
            winner = player1 if pred_winner == 0 else player2
            winner_prob = max(probabilities)
            
            # Guardar resultados
            results = {
                'player1': player1,
                'player2': player2,
                'surface': match.get('surface'),
                'predicted_winner': winner,
                'prediction_value': int(pred_winner),
                'confidence': winner_prob,
                'prob_player1_wins': probabilities[0],
                'prob_player2_wins': probabilities[1]
            }
            
            # Añadir características si están disponibles
            for col in X.columns:
                if len(X) > 0:  # Asegurar que tenemos datos
                    results[f'feature_{col}'] = X.iloc[0][col]
            
            predictions.append(results)
            
            # Mostrar resultado
            print(f"Predicción: {winner} ganará con {winner_prob:.2%} de probabilidad")
            print(f"  - Probabilidad {player1}: {probabilities[0]:.2%}")
            print(f"  - Probabilidad {player2}: {probabilities[1]:.2%}")
            
        except Exception as e:
            logging.error(f"Error prediciendo partido {i+1}: {e}")
            print(f"⚠️ Error en predicción: {e}")
    
    # Crear DataFrame con resultados
    if predictions:
        pred_df = pd.DataFrame(predictions)
        
        # Guardar a CSV
        pred_df.to_csv(output_file, index=False)
        print(f"\nPredicciones guardadas en: {output_file}")
        
        return pred_df
    else:
        print("No se pudieron generar predicciones.")
        return None

def plot_roc_curve(y_test, y_pred_proba, output_dir='ml-service/plots'):
    """
    Genera y guarda la curva ROC.
    
    Args:
        y_test: Etiquetas de prueba
        y_pred_proba: Probabilidades predichas
        output_dir: Directorio para guardar la gráfica
    """
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
    except Exception as e:
        logging.error(f"Error generando curva ROC: {e}")

def plot_confusion_matrix(cm, output_dir='ml-service/plots'):
    """
    Genera y guarda la matriz de confusión.
    
    Args:
        cm: Matriz de confusión
        output_dir: Directorio para guardar la gráfica
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
    except Exception as e:
        logging.error(f"Error generando matriz de confusión: {e}")

def test_model(model_path, test_data_path, output_dir='ml-service/results'):
    """
    Prueba el modelo con datos de prueba.
    
    Args:
        model_path: Ruta al modelo guardado
        test_data_path: Ruta a los datos de prueba
        output_dir: Directorio para guardar resultados
    """
    try:
        # Cargar modelo
        model = joblib.load(model_path)
        logging.info(f"Modelo cargado desde {model_path}")
        
        # Cargar datos de prueba
        test_data = pd.read_csv(test_data_path)
        logging.info(f"Datos de prueba cargados desde {test_data_path}")
        
        # Preparar características
        feature_engineering = TennisFeatureEngineering()
        X_test = feature_engineering.extract_features(test_data)
        y_test = test_data['winner']
        
        # Evaluar modelo
        metrics = evaluate_model(model, X_test, y_test)
        if metrics:
            logging.info("Métricas de evaluación:")
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':
                    logging.info(f"{metric}: {value:.4f}")
            
            # Generar visualizaciones
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            plot_roc_curve(y_test, y_pred_proba)
            plot_confusion_matrix(np.array(metrics['confusion_matrix']))
            
            # Guardar resultados
            os.makedirs(output_dir, exist_ok=True)
            results_file = os.path.join(output_dir, 'test_results.json')
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Resultados guardados en {results_file}")
            
    except Exception as e:
        logging.error(f"Error en prueba del modelo: {e}")

def main():
    """Función principal para ejecutar pruebas del modelo"""
    parser = argparse.ArgumentParser(description='Prueba el modelo de predicción de tenis')
    parser.add_argument('--model', required=True, help='Ruta al modelo guardado')
    parser.add_argument('--test_data', required=True, help='Ruta a los datos de prueba')
    parser.add_argument('--output_dir', default='ml-service/results', help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    test_model(args.model, args.test_data, args.output_dir)

if __name__ == '__main__':
    main()