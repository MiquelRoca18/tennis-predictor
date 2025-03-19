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
    classification_report, precision_recall_curve, average_precision_score
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
    format='%(asctime)s - %(levelname)s: %(message)s'
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

def evaluate_model(model, X_test, y_test, output_dir='evaluation_results'):
    """
    Evalúa el modelo con datos de prueba y genera visualizaciones.
    
    Args:
        model: Modelo a evaluar
        X_test: Características de prueba
        y_test: Etiquetas de prueba
        output_dir: Directorio para guardar resultados
        
    Returns:
        Diccionario con métricas
    """
    print("\n=== EVALUACIÓN DEL MODELO ===")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Jugador 1', 'Jugador 2'],
               yticklabels=['Jugador 1', 'Jugador 2'])
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Reporte de clasificación detallado
    class_report = classification_report(y_test, y_pred, target_names=['Jugador 1', 'Jugador 2'])
    print("\nReporte de Clasificación:")
    print(class_report)
    
    # Guardar reporte en archivo
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("REPORTE DE CLASIFICACIÓN\n")
        f.write("=======================\n\n")
        f.write(class_report)
    
    # Curva ROC
    try:
        # Calcular AUC
        auc = roc_auc_score(y_test, y_proba[:, 1])
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        print(f"AUC-ROC:   {auc:.4f}")
    except Exception as e:
        logging.error(f"Error generando curva ROC: {e}")
    
    # Curva Precision-Recall
    try:
        # Calcular average precision
        ap = average_precision_score(y_test, y_proba[:, 1])
        
        # Calcular curva precision-recall
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'Precision-Recall curve (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        plt.close()
        
        print(f"Average Precision: {ap:.4f}")
    except Exception as e:
        logging.error(f"Error generando curva Precision-Recall: {e}")
    
    # Análisis de errores
    try:
        # Crear DataFrame con resultados
        results_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
            'prob_player1': y_proba[:, 0],
            'prob_player2': y_proba[:, 1]
        })
        
        # Añadir indicador de predicción correcta
        results_df['correct'] = results_df['y_true'] == results_df['y_pred']
        
        # Añadir confianza de predicción
        results_df['confidence'] = results_df.apply(
            lambda row: row['prob_player1'] if row['y_pred'] == 0 else row['prob_player2'], 
            axis=1
        )
        
        # Guardar resultados detallados
        results_df.to_csv(os.path.join(output_dir, 'prediction_details.csv'), index=False)
        
        # Analizar errores por confianza
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_df, x='confidence', hue='correct', bins=20,
                    element="step", common_norm=False, alpha=0.5)
        plt.title('Distribución de Confianza vs. Predicción Correcta')
        plt.xlabel('Confianza de la Predicción')
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(output_dir, 'confidence_errors.png'))
        plt.close()
        
        # Calcular errores por nivel de confianza
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        error_rates = []
        
        for i in range(len(confidence_bins) - 1):
            lower = confidence_bins[i]
            upper = confidence_bins[i+1]
            
            bin_data = results_df[(results_df['confidence'] >= lower) & (results_df['confidence'] < upper)]
            if len(bin_data) > 0:
                error_rate = 1 - bin_data['correct'].mean()
                error_rates.append({
                    'bin': f"{lower:.1f} - {upper:.1f}",
                    'error_rate': error_rate,
                    'count': len(bin_data)
                })
        
        error_df = pd.DataFrame(error_rates)
        if not error_df.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=error_df, x='bin', y='error_rate')
            plt.title('Tasa de Error por Nivel de Confianza')
            plt.xlabel('Confianza de la Predicción')
            plt.ylabel('Tasa de Error')
            plt.savefig(os.path.join(output_dir, 'error_by_confidence.png'))
            plt.close()
            
            # Mostrar tabla de errores por confianza
            print("\nTasas de Error por Nivel de Confianza:")
            for _, row in error_df.iterrows():
                print(f"  Confianza {row['bin']}: {row['error_rate']:.4f} (n={row['count']})")
    except Exception as e:
        logging.error(f"Error en análisis de errores: {e}")
    
    print(f"\nResultados de evaluación guardados en: {os.path.abspath(output_dir)}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc if 'auc' in locals() else None,
        'ap': ap if 'ap' in locals() else None
    }

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

def main():
    parser = argparse.ArgumentParser(description="Evaluación avanzada de modelo de predicción de tenis")
    parser.add_argument('--model', type=str, help='Ruta al archivo del modelo')
    parser.add_argument('--data', type=str, default='tennis_matches.csv', help='Ruta al archivo de datos')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporción para datos de prueba')
    parser.add_argument('--external', type=str, help='Ruta a datos de prueba externos')
    parser.add_argument('--cv', action='store_true', help='Realizar validación cruzada')
    parser.add_argument('--folds', type=int, default=5, help='Número de folds para validación cruzada')
    parser.add_argument('--analyze', action='store_true', help='Realizar análisis detallado del modelo')
    parser.add_argument('--predict', type=str, help='Archivo CSV con partidos para predecir')
    parser.add_argument('--output-dir', type=str, default='model_evaluation', help='Directorio para resultados')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Cargar modelo
        model = load_model(args.model)
        
        # Si se solicita análisis del modelo
        if args.analyze:
            # Necesitamos los datos para el análisis de características
            fe = TennisFeatureEngineering(data_path=args.data)
            X, _ = fe.prepare_training_data()
            analyze_model(model, X)
        
        # Si se solicita validación cruzada
        if args.cv:
            fe = TennisFeatureEngineering(data_path=args.data)
            X, y = fe.prepare_training_data()
            cross_validate_model(model, X, y, n_folds=args.folds)
        
        # Si se solicita evaluación con datos externos
        if args.external:
            X_external, y_external = load_external_test_data(args.external)
            
            if y_external is not None:
                print("\n=== EVALUACIÓN CON DATOS EXTERNOS ===")
                external_results = evaluate_model(model, X_external, y_external, 
                                                output_dir=os.path.join(args.output_dir, 'external'))
        
        # Si no se piden datos externos, usar datos regulares divididos
        elif not args.predict:  # Solo si no estamos en modo predicción
            X_test, y_test = load_test_data(args.data, test_size=args.test_size)
            evaluate_model(model, X_test, y_test, output_dir=args.output_dir)
        
        # Si se solicita predicción de partidos específicos
        if args.predict:
            if os.path.exists(args.predict):
                # Cargar partidos a predecir
                test_matches_df = pd.read_csv(args.predict)
                test_matches = test_matches_df.to_dict('records')
                
                output_file = os.path.join(args.output_dir, 'match_predictions.csv')
                predict_matches(model, test_matches, output_file)
            else:
                print(f"Error: No se encontró el archivo de partidos: {args.predict}")
    
    except Exception as e:
        logging.error(f"Error en la evaluación del modelo: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()