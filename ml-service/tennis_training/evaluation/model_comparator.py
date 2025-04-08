"""
model_comparator.py

Este módulo implementa funcionalidades para comparar diferentes modelos de predicción
de tenis, facilitando la selección del mejor modelo y el análisis de las diferencias
en comportamiento entre ellos.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from .metrics import TennisMetrics


class ModelComparator:
    """
    Clase para comparar el rendimiento de diferentes modelos de predicción de tenis.
    Permite identificar las fortalezas y debilidades relativas de cada modelo y
    generar visualizaciones comparativas detalladas.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Inicializa el comparador de modelos.
        
        Args:
            output_dir: Directorio donde se guardarán los gráficos y resultados
        """
        self.metrics = TennisMetrics()
        self.models_results = {}
        self.models_metrics = {}
        
        # Configurar directorio de salida
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"model_comparison_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
    
    def add_model(self, 
                 model_name: str, 
                 y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 y_prob: np.ndarray,
                 additional_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Añade un modelo al comparador y calcula sus métricas.
        
        Args:
            model_name: Nombre identificativo del modelo
            y_true: Valores reales (1 para victoria, 0 para derrota)
            y_pred: Valores predichos (1 para victoria, 0 para derrota)
            y_prob: Probabilidades predichas para la clase positiva
            additional_data: DataFrame con datos adicionales como características o metadatos
            
        Returns:
            Diccionario con métricas calculadas para el modelo
        """
        if model_name in self.models_results:
            print(f"Advertencia: El modelo {model_name} ya existe. Se sobrescribirán sus datos.")
        
        # Guardar resultados
        self.models_results[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Calcular métricas
        metrics = self.metrics.calculate_all_metrics(y_true, y_pred, y_prob)
        self.models_metrics[model_name] = metrics
        
        # Guardar datos adicionales si se proporcionan
        if additional_data is not None:
            self.models_results[model_name]['additional_data'] = additional_data
        
        return metrics
    
    def add_model_from_dataframe(self, 
                                model_name: str, 
                                df: pd.DataFrame,
                                true_col: str = 'y_true', 
                                pred_col: str = 'y_pred', 
                                prob_col: str = 'y_prob') -> Dict[str, float]:
        """
        Añade un modelo al comparador a partir de un DataFrame.
        
        Args:
            model_name: Nombre identificativo del modelo
            df: DataFrame con resultados del modelo
            true_col: Nombre de columna para valores reales
            pred_col: Nombre de columna para valores predichos
            prob_col: Nombre de columna para probabilidades
            
        Returns:
            Diccionario con métricas calculadas para el modelo
        """
        # Verificar columnas
        required_cols = [true_col, pred_col, prob_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"La columna '{col}' no existe en el DataFrame")
        
        # Extraer datos relevantes
        y_true = df[true_col].values
        y_pred = df[pred_col].values
        y_prob = df[prob_col].values
        
        # Añadir modelo
        return self.add_model(model_name, y_true, y_pred, y_prob, df)
    
    def compare_all_models(self) -> pd.DataFrame:
        """
        Compara todos los modelos añadidos y genera un DataFrame comparativo.
        
        Returns:
            DataFrame con comparación de métricas entre modelos
        """
        if not self.models_metrics:
            raise ValueError("No hay modelos para comparar. Añade modelos primero.")
        
        # Usar el método de TennisMetrics para comparar
        models_list = list(self.models_metrics.values())
        model_names = list(self.models_metrics.keys())
        
        comparison_df = self.metrics.compare_models(models_list, model_names)
        
        # Guardar como CSV
        csv_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(csv_path)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'roc_auc') -> str:
        """
        Identifica el mejor modelo según una métrica específica.
        
        Args:
            metric: Métrica para evaluar (ej: 'accuracy', 'roc_auc', 'roi')
            
        Returns:
            Nombre del mejor modelo según la métrica
        """
        if not self.models_metrics:
            raise ValueError("No hay modelos para comparar. Añade modelos primero.")
        
        # Verificar que la métrica existe en todos los modelos
        for model_name, metrics in self.models_metrics.items():
            if metric not in metrics:
                raise ValueError(f"La métrica '{metric}' no existe en el modelo '{model_name}'")
        
        # Encontrar el mejor modelo
        best_model = None
        best_value = -float('inf')
        
        for model_name, metrics in self.models_metrics.items():
            value = metrics[metric]
            if value > best_value:
                best_value = value
                best_model = model_name
        
        return best_model
    
    def analyze_disagreements(self, model1: str, model2: str) -> Dict[str, Any]:
        """
        Analiza los casos donde los modelos difieren en sus predicciones.
        
        Args:
            model1: Nombre del primer modelo
            model2: Nombre del segundo modelo
            
        Returns:
            Diccionario con análisis de desacuerdos
        """
        if model1 not in self.models_results or model2 not in self.models_results:
            raise ValueError(f"Los modelos {model1} y/o {model2} no existen")
        
        # Extraer predicciones
        y_pred1 = self.models_results[model1]['y_pred']
        y_pred2 = self.models_results[model2]['y_pred']
        y_true = self.models_results[model1]['y_true']
        
        # Identificar desacuerdos
        disagreements = y_pred1 != y_pred2
        disagreement_indices = np.where(disagreements)[0]
        
        if len(disagreement_indices) == 0:
            return {"disagreement_count": 0, "message": "No hay desacuerdos entre los modelos"}
        
        # Analizar desacuerdos
        model1_correct = y_pred1[disagreements] == y_true[disagreements]
        model2_correct = y_pred2[disagreements] == y_true[disagreements]
        
        neither_correct = ~model1_correct & ~model2_correct
        both_correct = model1_correct & model2_correct
        only_model1_correct = model1_correct & ~model2_correct
        only_model2_correct = ~model1_correct & model2_correct
        
        # Calcular estadísticas
        results = {
            "disagreement_count": len(disagreement_indices),
            "disagreement_percentage": len(disagreement_indices) / len(y_true) * 100,
            "neither_correct_count": np.sum(neither_correct),
            "both_correct_count": np.sum(both_correct),
            f"{model1}_only_correct_count": np.sum(only_model1_correct),
            f"{model2}_only_correct_count": np.sum(only_model2_correct),
            f"{model1}_win_rate": np.sum(only_model1_correct) / np.sum(only_model1_correct + only_model2_correct) if np.sum(only_model1_correct + only_model2_correct) > 0 else 0
        }
        
        # Analizar probabilidades en desacuerdos
        y_prob1 = self.models_results[model1]['y_prob'][disagreements]
        y_prob2 = self.models_results[model2]['y_prob'][disagreements]
        
        results["avg_probability_difference"] = np.mean(np.abs(y_prob1 - y_prob2))
        results[f"{model1}_avg_probability_when_correct"] = np.mean(y_prob1[model1_correct]) if np.sum(model1_correct) > 0 else 0
        results[f"{model2}_avg_probability_when_correct"] = np.mean(y_prob2[model2_correct]) if np.sum(model2_correct) > 0 else 0
        
        # Guardar índices de desacuerdos para análisis posteriores
        results["disagreement_indices"] = disagreement_indices.tolist()
        
        # Guardar resultados como JSON
        with open(os.path.join(self.output_dir, f'disagreements_{model1}_vs_{model2}.json'), 'w') as f:
            # Crear una copia sin los índices para serializar
            serializable_results = {k: v for k, v in results.items() if k != "disagreement_indices"}
            json.dump(serializable_results, f, indent=4)
        
        # Visualizar desacuerdos
        self._plot_disagreements(model1, model2, y_prob1, y_prob2, model1_correct, model2_correct)
        
        return results
    
    def _plot_disagreements(self, model1, model2, y_prob1, y_prob2, model1_correct, model2_correct):
        """
        Genera visualizaciones para el análisis de desacuerdos entre modelos.
        
        Args:
            model1: Nombre del primer modelo
            model2: Nombre del segundo modelo
            y_prob1: Probabilidades del primer modelo en desacuerdos
            y_prob2: Probabilidades del segundo modelo en desacuerdos
            model1_correct: Máscara booleana de aciertos del primer modelo
            model2_correct: Máscara booleana de aciertos del segundo modelo
        """
        plt.figure(figsize=(12, 10))
        
        # Scatter plot de probabilidades
        plt.subplot(2, 1, 1)
        plt.scatter(y_prob1[model1_correct], y_prob2[model1_correct], 
                   alpha=0.6, label=f"Solo {model1} correcto", c='blue')
        plt.scatter(y_prob1[model2_correct], y_prob2[model2_correct], 
                   alpha=0.6, label=f"Solo {model2} correcto", c='red')
        
        neither_correct = ~model1_correct & ~model2_correct
        both_correct = model1_correct & model2_correct
        
        if np.any(neither_correct):
            plt.scatter(y_prob1[neither_correct], y_prob2[neither_correct], 
                       alpha=0.6, label="Ninguno correcto", c='gray')
        if np.any(both_correct):
            plt.scatter(y_prob1[both_correct], y_prob2[both_correct], 
                       alpha=0.6, label="Ambos correctos", c='green')
        
        plt.xlabel(f"Probabilidad {model1}")
        plt.ylabel(f"Probabilidad {model2}")
        plt.title(f"Probabilidades en Desacuerdos entre {model1} y {model2}")
        plt.grid(True)
        plt.legend()
        
        # Histograma de diferencias
        plt.subplot(2, 1, 2)
        diff = y_prob1 - y_prob2
        plt.hist(diff[model1_correct], alpha=0.5, bins=20, label=f"Solo {model1} correcto", color='blue')
        plt.hist(diff[model2_correct], alpha=0.5, bins=20, label=f"Solo {model2} correcto", color='red')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.xlabel(f"Diferencia de Probabilidad ({model1} - {model2})")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de Diferencias de Probabilidad en Desacuerdos")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'disagreements_{model1}_vs_{model2}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_calibration_differences(self, n_bins: int = 10) -> pd.DataFrame:
        """
        Analiza las diferencias de calibración entre los modelos.
        
        Args:
            n_bins: Número de bins para la calibración
            
        Returns:
            DataFrame con métricas de calibración por modelo
        """
        if not self.models_results:
            raise ValueError("No hay modelos para comparar. Añade modelos primero.")
        
        calibration_results = []
        
        for model_name, results in self.models_results.items():
            y_true = results['y_true']
            y_prob = results['y_prob']
            
            # Calcular métricas de calibración
            cal_metrics = self.metrics.calculate_calibration_metrics(y_true, y_prob, n_bins=n_bins)
            
            # Extraer métricas principales
            cal_results = {
                'model': model_name,
                'expected_calibration_error': cal_metrics['expected_calibration_error'],
                'maximum_calibration_error': cal_metrics['maximum_calibration_error']
            }
            
            calibration_results.append(cal_results)
        
        # Crear DataFrame
        cal_df = pd.DataFrame(calibration_results)
        
        # Guardar como CSV
        cal_df.to_csv(os.path.join(self.output_dir, 'calibration_comparison.csv'), index=False)
        
        # Generar visualización comparativa
        self._plot_calibration_comparison(n_bins)
        
        return cal_df
    
    def _plot_calibration_comparison(self, n_bins: int = 10):
        """
        Genera visualizaciones comparativas de calibración entre modelos.
        
        Args:
            n_bins: Número de bins para la calibración
        """
        plt.figure(figsize=(14, 10))
        
        # Línea de calibración perfecta
        plt.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
        
        # Añadir curva de cada modelo
        colors = plt.cm.tab10.colors
        for i, (model_name, results) in enumerate(self.models_results.items()):
            y_true = results['y_true']
            y_prob = results['y_prob']
            
            # Calcular datos de calibración
            bin_indices = np.floor(y_prob * n_bins).astype(int)
            bin_indices[bin_indices == n_bins] = n_bins - 1
            
            bin_sums = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)
            bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
            bin_counts = np.bincount(bin_indices, minlength=n_bins)
            
            # Evitar división por cero
            nonzero = bin_counts > 0
            bin_avg_pred = np.zeros(n_bins)
            bin_avg_true = np.zeros(n_bins)
            
            if np.any(nonzero):
                bin_avg_pred[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
                bin_avg_true[nonzero] = bin_true[nonzero] / bin_counts[nonzero]
            
            # Calcular ECE para etiqueta
            ece = np.sum(bin_counts * np.abs(bin_avg_pred - bin_avg_true)) / sum(bin_counts)
            
            # Calcular puntos para gráfico
            bin_edges = np.linspace(0, 1, n_bins + 1)[:-1]
            width = 1.0 / n_bins
            
            # Plotear calibración
            plt.plot(bin_edges[nonzero] + width/2, bin_avg_true[nonzero], 
                     marker='o', linestyle='-', color=colors[i % len(colors)],
                     label=f'{model_name} (ECE={ece:.4f})')
        
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Probabilidad predicha')
        plt.ylabel('Fracción de resultados positivos')
        plt.title('Comparación de Calibración entre Modelos')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, 'calibration_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_roc_curves(self):
        """
        Compara las curvas ROC de todos los modelos.
        
        Returns:
            Figura matplotlib
        """
        if not self.models_results:
            raise ValueError("No hay modelos para comparar. Añade modelos primero.")
        
        plt.figure(figsize=(12, 10))
        
        # Añadir curva de cada modelo
        colors = plt.cm.tab10.colors
        for i, (model_name, results) in enumerate(self.models_results.items()):
            y_true = results['y_true']
            y_prob = results['y_prob']
            
            # Calcular curva ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            
            # Calcular AUC para etiqueta
            auc = self.models_metrics[model_name]['roc_auc']
            
            # Plotear ROC
            plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})', 
                     color=colors[i % len(colors)], lw=2)
        
        # Línea de referencia
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Comparación de Curvas ROC')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Guardar figura
        plt.savefig(os.path.join(self.output_dir, 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_precision_recall_curves(self):
        """
        Compara las curvas de Precision-Recall de todos los modelos.
        
        Returns:
            Figura matplotlib
        """
        if not self.models_results:
            raise ValueError("No hay modelos para comparar. Añade modelos primero.")
        
        plt.figure(figsize=(12, 10))
        
        # Añadir curva de cada modelo
        colors = plt.cm.tab10.colors
        for i, (model_name, results) in enumerate(self.models_results.items()):
            y_true = results['y_true']
            y_prob = results['y_prob']
            
            # Calcular curva Precision-Recall
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            
            # Calcular F1 para etiqueta
            f1 = self.models_metrics[model_name]['f1']
            
            # Plotear Precision-Recall
            plt.plot(recall, precision, label=f'{model_name} (F1={f1:.4f})', 
                     color=colors[i % len(colors)], lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Comparación de Curvas Precision-Recall')
        plt.legend(loc='lower left')
        plt.grid(True)
        
        # Guardar figura
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curves_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_confusion_matrices(self):
        """
        Compara las matrices de confusión de todos los modelos.
        """
        if not self.models_results:
            raise ValueError("No hay modelos para comparar. Añade modelos primero.")
        
        # Calcular número de subplots necesarios
        n_models = len(self.models_results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]  # Convertir a lista para iterar más fácilmente
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()  # Convertir a 1D si solo hay una fila o columna
        
        for i, (model_name, results) in enumerate(self.models_results.items()):
            y_true = results['y_true']
            y_pred = results['y_pred']
            
            # Calcular matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            
            # Ubicar subplot
            row_idx = i // n_cols
            col_idx = i % n_cols
            
            if n_rows > 1 and n_cols > 1:
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[i]
            
            # Plotear matriz de confusión
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'Matriz de Confusión: {model_name}')
            ax.set_xlabel('Predicho')
            ax.set_ylabel('Real')
            ax.set_xticklabels(['Negativo', 'Positivo'])
            ax.set_yticklabels(['Negativo', 'Positivo'])
        
        # Ocultar subplots vacíos
        for i in range(n_models, n_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            
            if n_rows > 1 and n_cols > 1:
                axes[row_idx, col_idx].axis('off')
            elif i < len(axes):
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self) -> str:
        """
        Genera un informe completo comparando todos los modelos.
        
        Returns:
            Ruta al archivo de informe generado
        """
        if not self.models_results:
            raise ValueError("No hay modelos para comparar. Añade modelos primero.")
        
        # Generar comparaciones
        comparison_df = self.compare_all_models()
        self.compare_roc_curves()
        self.compare_precision_recall_curves()
        self.compare_confusion_matrices()
        cal_df = self.analyze_calibration_differences()
        
        # Identificar mejor modelo para diferentes métricas
        best_models = {}
        metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'expected_calibration_error', 'roi']
        
        for metric in metrics_to_check:
            try:
                best_model = self.get_best_model(metric)
                best_models[metric] = best_model
            except:
                pass
        
        # Crear informe
        report_lines = [
            "# Informe Comparativo de Modelos de Predicción de Tenis",
            "",
            f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Número de modelos comparados: {len(self.models_results)}",
            "",
            "## 1. Resumen de Métricas",
            ""
        ]
        
        # Tabla comparativa de métricas
        report_lines.append("| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC | ECE | ROI |")
        report_lines.append("| ------ | -------- | --------- | ------ | -- | ------- | --- | --- |")
        
        for model_name, metrics in self.models_metrics.items():
            report_lines.append(
                f"| {model_name} | "
                f"{metrics.get('accuracy', 0):.4f} | "
                f"{metrics.get('precision', 0):.4f} | "
                f"{metrics.get('recall', 0):.4f} | "
                f"{metrics.get('f1', 0):.4f} | "
                f"{metrics.get('roc_auc', 0):.4f} | "
                f"{metrics.get('expected_calibration_error', 0):.4f} | "
                f"{metrics.get('roi', 0):.4f} |"
            )
        
        report_lines.extend([
            "",
            "## 2. Mejores Modelos por Métrica",
            ""
        ])
        
        # Tabla de mejores modelos
        for metric, model in best_models.items():
            # Formatear nombre de la métrica
            metric_name = metric.replace('_', ' ').title()
            if metric == 'roc_auc':
                metric_name = 'ROC AUC'
            elif metric == 'expected_calibration_error':
                metric_name = 'Error de Calibración (menor es mejor)'
                
            report_lines.append(f"- **{metric_name}**: {model}")
        
        report_lines.extend([
            "",
            "## 3. Análisis de Desacuerdos",
            ""
        ])
        
        # Analizar desacuerdos entre parejas de modelos
        if len(self.models_results) >= 2:
            model_names = list(self.models_results.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    
                    report_lines.append(f"### 3.{i+1}.{j+1}. Desacuerdos entre {model1} y {model2}")
                    report_lines.append("")
                    
                    try:
                        disagreements = self.analyze_disagreements(model1, model2)
                        
                        report_lines.append(f"- Número de desacuerdos: {disagreements['disagreement_count']} ({disagreements['disagreement_percentage']:.2f}% del total)")
                        report_lines.append(f"- Casos donde ninguno acierta: {disagreements['neither_correct_count']}")
                        report_lines.append(f"- Casos donde solo {model1} acierta: {disagreements[f'{model1}_only_correct_count']}")
                        report_lines.append(f"- Casos donde solo {model2} acierta: {disagreements[f'{model2}_only_correct_count']}")
                        report_lines.append(f"- Tasa de victorias de {model1}: {disagreements[f'{model1}_win_rate']:.2f}")
                        report_lines.append(f"- Diferencia promedio de probabilidad: {disagreements['avg_probability_difference']:.4f}")
                    except Exception as e:
                        report_lines.append(f"Error al analizar desacuerdos: {str(e)}")
                    
                    report_lines.append("")
        
        report_lines.extend([
            "",
            "## 4. Visualizaciones",
            "",
            "Se han generado las siguientes visualizaciones:",
            ""
        ])
        
        for file in os.listdir(self.output_dir):
            if file.endswith('.png'):
                report_lines.append(f"- [{file}]({file})")
        
        # Guardar el informe
        report_path = os.path.join(self.output_dir, 'model_comparison_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return report_path


if __name__ == "__main__":
    # Ejemplo de uso
    comparator = ModelComparator("example_comparison")
    
    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 1000
    
    # Datos base
    elo_diff = np.random.normal(0, 100, n_samples)
    surface_advantage = np.random.normal(0, 20, n_samples)
    
    # Probabilidad "real" basada en ELO y ventaja de superficie
    true_prob = 1 / (1 + np.exp(-(elo_diff + surface_advantage) / 100))

    # Resultados reales
    y_true = (np.random.random(n_samples) < true_prob).astype(int)
    
    # Modelo 1: Baseline
    model1_noise = np.random.normal(0, 0.1, n_samples)
    model1_prob = np.clip(true_prob + model1_noise, 0.01, 0.99)
    model1_pred = (model1_prob > 0.5).astype(int)
    
    # Modelo 2: Ligeramente mejor
    model2_noise = np.random.normal(0, 0.08, n_samples)
    model2_prob = np.clip(true_prob + model2_noise, 0.01, 0.99)
    model2_pred = (model2_prob > 0.5).astype(int)
    
    # Modelo 3: Más conservador
    model3_prob = np.clip(0.5 + (true_prob - 0.5) * 0.8, 0.01, 0.99)
    model3_pred = (model3_prob > 0.5).astype(int)
    
    # Añadir modelos al comparador
    comparator.add_model("Baseline", y_true, model1_pred, model1_prob)
    comparator.add_model("Mejorado", y_true, model2_pred, model2_prob)
    comparator.add_model("Conservador", y_true, model3_pred, model3_prob)
    
    # Generar informe completo
    report_path = comparator.generate_comprehensive_report()
    print(f"Informe comparativo generado en: {report_path}")
    
    # Análisis específico de desacuerdos
    disagreements = comparator.analyze_disagreements("Baseline", "Mejorado")
    print(f"Desacuerdos entre Baseline y Mejorado: {disagreements['disagreement_count']} casos")