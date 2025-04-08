"""
metrics.py

Este módulo implementa métricas específicas para evaluar modelos de predicción de tenis.
Incluye métricas estándar de machine learning y métricas específicas para apuestas.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from scipy import stats


class TennisMetrics:
    """
    Clase para calcular y evaluar métricas específicas para modelos de predicción de tenis.
    Incluye métricas estándar y métricas específicas para apuestas y calibración.
    """
    
    def __init__(self):
        """Inicializa la clase TennisMetrics."""
        self.metrics_results = {}
        
    def calculate_standard_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calcula métricas estándar de clasificación.
        
        Args:
            y_true: Valores reales (1 para victoria, 0 para derrota)
            y_pred: Valores predichos (1 para victoria, 0 para derrota)
            y_prob: Probabilidades predichas para la clase positiva
            
        Returns:
            dict: Diccionario con las métricas calculadas
        """
        metrics = {}
        
        # Métricas básicas
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Métricas basadas en probabilidad (si están disponibles)
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['log_loss'] = log_loss(y_true, y_prob)
            metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        
        self.metrics_results.update(metrics)
        return metrics
    
    def calculate_calibration_metrics(self, y_true, y_prob, n_bins=10):
        """
        Calcula métricas de calibración para evaluar la calidad de las probabilidades.
        
        Args:
            y_true: Valores reales (1 para victoria, 0 para derrota)
            y_prob: Probabilidades predichas para la clase positiva
            n_bins: Número de bins para la calibración
            
        Returns:
            dict: Diccionario con las métricas de calibración
        """
        metrics = {}
        
        # Expected Calibration Error (ECE)
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
        
        # Expected Calibration Error
        ece = np.sum(bin_counts * np.abs(bin_avg_pred - bin_avg_true)) / sum(bin_counts)
        metrics['expected_calibration_error'] = ece
        
        # Maximum Calibration Error
        mce = np.max(np.abs(bin_avg_pred - bin_avg_true))
        metrics['maximum_calibration_error'] = mce
        
        # Guardar datos para diagrama de confiabilidad
        metrics['calibration_data'] = {
            'bin_pred': bin_avg_pred,
            'bin_true': bin_avg_true,
            'bin_counts': bin_counts,
            'bin_edges': np.linspace(0, 1, n_bins + 1)
        }
        
        self.metrics_results.update(metrics)
        return metrics
    
    def calculate_betting_metrics(self, y_true, y_prob, odds=None):
        """
        Calcula métricas específicas para apuestas y retorno de inversión.
        
        Args:
            y_true: Valores reales (1 para victoria, 0 para derrota)
            y_prob: Probabilidades predichas para la clase positiva
            odds: Cuotas de mercado para los eventos (si están disponibles)
            
        Returns:
            dict: Diccionario con métricas para evaluación de apuestas
        """
        metrics = {}
        
        if odds is None:
            # Si no hay cuotas, usamos cuotas simuladas basadas en las probabilidades predecidas
            odds = 1 / y_prob
        
        # Valor esperado (EV) para cada apuesta
        ev = (y_prob * (odds - 1)) - (1 - y_prob)
        metrics['average_ev'] = np.mean(ev)
        
        # Kelly Criterion
        kelly_fractions = (odds * y_prob - 1) / (odds - 1)
        kelly_fractions = np.clip(kelly_fractions, 0, 0.25)  # Limitado al 25% del bankroll
        metrics['average_kelly_fraction'] = np.mean(kelly_fractions)
        
        # Profit/Loss simulado con stakes Kelly
        initial_bankroll = 1000
        bankroll = initial_bankroll
        profits = []
        
        for i in range(len(y_true)):
            stake = bankroll * kelly_fractions[i]
            if y_true[i] == 1:
                # Ganancia
                profit = stake * (odds[i] - 1)
            else:
                # Pérdida
                profit = -stake
            
            bankroll += profit
            profits.append(profit)
        
        metrics['final_bankroll'] = bankroll
        metrics['roi'] = (bankroll - initial_bankroll) / initial_bankroll
        
        # Métricas adicionales de apuestas
        metrics['average_profit'] = np.mean(profits)
        metrics['profit_factor'] = abs(np.sum([p for p in profits if p > 0]) / np.sum([p for p in profits if p < 0]))
        metrics['win_rate'] = np.mean(y_true)
        
        # Apuestas positivas según criterio de Kelly
        positive_bets = kelly_fractions > 0
        metrics['positive_bets_percentage'] = np.mean(positive_bets)
        
        # Sharpe ratio (basado en los retornos)
        returns = np.array(profits) / initial_bankroll
        metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        self.metrics_results.update(metrics)
        return metrics
    
    def calculate_all_metrics(self, y_true, y_pred, y_prob, odds=None, n_bins=10):
        """
        Calcula todas las métricas disponibles.
        
        Args:
            y_true: Valores reales (1 para victoria, 0 para derrota)
            y_pred: Valores predichos (1 para victoria, 0 para derrota)
            y_prob: Probabilidades predichas para la clase positiva
            odds: Cuotas de mercado para los eventos (opcional)
            n_bins: Número de bins para calibración
            
        Returns:
            dict: Diccionario con todas las métricas calculadas
        """
        # Resetear resultados previos
        self.metrics_results = {}
        
        # Calcular todas las métricas
        self.calculate_standard_metrics(y_true, y_pred, y_prob)
        self.calculate_calibration_metrics(y_true, y_prob, n_bins)
        self.calculate_betting_metrics(y_true, y_prob, odds)
        
        return self.metrics_results
    
    def plot_calibration_curve(self, output_path=None):
        """
        Genera y guarda un diagrama de calibración (reliability diagram).
        
        Args:
            output_path: Ruta para guardar el gráfico (opcional)
            
        Returns:
            matplotlib.figure.Figure: Objeto figura con el diagrama
        """
        if 'calibration_data' not in self.metrics_results:
            raise ValueError("No hay datos de calibración disponibles. Ejecuta calculate_calibration_metrics primero.")
        
        cal_data = self.metrics_results['calibration_data']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Barras de frecuencia (histograma)
        width = 1.0 / len(cal_data['bin_counts'])
        bin_edges = cal_data['bin_edges'][:-1]
        ax2 = ax.twinx()
        ax2.bar(bin_edges + width/2, cal_data['bin_counts'] / np.sum(cal_data['bin_counts']), 
                width=width, alpha=0.3, color='gray', label='Frecuencia')
        ax2.set_ylabel('Frecuencia de predicciones')
        
        # Línea de calibración perfecta
        ax.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
        
        # Línea de calibración actual
        nonzero = cal_data['bin_counts'] > 0
        ax.plot(bin_edges[nonzero] + width/2, cal_data['bin_true'][nonzero], 
                marker='o', linestyle='-', label='Calibración del modelo')
        
        # Etiquetas y leyenda
        ax.set_xlabel('Probabilidad predicha')
        ax.set_ylabel('Fracción de resultados positivos')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title('Diagrama de Calibración (Reliability Diagram)')
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_metrics_report(self, detailed=True):
        """
        Genera un informe de texto formateado con las métricas calculadas.
        
        Args:
            detailed: Si True, incluye métricas detalladas
            
        Returns:
            str: Informe formateado con las métricas
        """
        if not self.metrics_results:
            return "No hay métricas disponibles. Ejecuta calculate_all_metrics primero."
        
        lines = ["# Informe de Métricas de Predicción de Tenis", ""]
        
        # Métricas estándar
        lines.append("## Métricas Estándar")
        if 'accuracy' in self.metrics_results:
            lines.append(f"- Accuracy: {self.metrics_results['accuracy']:.4f}")
        if 'precision' in self.metrics_results:
            lines.append(f"- Precision: {self.metrics_results['precision']:.4f}")
        if 'recall' in self.metrics_results:
            lines.append(f"- Recall: {self.metrics_results['recall']:.4f}")
        if 'f1' in self.metrics_results:
            lines.append(f"- F1 Score: {self.metrics_results['f1']:.4f}")
        if 'roc_auc' in self.metrics_results:
            lines.append(f"- ROC AUC: {self.metrics_results['roc_auc']:.4f}")
        
        # Matriz de confusión
        if all(k in self.metrics_results for k in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']):
            lines.append("\n## Matriz de Confusión")
            lines.append("```")
            lines.append(f"                | Predicción Positiva | Predicción Negativa")
            lines.append(f"----------------+--------------------+--------------------")
            lines.append(f"Actual Positivo | {self.metrics_results['true_positives']:18} | {self.metrics_results['false_negatives']:18}")
            lines.append(f"Actual Negativo | {self.metrics_results['false_positives']:18} | {self.metrics_results['true_negatives']:18}")
            lines.append("```")
        
        # Métricas de calibración
        if 'expected_calibration_error' in self.metrics_results:
            lines.append("\n## Métricas de Calibración")
            lines.append(f"- Error de Calibración Esperado (ECE): {self.metrics_results['expected_calibration_error']:.4f}")
            lines.append(f"- Error de Calibración Máximo (MCE): {self.metrics_results['maximum_calibration_error']:.4f}")
        
        # Métricas de apuestas
        if 'roi' in self.metrics_results:
            lines.append("\n## Métricas para Apuestas")
            lines.append(f"- ROI: {self.metrics_results['roi']:.4f}")
            lines.append(f"- Bankroll Final: {self.metrics_results['final_bankroll']:.2f}")
            lines.append(f"- Porcentaje de Apuestas Positivas: {self.metrics_results['positive_bets_percentage']:.4f}")
            lines.append(f"- Factor de Beneficio: {self.metrics_results['profit_factor']:.4f}")
            lines.append(f"- Sharpe Ratio: {self.metrics_results['sharpe_ratio']:.4f}")
        
        # Métricas detalladas (opcional)
        if detailed:
            lines.append("\n## Métricas Detalladas")
            for key, value in sorted(self.metrics_results.items()):
                if key not in ['calibration_data'] and key not in [
                    'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                    'true_positives', 'false_positives', 'true_negatives', 'false_negatives',
                    'expected_calibration_error', 'maximum_calibration_error',
                    'roi', 'final_bankroll', 'positive_bets_percentage', 'profit_factor', 'sharpe_ratio'
                ]:
                    if isinstance(value, (int, float)):
                        lines.append(f"- {key}: {value:.6f}")
        
        return "\n".join(lines)
    
    def compare_models(self, model_metrics_list, model_names=None):
        """
        Compara métricas entre múltiples modelos.
        
        Args:
            model_metrics_list: Lista de diccionarios con métricas de cada modelo
            model_names: Lista de nombres para los modelos (opcional)
            
        Returns:
            DataFrame con comparación de métricas entre modelos
        """
        if model_names is None:
            model_names = [f"Modelo {i+1}" for i in range(len(model_metrics_list))]
        
        if len(model_names) != len(model_metrics_list):
            raise ValueError("La lista de nombres debe tener la misma longitud que la lista de métricas")
        
        # Seleccionar métricas clave para comparación
        key_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
            'expected_calibration_error', 'roi', 'sharpe_ratio'
        ]
        
        comparison_data = {}
        
        for metric in key_metrics:
            metric_values = []
            for metrics_dict in model_metrics_list:
                if metric in metrics_dict:
                    metric_values.append(metrics_dict[metric])
                else:
                    metric_values.append(None)
            comparison_data[metric] = metric_values
        
        comparison_df = pd.DataFrame(comparison_data, index=model_names)
        return comparison_df

    def export_metrics(self, format='json', filepath=None):
        """
        Exporta las métricas calculadas en varios formatos.
        
        Args:
            format: Formato de exportación ('json', 'csv', 'prometheus', 'influxdb')
            filepath: Ruta para guardar el archivo (opcional)
            
        Returns:
            Datos formateados o ruta del archivo guardado
        """
        if not self.metrics_results:
            raise ValueError("No hay métricas para exportar. Ejecuta calculate_all_metrics primero.")
        
        # Filtrar datos complejos que no se pueden serializar fácilmente
        export_metrics = {k: v for k, v in self.metrics_results.items() 
                        if not isinstance(v, dict) and not isinstance(v, (list, np.ndarray))}
        
        if format == 'json':
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(export_metrics, f, indent=4)
                return filepath
            return json.dumps(export_metrics, indent=4)
        
        elif format == 'csv':
            df = pd.DataFrame([export_metrics])
            if filepath:
                df.to_csv(filepath, index=False)
                return filepath
            return df.to_csv(index=False)
        
        elif format == 'prometheus':
            # Formato para Prometheus Pushgateway
            lines = []
            for metric, value in export_metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f'tennis_prediction_{metric} {value}')
            result = '\n'.join(lines)
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(result)
                return filepath
            return result
        
        elif format == 'influxdb':
            # Formato para InfluxDB Line Protocol
            timestamp = int(datetime.now().timestamp() * 1e9)  # nanosegundos
            lines = []
            for metric, value in export_metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f'tennis_prediction,metric={metric} value={value} {timestamp}')
            result = '\n'.join(lines)
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(result)
                return filepath
            return result
        
        else:
            raise ValueError(f"Formato '{format}' no soportado. Opciones: json, csv, prometheus, influxdb")

    def calculate_temporal_metrics(self, y_true, y_pred, y_prob, timestamps, window_size=30):
        """
        Calcula métricas en ventanas temporales para evaluar la evolución del rendimiento.
        
        Args:
            y_true: Valores reales (1 para victoria, 0 para derrota)
            y_pred: Valores predichos (1 para victoria, 0 para derrota)
            y_prob: Probabilidades predichas para la clase positiva
            timestamps: Lista/array de timestamps (puede ser datetime o numérico)
            window_size: Tamaño de la ventana deslizante en días
            
        Returns:
            DataFrame con métricas a lo largo del tiempo
        """
        if len(y_true) != len(timestamps):
            raise ValueError("La longitud de y_true y timestamps debe ser igual")
        
        # Convertir a DataFrame para procesamiento temporal
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        })
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp')
        
        # Crear ventanas deslizantes
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        
        window_starts = []
        window_ends = []
        accuracies = []
        roc_aucs = []
        calibration_errors = []
        
        current_date = min_date
        
        while current_date <= max_date:
            window_end = current_date + pd.Timedelta(days=window_size)
            window_data = df[(df['timestamp'] >= current_date) & (df['timestamp'] < window_end)]
            
            if len(window_data) >= 20:  # Mínimo de datos para calcular métricas
                window_y_true = window_data['y_true'].values
                window_y_pred = window_data['y_pred'].values
                window_y_prob = window_data['y_prob'].values
                
                # Calcular métricas para esta ventana
                window_metrics = self.calculate_standard_metrics(window_y_true, window_y_pred, window_y_prob)
                window_cal_metrics = self.calculate_calibration_metrics(window_y_true, window_y_prob)
                
                window_starts.append(current_date)
                window_ends.append(window_end)
                accuracies.append(window_metrics['accuracy'])
                roc_aucs.append(window_metrics.get('roc_auc', None))
                calibration_errors.append(window_cal_metrics.get('expected_calibration_error', None))
            
            current_date += pd.Timedelta(days=window_size // 2)  # Ventanas solapadas
        
        # Crear DataFrame de resultados
        temporal_metrics = pd.DataFrame({
            'window_start': window_starts,
            'window_end': window_ends,
            'accuracy': accuracies,
            'roc_auc': roc_aucs,
            'expected_calibration_error': calibration_errors
        })
        
        return temporal_metrics


if __name__ == "__main__":
    # Ejemplo de uso
    metrics = TennisMetrics()
    
    # Datos simulados
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.random(100) * 0.5 + 0.25  # Probabilidades entre 0.25 y 0.75
    y_pred = (y_prob > 0.5).astype(int)
    odds = 1 / y_prob * 0.9  # Odds con margen del 10%
    
    # Calcular todas las métricas
    results = metrics.calculate_all_metrics(y_true, y_pred, y_prob, odds)
    
    # Imprimir informe
    print(metrics.get_metrics_report())
    
    # Generar gráfico de calibración
    metrics.plot_calibration_curve('calibration_curve.png')
    
    # Ejemplo de comparación de modelos
    model2_y_prob = np.clip(y_prob + np.random.normal(0, 0.05, len(y_prob)), 0, 1)
    model2_y_pred = (model2_y_prob > 0.5).astype(int)
    model2_results = metrics.calculate_all_metrics(y_true, model2_y_pred, model2_y_prob, odds)
    
    comparison = metrics.compare_models([results, model2_results], ["Modelo Base", "Modelo Mejorado"])
    print("\nComparación de modelos:")
    print(comparison)
    
    # Ejemplo de exportación de métricas
    json_output = metrics.export_metrics(format='json')
    print("\nExportación JSON:")
    print(json_output[:200] + "...")  