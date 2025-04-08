"""
temporal_analysis.py

Este módulo implementa funcionalidades para evaluar la evolución del rendimiento
de modelos de predicción de tenis a lo largo del tiempo, identificando patrones
temporales, tendencias y potenciales causas de degradación o mejora.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.linear_model import LinearRegression

from .metrics import TennisMetrics


class TemporalAnalyzer:
    """
    Clase para analizar la evolución temporal del rendimiento de modelos de 
    predicción de tenis, permitiendo identificar patrones estacionales,
    tendencias y cambios significativos.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Inicializa el analizador temporal.
        
        Args:
            output_dir: Directorio donde se guardarán los gráficos y resultados
        """
        self.metrics = TennisMetrics()
        self.data = None
        self.temporal_metrics = None
        
        # Configurar directorio de salida
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"temporal_analysis_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, 
                 df: pd.DataFrame = None,
                 csv_path: str = None,
                 date_column: str = 'match_date',
                 true_column: str = 'y_true',
                 pred_column: str = 'y_pred',
                 prob_column: str = 'y_prob') -> pd.DataFrame:
        """
        Carga los datos para análisis temporal.
        
        Args:
            df: DataFrame con resultados y fechas (opcional)
            csv_path: Ruta al archivo CSV con resultados (opcional)
            date_column: Nombre de la columna de fecha
            true_column: Nombre de la columna de valores reales
            pred_column: Nombre de la columna de valores predichos
            prob_column: Nombre de la columna de probabilidades
            
        Returns:
            DataFrame con los datos cargados
        """
        if df is not None:
            self.data = df.copy()
        elif csv_path:
            self.data = pd.read_csv(csv_path)
        else:
            raise ValueError("Debe proporcionar un DataFrame o ruta CSV")
        
        # Verificar columnas requeridas
        required_columns = [date_column, true_column, pred_column, prob_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        # Convertir columna de fecha a datetime si es necesario
        if not pd.api.types.is_datetime64_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Ordenar por fecha
        self.data = self.data.sort_values(date_column)
        
        # Normalizar nombres de columnas
        self.data_columns = {
            'date': date_column,
            'true': true_column,
            'pred': pred_column,
            'prob': prob_column
        }
        
        return self.data
    
    def calculate_window_metrics(self, 
                               window_size: int = 30, 
                               step_size: int = 15,
                               min_samples: int = 20) -> pd.DataFrame:
        """
        Calcula métricas en ventanas temporales deslizantes.
        
        Args:
            window_size: Tamaño de ventana en días
            step_size: Tamaño de paso entre ventanas en días
            min_samples: Número mínimo de muestras para calcular métricas
            
        Returns:
            DataFrame con métricas por ventana temporal
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Ejecuta load_data primero.")
        
        # Extraer columnas relevantes
        date_col = self.data_columns['date']
        true_col = self.data_columns['true']
        pred_col = self.data_columns['pred']
        prob_col = self.data_columns['prob']
        
        # Fechas mínima y máxima
        min_date = self.data[date_col].min()
        max_date = self.data[date_col].max()
        
        # Configurar ventanas deslizantes
        window_starts = []
        window_ends = []
        current_date = min_date
        
        while current_date <= max_date:
            window_end = current_date + timedelta(days=window_size)
            window_starts.append(current_date)
            window_ends.append(window_end)
            current_date += timedelta(days=step_size)
        
        # Calcular métricas para cada ventana
        window_metrics = []
        
        for start_date, end_date in zip(window_starts, window_ends):
            # Filtrar datos para la ventana actual
            window_data = self.data[(self.data[date_col] >= start_date) & 
                                    (self.data[date_col] < end_date)]
            
            # Verificar cantidad mínima de muestras
            if len(window_data) < min_samples:
                continue
            
            # Calcular métricas
            y_true = window_data[true_col].values
            y_pred = window_data[pred_col].values
            y_prob = window_data[prob_col].values
            
            # Obtener cuotas si están disponibles
            odds = None
            if 'odds_player1' in window_data.columns:
                odds = window_data['odds_player1'].values
            
            # Calcular métricas para esta ventana
            metrics = self.metrics.calculate_all_metrics(y_true, y_pred, y_prob, odds)
            
            # Añadir fechas de ventana
            metrics['window_start'] = start_date
            metrics['window_end'] = end_date
            metrics['window_center'] = start_date + (end_date - start_date) / 2
            metrics['sample_count'] = len(window_data)
            
            window_metrics.append(metrics)
        
        # Crear DataFrame con los resultados
        if not window_metrics:
            raise ValueError("No se pudieron calcular métricas para ninguna ventana temporal")
        
        self.temporal_metrics = pd.DataFrame(window_metrics)
        
        # Guardar como CSV
        self.temporal_metrics.to_csv(os.path.join(self.output_dir, 'temporal_metrics.csv'), index=False)
        
        return self.temporal_metrics
    
    def analyze_trends(self, metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Analiza tendencias en las métricas a lo largo del tiempo.
        
        Args:
            metrics: Lista de métricas a analizar (si es None, se analizan las principales)
            
        Returns:
            Diccionario con análisis de tendencias por métrica
        """
        if self.temporal_metrics is None:
            raise ValueError("No hay métricas temporales. Ejecuta calculate_window_metrics primero.")
        
        # Métricas por defecto si no se especifican
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
                      'expected_calibration_error', 'roi']
        
        # Filtrar métricas disponibles
        available_metrics = [m for m in metrics if m in self.temporal_metrics.columns]
        if not available_metrics:
            raise ValueError(f"Ninguna de las métricas especificadas está disponible: {metrics}")
        
        # Análisis de tendencias
        trend_results = {}
        
        for metric in available_metrics:
            # Preparar datos para regresión
            X = (self.temporal_metrics['window_center'] - self.temporal_metrics['window_center'].min()).dt.total_seconds().values.reshape(-1, 1)
            y = self.temporal_metrics[metric].values
            
            # Ajustar regresión lineal
            model = LinearRegression()
            model.fit(X, y)
            
            # Predecir valores para visualizar tendencia
            y_pred = model.predict(X)
            
            # Calcular estadísticas
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(X, y)
            
            # Normalizar pendiente a cambio por año
            seconds_per_year = 365.25 * 24 * 60 * 60
            slope_per_year = slope * seconds_per_year
            
            # Calcular otros estadísticos
            mean_value = np.mean(y)
            std_value = np.std(y)
            min_value = np.min(y)
            max_value = np.max(y)
            relative_change = slope_per_year / mean_value if mean_value != 0 else 0
            
            # Guardar resultados
            trend_results[metric] = {
                'slope': slope,
                'slope_per_year': slope_per_year,
                'intercept': intercept,
                'r_squared': r_squared,
                'mean': mean_value,
                'std': std_value,
                'min': min_value,
                'max': max_value,
                'relative_change_per_year': relative_change,
                'trend_direction': 'up' if slope > 0 else 'down',
                'significance': 'high' if abs(r_squared) > 0.5 else ('medium' if abs(r_squared) > 0.25 else 'low')
            }
        
        # Guardar resultados
        with open(os.path.join(self.output_dir, 'trend_analysis.json'), 'w') as f:
            json.dump(trend_results, f, indent=4)
        
        # Generar visualizaciones
        self._plot_temporal_trends(available_metrics, trend_results)
        
        return trend_results
    
    def _plot_temporal_trends(self, metrics, trend_results):
        """
        Genera visualizaciones de tendencias temporales.
        
        Args:
            metrics: Lista de métricas a visualizar
            trend_results: Resultados del análisis de tendencias
        """
        # Determinar número de filas y columnas para subplots
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])  # Convertir a array para índice consistente
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Datos para gráfico
            dates = self.temporal_metrics['window_center']
            values = self.temporal_metrics[metric]
            
            # Graficar valores
            ax.plot(dates, values, 'o-', label='Valores observados')
            
            # Graficar línea de tendencia
            trend = trend_results[metric]
            
            # Fechas para línea de tendencia
            x_dates = [self.temporal_metrics['window_center'].min(), self.temporal_metrics['window_center'].max()]
            
            # Calcular valores de tendencia
            x_seconds = [(d - self.temporal_metrics['window_center'].min()).total_seconds() for d in x_dates]
            y_trend = [trend['intercept'] + trend['slope'] * x for x in x_seconds]
            
            # Graficar tendencia
            sign = '+' if trend['slope_per_year'] >= 0 else ''
            ax.plot(x_dates, y_trend, 'r--', 
                   label=f"Tendencia: {sign}{trend['slope_per_year']:.6f}/año (R²={trend['r_squared']:.2f})")
            
            # Configurar gráfico
            ax.set_title(f"Evolución de {metric}")
            ax.set_xlabel('Fecha')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)
            
            # Formato de fecha en eje X
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Ocultar ejes vacíos
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temporal_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def detect_change_points(self, 
                           metric: str = 'accuracy', 
                           window_size: int = 5) -> Dict[str, Any]:
        """
        Detecta puntos de cambio significativos en la evolución temporal.
        
        Args:
            metric: Métrica a analizar
            window_size: Tamaño de ventana para detección de cambios
            
        Returns:
            Diccionario con puntos de cambio detectados
        """
        if self.temporal_metrics is None:
            raise ValueError("No hay métricas temporales. Ejecuta calculate_window_metrics primero.")
        
        if metric not in self.temporal_metrics.columns:
            raise ValueError(f"La métrica {metric} no está disponible")
        
        # Verificar suficientes datos
        if len(self.temporal_metrics) < window_size * 2:
            return {"message": "Insuficientes datos para detectar puntos de cambio"}
        
        # Calcular cambios entre ventanas consecutivas
        values = self.temporal_metrics[metric].values
        dates = self.temporal_metrics['window_center'].values
        
        # Calcular diferencias relativas
        rolling_mean = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        
        # Detectar cambios significativos
        changes = []
        thresholds = np.std(values) * 1.5  # Umbral: 1.5 desviaciones estándar
        
        for i in range(len(rolling_mean) - 1):
            diff = rolling_mean[i+1] - rolling_mean[i]
            if abs(diff) > thresholds:
                idx = i + window_size // 2
                changes.append({
                    'date': dates[idx],
                    'change': diff,
                    'relative_change': diff / rolling_mean[i] if rolling_mean[i] != 0 else 0,
                    'direction': 'up' if diff > 0 else 'down',
                    'significance': abs(diff) / thresholds
                })
        
        # Ordenar por significancia
        changes.sort(key=lambda x: abs(x['significance']), reverse=True)
        
        # Guardar y visualizar resultados
        result = {
            'metric': metric,
            'threshold': thresholds,
            'change_points': changes
        }
        
        # Guardar como JSON
        with open(os.path.join(self.output_dir, f'change_points_{metric}.json'), 'w') as f:
            # Convertir fechas a formato serializable
            serializable_changes = []
            for change in changes:
                change_copy = change.copy()
                if isinstance(change_copy['date'], np.datetime64):
                    change_copy['date'] = str(change_copy['date'])
                serializable_changes.append(change_copy)
                
            json.dump({
                'metric': metric,
                'threshold': float(thresholds),
                'change_points': serializable_changes
            }, f, indent=4)
        
        # Visualizar cambios
        self._plot_change_points(metric, values, dates, changes, thresholds)
        
        return result
    
    def _plot_change_points(self, metric, values, dates, changes, threshold):
        """
        Visualiza los puntos de cambio detectados.
        
        Args:
            metric: Nombre de la métrica
            values: Valores de la métrica
            dates: Fechas correspondientes
            changes: Lista de puntos de cambio detectados
            threshold: Umbral utilizado para la detección
        """
        plt.figure(figsize=(12, 6))
        
        # Graficar evolución de la métrica
        plt.plot(dates, values, 'b-', label=metric)
        
        # Añadir puntos de cambio
        for change in changes:
            change_date = change['date']
            change_idx = np.where(dates == change_date)[0][0]
            
            # Determinar color según dirección
            color = 'g' if change['direction'] == 'up' else 'r'
            
            # Marcar punto de cambio
            plt.scatter(change_date, values[change_idx], color=color, s=100, 
                       zorder=5, label='_nolegend_')
            
            # Añadir anotación
            plt.annotate(f"{change['change']:.4f}", 
                        (change_date, values[change_idx]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color=color))
        
        # Configurar gráfico
        plt.title(f"Puntos de Cambio en {metric}")
        plt.xlabel('Fecha')
        plt.ylabel(metric)
        plt.grid(True)
        
        # Formato de fecha
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'change_points_{metric}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_seasonality(self, 
                          metric: str = 'accuracy',
                          by: str = 'month') -> pd.DataFrame:
        """
        Analiza patrones estacionales en los datos.
        
        Args:
            metric: Métrica a analizar
            by: Tipo de análisis estacional ('month', 'quarter', 'dayofweek')
            
        Returns:
            DataFrame con análisis estacional
        """
        if self.data is None or self.temporal_metrics is None:
            raise ValueError("Datos o métricas temporales no disponibles")
        
        if metric not in self.temporal_metrics.columns:
            raise ValueError(f"La métrica {metric} no está disponible")
        
        # Extraer fecha y métrica
        date_col = self.data_columns['date']
        true_col = self.data_columns['true']
        pred_col = self.data_columns['pred']
        prob_col = self.data_columns['prob']
        
        # Preparar datos para análisis estacional
        if by == 'month':
            self.data['period'] = self.data[date_col].dt.month
            period_name = 'Mes'
            labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        elif by == 'quarter':
            self.data['period'] = self.data[date_col].dt.quarter
            period_name = 'Trimestre'
            labels = ['T1', 'T2', 'T3', 'T4']
        elif by == 'dayofweek':
            self.data['period'] = self.data[date_col].dt.dayofweek
            period_name = 'Día de la Semana'
            labels = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        else:
            raise ValueError("El parámetro 'by' debe ser 'month', 'quarter' o 'dayofweek'")
        
        # Calcular métricas por periodo
        seasonal_metrics = []
        
        for period, group in self.data.groupby('period'):
            if len(group) < 20:  # Mínimo de muestras
                continue
                
            # Calcular métricas
            y_true = group[true_col].values
            y_pred = group[pred_col].values
            y_prob = group[prob_col].values
            
            # Obtener cuotas si están disponibles
            odds = None
            if 'odds_player1' in group.columns:
                odds = group['odds_player1'].values
            
            # Calcular métricas para este periodo
            metrics = self.metrics.calculate_all_metrics(y_true, y_pred, y_prob, odds)
            metrics['period'] = int(period)
            metrics['sample_count'] = len(group)
            seasonal_metrics.append(metrics)
        
        if not seasonal_metrics:
            raise ValueError("Insuficientes datos para análisis estacional")
        
        # Crear DataFrame con resultados
        seasonal_df = pd.DataFrame(seasonal_metrics)
        
        # Añadir etiquetas
        period_labels = {i+1: label for i, label in enumerate(labels)}
        seasonal_df['period_label'] = seasonal_df['period'].map(period_labels)
        
        # Ordenar por periodo
        seasonal_df = seasonal_df.sort_values('period')
        
        # Guardar como CSV
        seasonal_df.to_csv(os.path.join(self.output_dir, f'seasonality_{by}_{metric}.csv'), index=False)
        
        # Visualizar resultados
        self._plot_seasonality(seasonal_df, metric, period_name)
        
        return seasonal_df
    
    def _plot_seasonality(self, seasonal_df, metric, period_name):
        """
        Visualiza el análisis estacional.
        
        Args:
            seasonal_df: DataFrame con datos estacionales
            metric: Métrica analizada
            period_name: Nombre del periodo (Mes, Trimestre, etc.)
        """
        plt.figure(figsize=(12, 6))
        
        # Graficar la métrica por periodo
        bars = plt.bar(seasonal_df['period_label'], seasonal_df[metric])
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)
        
        # Calcular media global para referencia
        mean_value = seasonal_df[metric].mean()
        plt.axhline(y=mean_value, color='r', linestyle='--', 
                   label=f'Media: {mean_value:.4f}')
        
        # Configurar gráfico
        plt.title(f"{metric} por {period_name}")
        plt.xlabel(period_name)
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'seasonality_{period_name.lower()}_{metric}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self) -> str:
        """
        Genera un informe completo con el análisis temporal.
        
        Returns:
            Ruta al archivo de informe generado
        """
        if self.temporal_metrics is None:
            raise ValueError("No hay métricas temporales. Ejecuta calculate_window_metrics primero.")
        
        # Analizar tendencias en métricas principales
        main_metrics = ['accuracy', 'roc_auc', 'expected_calibration_error', 'roi']
        available_metrics = [m for m in main_metrics if m in self.temporal_metrics.columns]
        
        trend_results = self.analyze_trends(available_metrics)
        
        # Detectar puntos de cambio
        change_points = {}
        for metric in available_metrics:
            try:
                change_points[metric] = self.detect_change_points(metric)
            except Exception as e:
                print(f"Error al detectar puntos de cambio para {metric}: {str(e)}")
        
        # Análisis estacional
        try:
            monthly_seasonality = self.analyze_seasonality('accuracy', 'month')
            quarterly_seasonality = self.analyze_seasonality('accuracy', 'quarter')
        except Exception as e:
            print(f"Error en análisis estacional: {str(e)}")
        
        # Crear informe
        report_lines = [
            "# Informe de Análisis Temporal de Predicciones de Tenis",
            "",
            f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Periodo analizado: {self.temporal_metrics['window_start'].min().strftime('%Y-%m-%d')} a {self.temporal_metrics['window_end'].max().strftime('%Y-%m-%d')}",
            "",
            "## 1. Resumen de Tendencias",
            ""
        ]
        
        # Tabla de tendencias
        report_lines.append("| Métrica | Tendencia Anual | Cambio Relativo | R² | Dirección | Significancia |")
        report_lines.append("| ------- | -------------- | --------------- | -- | --------- | ------------- |")
        
        for metric, trend in trend_results.items():
            report_lines.append(
                f"| {metric} | "
                f"{trend['slope_per_year']:.6f} | "
                f"{trend['relative_change_per_year']:.2%} | "
                f"{trend['r_squared']:.3f} | "
                f"{trend['trend_direction']} | "
                f"{trend['significance']} |"
            )
        
        # Añadir interpretación
        report_lines.append("")
        for metric, trend in trend_results.items():
            direction = "mejorando" if (trend['trend_direction'] == 'up' and metric != 'expected_calibration_error') or (trend['trend_direction'] == 'down' and metric == 'expected_calibration_error') else "empeorando"
            significance = "altamente significativa" if trend['significance'] == 'high' else ("moderadamente significativa" if trend['significance'] == 'medium' else "no significativa")
            
            report_lines.append(f"- **{metric}**: Está {direction} a un ritmo de {trend['slope_per_year']:.6f} unidades por año ({trend['relative_change_per_year']:.2%}). Esta tendencia es {significance} (R²={trend['r_squared']:.3f}).")
        
        report_lines.extend([
            "",
            "## 2. Puntos de Cambio Significativos",
            ""
        ])
        
        # Añadir puntos de cambio para cada métrica
        for metric, cp_data in change_points.items():
            report_lines.append(f"### 2.{list(change_points.keys()).index(metric) + 1}. Puntos de cambio en {metric}")
            report_lines.append("")
            
            if 'change_points' in cp_data and cp_data['change_points']:
                report_lines.append("| Fecha | Cambio | Cambio Relativo | Dirección | Significancia |")
                report_lines.append("| ----- | ------ | --------------- | --------- | ------------- |")
                
                for i, cp in enumerate(cp_data['change_points'][:5]):  # Mostrar los 5 más significativos
                    cp_date = cp['date']
                    if isinstance(cp_date, np.datetime64):
                        cp_date = pd.Timestamp(cp_date).strftime('%Y-%m-%d')
                    
                    report_lines.append(
                        f"| {cp_date} | "
                        f"{cp['change']:.4f} | "
                        f"{cp['relative_change']:.2%} | "
                        f"{cp['direction']} | "
                        f"{cp['significance']:.2f} |"
                    )
            else:
                report_lines.append("No se detectaron puntos de cambio significativos.")
            
            report_lines.append("")
        
        report_lines.extend([
            "",
            "## 3. Análisis Estacional",
            ""
        ])
        
        # Añadir análisis estacional por mes
        try:
            report_lines.append("### 3.1. Variación Mensual")
            report_lines.append("")
            report_lines.append("| Mes | Accuracy | ROC AUC | ECE | ROI | Muestras |")
            report_lines.append("| --- | -------- | ------- | --- | --- | -------- |")
            
            for _, row in monthly_seasonality.iterrows():
                report_lines.append(
                    f"| {row['period_label']} | "
                    f"{row.get('accuracy', 0):.4f} | "
                    f"{row.get('roc_auc', 0):.4f} | "
                    f"{row.get('expected_calibration_error', 0):.4f} | "
                    f"{row.get('roi', 0):.4f} | "
                    f"{row['sample_count']} |"
                )
            
            # Encontrar mejor y peor mes
            best_month = monthly_seasonality.loc[monthly_seasonality['accuracy'].idxmax()]
            worst_month = monthly_seasonality.loc[monthly_seasonality['accuracy'].idxmin()]
            
            report_lines.append("")
            report_lines.append(f"- **Mejor rendimiento**: {best_month['period_label']} (Accuracy: {best_month['accuracy']:.4f})")
            report_lines.append(f"- **Peor rendimiento**: {worst_month['period_label']} (Accuracy: {worst_month['accuracy']:.4f})")
        except:
            report_lines.append("Análisis estacional mensual no disponible.")
        
        # Añadir análisis estacional por trimestre
        try:
            report_lines.append("")
            report_lines.append("### 3.2. Variación Trimestral")
            report_lines.append("")
            report_lines.append("| Trimestre | Accuracy | ROC AUC | ECE | ROI | Muestras |")
            report_lines.append("| --------- | -------- | ------- | --- | --- | -------- |")
            
            for _, row in quarterly_seasonality.iterrows():
                report_lines.append(
                    f"| {row['period_label']} | "
                    f"{row.get('accuracy', 0):.4f} | "
                    f"{row.get('roc_auc', 0):.4f} | "
                    f"{row.get('expected_calibration_error', 0):.4f} | "
                    f"{row.get('roi', 0):.4f} | "
                    f"{row['sample_count']} |"
                )
        except:
            report_lines.append("Análisis estacional trimestral no disponible.")
        
        report_lines.extend([
            "",
            "## 4. Visualizaciones",
            "",
            "Las siguientes visualizaciones se han generado para este análisis:",
            ""
        ])
        
        # Listar todas las visualizaciones
        for file in os.listdir(self.output_dir):
            if file.endswith('.png'):
                report_lines.append(f"- [{file}]({file})")
        
        # Añadir conclusiones
        report_lines.extend([
            "",
            "## 5. Conclusiones y Recomendaciones",
            ""
        ])
        
        # Generar conclusiones basadas en los análisis
        significant_trends = [m for m, t in trend_results.items() if t['significance'] != 'low']
        
        if significant_trends:
            report_lines.append("### Tendencias Significativas")
            for metric in significant_trends:
                trend = trend_results[metric]
                direction = "mejorando" if (trend['trend_direction'] == 'up' and metric != 'expected_calibration_error') or (trend['trend_direction'] == 'down' and metric == 'expected_calibration_error') else "empeorando"
                
                report_lines.append(f"- El rendimiento en **{metric}** está **{direction}** significativamente a lo largo del tiempo.")
        else:
            report_lines.append("- No se observan tendencias significativas en el rendimiento a lo largo del tiempo.")
        
        # Añadir recomendaciones basadas en el análisis
        report_lines.append("")
        report_lines.append("### Recomendaciones")
        
        # Recomendaciones basadas en tendencias
        negative_trends = [m for m, t in trend_results.items() 
                         if (t['trend_direction'] == 'down' and m != 'expected_calibration_error') or 
                            (t['trend_direction'] == 'up' and m == 'expected_calibration_error')]
        
        if negative_trends:
            report_lines.append("1. **Investigar causas de degradación**: Revisar cambios en el modelo o en los datos que podrían explicar el empeoramiento en:")
            for metric in negative_trends:
                report_lines.append(f"   - {metric}")
        
        # Recomendaciones basadas en estacionalidad
        try:
            worst_month_name = worst_month['period_label']
            report_lines.append(f"2. **Mejorar rendimiento estacional**: El modelo muestra un peor rendimiento en {worst_month_name}. Considerar:")
            report_lines.append("   - Entrenar modelos específicos para diferentes épocas del año")
            report_lines.append("   - Incorporar características estacionales al modelo")
        except:
            pass
        
        # Recomendaciones generales
        report_lines.append("3. **Monitoreo continuo**: Implementar un sistema de monitoreo continuo para detectar cambios significativos en el rendimiento")
        report_lines.append("4. **Reentrenamiento periódico**: Establecer un calendario de reentrenamiento basado en los patrones temporales observados")
        
        # Guardar el informe
        report_path = os.path.join(self.output_dir, 'temporal_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return report_path


if __name__ == "__main__":
    # Ejemplo de uso
    analyzer = TemporalAnalyzer("example_temporal_analysis")
    
    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 1000
    
    # Generar fechas en un rango de 2 años
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Simular tendencia temporal (mejora gradual)
    time_factor = np.linspace(0, 0.1, n_samples)  # Mejora gradual
    
    # Añadir estacionalidad (mes)
    months = [d.month for d in dates]
    month_factors = {
        1: -0.05, 2: -0.03, 3: 0, 4: 0.02, 5: 0.03, 6: 0.04,
        7: 0.03, 8: 0.01, 9: 0, 10: -0.01, 11: -0.02, 12: -0.04
    }
    seasonal_factor = np.array([month_factors[m] for m in months])
    
    # Simulación base
    elo_diff = np.random.normal(0, 100, n_samples)
    surface_advantage = np.random.normal(0, 20, n_samples)
    
    # Probabilidad "real" basada en ELO, ventaja de superficie, tendencia y estacionalidad
    base_prob = 1 / (1 + np.exp(-(elo_diff + surface_advantage) / 100))
    true_prob = np.clip(base_prob + time_factor + seasonal_factor, 0.01, 0.99)
    
    # Resultados reales
    y_true = (np.random.random(n_samples) < true_prob).astype(int)
    
    # Probabilidad predicha (con algo de ruido)
    y_prob = np.clip(true_prob + np.random.normal(0, 0.1, n_samples), 0.01, 0.99)
    
    # Predicciones
    y_pred = (y_prob > 0.5).astype(int)
    
    # Crear DataFrame
    results = pd.DataFrame({
        'match_date': dates,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'player1_elo': np.random.normal(1500, 200, n_samples),
        'player2_elo': np.random.normal(1500, 200, n_samples),
        'elo_difference': elo_diff,
        'surface': np.random.choice(['hard', 'clay', 'grass', 'carpet'], n_samples),
        'tourney_level': np.random.choice(['grand_slam', 'atp1000', 'atp500', 'atp250', 'challenger'], n_samples)
    })
    
    # Cargar datos
    analyzer.load_data(results, date_column='match_date')
    
    # Calcular métricas en ventanas temporales
    analyzer.calculate_window_metrics(window_size=30, step_size=15)
    
    # Generar informe completo
    report_path = analyzer.generate_comprehensive_report()
    print(f"Análisis temporal completado. Informe generado en: {report_path}")