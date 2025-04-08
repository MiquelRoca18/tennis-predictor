"""
bias_detector.py

Este módulo implementa funcionalidades para identificar sesgos sistemáticos en modelos
de predicción de tenis, tanto en las probabilidades generadas como en el rendimiento
con diferentes tipos de jugadores, superficies o condiciones.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats

from .metrics import TennisMetrics


class BiasDetector:
    """
    Clase para detectar y analizar sesgos sistemáticos en los modelos de predicción de tenis.
    Permite identificar problemas como calibración asimétrica, sesgos por tipo de jugador,
    superficie, o nivel de torneo.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Inicializa el detector de sesgos.
        
        Args:
            output_dir: Directorio donde se guardarán los gráficos y resultados
        """
        self.metrics = TennisMetrics()
        self.data = None
        self.bias_results = {}
        
        # Configurar directorio de salida
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"bias_analysis_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, 
                 df: pd.DataFrame = None,
                 csv_path: str = None) -> pd.DataFrame:
        """
        Carga los datos para análisis de sesgos.
        
        Args:
            df: DataFrame con resultados y características (opcional)
            csv_path: Ruta al archivo CSV con resultados (opcional)
            
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
        required_columns = ['y_true', 'y_pred', 'y_prob']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            # Intentar con nombres alternativos
            column_mapping = {
                'winner_actual': 'y_true',
                'winner_predicted': 'y_pred',
                'probability': 'y_prob',
                'probability_player1': 'y_prob'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in self.data.columns and new_name in missing_columns:
                    self.data[new_name] = self.data[old_name]
                    missing_columns.remove(new_name)
        
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        return self.data
    
    def analyze_probability_calibration(self, n_bins: int = 10) -> Dict[str, Any]:
        """
        Analiza la calibración de probabilidades para detectar sesgos.
        
        Args:
            n_bins: Número de bins para el análisis de calibración
            
        Returns:
            Diccionario con resultados del análisis de calibración
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Ejecuta load_data primero.")
        
        # Calcular métricas de calibración
        y_true = self.data['y_true'].values
        y_prob = self.data['y_prob'].values
        
        cal_metrics = self.metrics.calculate_calibration_metrics(y_true, y_prob, n_bins)
        cal_data = cal_metrics['calibration_data']
        
        # Calcular sesgos de calibración
        bin_edges = cal_data['bin_edges']
        bin_true = cal_data['bin_true']
        bin_pred = cal_data['bin_pred']
        bin_counts = cal_data['bin_counts']
        
        # Identificar bins con suficientes muestras
        valid_bins = bin_counts >= 10
        
        # Calcular bias por bin
        bin_bias = np.zeros_like(bin_pred)
        bin_bias[valid_bins] = bin_pred[valid_bins] - bin_true[valid_bins]
        
        # Calcular métricas globales de sesgo
        bias_metrics = {
            'expected_calibration_error': cal_metrics['expected_calibration_error'],
            'maximum_calibration_error': cal_metrics['maximum_calibration_error'],
            'mean_bias': np.mean(bin_bias[valid_bins]),
            'bias_std': np.std(bin_bias[valid_bins]),
            'bias_range': (np.min(bin_bias[valid_bins]), np.max(bin_bias[valid_bins])),
        }
        
        # Detectar patrón de sesgo sistemático
        overconfidence = np.sum((bin_pred > bin_true) & valid_bins)
        underconfidence = np.sum((bin_pred < bin_true) & valid_bins)
        total_valid_bins = np.sum(valid_bins)
        
        if total_valid_bins > 0:
            if overconfidence / total_valid_bins > 0.7:
                bias_metrics['pattern'] = 'overconfidence'
                bias_metrics['pattern_strength'] = overconfidence / total_valid_bins
            elif underconfidence / total_valid_bins > 0.7:
                bias_metrics['pattern'] = 'underconfidence'
                bias_metrics['pattern_strength'] = underconfidence / total_valid_bins
            else:
                bias_metrics['pattern'] = 'mixed'
                bias_metrics['pattern_strength'] = max(overconfidence, underconfidence) / total_valid_bins
        else:
            bias_metrics['pattern'] = 'unknown'
            bias_metrics['pattern_strength'] = 0
        
        # Guardar resultados
        self.bias_results['calibration'] = {
            'metrics': bias_metrics,
            'bin_data': {
                'bin_centers': [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)],
                'bin_true': bin_true.tolist(),
                'bin_pred': bin_pred.tolist(),
                'bin_bias': bin_bias.tolist(),
                'bin_counts': bin_counts.tolist(),
                'valid_bins': valid_bins.tolist()
            }
        }
        
        # Visualizar resultados
        self._plot_calibration_bias(n_bins)
        
        return self.bias_results['calibration']
    
    def _plot_calibration_bias(self, n_bins: int):
        """
        Genera visualizaciones para el análisis de sesgo de calibración.
        
        Args:
            n_bins: Número de bins utilizados
        """
        if 'calibration' not in self.bias_results:
            return
        
        cal_data = self.bias_results['calibration']['bin_data']
        metrics = self.bias_results['calibration']['metrics']
        
        bin_centers = np.array(cal_data['bin_centers'])
        bin_true = np.array(cal_data['bin_true'])
        bin_pred = np.array(cal_data['bin_pred'])
        bin_bias = np.array(cal_data['bin_bias'])
        bin_counts = np.array(cal_data['bin_counts'])
        valid_bins = np.array(cal_data['valid_bins'])
        
        # Figura completa
        plt.figure(figsize=(14, 12))
        
        # 1. Gráfico de calibración
        plt.subplot(2, 1, 1)
        plt.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
        plt.scatter(bin_centers[valid_bins], bin_true[valid_bins], 
                   s=bin_counts[valid_bins]/np.max(bin_counts)*200, 
                   alpha=0.7, label='Observado')
        
        # Añadir línea de tendencia
        if np.sum(valid_bins) > 2:
            z = np.polyfit(bin_centers[valid_bins], bin_true[valid_bins], 1)
            p = np.poly1d(z)
            plt.plot(bin_centers, p(bin_centers), "r--", 
                     label=f'Tendencia (y={z[0]:.3f}x+{z[1]:.3f})')
        
        plt.xlabel('Probabilidad predicha')
        plt.ylabel('Frecuencia observada')
        plt.title(f'Análisis de Calibración (ECE={metrics["expected_calibration_error"]:.4f})')
        plt.legend()
        plt.grid(True)
        
        # 2. Gráfico de bias
        plt.subplot(2, 1, 2)
        colors = ['r' if b > 0 else 'g' for b in bin_bias[valid_bins]]
        plt.bar(bin_centers[valid_bins], bin_bias[valid_bins], color=colors, alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-')
        
        # Añadir textos descriptivos
        if metrics['pattern'] == 'overconfidence':
            bias_text = "Patrón: Sobreconfianza sistemática"
        elif metrics['pattern'] == 'underconfidence':
            bias_text = "Patrón: Subconfianza sistemática"
        else:
            bias_text = "Patrón: Sesgo mixto"
        
        plt.annotate(bias_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
        
        plt.xlabel('Probabilidad predicha')
        plt.ylabel('Sesgo (Predicha - Observada)')
        plt.title(f'Sesgo de Calibración (Media={metrics["mean_bias"]:.4f})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'calibration_bias.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_feature_bias(self, 
                            feature_name: str,
                            is_categorical: bool = None,
                            n_bins: int = 5) -> Dict[str, Any]:
        """
        Analiza sesgos en función de una característica específica.
        
        Args:
            feature_name: Nombre de la característica a analizar
            is_categorical: Si la característica es categórica (auto-detectada si es None)
            n_bins: Número de bins para características numéricas
            
        Returns:
            Diccionario con resultados del análisis de sesgo
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Ejecuta load_data primero.")
        
        if feature_name not in self.data.columns:
            raise ValueError(f"La característica '{feature_name}' no existe en los datos")
        
        # Detectar tipo de característica si no se especifica
        if is_categorical is None:
            is_categorical = pd.api.types.is_categorical_dtype(self.data[feature_name]) or \
                             pd.api.types.is_object_dtype(self.data[feature_name]) or \
                             self.data[feature_name].nunique() < 10
        
        # Analizar según tipo de característica
        if is_categorical:
            return self._analyze_categorical_feature(feature_name)
        else:
            return self._analyze_numerical_feature(feature_name, n_bins)
    
    def _analyze_categorical_feature(self, feature_name: str) -> Dict[str, Any]:
        """
        Analiza sesgos para una característica categórica.
        
        Args:
            feature_name: Nombre de la característica
            
        Returns:
            Diccionario con resultados del análisis
        """
        # Calcular métricas por categoría
        categories = []
        category_metrics = []
        category_counts = []
        
        for category, group in self.data.groupby(feature_name):
            if len(group) < 20:  # Mínimo de muestras
                continue
            
            y_true = group['y_true'].values
            y_pred = group['y_pred'].values
            y_prob = group['y_prob'].values
            
            # Calcular métricas básicas
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'mean_probability': np.mean(y_prob),
                'mean_outcome': np.mean(y_true),
                'count': len(group)
            }
            
            if len(np.unique(y_true)) > 1:  # Solo calcular AUC si hay ambas clases
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            
            # Calcular sesgo de probabilidad
            metrics['probability_bias'] = metrics['mean_probability'] - metrics['mean_outcome']
            
            categories.append(category)
            category_metrics.append(metrics)
            category_counts.append(len(group))
        
        if not categories:
            return {"message": f"Insuficientes datos para analizar la característica {feature_name}"}
        
        # Convertir a DataFrame para análisis
        results_df = pd.DataFrame(category_metrics, index=categories)
        
        # Calcular estadísticas de sesgo
        bias_values = results_df['probability_bias'].values
        
        # Prueba estadística para verificar si hay sesgo sistemático
        t_stat, p_value = stats.ttest_1samp(bias_values, 0)
        
        bias_stats = {
            'mean_bias': np.mean(bias_values),
            'bias_std': np.std(bias_values),
            'bias_range': (np.min(bias_values), np.max(bias_values)),
            't_statistic': t_stat,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05
        }
        
        # Identificar categorías con mayor sesgo
        abs_bias = np.abs(bias_values)
        worst_idx = np.argsort(abs_bias)[-3:]  # Top 3 peores
        worst_categories = []
        
        for idx in worst_idx[::-1]:  # Reverse para orden descendente
            cat = categories[idx]
            bias = bias_values[idx]
            count = category_counts[idx]
            
            worst_categories.append({
                'category': cat,
                'bias': bias,
                'overconfident' if bias > 0 else 'underconfident': True,
                'sample_count': count
            })
        
        # Guardar resultados
        self.bias_results[feature_name] = {
            'type': 'categorical',
            'statistics': bias_stats,
            'worst_categories': worst_categories,
            'all_categories': [{
                'category': cat,
                'metrics': metrics,
                'sample_count': count
            } for cat, metrics, count in zip(categories, category_metrics, category_counts)]
        }
        
        # Visualizar resultados
        self._plot_categorical_bias(feature_name, results_df, categories)
        
        return self.bias_results[feature_name]
    
    def _plot_categorical_bias(self, feature_name, results_df, categories):
        """
        Genera visualizaciones para el análisis de sesgo en características categóricas.
        
        Args:
            feature_name: Nombre de la característica
            results_df: DataFrame con resultados por categoría
            categories: Lista de categorías
        """
        plt.figure(figsize=(12, 10))
        
        # Ordenar categorías por sesgo absoluto
        sorted_idx = np.argsort(np.abs(results_df['probability_bias'].values))
        sorted_cats = [categories[i] for i in sorted_idx]
        
        # 1. Gráfico de sesgo por categoría
        plt.subplot(2, 1, 1)
        bars = plt.barh(sorted_cats, [results_df.loc[cat, 'probability_bias'] for cat in sorted_cats])
        
        # Colorear según dirección del sesgo
        for i, bar in enumerate(bars):
            bias = results_df.loc[sorted_cats[i], 'probability_bias']
            bar.set_color('r' if bias > 0 else 'g')
        
        plt.axvline(x=0, color='k', linestyle='-')
        plt.xlabel('Sesgo de Probabilidad (Predicha - Observada)')
        plt.ylabel(feature_name)
        plt.title(f'Sesgo por {feature_name}')
        plt.grid(True)
        
        # 2. Gráfico de probabilidad predicha vs observada
        plt.subplot(2, 1, 2)
        plt.scatter(
            [results_df.loc[cat, 'mean_outcome'] for cat in categories],
            [results_df.loc[cat, 'mean_probability'] for cat in categories],
            s=[results_df.loc[cat, 'count']/10 for cat in categories],
            alpha=0.7
        )
        
        # Añadir línea de referencia
        plt.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
        
        # Añadir etiquetas para los puntos
        for cat in categories:
            plt.annotate(
                str(cat),
                (results_df.loc[cat, 'mean_outcome'], results_df.loc[cat, 'mean_probability']),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.xlabel('Tasa de Éxito Observada')
        plt.ylabel('Probabilidad Media Predicha')
        plt.title(f'Calibración por {feature_name}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'bias_{feature_name}_categorical.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_numerical_feature(self, feature_name: str, n_bins: int = 5) -> Dict[str, Any]:
        """
        Analiza sesgos para una característica numérica.
        
        Args:
            feature_name: Nombre de la característica
            n_bins: Número de bins para discretizar
            
        Returns:
            Diccionario con resultados del análisis
        """
        # Verificar que la característica es numérica
        if not pd.api.types.is_numeric_dtype(self.data[feature_name]):
            raise ValueError(f"La característica '{feature_name}' no es numérica")
        
        # Crear bins para la característica
        try:
            bins = pd.qcut(self.data[feature_name], n_bins, duplicates='drop')
        except:
            # Si falla qcut (e.g., demasiados valores duplicados), usar cut
            bins = pd.cut(self.data[feature_name], n_bins)
        
        self.data['temp_bin'] = bins
        
        # Calcular métricas por bin
        bin_metrics = []
        bin_labels = []
        bin_counts = []
        bin_values = []
        
        for bin_label, group in self.data.groupby('temp_bin'):
            if len(group) < 20:  # Mínimo de muestras
                continue
                
            y_true = group['y_true'].values
            y_pred = group['y_pred'].values
            y_prob = group['y_prob'].values
            
            # Valor medio de la característica en este bin
            bin_value = group[feature_name].mean()
            
            # Calcular métricas
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'mean_probability': np.mean(y_prob),
                'mean_outcome': np.mean(y_true),
                'count': len(group)
            }
            
            if len(np.unique(y_true)) > 1:  # Solo calcular AUC si hay ambas clases
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            
            # Calcular sesgo de probabilidad
            metrics['probability_bias'] = metrics['mean_probability'] - metrics['mean_outcome']
            
            bin_metrics.append(metrics)
            bin_labels.append(str(bin_label))
            bin_counts.append(len(group))
            bin_values.append(bin_value)
        
        # Limpiar columna temporal
        self.data.drop('temp_bin', axis=1, inplace=True)
        
        if not bin_labels:
            return {"message": f"Insuficientes datos para analizar la característica {feature_name}"}
        
        # Convertir a DataFrame para análisis
        results_df = pd.DataFrame(bin_metrics, index=bin_labels)
        
        # Detectar tendencia en el sesgo
        X = np.array(bin_values).reshape(-1, 1)
        y = results_df['probability_bias'].values
        
        # Ajustar regresión lineal para detectar tendencia
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calcular coeficiente de correlación
        correlation = np.corrcoef(bin_values, y)[0, 1] if len(bin_values) > 1 else 0
        
        # Calcular estadísticas de sesgo
        bias_stats = {
            'mean_bias': np.mean(y),
            'bias_std': np.std(y),
            'bias_range': (np.min(y), np.max(y)),
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'correlation': correlation,
            'trend_significance': 'high' if abs(correlation) > 0.7 else ('medium' if abs(correlation) > 0.4 else 'low')
        }
        
        # Determinar tipo de sesgo
        if abs(correlation) > 0.4:
            if correlation > 0:
                bias_stats['trend_pattern'] = 'increasing_bias'
                bias_stats['interpretation'] = f'El sesgo aumenta con valores mayores de {feature_name}'
            else:
                bias_stats['trend_pattern'] = 'decreasing_bias'
                bias_stats['interpretation'] = f'El sesgo disminuye con valores mayores de {feature_name}'
        else:
            bias_stats['trend_pattern'] = 'no_clear_trend'
            bias_stats['interpretation'] = f'No hay una tendencia clara del sesgo con respecto a {feature_name}'
        
        # Guardar resultados
        self.bias_results[feature_name] = {
            'type': 'numerical',
            'statistics': bias_stats,
            'bins': [{
                'bin_label': label,
                'bin_value': value,
                'metrics': metrics,
                'sample_count': count
            } for label, value, metrics, count in zip(bin_labels, bin_values, bin_metrics, bin_counts)]
        }
        
        # Visualizar resultados
        self._plot_numerical_bias(feature_name, results_df, bin_values, bin_labels)
        
        return self.bias_results[feature_name]
    
    def _plot_numerical_bias(self, feature_name, results_df, bin_values, bin_labels):
        """
        Genera visualizaciones para el análisis de sesgo en características numéricas.
        
        Args:
            feature_name: Nombre de la característica
            results_df: DataFrame con resultados por bin
            bin_values: Valores representativos de cada bin
            bin_labels: Etiquetas de los bins
        """
        bias_stats = self.bias_results[feature_name]['statistics']
        
        plt.figure(figsize=(12, 10))
        
        # 1. Gráfico de sesgo vs valor de característica
        plt.subplot(2, 1, 1)
        plt.scatter(bin_values, results_df['probability_bias'].values, alpha=0.7, s=100)
        
        # Añadir línea de tendencia
        x_range = np.linspace(min(bin_values), max(bin_values), 100)
        y_pred = bias_stats['intercept'] + bias_stats['slope'] * x_range
        plt.plot(x_range, y_pred, 'r--', 
               label=f"Tendencia: y={bias_stats['slope']:.6f}x+{bias_stats['intercept']:.4f}")
        
        plt.axhline(y=0, color='k', linestyle='-')
        plt.xlabel(feature_name)
        plt.ylabel('Sesgo de Probabilidad')
        plt.title(f'Sesgo vs {feature_name} (Correlación: {bias_stats["correlation"]:.4f})')
        plt.legend()
        plt.grid(True)
        
        # 2. Gráfico de probabilidad predicha vs observada
        plt.subplot(2, 1, 2)
        plt.scatter(
            results_df['mean_outcome'].values,
            results_df['mean_probability'].values,
            s=results_df['count'].values/5,
            alpha=0.7
        )
        
        # Añadir línea de referencia
        plt.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
        
        # Añadir etiquetas para los puntos
        for i, label in enumerate(bin_labels):
            plt.annotate(
                label,
                (results_df['mean_outcome'].values[i], results_df['mean_probability'].values[i]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.xlabel('Tasa de Éxito Observada')
        plt.ylabel('Probabilidad Media Predicha')
        plt.title(f'Calibración por {feature_name}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'bias_{feature_name}_numerical.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_joint_biases(self, 
                            feature1: str, 
                            feature2: str,
                            is_categorical1: bool = None,
                            is_categorical2: bool = None) -> Dict[str, Any]:
        """
        Analiza sesgos conjuntos para dos características.
        
        Args:
            feature1: Nombre de la primera característica
            feature2: Nombre de la segunda característica
            is_categorical1: Si la primera característica es categórica
            is_categorical2: Si la segunda característica es categórica
            
        Returns:
            Diccionario con resultados del análisis
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Ejecuta load_data primero.")
        
        if feature1 not in self.data.columns or feature2 not in self.data.columns:
            raise ValueError(f"Las características {feature1} y/o {feature2} no existen en los datos")
        
        # Detectar tipos de características si no se especifican
        if is_categorical1 is None:
            is_categorical1 = pd.api.types.is_categorical_dtype(self.data[feature1]) or \
                             pd.api.types.is_object_dtype(self.data[feature1]) or \
                             self.data[feature1].nunique() < 10
        
        if is_categorical2 is None:
            is_categorical2 = pd.api.types.is_categorical_dtype(self.data[feature2]) or \
                             pd.api.types.is_object_dtype(self.data[feature2]) or \
                             self.data[feature2].nunique() < 10
        
        # Preparar datos según tipos
        if is_categorical1:
            # Usar valores originales para categóricas
            groups1 = self.data[feature1]
        else:
            # Discretizar para numéricas
            groups1 = pd.qcut(self.data[feature1], 3, duplicates='drop')
        
        if is_categorical2:
            # Usar valores originales para categóricas
            groups2 = self.data[feature2]
        else:
            # Discretizar para numéricas
            groups2 = pd.qcut(self.data[feature2], 3, duplicates='drop')
        
        # Crear grupos conjuntos
        self.data['temp_group1'] = groups1
        self.data['temp_group2'] = groups2
        
        # Calcular métricas por combinación de grupos
        joint_metrics = []
        
        for (group1, group2), group_data in self.data.groupby(['temp_group1', 'temp_group2']):
            if len(group_data) < 20:  # Mínimo de muestras
                continue
                
            y_true = group_data['y_true'].values
            y_pred = group_data['y_pred'].values
            y_prob = group_data['y_prob'].values
            
            # Calcular métricas
            metrics = {
                'group1': str(group1),
                'group2': str(group2),
                'accuracy': accuracy_score(y_true, y_pred),
                'mean_probability': np.mean(y_prob),
                'mean_outcome': np.mean(y_true),
                'probability_bias': np.mean(y_prob) - np.mean(y_true),
                'sample_count': len(group_data)
            }
            
            if len(np.unique(y_true)) > 1:  # Solo calcular AUC si hay ambas clases
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            
            joint_metrics.append(metrics)
        
        # Limpiar columnas temporales
        self.data.drop(['temp_group1', 'temp_group2'], axis=1, inplace=True)
        
        if not joint_metrics:
            return {"message": f"Insuficientes datos para analizar el sesgo conjunto de {feature1} y {feature2}"}
        
        # Convertir a DataFrame
        joint_df = pd.DataFrame(joint_metrics)
        
        # Analizar interacción de sesgos (ANOVA)
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            
            # Crear modelo ANOVA
            model = ols('probability_bias ~ C(group1) + C(group2) + C(group1):C(group2)', data=joint_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Extraer p-valores
            p_main1 = anova_table.loc['C(group1)', 'PR(>F)']
            p_main2 = anova_table.loc['C(group2)', 'PR(>F)']
            p_interaction = anova_table.loc['C(group1):C(group2)', 'PR(>F)']
            
            # Interpretar resultados
            interaction_stats = {
                'feature1_effect_pvalue': p_main1,
                'feature2_effect_pvalue': p_main2,
                'interaction_pvalue': p_interaction,
                'feature1_significant': p_main1 < 0.05,
                'feature2_significant': p_main2 < 0.05,
                'interaction_significant': p_interaction < 0.05
            }
        except:
            # Si falla ANOVA, usar análisis más simple
            group1_effect = joint_df.groupby('group1')['probability_bias'].mean().std()
            group2_effect = joint_df.groupby('group2')['probability_bias'].mean().std()
            
            interaction_stats = {
                'feature1_effect_strength': group1_effect,
                'feature2_effect_strength': group2_effect,
                'feature1_stronger': group1_effect > group2_effect,
                'feature2_stronger': group2_effect > group1_effect
            }
        
        # Guardar resultados
        joint_key = f"{feature1}_x_{feature2}"
        self.bias_results[joint_key] = {
            'features': [feature1, feature2],
            'is_categorical': [is_categorical1, is_categorical2],
            'statistics': interaction_stats,
            'joint_groups': joint_metrics
        }
        
        # Visualizar resultados
        self._plot_joint_bias(feature1, feature2, joint_df)
        
        return self.bias_results[joint_key]
    
    def _plot_joint_bias(self, feature1, feature2, joint_df):
        """
        Genera visualizaciones para el análisis de sesgo conjunto.
        
        Args:
            feature1: Nombre de la primera característica
            feature2: Nombre de la segunda característica
            joint_df: DataFrame con métricas conjuntas
        """
        plt.figure(figsize=(12, 10))
        
        # Crear matriz pivot para heatmap
        pivot_df = joint_df.pivot_table(
            index='group1', 
            columns='group2', 
            values='probability_bias',
            aggfunc='mean'
        )
        
        # Heatmap de sesgo
        plt.subplot(2, 1, 1)
        sns.heatmap(pivot_df, annot=True, cmap='RdBu_r', center=0, fmt='.3f',
                   linewidths=.5, cbar_kws={'label': 'Sesgo de Probabilidad'})
        plt.title(f'Mapa de Calor de Sesgo: {feature1} vs {feature2}')
        plt.ylabel(feature1)
        plt.xlabel(feature2)
        
        # Scatter plot de probabilidad predicha vs observada
        plt.subplot(2, 1, 2)
        scatter = plt.scatter(
            joint_df['mean_outcome'], 
            joint_df['mean_probability'],
            s=joint_df['sample_count']/5,
            c=joint_df['probability_bias'],
            cmap='RdBu_r',
            alpha=0.7
        )
        
        # Añadir línea de referencia
        plt.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
        
        # Añadir barra de color
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sesgo de Probabilidad')
        
        # Añadir etiquetas para puntos relevantes
        for _, row in joint_df.nlargest(3, 'sample_count').iterrows():
            plt.annotate(
                f"{row['group1']}, {row['group2']}",
                (row['mean_outcome'], row['mean_probability']),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.xlabel('Tasa de Éxito Observada')
        plt.ylabel('Probabilidad Media Predicha')
        plt.title(f'Calibración por Grupos: {feature1} y {feature2}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'bias_joint_{feature1}_{feature2}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_all_features(self, 
                           categorical_features: List[str] = None,
                           numerical_features: List[str] = None,
                           min_samples_per_group: int = 20) -> Dict[str, Any]:
        """
        Analiza sesgos para todas las características especificadas.
        
        Args:
            categorical_features: Lista de características categóricas (si es None, auto-detecta)
            numerical_features: Lista de características numéricas (si es None, auto-detecta)
            min_samples_per_group: Número mínimo de muestras por grupo
            
        Returns:
            Diccionario con resumen de resultados
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Ejecuta load_data primero.")
        
        # Auto-detectar características si no se especifican
        if categorical_features is None and numerical_features is None:
            # Excluir columnas de resultados y otras no relevantes
            exclude_cols = ['y_true', 'y_pred', 'y_prob', 'match_id', 'date', 'id', 'index']
            potential_features = [c for c in self.data.columns if c not in exclude_cols]
            
            categorical_features = []
            numerical_features = []
            
            for col in potential_features:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    if self.data[col].nunique() < 10:
                        categorical_features.append(col)
                    else:
                        numerical_features.append(col)
                else:
                    categorical_features.append(col)
        
        # Inicializar resultados resumidos
        summary = {
            'calibration': self.analyze_probability_calibration(),
            'categorical_features': {},
            'numerical_features': {},
            'significant_biases': []
        }
        
        # Analizar características categóricas
        if categorical_features:
            for feature in categorical_features:
                try:
                    result = self.analyze_feature_bias(feature, is_categorical=True)
                    summary['categorical_features'][feature] = result
                    
                    # Verificar si hay sesgo significativo
                    if 'statistics' in result and result['statistics'].get('statistically_significant', False):
                        summary['significant_biases'].append({
                            'feature': feature,
                            'type': 'categorical',
                            'mean_bias': result['statistics']['mean_bias'],
                            'p_value': result['statistics'].get('p_value', None)
                        })
                except Exception as e:
                    print(f"Error al analizar la característica categórica {feature}: {str(e)}")
        
        # Analizar características numéricas
        if numerical_features:
            for feature in numerical_features:
                try:
                    result = self.analyze_feature_bias(feature, is_categorical=False)
                    summary['numerical_features'][feature] = result
                    
                    # Verificar si hay tendencia de sesgo significativa
                    if 'statistics' in result and result['statistics'].get('trend_significance') in ['medium', 'high']:
                        summary['significant_biases'].append({
                            'feature': feature,
                            'type': 'numerical',
                            'correlation': result['statistics']['correlation'],
                            'trend_pattern': result['statistics']['trend_pattern']
                        })
                except Exception as e:
                    print(f"Error al analizar la característica numérica {feature}: {str(e)}")
        
        # Ordenar sesgos significativos por magnitud
        summary['significant_biases'].sort(key=lambda x: abs(x.get('mean_bias', 0)) if 'mean_bias' in x else abs(x.get('correlation', 0)), reverse=True)
        
        # Generar informe completo
        report_path = self.generate_bias_report(summary)
        summary['report_path'] = report_path
        
        return summary
    
    def analyze_prediction_bias(self) -> Dict[str, Any]:
        """
        Analiza sesgos entre predicciones positivas y negativas.
        
        Returns:
            Diccionario con resultados del análisis
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Ejecuta load_data primero.")
        
        # Separar por predicción
        pos_pred = self.data[self.data['y_pred'] == 1]
        neg_pred = self.data[self.data['y_pred'] == 0]
        
        # Calcular métricas para cada grupo
        pos_metrics = {
            'count': len(pos_pred),
            'accuracy': accuracy_score(pos_pred['y_true'], pos_pred['y_pred']),
            'mean_probability': pos_pred['y_prob'].mean(),
            'mean_outcome': pos_pred['y_true'].mean(),
            'probability_bias': pos_pred['y_prob'].mean() - pos_pred['y_true'].mean()
        }
        
        neg_metrics = {
            'count': len(neg_pred),
            'accuracy': accuracy_score(neg_pred['y_true'], neg_pred['y_pred']),
            'mean_probability': neg_pred['y_prob'].mean(),
            'mean_outcome': neg_pred['y_true'].mean(),
            'probability_bias': neg_pred['y_prob'].mean() - neg_pred['y_true'].mean()
        }
        
        # Calcular estadísticas por rango de probabilidad
        prob_ranges = []
        
        for lower, upper in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
            range_data = self.data[(self.data['y_prob'] >= lower) & (self.data['y_prob'] < upper)]
            
            if len(range_data) < 20:  # Mínimo de muestras
                continue
                
            range_metrics = {
                'range': f"{lower:.1f}-{upper:.1f}",
                'count': len(range_data),
                'mean_probability': range_data['y_prob'].mean(),
                'mean_outcome': range_data['y_true'].mean(),
                'probability_bias': range_data['y_prob'].mean() - range_data['y_true'].mean()
            }
            
            prob_ranges.append(range_metrics)
        
        # Guardar resultados
        prediction_bias = {
            'positive_predictions': pos_metrics,
            'negative_predictions': neg_metrics,
            'probability_ranges': prob_ranges,
            'bias_difference': pos_metrics['probability_bias'] - neg_metrics['probability_bias'],
            'asymmetric_bias': abs(pos_metrics['probability_bias'] - neg_metrics['probability_bias']) > 0.05
        }
        
        self.bias_results['prediction_bias'] = prediction_bias
        
        # Visualizar resultados
        self._plot_prediction_bias(prediction_bias, prob_ranges)
        
        return prediction_bias
    
    def _plot_prediction_bias(self, prediction_bias, prob_ranges):
        """
        Genera visualizaciones para el análisis de sesgo de predicción.
        
        Args:
            prediction_bias: Resultados del análisis de sesgo de predicción
            prob_ranges: Lista de métricas por rango de probabilidad
        """
        plt.figure(figsize=(12, 10))
        
        # 1. Comparación de sesgo entre predicciones positivas y negativas
        plt.subplot(2, 1, 1)
        classes = ['Negativas (0)', 'Positivas (1)']
        biases = [prediction_bias['negative_predictions']['probability_bias'], 
                 prediction_bias['positive_predictions']['probability_bias']]
        
        bars = plt.bar(classes, biases)
        
        # Colorear según dirección del sesgo
        for bar, bias in zip(bars, biases):
            bar.set_color('r' if bias > 0 else 'g')
        
        plt.axhline(y=0, color='k', linestyle='-')
        plt.ylabel('Sesgo de Probabilidad')
        plt.title('Comparación de Sesgo entre Predicciones Positivas y Negativas')
        plt.grid(True, axis='y')
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.01 if height >= 0 else -0.03),
                   f'{height:.4f}',
                   ha='center', va='bottom' if height >= 0 else 'top')
        
        # 2. Sesgo por rango de probabilidad
        plt.subplot(2, 1, 2)
        
        ranges = [r['range'] for r in prob_ranges]
        range_biases = [r['probability_bias'] for r in prob_ranges]
        counts = [r['count'] for r in prob_ranges]
        
        # Normalizar tamaños para el área de las barras
        max_count = max(counts)
        normalized_widths = [0.5 * (count / max_count) + 0.5 for count in counts]
        
        bars = plt.bar(ranges, range_biases, width=normalized_widths)
        
        # Colorear según dirección del sesgo
        for bar, bias in zip(bars, range_biases):
            bar.set_color('r' if bias > 0 else 'g')
        
        plt.axhline(y=0, color='k', linestyle='-')
        plt.xlabel('Rango de Probabilidad Predicha')
        plt.ylabel('Sesgo de Probabilidad')
        plt.title('Sesgo por Rango de Probabilidad')
        plt.grid(True, axis='y')
        
        # Añadir valores sobre las barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.01 if height >= 0 else -0.03),
                   f'{height:.4f}\n(n={counts[i]})',
                   ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'prediction_bias.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_bias_report(self, summary=None) -> str:
        """
        Genera un informe completo con todos los análisis de sesgo.
        
        Args:
            summary: Resultados resumidos de analyze_all_features (opcional)
            
        Returns:
            Ruta al archivo de informe generado
        """
        if not self.bias_results and summary is None:
            raise ValueError("No hay resultados de análisis de sesgo disponibles")
        
        # Usar resumen proporcionado o resultados actuales
        if summary is None:
            summary = {
                'calibration': self.bias_results.get('calibration', {}),
                'categorical_features': {k: v for k, v in self.bias_results.items() 
                                       if 'type' in v and v['type'] == 'categorical'},
                'numerical_features': {k: v for k, v in self.bias_results.items() 
                                     if 'type' in v and v['type'] == 'numerical'},
                'significant_biases': []
            }
        
        # Crear informe
        report_lines = [
            "# Informe de Análisis de Sesgos en Predicciones de Tenis",
            "",
            f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. Resumen Ejecutivo",
            ""
        ]
        
        # Calibración general
        if 'calibration' in summary and 'metrics' in summary['calibration']:
            cal_metrics = summary['calibration']['metrics']
            
            report_lines.append("### 1.1. Calibración General")
            report_lines.append("")
            report_lines.append(f"- Error de Calibración Esperado (ECE): {cal_metrics.get('expected_calibration_error', 0):.4f}")
            report_lines.append(f"- Error de Calibración Máximo (MCE): {cal_metrics.get('maximum_calibration_error', 0):.4f}")
            report_lines.append(f"- Sesgo Medio: {cal_metrics.get('mean_bias', 0):.4f}")
            
            if 'pattern' in cal_metrics:
                if cal_metrics['pattern'] == 'overconfidence':
                    report_lines.append("- **Patrón Detectado**: Sobreconfianza sistemática")
                elif cal_metrics['pattern'] == 'underconfidence':
                    report_lines.append("- **Patrón Detectado**: Subconfianza sistemática")
                else:
                    report_lines.append("- **Patrón Detectado**: Sesgo mixto sin tendencia clara")
            
            report_lines.append("")
        
        # Sesgos significativos
        report_lines.append("### 1.2. Sesgos Significativos Detectados")
        report_lines.append("")
        
        if summary['significant_biases']:
            for i, bias in enumerate(summary['significant_biases'][:5]):  # Top 5
                feature = bias['feature']
                
                if bias['type'] == 'categorical':
                    report_lines.append(f"{i+1}. **{feature}** (categórica): Sesgo medio de {bias.get('mean_bias', 0):.4f}")
                    if 'p_value' in bias:
                        report_lines.append(f"   - Significancia estadística: p-valor = {bias['p_value']:.4f}")
                else:
                    report_lines.append(f"{i+1}. **{feature}** (numérica): Correlación de sesgo = {bias.get('correlation', 0):.4f}")
                    if 'trend_pattern' in bias:
                        pattern = bias['trend_pattern']
                        if pattern == 'increasing_bias':
                            report_lines.append(f"   - El sesgo aumenta con valores mayores de {feature}")
                        elif pattern == 'decreasing_bias':
                            report_lines.append(f"   - El sesgo disminuye con valores mayores de {feature}")
        else:
            report_lines.append("No se detectaron sesgos significativos en las características analizadas.")
        
        report_lines.append("")
        
        # Análisis detallado de calibración
        report_lines.extend([
            "## 2. Análisis Detallado de Calibración",
            ""
        ])
        
        if 'calibration' in summary and 'metrics' in summary['calibration']:
            cal_metrics = summary['calibration']['metrics']
            
            report_lines.append("| Métrica | Valor |")
            report_lines.append("| ------- | ----- |")
            report_lines.append(f"| ECE | {cal_metrics.get('expected_calibration_error', 0):.4f} |")
            report_lines.append(f"| MCE | {cal_metrics.get('maximum_calibration_error', 0):.4f} |")
            report_lines.append(f"| Sesgo Medio | {cal_metrics.get('mean_bias', 0):.4f} |")
            report_lines.append(f"| Desviación Estándar del Sesgo | {cal_metrics.get('bias_std', 0):.4f} |")
            report_lines.append(f"| Rango de Sesgo | {cal_metrics.get('bias_range', (0,0))[0]:.4f} a {cal_metrics.get('bias_range', (0,0))[1]:.4f} |")
            
            if 'pattern' in cal_metrics and 'pattern_strength' in cal_metrics:
                pattern = cal_metrics['pattern']
                strength = cal_metrics['pattern_strength']

                report_lines.append(f"| Patrón de Sesgo | {pattern} |")
                report_lines.append(f"| Fuerza del Patrón | {strength:.2f} |")
        
        # Análisis de características categóricas
        cat_features = summary.get('categorical_features', {})
        if cat_features:
            report_lines.extend([
                "",
                "## 3. Análisis de Características Categóricas",
                ""
            ])
            
            for i, (feature, results) in enumerate(cat_features.items()):
                if 'statistics' not in results:
                    continue
                    
                stats = results['statistics']
                
                report_lines.append(f"### 3.{i+1}. {feature}")
                report_lines.append("")
                report_lines.append(f"- Sesgo medio: {stats.get('mean_bias', 0):.4f}")
                report_lines.append(f"- Desviación estándar: {stats.get('bias_std', 0):.4f}")
                
                if 'p_value' in stats:
                    significance = "Sí" if stats.get('statistically_significant', False) else "No"
                    report_lines.append(f"- p-valor: {stats.get('p_value', 1):.4f}")
                    report_lines.append(f"- Estadísticamente significativo: {significance}")
                
                report_lines.append("")
                
                # Añadir información sobre las peores categorías
                if 'worst_categories' in results:
                    report_lines.append("#### Categorías con mayor sesgo:")
                    report_lines.append("")
                    
                    for cat in results['worst_categories']:
                        bias_type = "sobreconfianza" if cat.get('bias', 0) > 0 else "subconfianza"
                        report_lines.append(f"- **{cat['category']}**: {cat['bias']:.4f} ({bias_type}, n={cat['sample_count']})")
                
                report_lines.append("")
        
        # Análisis de características numéricas
        num_features = summary.get('numerical_features', {})
        if num_features:
            report_lines.extend([
                "",
                "## 4. Análisis de Características Numéricas",
                ""
            ])
            
            for i, (feature, results) in enumerate(num_features.items()):
                if 'statistics' not in results:
                    continue
                    
                stats = results['statistics']
                
                report_lines.append(f"### 4.{i+1}. {feature}")
                report_lines.append("")
                report_lines.append(f"- Pendiente de tendencia: {stats.get('slope', 0):.6f}")
                report_lines.append(f"- Correlación: {stats.get('correlation', 0):.4f}")
                report_lines.append(f"- Significancia de tendencia: {stats.get('trend_significance', 'desconocida')}")
                
                if 'interpretation' in stats:
                    report_lines.append(f"- Interpretación: {stats['interpretation']}")
                
                report_lines.append("")
        
        # Análisis de sesgo de predicción
        if 'prediction_bias' in self.bias_results:
            pred_bias = self.bias_results['prediction_bias']
            
            report_lines.extend([
                "",
                "## 5. Análisis de Sesgo por Tipo de Predicción",
                ""
            ])
            
            pos_metrics = pred_bias['positive_predictions']
            neg_metrics = pred_bias['negative_predictions']
            
            report_lines.append("### 5.1. Comparación de Predicciones Positivas vs Negativas")
            report_lines.append("")
            report_lines.append("| Métrica | Predicciones Positivas | Predicciones Negativas |")
            report_lines.append("| ------- | ---------------------- | ---------------------- |")
            report_lines.append(f"| Cantidad | {pos_metrics['count']} | {neg_metrics['count']} |")
            report_lines.append(f"| Accuracy | {pos_metrics['accuracy']:.4f} | {neg_metrics['accuracy']:.4f} |")
            report_lines.append(f"| Probabilidad Media | {pos_metrics['mean_probability']:.4f} | {neg_metrics['mean_probability']:.4f} |")
            report_lines.append(f"| Tasa Real de Positivos | {pos_metrics['mean_outcome']:.4f} | {neg_metrics['mean_outcome']:.4f} |")
            report_lines.append(f"| Sesgo | {pos_metrics['probability_bias']:.4f} | {neg_metrics['probability_bias']:.4f} |")
            
            report_lines.append("")
            report_lines.append(f"- Diferencia de sesgo: {pred_bias['bias_difference']:.4f}")
            
            asymmetric = "Sí" if pred_bias['asymmetric_bias'] else "No"
            report_lines.append(f"- Sesgo asimétrico: {asymmetric}")
            
            if pred_bias['probability_ranges']:
                report_lines.append("")
                report_lines.append("### 5.2. Sesgo por Rango de Probabilidad")
                report_lines.append("")
                report_lines.append("| Rango | Cantidad | Probabilidad Media | Tasa Real | Sesgo |")
                report_lines.append("| ----- | -------- | ------------------ | --------- | ----- |")
                
                for r in pred_bias['probability_ranges']:
                    report_lines.append(f"| {r['range']} | {r['count']} | {r['mean_probability']:.4f} | {r['mean_outcome']:.4f} | {r['probability_bias']:.4f} |")
        
        # Visualizaciones
        report_lines.extend([
            "",
            "## 6. Visualizaciones",
            "",
            "Se han generado las siguientes visualizaciones para este análisis:",
            ""
        ])
        
        for file in os.listdir(self.output_dir):
            if file.endswith('.png'):
                report_lines.append(f"- ![{file}]({file})")
        
        # Recomendaciones
        report_lines.extend([
            "",
            "## 7. Conclusiones y Recomendaciones",
            ""
        ])
        
        # Añadir conclusiones y recomendaciones basadas en los análisis
        cal_pattern = summary.get('calibration', {}).get('metrics', {}).get('pattern', '')
        
        # Recomendaciones basadas en calibración
        if cal_pattern == 'overconfidence':
            report_lines.append("1. **Mejorar calibración**: El modelo muestra una tendencia significativa a la sobreconfianza. Se recomienda:")
            report_lines.append("   - Aplicar técnicas de calibración como Platt Scaling o Isotonic Regression")
            report_lines.append("   - Ajustar las probabilidades con un factor de corrección basado en la curva de calibración")
        elif cal_pattern == 'underconfidence':
            report_lines.append("1. **Mejorar calibración**: El modelo muestra una tendencia significativa a la subconfianza. Se recomienda:")
            report_lines.append("   - Aplicar técnicas de calibración como Platt Scaling o Isotonic Regression")
            report_lines.append("   - Ajustar las probabilidades para ser más cercanas a los extremos")
        
        # Recomendaciones basadas en sesgos significativos
        if summary['significant_biases']:
            report_lines.append("2. **Corregir sesgos específicos**: Se han detectado sesgos significativos en las siguientes características:")
            
            for i, bias in enumerate(summary['significant_biases'][:3]):  # Top 3
                feature = bias['feature']
                if bias['type'] == 'categorical':
                    report_lines.append(f"   - **{feature}**: Considerar entrenar modelos específicos para diferentes categorías o incluir interacciones en el modelo")
                else:
                    report_lines.append(f"   - **{feature}**: Revisar la relación entre esta característica y la probabilidad predicha, posiblemente añadiendo términos no lineales")
        
        # Recomendaciones basadas en sesgo de predicción
        if 'prediction_bias' in self.bias_results and self.bias_results['prediction_bias']['asymmetric_bias']:
            report_lines.append("3. **Corregir sesgo asimétrico**: El modelo muestra un sesgo asimétrico entre predicciones positivas y negativas. Se recomienda:")
            report_lines.append("   - Ajustar el umbral de decisión")
            report_lines.append("   - Aplicar diferentes factores de corrección para predicciones positivas y negativas")
        
        # Guardar el informe
        report_path = os.path.join(self.output_dir, 'bias_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return report_path


if __name__ == "__main__":
    # Ejemplo de uso
    detector = BiasDetector("example_bias_analysis")
    
    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 1000
    
    # Simular características que influyen en el sesgo
    surface = np.random.choice(['hard', 'clay', 'grass', 'carpet'], n_samples)
    player_rank = np.random.randint(1, 500, n_samples)
    tournament_level = np.random.choice(['grand_slam', 'atp1000', 'atp500', 'atp250', 'challenger'], n_samples)
    
    # Simular sesgo por superficie
    surface_bias = {
        'hard': 0.05,
        'clay': -0.03,
        'grass': 0.08,
        'carpet': 0
    }
    
    # Simular sesgo por ranking (más sobreconfianza para jugadores de alto ranking)
    rank_factor = (500 - player_rank) / 500  # 0-1 escala, mayor para mejores rankings
    rank_bias = 0.1 * rank_factor  # Más sobreconfianza para mejores jugadores
    
    # Probabilidad base "real"
    base_prob = np.random.beta(5, 5, n_samples)  # Centrada en 0.5
    
    # Añadir sesgo a las predicciones
    surface_bias_values = np.array([surface_bias[s] for s in surface])
    
    # La probabilidad "predicha" tiene sesgo
    biased_prob = np.clip(base_prob + surface_bias_values + rank_bias, 0.01, 0.99)
    
    # Resultados reales y predicciones
    y_true = (np.random.random(n_samples) < base_prob).astype(int)
    y_pred = (biased_prob > 0.5).astype(int)
    
    # Crear DataFrame
    data = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': biased_prob,
        'surface': surface,
        'player_rank': player_rank,
        'tournament_level': tournament_level,
        'elo_difference': np.random.normal(0, 100, n_samples)
    })
    
    # Cargar datos
    detector.load_data(data)
    
    # Analizar calibración general
    calibration_results = detector.analyze_probability_calibration()
    print("Análisis de calibración completado.")
    
    # Analizar sesgos por características
    detector.analyze_feature_bias('surface', is_categorical=True)
    detector.analyze_feature_bias('player_rank', is_categorical=False)
    detector.analyze_feature_bias('tournament_level', is_categorical=True)
    print("Análisis de características completado.")
    
    # Analizar sesgo de predicción
    detector.analyze_prediction_bias()
    print("Análisis de sesgo de predicción completado.")
    
    # Generar informe completo
    report_path = detector.generate_bias_report()
    print(f"Informe de análisis de sesgo generado en: {report_path}")