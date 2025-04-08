"""
results_analyzer.py

Este módulo implementa funcionalidades para analizar en detalle los resultados
de las predicciones de partidos de tenis, segmentando por diversas categorías
y generando insights sobre el rendimiento del modelo.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Union, Any

from .metrics import TennisMetrics


class ResultsAnalyzer:
    """
    Clase para analizar detalladamente los resultados de predicciones de tenis.
    Permite segmentar análisis por superficie, torneo, tipo de jugador y otros factores.
    También identifica patrones en predicciones incorrectas y calcula retornos de inversión.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Inicializa el analizador de resultados.
        
        Args:
            output_dir: Directorio donde se guardarán los gráficos y resultados
        """
        self.metrics = TennisMetrics()
        self.results_df = None
        self.segmented_metrics = {}
        
        # Configurar directorio de salida
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"results_analysis_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
    
    def load_results(self, 
                    results_df: pd.DataFrame = None, 
                    csv_path: str = None) -> pd.DataFrame:
        """
        Carga los resultados de predicciones desde un DataFrame o archivo CSV.
        
        Args:
            results_df: DataFrame con resultados (opcional)
            csv_path: Ruta al archivo CSV con resultados (opcional)
            
        Returns:
            DataFrame con los resultados cargados
            
        Nota: El DataFrame debe contener al menos las columnas:
            - y_true o winner_actual (1 para victoria del jugador 1, 0 para derrota)
            - y_pred o winner_predicted (1 para victoria predicha del jugador 1, 0 para derrota)
            - y_prob o probability (probabilidad predicha para la victoria del jugador 1)
        """
        if results_df is not None:
            self.results_df = results_df.copy()
        elif csv_path:
            self.results_df = pd.read_csv(csv_path)
        else:
            raise ValueError("Debe proporcionar un DataFrame o ruta CSV")
        
        # Normalizar nombres de columnas
        column_mapping = {
            'winner_actual': 'y_true',
            'winner_predicted': 'y_pred',
            'probability': 'y_prob',
            'probability_player1': 'y_prob'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in self.results_df.columns and new_name not in self.results_df.columns:
                self.results_df[new_name] = self.results_df[old_name]
        
        # Verificar columnas requeridas
        required_columns = ['y_true', 'y_pred', 'y_prob']
        missing_columns = [col for col in required_columns if col not in self.results_df.columns]
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        return self.results_df
    
    def calculate_overall_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas generales para todo el conjunto de datos.
        
        Returns:
            Diccionario con métricas generales
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        y_true = self.results_df['y_true'].values
        y_pred = self.results_df['y_pred'].values
        y_prob = self.results_df['y_prob'].values
        
        # Calcular cuotas si están disponibles, de lo contrario usar None
        odds = None
        if 'odds_player1' in self.results_df.columns:
            odds = self.results_df['odds_player1'].values
        
        return self.metrics.calculate_all_metrics(y_true, y_pred, y_prob, odds)
    
    def segment_by_category(self, 
                           category_column: str, 
                           min_matches: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Segmenta los resultados por una categoría y calcula métricas para cada segmento.
        
        Args:
            category_column: Nombre de la columna para segmentar (ej: 'surface', 'tournament_level')
            min_matches: Número mínimo de partidos para incluir un segmento
            
        Returns:
            Diccionario con métricas por segmento
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        if category_column not in self.results_df.columns:
            raise ValueError(f"La columna {category_column} no existe en los resultados")
        
        segment_metrics = {}
        for category_value, group in self.results_df.groupby(category_column):
            if len(group) < min_matches:
                continue
                
            y_true = group['y_true'].values
            y_pred = group['y_pred'].values
            y_prob = group['y_prob'].values
            
            # Calcular cuotas si están disponibles
            odds = None
            if 'odds_player1' in group.columns:
                odds = group['odds_player1'].values
            
            # Calcular métricas para este segmento
            metrics = self.metrics.calculate_all_metrics(y_true, y_pred, y_prob, odds)
            segment_metrics[str(category_value)] = metrics
        
        # Guardar para uso posterior
        self.segmented_metrics[category_column] = segment_metrics
        
        return segment_metrics
    
    def analyze_by_surface(self, min_matches: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Analiza los resultados segmentados por superficie.
        
        Args:
            min_matches: Número mínimo de partidos para incluir una superficie
            
        Returns:
            Diccionario con métricas por superficie
        """
        return self.segment_by_category('surface', min_matches)
    
    def analyze_by_tournament_level(self, min_matches: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Analiza los resultados segmentados por nivel de torneo.
        
        Args:
            min_matches: Número mínimo de partidos para incluir un nivel
            
        Returns:
            Diccionario con métricas por nivel de torneo
        """
        return self.segment_by_category('tourney_level', min_matches)
    
    def analyze_by_player_ranking(self, 
                                 bins: List[int] = [1, 10, 50, 100, 500, 2000],
                                 player_column: str = 'player1_rank') -> Dict[str, Dict[str, float]]:
        """
        Analiza los resultados segmentados por rango de ranking de los jugadores.
        
        Args:
            bins: Límites de las categorías de ranking
            player_column: Columna con el ranking del jugador
            
        Returns:
            Diccionario con métricas por categoría de ranking
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        if player_column not in self.results_df.columns:
            raise ValueError(f"La columna {player_column} no existe en los resultados")
        
        # Crear categorías de ranking
        self.results_df['ranking_category'] = pd.cut(
            self.results_df[player_column], 
            bins=bins, 
            labels=[f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        )
        
        return self.segment_by_category('ranking_category')
    
    def analyze_by_match_importance(self, 
                                   round_importance: Dict[str, int] = None) -> Dict[str, Dict[str, float]]:
        """
        Analiza los resultados segmentados por importancia del partido.
        
        Args:
            round_importance: Diccionario con valores de importancia por ronda
            
        Returns:
            Diccionario con métricas por nivel de importancia
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        if 'round' not in self.results_df.columns:
            raise ValueError("La columna 'round' no existe en los resultados")
        
        # Importancia por defecto si no se proporciona
        if round_importance is None:
            round_importance = {
                'F': 5,      # Final
                'SF': 4,     # Semifinal
                'QF': 3,     # Cuartos de final
                'R16': 2,    # Octavos de final
                'R32': 1,    # Dieciseisavos de final
                'R64': 1,    # Primera ronda
                'R128': 1    # Primera ronda
            }
        
        # Asignar nivel de importancia
        self.results_df['match_importance'] = self.results_df['round'].map(
            lambda r: round_importance.get(r, 1)
        )
        
        return self.segment_by_category('match_importance')
    
    def analyze_by_confidence_level(self, bins: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Analiza los resultados segmentados por nivel de confianza (probabilidad).
        
        Args:
            bins: Número de bins para segmentar las probabilidades
            
        Returns:
            Diccionario con métricas por nivel de confianza
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        # Crear categorías de confianza
        self.results_df['confidence_level'] = pd.qcut(
            self.results_df['y_prob'], 
            q=bins, 
            labels=[f"Bin {i+1}" for i in range(bins)]
        )
        
        return self.segment_by_category('confidence_level')
    
    def analyze_incorrect_predictions(self) -> pd.DataFrame:
        """
        Analiza patrones en las predicciones incorrectas.
        
        Returns:
            DataFrame con análisis de predicciones incorrectas
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        # Filtrar predicciones incorrectas
        incorrect = self.results_df[self.results_df['y_true'] != self.results_df['y_pred']]
        
        # Crear análisis
        analysis = {}
        
        # Distribución de incorrectas por superficie
        if 'surface' in incorrect.columns:
            analysis['by_surface'] = incorrect['surface'].value_counts(normalize=True).to_dict()
        
        # Distribución de incorrectas por nivel de torneo
        if 'tourney_level' in incorrect.columns:
            analysis['by_tourney_level'] = incorrect['tourney_level'].value_counts(normalize=True).to_dict()
        
        # Distribución de incorrectas por ronda
        if 'round' in incorrect.columns:
            analysis['by_round'] = incorrect['round'].value_counts(normalize=True).to_dict()
        
        # Análisis de falsas positivas vs falsas negativas
        false_positives = incorrect[incorrect['y_pred'] == 1]
        false_negatives = incorrect[incorrect['y_pred'] == 0]
        
        analysis['false_positive_count'] = len(false_positives)
        analysis['false_negative_count'] = len(false_negatives)
        
        # Características promedio de falsas positivas
        if len(false_positives) > 0:
            fp_features = {}
            for col in false_positives.columns:
                if col not in ['y_true', 'y_pred', 'y_prob'] and pd.api.types.is_numeric_dtype(false_positives[col]):
                    fp_features[col] = false_positives[col].mean()
            analysis['false_positive_avg_features'] = fp_features
        
        # Características promedio de falsas negativas
        if len(false_negatives) > 0:
            fn_features = {}
            for col in false_negatives.columns:
                if col not in ['y_true', 'y_pred', 'y_prob'] and pd.api.types.is_numeric_dtype(false_negatives[col]):
                    fn_features[col] = false_negatives[col].mean()
            analysis['false_negative_avg_features'] = fn_features
        
        # Guardar resultados como DataFrame
        analysis_df = pd.DataFrame.from_dict({k: [v] if not isinstance(v, dict) else [json.dumps(v)] 
                                           for k, v in analysis.items()}).T
        analysis_df.columns = ['value']
        
        # Guardar como JSON también
        with open(os.path.join(self.output_dir, 'incorrect_predictions_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=4)
        
        return analysis_df
    
    def calculate_roi_by_threshold(self, 
                                 thresholds: List[float] = None,
                                 stake: float = 100) -> pd.DataFrame:
        """
        Calcula el ROI para diferentes umbrales de probabilidad.
        
        Args:
            thresholds: Lista de umbrales de probabilidad
            stake: Apuesta por defecto
            
        Returns:
            DataFrame con ROI por umbral
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        if 'odds_player1' not in self.results_df.columns:
            raise ValueError("La columna 'odds_player1' no existe en los resultados")
        
        if thresholds is None:
            thresholds = np.linspace(0.5, 0.9, 9)
        
        roi_results = []
        
        for threshold in thresholds:
            # Filtrar partidos donde la probabilidad supera el umbral
            filtered = self.results_df[self.results_df['y_prob'] >= threshold]
            
            if len(filtered) == 0:
                roi_results.append({
                    'threshold': threshold,
                    'bets_count': 0,
                    'win_rate': 0,
                    'roi': 0,
                    'profit': 0
                })
                continue
            
            # Calcular ganancias/pérdidas
            filtered['profit'] = np.where(
                filtered['y_true'] == 1,
                stake * (filtered['odds_player1'] - 1),  # Ganancia
                -stake                                   # Pérdida
            )
            
            total_investment = stake * len(filtered)
            total_profit = filtered['profit'].sum()
            roi = total_profit / total_investment if total_investment > 0 else 0
            
            roi_results.append({
                'threshold': threshold,
                'bets_count': len(filtered),
                'win_rate': filtered['y_true'].mean(),
                'roi': roi,
                'profit': total_profit
            })
        
        roi_df = pd.DataFrame(roi_results)
        
        # Guardar como CSV
        roi_df.to_csv(os.path.join(self.output_dir, 'roi_by_threshold.csv'), index=False)
        
        return roi_df
    
    def calculate_feature_importance(self) -> pd.DataFrame:
        """
        Calcula la importancia de las características basándose en cómo afectan a la precisión.
        
        Returns:
            DataFrame con importancia de características
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        # Identificar columnas numéricas (posibles características)
        feature_cols = [col for col in self.results_df.columns 
                      if col not in ['y_true', 'y_pred', 'y_prob', 'match_id', 'player1', 'player2'] 
                      and pd.api.types.is_numeric_dtype(self.results_df[col])]
        
        if not feature_cols:
            raise ValueError("No se encontraron columnas numéricas que podrían ser características")
        
        # Calcular importancia por correlación con predicciones correctas
        self.results_df['correct_prediction'] = (self.results_df['y_true'] == self.results_df['y_pred']).astype(int)
        
        importance_results = []
        
        for feature in feature_cols:
            correlation = self.results_df[[feature, 'correct_prediction']].corr().iloc[0, 1]
            importance_results.append({
                'feature': feature,
                'correlation_with_correctness': correlation
            })
        
        # Calcular importancia por diferencia media entre predicciones correctas e incorrectas
        for feature in feature_cols:
            correct_mean = self.results_df[self.results_df['correct_prediction'] == 1][feature].mean()
            incorrect_mean = self.results_df[self.results_df['correct_prediction'] == 0][feature].mean()
            
            # Encontrar la importancia correspondiente
            idx = next(i for i, d in enumerate(importance_results) if d['feature'] == feature)
            importance_results[idx]['correct_mean'] = correct_mean
            importance_results[idx]['incorrect_mean'] = incorrect_mean
            importance_results[idx]['mean_difference'] = abs(correct_mean - incorrect_mean)
        
        importance_df = pd.DataFrame(importance_results)
        importance_df = importance_df.sort_values('correlation_with_correctness', ascending=False)
        
        # Guardar como CSV
        importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        
        return importance_df
    
    def generate_plots(self) -> None:
        """
        Genera múltiples visualizaciones para analizar las predicciones.
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        self._plot_roc_curve()
        self._plot_precision_recall_curve()
        self._plot_calibration_curve()
        
        # Gráficos adicionales si hay segmentación
        if 'surface' in self.segmented_metrics:
            self._plot_performance_by_segment('surface', 'Rendimiento por Superficie')
            
        if 'tourney_level' in self.segmented_metrics:
            self._plot_performance_by_segment('tourney_level', 'Rendimiento por Nivel de Torneo')
            
        if hasattr(self, 'roi_by_threshold_df'):
            self._plot_roi_by_threshold()
    
    def _plot_roc_curve(self) -> None:
        """Genera y guarda la curva ROC."""
        y_true = self.results_df['y_true'].values
        y_prob = self.results_df['y_prob'].values
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {self.metrics.metrics_results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self) -> None:
        """Genera y guarda la curva de precisión-recall."""
        y_true = self.results_df['y_true'].values
        y_prob = self.results_df['y_prob'].values
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Curva de Precisión-Recall')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self) -> None:
        """Genera y guarda la curva de calibración."""
        # Usar la función existente en la clase TennisMetrics
        if 'calibration_data' in self.metrics.metrics_results:
            fig = self.metrics.plot_calibration_curve()
            plt.savefig(os.path.join(self.output_dir, 'calibration_curve.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def _plot_performance_by_segment(self, segment_name: str, title: str) -> None:
        """
        Genera y guarda un gráfico de barras con métricas por segmento.
        
        Args:
            segment_name: Nombre del segmento (ej: 'surface')
            title: Título del gráfico
        """
        if segment_name not in self.segmented_metrics:
            return
        
        # Extraer métricas por segmento
        segments = []
        accuracy = []
        roc_auc = []
        roi = []
        
        for segment, metrics in self.segmented_metrics[segment_name].items():
            segments.append(segment)
            accuracy.append(metrics.get('accuracy', 0))
            roc_auc.append(metrics.get('roc_auc', 0))
            roi.append(metrics.get('roi', 0))
        
        # Crear DataFrame para el gráfico
        df = pd.DataFrame({
            'Segmento': segments,
            'Accuracy': accuracy,
            'ROC AUC': roc_auc,
            'ROI': roi
        })
        
        # Ordenar por accuracy
        df = df.sort_values('Accuracy', ascending=False)
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(df))
        width = 0.25
        
        plt.bar(x - width, df['Accuracy'], width, label='Accuracy')
        plt.bar(x, df['ROC AUC'], width, label='ROC AUC')
        plt.bar(x + width, df['ROI'], width, label='ROI')
        
        plt.xlabel('Segmento')
        plt.ylabel('Valor')
        plt.title(title)
        plt.xticks(x, df['Segmento'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f'performance_by_{segment_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roi_by_threshold(self) -> None:
        """Genera y guarda un gráfico de ROI por umbral de probabilidad."""
        if not hasattr(self, 'roi_by_threshold_df'):
            return
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.roi_by_threshold_df['threshold'], self.roi_by_threshold_df['roi'], 'o-', lw=2)
        plt.xlabel('Umbral de Probabilidad')
        plt.ylabel('ROI')
        plt.title('ROI por Umbral de Probabilidad')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.roi_by_threshold_df['threshold'], self.roi_by_threshold_df['bets_count'], 'o-', lw=2)
        plt.xlabel('Umbral de Probabilidad')
        plt.ylabel('Número de Apuestas')
        plt.title('Cantidad de Apuestas por Umbral')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roi_by_threshold.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self) -> str:
        """
        Genera un informe completo con todas las métricas y análisis.
        
        Returns:
            Ruta al archivo de informe generado
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        report_lines = [
            "# Informe de Análisis de Predicciones de Tenis",
            "",
            f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total de partidos analizados: {len(self.results_df)}",
            "",
            "## 1. Métricas Generales",
            ""
        ]
        
        # Añadir métricas generales
        report_lines.append(self.metrics.get_metrics_report(detailed=False))
        
        # Añadir análisis por segmentos
        report_lines.extend([
            "",
            "## 2. Análisis por Segmentos",
            ""
        ])
        
        for segment_name, segment_metrics in self.segmented_metrics.items():
            report_lines.append(f"### 2.{len(report_lines) - 14}. Análisis por {segment_name}")
            report_lines.append("")
            
            # Crear tabla con métricas principales
            report_lines.append("| Segmento | Accuracy | Precision | Recall | F1 | ROC AUC | ROI |")
            report_lines.append("| -------- | -------- | --------- | ------ | -- | ------- | --- |")
            
            for segment, metrics in segment_metrics.items():
                report_lines.append(
                    f"| {segment} | "
                    f"{metrics.get('accuracy', 0):.4f} | "
                    f"{metrics.get('precision', 0):.4f} | "
                    f"{metrics.get('recall', 0):.4f} | "
                    f"{metrics.get('f1', 0):.4f} | "
                    f"{metrics.get('roc_auc', 0):.4f} | "
                    f"{metrics.get('roi', 0):.4f} |"
                )
            
            report_lines.append("")
        
        # Añadir análisis de predicciones incorrectas
        report_lines.extend([
            "",
            "## 3. Análisis de Predicciones Incorrectas",
            ""
        ])
        
        incorrect_analysis = self.analyze_incorrect_predictions()
        for index, row in incorrect_analysis.iterrows():
            value = row['value']
            if isinstance(value, str) and value.startswith('{'):
                try:
                    value_dict = json.loads(value)
                    report_lines.append(f"### {index}")
                    report_lines.append("")
                    for k, v in value_dict.items():
                        report_lines.append(f"- {k}: {v}")
                    report_lines.append("")
                except:
                    report_lines.append(f"### {index}")
                    report_lines.append(f"{value}")
                    report_lines.append("")
            else:
                report_lines.append(f"### {index}")
                report_lines.append(f"{value}")
                report_lines.append("")
        
        # Añadir análisis ROI por umbral
        try:
            roi_df = self.calculate_roi_by_threshold()
            self.roi_by_threshold_df = roi_df
            
            report_lines.extend([
                "",
                "## 4. Análisis de ROI por Umbral de Probabilidad",
                "",
                "| Umbral | Apuestas | Tasa de Acierto | ROI | Beneficio |",
                "| ------ | -------- | --------------- | --- | --------- |"
            ])
            
            for _, row in roi_df.iterrows():
                report_lines.append(
                    f"| {row['threshold']:.2f} | "
                    f"{row['bets_count']} | "
                    f"{row['win_rate']:.4f} | "
                    f"{row['roi']:.4f} | "
                    f"{row['profit']:.2f} |"
                )
            
            report_lines.append("")
        except Exception as e:
            report_lines.extend([
                "",
                "## 4. Análisis de ROI por Umbral de Probabilidad",
                "",
                f"No se pudo calcular: {str(e)}",
                ""
            ])
        
        # Añadir análisis de importancia de características
        try:
            importance_df = self.calculate_feature_importance()
            
            report_lines.extend([
                "",
                "## 5. Importancia de Características",
                "",
                "| Característica | Correlación con Acierto | Diferencia Media |",
                "| -------------- | ----------------------- | ---------------- |"
            ])
            
            for _, row in importance_df.head(10).iterrows():
                report_lines.append(
                    f"| {row['feature']} | "
                    f"{row['correlation_with_correctness']:.4f} | "
                    f"{row['mean_difference']:.4f} |"
                )
            
            report_lines.append("")
        except Exception as e:
            report_lines.extend([
                "",
                "## 5. Importancia de Características",
                "",
                f"No se pudo calcular: {str(e)}",
                ""
            ])
        
        # Generar visualizaciones
        report_lines.extend([
            "",
            "## 6. Visualizaciones",
            "",
            "Se han generado las siguientes visualizaciones en el directorio de resultados:",
            ""
        ])
        
        # Generar todas las visualizaciones
        try:
            self.generate_plots()
            for file in os.listdir(self.output_dir):
                if file.endswith('.png'):
                    report_lines.append(f"- {file}")
        except Exception as e:
            report_lines.append(f"Error al generar visualizaciones: {str(e)}")
        
        # Guardar el informe
        report_path = os.path.join(self.output_dir, 'comprehensive_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return report_path
    
    def compare_model_versions(self, results_list, model_names=None):
        """
        Compara los resultados de diferentes versiones de modelos.
        
        Args:
            results_list: Lista de DataFrames con resultados de cada modelo
            model_names: Lista de nombres para los modelos (opcional)
        
        Returns:
            DataFrame con comparación de métricas y visualizaciones
        """
        if model_names is None:
            model_names = [f"Modelo {i+1}" for i in range(len(results_list))]
        
        if len(model_names) != len(results_list):
            raise ValueError("La lista de nombres debe tener la misma longitud que la lista de resultados")
        
        comparison_metrics = []
        
        # Calcular métricas para cada modelo
        for i, results_df in enumerate(results_list):
            # Guardar resultado actual
            original_results = self.results_df
            
            # Cargar resultados del modelo a comparar
            self.load_results(results_df)
            
            # Calcular métricas
            metrics = self.calculate_overall_metrics()
            metrics['model_name'] = model_names[i]
            comparison_metrics.append(metrics)
            
            # Restaurar resultados originales
            self.results_df = original_results
        
        # Crear DataFrame de comparación
        metrics_df = pd.DataFrame(comparison_metrics)
        
        # Generar visualizaciones comparativas
        self._plot_model_comparison(metrics_df)
        
        return metrics_df

    def _plot_model_comparison(self, metrics_df):
        """
        Genera visualizaciones comparativas entre modelos.
        
        Args:
            metrics_df: DataFrame con métricas de diferentes modelos
        """
        plt.figure(figsize=(12, 10))
        
        # Seleccionar métricas principales para la comparación
        main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        available_metrics = [m for m in main_metrics if m in metrics_df.columns]
        
        for i, metric in enumerate(available_metrics):
            plt.subplot(len(available_metrics), 1, i+1)
            
            # Ordenar por valor de métrica
            sorted_df = metrics_df.sort_values(metric)
            
            plt.barh(sorted_df['model_name'], sorted_df[metric])
            plt.xlabel(metric.upper())
            plt.xlim(0, 1)
            plt.grid(True, axis='x')
            
            # Añadir valores
            for j, v in enumerate(sorted_df[metric]):
                plt.text(v + 0.01, j, f"{v:.4f}", va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_temporal_performance(self, date_column='match_date'):
        """
        Analiza la evolución del rendimiento del modelo a lo largo del tiempo.
        
        Args:
            date_column: Nombre de la columna con fechas de partidos
            
        Returns:
            DataFrame con métricas temporales
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        if date_column not in self.results_df.columns:
            raise ValueError(f"La columna {date_column} no existe en los resultados")
        
        # Asegurar que la columna de fecha es datetime
        if not pd.api.types.is_datetime64_dtype(self.results_df[date_column]):
            self.results_df[date_column] = pd.to_datetime(self.results_df[date_column])
        
        # Ordenar por fecha
        self.results_df = self.results_df.sort_values(date_column)
        
        # Calcular métricas temporales (ventana móvil)
        temporal_metrics = self.metrics.calculate_temporal_metrics(
            self.results_df['y_true'].values,
            self.results_df['y_pred'].values,
            self.results_df['y_prob'].values,
            self.results_df[date_column].values,
            window_size=30  # 30 días por ventana
        )
        
        # Guardar como CSV
        temporal_metrics.to_csv(os.path.join(self.output_dir, 'temporal_metrics.csv'), index=False)
        
        # Visualizar evolución temporal
        self._plot_temporal_performance(temporal_metrics)
        
        return temporal_metrics

    def _plot_temporal_performance(self, temporal_metrics):
        """
        Genera visualizaciones de la evolución del rendimiento a lo largo del tiempo.
        
        Args:
            temporal_metrics: DataFrame con métricas temporales
        """
        plt.figure(figsize=(14, 10))
        
        # Plot de accuracy
        plt.subplot(3, 1, 1)
        plt.plot(temporal_metrics['window_start'], temporal_metrics['accuracy'], 'o-')
        plt.title('Evolución de Accuracy')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Plot de ROC AUC
        plt.subplot(3, 1, 2)
        plt.plot(temporal_metrics['window_start'], temporal_metrics['roc_auc'], 'o-')
        plt.title('Evolución de ROC AUC')
        plt.ylabel('ROC AUC')
        plt.grid(True)
        
        # Plot de error de calibración
        plt.subplot(3, 1, 3)
        plt.plot(temporal_metrics['window_start'], temporal_metrics['expected_calibration_error'], 'o-')
        plt.title('Evolución del Error de Calibración')
        plt.ylabel('ECE')
        plt.xlabel('Fecha')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temporal_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def detect_systematic_biases(self):
        """
        Detecta sesgos sistemáticos en las predicciones del modelo.
        
        Returns:
            DataFrame con análisis de sesgos
        """
        if self.results_df is None:
            raise ValueError("No hay resultados cargados. Utiliza load_results primero.")
        
        bias_results = {}
        
        # 1. Sesgo en probabilidades (sobrestimación/subestimación)
        avg_pred_prob = self.results_df['y_prob'].mean()
        avg_actual = self.results_df['y_true'].mean()
        prob_bias = avg_pred_prob - avg_actual
        
        bias_results['probability_bias'] = {
            'predicted_average': avg_pred_prob,
            'actual_average': avg_actual,
            'bias': prob_bias,
            'conclusion': 'Sobrestimación' if prob_bias > 0.02 else ('Subestimación' if prob_bias < -0.02 else 'No significativo')
        }
        
        # 2. Sesgo por nivel de confianza
        confidence_bins = pd.qcut(self.results_df['y_prob'], 10, labels=False)
        self.results_df['confidence_bin'] = confidence_bins
        
        bin_stats = self.results_df.groupby('confidence_bin').agg({
            'y_prob': 'mean',
            'y_true': 'mean',
            'y_pred': 'count'
        }).reset_index()
        
        bin_stats['bias'] = bin_stats['y_prob'] - bin_stats['y_true']
        bin_stats['abs_bias'] = abs(bin_stats['bias'])
        
        max_bias_bin = bin_stats.loc[bin_stats['abs_bias'].idxmax()]
        
        bias_results['confidence_level_bias'] = {
            'max_bias_bin': int(max_bias_bin['confidence_bin']),
            'predicted_prob': max_bias_bin['y_prob'],
            'actual_rate': max_bias_bin['y_true'],
            'bias': max_bias_bin['bias'],
            'bin_count': max_bias_bin['y_pred']
        }
        
        # 3. Detección de sesgos por características
        feature_cols = [col for col in self.results_df.columns 
                    if col not in ['y_true', 'y_pred', 'y_prob', 'match_id', 'player1', 'player2',
                                    'confidence_bin'] 
                    and pd.api.types.is_numeric_dtype(self.results_df[col])]
        
        feature_biases = {}
        for feature in feature_cols:
            # Dividir en bins
            if self.results_df[feature].nunique() > 10:
                self.results_df[f'{feature}_bin'] = pd.qcut(
                    self.results_df[feature], 
                    5, 
                    labels=False, 
                    duplicates='drop'
                )
                group_col = f'{feature}_bin'
            else:
                group_col = feature
            
            # Calcular bias por grupo
            group_stats = self.results_df.groupby(group_col).agg({
                'y_prob': 'mean',
                'y_true': 'mean',
                'y_pred': 'count'
            }).reset_index()
            
            group_stats['bias'] = group_stats['y_prob'] - group_stats['y_true']
            group_stats['abs_bias'] = abs(group_stats['bias'])
            
            # Si hay un bias consistente (mismo signo en todos los bins)
            if (group_stats['bias'] > 0.05).all() or (group_stats['bias'] < -0.05).all():
                direction = 'positivo' if group_stats['bias'].mean() > 0 else 'negativo'
                feature_biases[feature] = {
                    'direction': direction,
                    'avg_bias': group_stats['bias'].mean(),
                    'significance': 'Alto' if abs(group_stats['bias'].mean()) > 0.1 else 'Medio'
                }
        
        bias_results['feature_biases'] = feature_biases
        
        # Guardar resultados
        with open(os.path.join(self.output_dir, 'bias_analysis.json'), 'w') as f:
            json.dump(bias_results, f, indent=4)
        
        # Generar visualización de sesgos
        self._plot_bias_analysis(bin_stats, feature_biases)
        
        return bias_results

    def _plot_bias_analysis(self, bin_stats, feature_biases):
        """
        Genera visualizaciones para el análisis de sesgos.
        
        Args:
            bin_stats: DataFrame con estadísticas por bin de confianza
            feature_biases: Diccionario con sesgos por característica
        """
        plt.figure(figsize=(14, 10))
        
        # Gráfico de calibración
        plt.subplot(2, 1, 1)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfecta calibración')
        plt.plot(bin_stats['y_prob'], bin_stats['y_true'], 'o-', label='Calibración actual')
        plt.xlabel('Probabilidad predicha')
        plt.ylabel('Tasa de éxito real')
        plt.title('Análisis de Calibración')
        plt.legend()
        plt.grid(True)
        
        # Gráfico de sesgo por nivel de confianza
        plt.subplot(2, 1, 2)
        plt.bar(bin_stats['confidence_bin'], bin_stats['bias'])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Bin de confianza (0=baja, 9=alta)')
        plt.ylabel('Sesgo (predicha - real)')
        plt.title('Sesgo por Nivel de Confianza')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'bias_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Si hay sesgos por características, graficarlos
        if feature_biases:
            plt.figure(figsize=(14, len(feature_biases) * 3))
            
            for i, (feature, bias_info) in enumerate(feature_biases.items()):
                plt.subplot(len(feature_biases), 1, i+1)
                
                # Obtener datos para este feature
                if f'{feature}_bin' in self.results_df.columns:
                    group_col = f'{feature}_bin'
                else:
                    group_col = feature
                    
                group_stats = self.results_df.groupby(group_col).agg({
                    'y_prob': 'mean',
                    'y_true': 'mean'
                }).reset_index()
                
                plt.bar(range(len(group_stats)), group_stats['y_prob'] - group_stats['y_true'])
                plt.axhline(y=0, color='r', linestyle='-')
                plt.title(f'Sesgo para {feature} ({bias_info["direction"]}, {bias_info["significance"]})')
                plt.ylabel('Sesgo')
                plt.xlabel(feature)
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_biases.png'), dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    # Ejemplo de uso
    analyzer = ResultsAnalyzer("example_results")
    
    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 1000
    
    # Simulación básica
    elo_diff = np.random.normal(0, 100, n_samples)
    surface_advantage = np.random.normal(0, 20, n_samples)
    
    # Probabilidad "real" basada en ELO y ventaja de superficie
    true_prob = 1 / (1 + np.exp(-(elo_diff + surface_advantage) / 100))
    
    # Resultados reales
    y_true = (np.random.random(n_samples) < true_prob).astype(int)
    
    # Probabilidad predicha (con algo de ruido)
    y_prob = np.clip(true_prob + np.random.normal(0, 0.1, n_samples), 0.01, 0.99)
    
    # Predicciones
    y_pred = (y_prob > 0.5).astype(int)
    
    # Cuotas (inversamente proporcionales a la probabilidad con margen)
    odds = 0.9 / y_prob
    
    # Crear DataFrame
    results = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'odds_player1': odds,
        'player1_elo': np.random.normal(1500, 200, n_samples),
        'player2_elo': np.random.normal(1500, 200, n_samples),
        'player1_rank': np.random.randint(1, 500, n_samples),
        'player2_rank': np.random.randint(1, 500, n_samples),
        'elo_difference': elo_diff,
        'surface': np.random.choice(['hard', 'clay', 'grass', 'carpet'], n_samples),
        'tourney_level': np.random.choice(['grand_slam', 'atp1000', 'atp500', 'atp250', 'challenger'], n_samples),
        'round': np.random.choice(['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F'], n_samples)
    })
    
    # Cargar resultados
    analyzer.load_results(results)
    
    # Calcular métricas generales
    metrics = analyzer.calculate_overall_metrics()
    print("Métricas generales calculadas.")
    
    # Análisis por segmentos
    analyzer.analyze_by_surface()
    analyzer.analyze_by_tournament_level()
    analyzer.analyze_by_player_ranking()
    analyzer.analyze_by_match_importance()
    print("Análisis por segmentos completado.")
    
    # Generar informe completo
    report_path = analyzer.generate_comprehensive_report()
    print(f"Informe completo generado en: {report_path}")
    
    # Ejemplos de las nuevas funcionalidades
    print("\nDetectando sesgos sistemáticos...")
    biases = analyzer.detect_systematic_biases()
    print("Análisis de sesgos completado.")

    # Ejemplo de análisis temporal (añadiendo una columna de fecha)
    results['match_date'] = pd.date_range(start='2023-01-01', periods=len(results), freq='D')
    print("\nAnalizando rendimiento temporal...")
    temporal_results = analyzer.analyze_temporal_performance(date_column='match_date')
    print("Análisis temporal completado.")

    # Ejemplo de comparación de modelos
    model2_y_prob = np.clip(y_prob + np.random.normal(0, 0.05, len(y_prob)), 0, 1)
    model2_y_pred = (model2_y_prob > 0.5).astype(int)
    model2_results = results.copy()
    model2_results['y_prob'] = model2_y_prob
    model2_results['y_pred'] = model2_y_pred

    print("\nComparando versiones de modelos...")
    model_comparison = analyzer.compare_model_versions([results, model2_results], ["Modelo Original", "Modelo V2"])
    print("Comparación de modelos completada.")
