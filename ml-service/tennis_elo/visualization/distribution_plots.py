"""
visualization/distribution_plots.py

Módulo para crear visualizaciones relacionadas con distribuciones de 
ratings ELO en tenis. Permite generar gráficos de distribución general,
por superficie y comparativos, con estadísticas asociadas.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from scipy import stats

# Configurar logger
logger = logging.getLogger(__name__)

def plot_rating_distribution(ratings: Dict[str, float],
                           surface: Optional[str] = None, 
                           min_matches: int = 10,
                           player_match_counts: Optional[Dict[str, int]] = None,
                           distribution_stats: Optional[Dict] = None,
                           top_players: Optional[pd.DataFrame] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Genera un gráfico con la distribución de ratings ELO.
    
    Args:
        ratings: Diccionario con ratings {player_id: rating}
        surface: Superficie específica (opcional)
        min_matches: Número mínimo de partidos jugados
        player_match_counts: Diccionario con conteo de partidos por jugador
        distribution_stats: Estadísticas precalculadas (opcional)
        top_players: DataFrame con mejores jugadores (opcional)
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Filtrar jugadores con pocos partidos si se proporcionan conteos
    if player_match_counts:
        filtered_ratings = {
            player_id: rating 
            for player_id, rating in ratings.items()
            if player_match_counts.get(player_id, 0) >= min_matches
        }
    else:
        filtered_ratings = ratings
    
    if not filtered_ratings:
        logger.warning("No hay suficientes datos para graficar distribución")
        return plt.figure()
    
    # Calcular estadísticas si no se proporcionan
    if not distribution_stats:
        distribution_stats = analyze_ratings_distribution(filtered_ratings)
    
    # Crear figura compuesta de dos gráficos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # 1. Histograma con distribución
    if 'distribution' in distribution_stats:
        bin_edges = distribution_stats['distribution']['bin_edges']
        bin_counts = distribution_stats['distribution']['bin_counts']
        bin_centers = distribution_stats['distribution']['bin_centers']
        
        ax1.bar(bin_centers, bin_counts, 
                width=bin_edges[1]-bin_edges[0] if len(bin_edges) > 1 else 50, 
                alpha=0.7, color='royalblue',
                edgecolor='navy')
    else:
        # Crear histograma si no hay distribución precalculada
        ratings_values = list(filtered_ratings.values())
        ax1.hist(ratings_values, bins=20, alpha=0.7, color='royalblue', edgecolor='navy')
    
    # Añadir curva normal
    if distribution_stats['count'] > 30:
        x = np.linspace(distribution_stats['min'] - 50, distribution_stats['max'] + 50, 100)
        pdf = stats.norm.pdf(x, distribution_stats['mean'], distribution_stats['std'])
        
        # Escalar PDF para que coincida con el histograma
        if 'distribution' in distribution_stats:
            bin_counts = distribution_stats['distribution']['bin_counts']
            bin_edges = distribution_stats['distribution']['bin_edges']
            pdf_scaled = pdf * sum(bin_counts) * (bin_edges[1]-bin_edges[0] if len(bin_edges) > 1 else 50)
        else:
            pdf_scaled = pdf * len(filtered_ratings) * 50  # Valor aproximado
            
        ax1.plot(x, pdf_scaled, 'r-', linewidth=2, alpha=0.7, label='Distribución normal')
    
    # Añadir líneas verticales para percentiles
    percentiles = distribution_stats['percentiles']
    ax1.axvline(x=percentiles['25'], color='gray', linestyle='--', alpha=0.7, 
                label=f"Percentil 25: {percentiles['25']:.0f}")
    ax1.axvline(x=percentiles['50'], color='black', linestyle='--', alpha=0.7, 
                label=f"Mediana: {percentiles['50']:.0f}")
    ax1.axvline(x=percentiles['75'], color='gray', linestyle='--', alpha=0.7, 
                label=f"Percentil 75: {percentiles['75']:.0f}")
    
    # Ajustes al gráfico principal
    if surface:
        ax1.set_title(f'Distribución de ratings ELO en {surface.upper()} (N={distribution_stats["count"]})', fontsize=16)
    else:
        ax1.set_title(f'Distribución de ratings ELO (N={distribution_stats["count"]})', fontsize=16)
    
    ax1.set_xlabel('Rating ELO', fontsize=12)
    ax1.set_ylabel('Número de jugadores', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Gráfico inferior con top jugadores
    if top_players is not None and not top_players.empty:
        # Ordenar de mejor a peor para el gráfico
        top_players = top_players.sort_values('elo_rating', ascending=True)
        
        # Crear nombres legibles para etiquetas
        if 'player_name' in top_players.columns:
            labels = top_players['player_name']
        else:
            labels = top_players['player_id']
        
        # Crear gráfico de barras horizontales
        bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_players)))
        
        # Gráfico de barras con top jugadores
        bars = ax2.barh(labels, top_players['elo_rating'], color=bar_colors, alpha=0.7)
        
        # Añadir valores numéricos a las barras
        for i, bar in enumerate(bars):
            ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                    f"{top_players['elo_rating'].iloc[i]:.0f}", 
                    va='center', fontsize=9)
        
        # Ajustes al gráfico de top jugadores
        ax2.set_title(f'Top {len(top_players)} jugadores por rating ELO', fontsize=14)
        ax2.set_xlabel('Rating ELO', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Mantener el mismo rango de X que el histograma principal
        ax2.set_xlim(ax1.get_xlim())
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de distribución guardado en {save_path}")
    
    return fig

def analyze_ratings_distribution(ratings: Dict[str, float], 
                              percentiles: List[int] = [5, 25, 50, 75, 95],
                              bins: int = 20) -> Dict:
    """
    Analiza la distribución de los ratings ELO.
    
    Args:
        ratings: Diccionario con ratings {player_id: rating}
        percentiles: Lista de percentiles a calcular
        bins: Número de bins para histograma
        
    Returns:
        Diccionario con estadísticas de la distribución
    """
    ratings_values = list(ratings.values())
    
    if not ratings_values:
        # Valores por defecto si no hay datos
        return {
            'count': 0,
            'mean': 1500,
            'std': 0,
            'min': 1500,
            'max': 1500,
            'percentiles': {str(p): 1500 for p in percentiles}
        }
    
    # Calcular estadísticas
    ratings_array = np.array(ratings_values)
    
    # Cálculos estadísticos básicos
    stats = {
        'count': len(ratings),
        'mean': float(np.mean(ratings_array)),
        'std': float(np.std(ratings_array)),
        'min': float(np.min(ratings_array)),
        'max': float(np.max(ratings_array)),
        'percentiles': {
            str(p): float(np.percentile(ratings_array, p)) for p in percentiles
        }
    }
    
    # Añadir distribución por bins para visualización
    hist, edges = np.histogram(ratings_array, bins=bins)
    
    # Convertir a porcentajes
    total = hist.sum()
    percentages = (hist / total * 100).tolist() if total > 0 else [0] * len(hist)
    
    # Añadir al resultado
    stats['distribution'] = {
        'bin_counts': hist.tolist(),
        'bin_percentages': percentages,
        'bin_edges': edges.tolist(),
        'bin_centers': [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
    }
    
    # Verificar normalidad (para análisis)
    if len(ratings_array) > 20:  # Solo para muestras suficientemente grandes
        try:
            normality_test = stats.normaltest(ratings_array)
            stats['normality'] = {
                'statistic': float(normality_test.statistic),
                'p_value': float(normality_test.pvalue),
                'is_normal': float(normality_test.pvalue) > 0.05
            }
        except Exception as e:
            logger.warning(f"Error calculando test de normalidad: {str(e)}")
    
    return stats

def plot_rating_distributions_comparison(distributions: Dict[str, Dict],
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Compara distribuciones de ratings entre diferentes categorías (superficies, periodos, etc.)
    
    Args:
        distributions: Diccionario con estadísticas de distribución por categoría
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        
    Returns:
        Objeto Figure de matplotlib
    """
    if not distributions:
        logger.warning("No hay distribuciones para comparar")
        return plt.figure()
    
    # Crear figura principal
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # 1. Gráfico de densidades KDE para comparar distribuiones
    ax1 = axes[0]
    
    # Paleta de colores
    colors = plt.cm.tab10.colors
    
    # Datos para leyenda y estadísticas
    legend_data = []
    
    # Crear un array común para todas las distribuciones (para el eje X)
    min_rating = min([d['min'] for d in distributions.values() if 'min' in d])
    max_rating = max([d['max'] for d in distributions.values() if 'max' in d])
    
    x = np.linspace(min_rating - 50, max_rating + 50, 1000)
    
    # Graficar KDE para cada distribución
    for i, (category, stats) in enumerate(distributions.items()):
        if 'mean' not in stats or 'std' not in stats or stats['count'] == 0:
            continue
            
        # Crear una distribución normal con los parámetros
        mean = stats['mean']
        std = stats['std']
        
        # Graficar distribución normal
        pdf = stats.norm.pdf(x, mean, std)
        
        # Normalizar para que todas las curvas tengan área = 1
        # Esto hace que sean comparables independientemente del número de jugadores
        pdf_norm = pdf / (pdf.sum() * (x[1] - x[0]))
        
        # Graficar con etiqueta
        count = stats['count']
        ax1.plot(x, pdf_norm, '-', color=colors[i % len(colors)], linewidth=2, 
                label=f"{category} (n={count}, μ={mean:.0f})")
        
        # Añadir línea vertical para la media
        ax1.axvline(x=mean, color=colors[i % len(colors)], linestyle='--', alpha=0.5)
        
        # Guardar para estadísticas
        legend_data.append({
            'category': category,
            'mean': mean,
            'std': std,
            'count': count,
            'color': colors[i % len(colors)]
        })
    
    # Configurar el primer gráfico
    ax1.set_title('Comparación de Distribuciones de Rating ELO', fontsize=16)
    ax1.set_xlabel('Rating ELO', fontsize=12)
    ax1.set_ylabel('Densidad', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Gráfico de barras para comparar estadísticas clave
    ax2 = axes[1]
    
    # Preparar datos para gráfico de barras
    categories = [d['category'] for d in legend_data]
    means = [d['mean'] for d in legend_data]
    colors_bar = [d['color'] for d in legend_data]
    
    # Gráfico de barras para medias
    bars = ax2.bar(categories, means, color=colors_bar, alpha=0.7)
    
    # Añadir valores numéricos a las barras
    for bar, mean, std in zip(bars, means, [d['std'] for d in legend_data]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{mean:.0f}±{std:.0f}", ha='center', va='bottom', fontsize=10)
    
    # Configurar el segundo gráfico
    ax2.set_title('Comparación de Ratings Medios', fontsize=14)
    ax2.set_ylabel('Rating ELO Medio', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Ajustar valores del eje Y para mejor visualización
    min_mean = min(means) - 50
    max_mean = max(means) + 100
    ax2.set_ylim(min_mean, max_mean)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico comparativo guardado en {save_path}")
    
    return fig

def plot_rating_evolution_by_date(historical_ratings: Dict[datetime, Dict[str, float]],
                               time_periods: List[datetime],
                               min_players: int = 10,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Genera un gráfico de evolución de la distribución de ratings a lo largo del tiempo.
    
    Args:
        historical_ratings: Diccionario con ratings por fecha {fecha: {player_id: rating}}
        time_periods: Lista de fechas para las cuales mostrar la distribución
        min_players: Número mínimo de jugadores para incluir un período
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        
    Returns:
        Objeto Figure de matplotlib
    """
    if not historical_ratings or not time_periods:
        logger.warning("No hay datos históricos suficientes para el gráfico")
        return plt.figure()
    
    # Crear figura principal
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    ax1 = axes[0]  # Para gráfico de violín / boxplot
    ax2 = axes[1]  # Para gráfico de evolución de medias
    
    # Recopilar datos para cada período
    plot_data = []
    stats_by_period = {}
    
    for date in time_periods:
        # Encontrar la fecha más cercana en los datos
        closest_date = min(historical_ratings.keys(), 
                         key=lambda d: abs((d - date).total_seconds()))
        
        ratings = historical_ratings.get(closest_date, {})
        
        if len(ratings) >= min_players:
            # Analizar distribución
            stats = analyze_ratings_distribution(ratings)
            stats_by_period[closest_date] = stats
            
            # Preparar datos para violín / boxplot
            for player_id, rating in ratings.items():
                plot_data.append({
                    'date': closest_date,
                    'rating': rating
                })
    
    if not plot_data or len(stats_by_period) < 2:
        logger.warning("Datos insuficientes para graficar evolución temporal")
        return plt.figure()
    
    # Convertir a DataFrame
    df = pd.DataFrame(plot_data)
    
    # 1. Gráfico de violín para mostrar distribuciones
    sns.violinplot(x='date', y='rating', data=df, ax=ax1, 
                  palette='viridis', inner='box', cut=0)
    
    # Configurar el primer gráfico
    ax1.set_title('Evolución de la Distribución de Ratings ELO', fontsize=16)
    ax1.set_xlabel('')  # Quitar etiqueta del eje X para evitar duplicidad
    ax1.set_ylabel('Rating ELO', fontsize=12)
    
    # Rotar etiquetas de fecha
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Gráfico de línea para la evolución de estadísticas clave
    dates = sorted(stats_by_period.keys())
    means = [stats_by_period[d]['mean'] for d in dates]
    p25 = [stats_by_period[d]['percentiles']['25'] for d in dates]
    p75 = [stats_by_period[d]['percentiles']['75'] for d in dates]
    
    # Graficar línea de medias
    ax2.plot(dates, means, 'o-', color='royalblue', linewidth=2, 
            label='Media')
    
    # Graficar área para los percentiles 25-75
    ax2.fill_between(dates, p25, p75, color='royalblue', alpha=0.2,
                    label='Rango 25-75%')
    
    # Configurar el segundo gráfico
    ax2.set_title('Evolución del Rating ELO Medio', fontsize=14)
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Rating ELO', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Rotar etiquetas de fecha
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de evolución temporal guardado en {save_path}")
    
    return fig