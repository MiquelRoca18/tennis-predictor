"""
visualization/player_comparison.py

Módulo para crear visualizaciones comparativas entre jugadores de tenis.
Permite comparar rendimiento, estadísticas y evolución histórica.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any, Set

# Configurar logger
logger = logging.getLogger(__name__)

def compare_players_radar(player_stats: Dict[str, Dict], 
                         metrics: List[str] = None,
                         title: str = "Comparación de Jugadores", 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Crea un gráfico de radar (spider/polar) para comparar múltiples jugadores en varias métricas.
    
    Args:
        player_stats: Diccionario con estadísticas por jugador {player_id: stats_dict}
        metrics: Lista de métricas a incluir (default: usa todas las comunes)
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    if not player_stats:
        logger.warning("No se proporcionaron estadísticas de jugadores para comparar")
        return plt.figure()  # Figura vacía
    
    # Si no se especifican métricas, usar las comunes a todos los jugadores
    if metrics is None:
        # Encontrar las claves comunes a todos los jugadores
        common_keys = set.intersection(*[set(stats.keys()) for stats in player_stats.values()])
        # Filtrar solo métricas numéricas
        metrics = [key for key in common_keys 
                  if any(isinstance(stats.get(key), (int, float)) 
                       for stats in player_stats.values())]
        
        # Si no hay métricas comunes, mostrar error
        if not metrics:
            logger.error("No se encontraron métricas comunes para comparar")
            return plt.figure()
    
    # Normalizar valores para comparar en la misma escala (0-1)
    normalized_stats = {}
    
    for metric in metrics:
        # Obtener valores para todos los jugadores
        values = [stats.get(metric, 0) for stats in player_stats.values() 
                 if isinstance(stats.get(metric), (int, float))]
        
        if not values:
            continue
            
        min_val = min(values)
        max_val = max(values)
        
        # Evitar división por cero
        range_val = max_val - min_val
        
        for player_id, stats in player_stats.items():
            if player_id not in normalized_stats:
                normalized_stats[player_id] = {}
                
            # Normalizar solo si es numérico
            if isinstance(stats.get(metric), (int, float)):
                if range_val > 0:
                    normalized_stats[player_id][metric] = (stats.get(metric, 0) - min_val) / range_val
                else:
                    normalized_stats[player_id][metric] = 0.5  # Valor medio si no hay rango
            else:
                normalized_stats[player_id][metric] = 0  # Valor por defecto
    
    # Crear gráfico polar
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Número de variables
    N = len(metrics)
    
    # Ángulos para cada variable (en radianes)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Cerrar el gráfico repitiendo el primer ángulo
    angles += angles[:1]
    
    # Configurar etiquetas
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Configurar límites y grillas
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    # Paleta de colores
    colors = plt.cm.tab10.colors
    
    # Graficar cada jugador
    for i, (player_id, stats) in enumerate(normalized_stats.items()):
        values = [stats.get(metric, 0) for metric in metrics]
        # Cerrar el polígono repitiendo el primer valor
        values += values[:1]
        
        # Dibujar el polígono
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
               label=player_id, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # Añadir título y leyenda
    plt.title(title, size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def plot_elo_vs_win_rate(processor, player_ids: List[str], 
                       surface: Optional[str] = None,
                       min_matches: int = 10,
                       player_names: Optional[Dict[str, str]] = None,
                       highlight_players: Optional[List[str]] = None,
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Crea un gráfico de dispersión que muestra la relación entre ELO y win rate,
    resaltando los jugadores seleccionados.
    
    Args:
        processor: Instancia del procesador ELO que contiene datos de jugadores
        player_ids: Lista de IDs de jugadores a resaltar específicamente
        surface: Superficie específica para análisis (opcional)
        min_matches: Número mínimo de partidos jugados para incluir jugadores
        player_names: Diccionario de mapeo de IDs a nombres
        highlight_players: Lista adicional de jugadores a resaltar
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Verificar que hay datos de jugadores
    if not hasattr(processor, 'player_match_count') or not processor.player_match_count:
        logger.warning("No hay datos de partidos jugados disponibles")
        return plt.figure()
    
    # Combinar las listas de jugadores a resaltar
    if highlight_players:
        players_to_highlight = set(player_ids).union(set(highlight_players))
    else:
        players_to_highlight = set(player_ids)
    
    # Preparar datos para el gráfico
    scatter_data = []
    
    # Si se especifica una superficie, usar datos específicos
    if surface:
        surface = surface.lower()
        # Asegurar que es una superficie válida
        if surface not in processor.player_ratings_by_surface:
            logger.warning(f"Superficie '{surface}' no encontrada, usando datos generales")
            surface = None
    
    # Recopilar datos de todos los jugadores con suficientes partidos
    for player_id, match_count in processor.player_match_count.items():
        if match_count < min_matches:
            continue
        
        # Determinar si este jugador debe ser resaltado
        is_highlighted = player_id in players_to_highlight
        
        # Obtener nombre del jugador
        player_name = player_names.get(player_id, player_id) if player_names else player_id
        
        # Rating ELO
        if surface:
            elo = processor.get_player_rating(player_id, surface)
            matches_surface = processor.player_match_count_by_surface[surface].get(player_id, 0)
            
            # Si no hay suficientes partidos en esta superficie, omitir
            if matches_surface < min_matches:
                continue
        else:
            elo = processor.get_player_rating(player_id)
        
        # Obtener win rate
        if hasattr(processor, 'player_match_history') and processor.player_match_history:
            matches = processor.player_match_history.get(player_id, [])
            if surface:
                # Filtrar por superficie
                matches = [m for m in matches if 'surface' in m and m['surface'] == surface]
            
            total = len(matches)
            if total >= min_matches:
                wins = sum(1 for m in matches if m['result'] == 'win')
                win_rate = wins / total
                
                scatter_data.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'elo': elo,
                    'win_rate': win_rate,
                    'matches': total,
                    'is_highlighted': is_highlighted
                })
        elif hasattr(processor, 'get_player_details'):
            # Alternativa usando get_player_details
            details = processor.get_player_details(player_id)
            
            if 'stats' in details:
                # Datos específicos por superficie
                if 'by_surface' in details['stats'] and surface in details['stats']['by_surface']:
                    surface_stats = details['stats']['by_surface'][surface]
                    if 'matches' in surface_stats and surface_stats['matches'] >= min_matches:
                        win_rate = surface_stats.get('win_rate', 0)
                        matches = surface_stats.get('matches', 0)
                        
                        scatter_data.append({
                            'player_id': player_id,
                            'player_name': player_name,
                            'elo': elo,
                            'win_rate': win_rate,
                            'matches': matches,
                            'is_highlighted': is_highlighted
                        })
                else:
                    # Datos generales
                    win_rate = details['stats'].get('win_rate', 0)
                    matches = details['stats'].get('total_matches', 0)
                    
                    if matches >= min_matches:
                        scatter_data.append({
                            'player_id': player_id,
                            'player_name': player_name,
                            'elo': elo,
                            'win_rate': win_rate,
                            'matches': matches,
                            'is_highlighted': is_highlighted
                        })
        else:
            # Si no hay más datos, intentar con contadores básicos de partidos
            wins = processor.player_recent_form.get(player_id, {}).get('wins', 0)
            if match_count > 0 and 'wins' in processor.player_recent_form.get(player_id, {}):
                win_rate = wins / match_count
                
                scatter_data.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'elo': elo,
                    'win_rate': win_rate,
                    'matches': match_count,
                    'is_highlighted': is_highlighted
                })
    
    if not scatter_data:
        logger.warning("No se encontraron datos suficientes para el gráfico")
        return plt.figure()
    
    # Convertir a DataFrame
    df = pd.DataFrame(scatter_data)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # Tamaño de punto base según número de partidos
    sizes = df['matches'].map(lambda x: 30 + (x / df['matches'].max() * 150))
    
    # Gráfico de dispersión principal (todos los jugadores)
    scatter = ax.scatter(df[~df['is_highlighted']]['elo'], 
                       df[~df['is_highlighted']]['win_rate'],
                       s=sizes[~df['is_highlighted']],
                       alpha=0.4,
                       color='gray',
                       edgecolor='darkgray')
    
    # Resaltar jugadores seleccionados
    highlighted = df[df['is_highlighted']]
    if not highlighted.empty:
        colors = plt.cm.tab10.colors
        
        for i, (_, row) in enumerate(highlighted.iterrows()):
            ax.scatter(row['elo'], row['win_rate'],
                     s=30 + (row['matches'] / df['matches'].max() * 200),
                     color=colors[i % len(colors)],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.5)
            
            # Añadir etiqueta con nombre
            ax.text(row['elo'], row['win_rate'] + 0.01,
                  row['player_name'],
                  ha='center', va='bottom',
                  fontsize=10, fontweight='bold',
                  color=colors[i % len(colors)])
    
    # Añadir línea de regresión para ver tendencia
    try:
        sns.regplot(x='elo', y='win_rate', data=df, 
                  scatter=False, ax=ax, 
                  line_kws={'color': 'red', 'linestyle': '--', 'alpha': 0.7})
    except Exception as e:
        logger.warning(f"No se pudo añadir línea de regresión: {str(e)}")
    
    # Configurar gráfico
    if surface:
        title = f'Relación entre Rating ELO y Porcentaje de Victorias en {surface.upper()}'
    else:
        title = 'Relación entre Rating ELO y Porcentaje de Victorias'
    
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Rating ELO', fontsize=12)
    ax.set_ylabel('Porcentaje de Victorias', fontsize=12)
    
    # Formato del eje Y como porcentaje
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    # Añadir grid
    ax.grid(True, alpha=0.3)
    
    # Añadir leyenda para tamaño de puntos
    # Crear puntos de referencia para la leyenda
    sizes_legend = [min(df['matches']), 
                   df['matches'].quantile(0.5), 
                   max(df['matches'])]
    
    # Redondear a enteros
    sizes_legend = [int(s) for s in sizes_legend]
    
    # Crear leyenda para tamaños
    for size, matches in zip([30, 100, 180], sizes_legend):
        ax.scatter([], [], s=size, color='gray', alpha=0.5,
                 label=f'{matches} partidos')
    
    # Añadir leyenda con ubicación personalizada
    ax.legend(title="Partidos jugados", 
            loc='upper left', 
            scatterpoints=1,
            frameon=True,
            fontsize=9)
    
    # Ajustar márgenes
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def compare_h2h_stats(h2h_data: Dict, 
                     player_names: Optional[Dict[str, str]] = None,
                     min_matches: int = 1,
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Crea un gráfico comparativo de estadísticas head-to-head entre jugadores.
    
    Args:
        h2h_data: Diccionario con datos h2h {player1_id: {player2_id: {wins: X, losses: Y}}}
        player_names: Diccionario de mapeo de IDs a nombres {player_id: name}
        min_matches: Número mínimo de partidos para incluir en la comparación
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    if not h2h_data:
        logger.warning("No se proporcionaron datos head-to-head para comparar")
        return plt.figure()
    
    # Extraer datos para el gráfico
    matchups = []
    
    for player1_id, opponents in h2h_data.items():
        player1_name = player_names.get(player1_id, player1_id) if player_names else player1_id
        
        for player2_id, record in opponents.items():
            player2_name = player_names.get(player2_id, player2_id) if player_names else player2_id
            
            # Verificar si es un diccionario con estructura esperada
            if isinstance(record, dict) and 'wins' in record and 'losses' in record:
                wins = record.get('wins', 0)
                losses = record.get('losses', 0)
                total = wins + losses
                
                # Solo incluir si tienen suficientes partidos
                if total >= min_matches:
                    win_rate = wins / total if total > 0 else 0
                    
                    matchups.append({
                        'player1_id': player1_id,
                        'player1_name': player1_name,
                        'player2_id': player2_id,
                        'player2_name': player2_name,
                        'wins': wins,
                        'losses': losses,
                        'total': total,
                        'win_rate': win_rate
                    })
    
    if not matchups:
        logger.warning(f"No se encontraron enfrentamientos con al menos {min_matches} partidos")
        return plt.figure()
    
    # Convertir a DataFrame para facilitar el manejo
    df = pd.DataFrame(matchups)
    
    # Ordenar por número total de partidos
    df = df.sort_values('total', ascending=False)
    
    # Limitar a los 15 enfrentamientos con más partidos
    if len(df) > 15:
        df = df.head(15)
    
    # Crear etiquetas para el eje Y
    if player_names:
        labels = [f"{row['player1_name']} vs {row['player2_name']}" for _, row in df.iterrows()]
    else:
        labels = [f"{row['player1_id']} vs {row['player2_id']}" for _, row in df.iterrows()]
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Barras apiladas para victorias y derrotas
    bar_height = 0.8
    
    # Crear barras
    ax.barh(range(len(df)), df['wins'], bar_height, label='Victorias', color='mediumseagreen')
    ax.barh(range(len(df)), df['losses'], bar_height, left=df['wins'], label='Derrotas', color='indianred')
    
    # Añadir etiquetas con porcentajes
    for i, (_, row) in enumerate(df.iterrows()):
        win_rate = row['win_rate'] * 100
        
        # Posición para el texto
        wins_width = row['wins']
        
        # Añadir porcentaje en la barra de victorias
        if wins_width > 0:
            ax.text(wins_width / 2, i, f"{win_rate:.0f}%", 
                   ha='center', va='center', color='white', fontweight='bold')
        
        # Añadir texto con número total de partidos
        ax.text(row['total'] + 0.5, i, f"({row['total']} partidos)", 
               ha='left', va='center', color='black')
    
    # Configurar ejes
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Número de partidos')
    ax.invert_yaxis()  # Para que el orden sea de arriba hacia abajo
    
    # Añadir título y leyenda
    plt.title('Comparación Head-to-Head', fontsize=15)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def compare_tournament_performance(player_stats: Dict[str, Dict], 
                                 player_names: Optional[Dict[str, str]] = None, 
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Compara el rendimiento de múltiples jugadores por nivel de torneo.
    
    Args:
        player_stats: Diccionario con estadísticas por jugador {player_id: stats_dict}
        player_names: Diccionario de mapeo de IDs a nombres {player_id: name}
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    if not player_stats:
        logger.warning("No se proporcionaron estadísticas de jugadores para comparar")
        return plt.figure()
    
    # Preparar datos para el gráfico
    tourney_data = []
    
    for player_id, stats in player_stats.items():
        player_name = player_names.get(player_id, player_id) if player_names else player_id
        
        # Extraer estadísticas por nivel de torneo si están disponibles
        if 'stats' in stats and 'by_tourney_level' in stats['stats']:
            tourney_stats = stats['stats']['by_tourney_level']
            
            for level, level_data in tourney_stats.items():
                if isinstance(level_data, dict):
                    matches = level_data.get('matches', 0)
                    win_rate = level_data.get('win_rate', 0)
                    
                    # Solo incluir si hay partidos
                    if matches > 0:
                        tourney_data.append({
                            'player_id': player_id,
                            'player_name': player_name,
                            'tourney_level': level,
                            'matches': matches,
                            'win_rate': win_rate,
                            'wins': level_data.get('wins', 0),
                            'losses': level_data.get('losses', 0)
                        })
    
    if not tourney_data:
        logger.warning("No se encontraron datos de rendimiento por nivel de torneo")
        return plt.figure()
    
    # Convertir a DataFrame
    df = pd.DataFrame(tourney_data)
    
    # Mapear códigos de torneo a nombres más descriptivos
    tourney_name_mapping = {
        'G': 'Grand Slam',
        'M': 'Masters 1000',
        'A': 'ATP 500',
        'D': 'ATP 250',
        'F': 'Tour Finals',
        'C': 'Challenger',
        'S': 'Satellite/ITF',
        'O': 'Other'
    }
    
    # Añadir nombres descriptivos
    df['tourney_name'] = df['tourney_level'].map(lambda x: tourney_name_mapping.get(x, x))
    
    # Ordenar niveles por importancia
    level_order = ['G', 'F', 'M', 'A', 'D', 'C', 'S', 'O']
    
    # Asignar valor numérico para ordenar
    df['level_rank'] = df['tourney_level'].map(lambda x: level_order.index(x) if x in level_order else 999)
    df = df.sort_values('level_rank')
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 2])
    
    # 1. Gráfico de barras para win rate por nivel de torneo
    levels = df['tourney_level'].unique()
    players = df['player_id'].unique()
    
    # Obtener nombres de torneos en el orden correcto
    level_names = [tourney_name_mapping.get(level, level) for level in levels]
    
    bar_width = 0.8 / len(players)
    positions = np.arange(len(levels))
    
    # Paleta de colores
    colors = plt.cm.tab10.colors
    
    # Graficar win rate por jugador y nivel de torneo
    for i, player_id in enumerate(players):
        player_name = player_names.get(player_id, player_id) if player_names else player_id
        player_data = df[df['player_id'] == player_id]
        
        # Preparar datos para cada nivel
        win_rates = []
        for level in levels:
            level_row = player_data[player_data['tourney_level'] == level]
            win_rate = level_row['win_rate'].values[0] if not level_row.empty else 0
            win_rates.append(win_rate)
        
        # Posición de las barras
        bar_positions = positions + (i * bar_width) - (bar_width * (len(players) - 1) / 2)
        
        # Graficar barras
        bars = ax1.bar(bar_positions, win_rates, bar_width, label=player_name, 
                     color=colors[i % len(colors)])
        
        # Añadir etiquetas de porcentaje
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, height,
                       f"{height*100:.0f}%", ha='center', va='bottom', 
                       fontsize=9, color='black')
    
    # Configurar primer gráfico
    ax1.set_ylabel('Porcentaje de victorias')
    ax1.set_title('Comparación de rendimiento por nivel de torneo', fontsize=15)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(level_names)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Gráfico de barras para número de partidos jugados por nivel
    for i, player_id in enumerate(players):
        player_name = player_names.get(player_id, player_id) if player_names else player_id
        player_data = df[df['player_id'] == player_id]
        
        # Preparar datos para cada nivel
        matches = []
        for level in levels:
            level_row = player_data[player_data['tourney_level'] == level]
            match_count = level_row['matches'].values[0] if not level_row.empty else 0
            matches.append(match_count)
        
        # Posición de las barras
        bar_positions = positions + (i * bar_width) - (bar_width * (len(players) - 1) / 2)
        
        # Graficar barras
        bars = ax2.bar(bar_positions, matches, bar_width, label=player_name, 
                     color=colors[i % len(colors)])
        
        # Añadir etiquetas con número de partidos
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, height,
                       f"{int(height)}", ha='center', va='bottom', 
                       fontsize=9, color='black')
    
    # Configurar segundo gráfico
    ax2.set_ylabel('Número de partidos')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(level_names)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def compare_player_form_trends(processor, player_ids: List[str], 
                             window_size: int = 10,
                             player_names: Optional[Dict[str, str]] = None,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Compara las tendencias de forma (rachas) entre varios jugadores.
    
    Args:
        processor: Instancia del procesador ELO que contiene datos de partidos
        player_ids: Lista de IDs de jugadores a comparar
        window_size: Tamaño de la ventana para calcular tendencias
        player_names: Diccionario de mapeo de IDs a nombres
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Verificar que hay datos de partidos por jugador
    if not hasattr(processor, 'player_match_history') or not processor.player_match_history:
        logger.warning("No hay datos de historial de partidos disponibles")
        return plt.figure()
    
    # Preparar datos para el gráfico
    form_data = []
    
    for player_id in player_ids:
        matches = processor.player_match_history.get(player_id, [])
        
        if not matches:
            logger.warning(f"No hay datos para el jugador {player_id}")
            continue
        
        # Ordenar partidos por fecha
        sorted_matches = sorted(matches, key=lambda x: x['date'] if 'date' in x else datetime.min)
        
        # Calcular forma móvil (rachas)
        for i in range(len(sorted_matches)):
            # Tomar ventana de partidos anteriores
            start_idx = max(0, i - window_size + 1)
            window = sorted_matches[start_idx:i+1]
            
            if not window:
                continue
                
            # Calcular ratio de victorias en la ventana
            wins = sum(1 for m in window if m['result'] == 'win')
            win_ratio = wins / len(window)
            
            # Añadir punto de datos
            form_data.append({
                'player_id': player_id,
                'date': window[-1]['date'],  # Fecha del partido más reciente
                'win_ratio': win_ratio,
                'window_size': len(window)
            })
    
    if not form_data:
        logger.warning("No se pudieron calcular tendencias de forma")
        return plt.figure()
    
    # Convertir a DataFrame
    df = pd.DataFrame(form_data)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Paleta de colores
    colors = plt.cm.tab10.colors
    
    # Graficar tendencia para cada jugador
    for i, player_id in enumerate(player_ids):
        player_data = df[df['player_id'] == player_id]
        
        if player_data.empty:
            continue
            
        # Ordenar por fecha
        player_data = player_data.sort_values('date')
        
        # Nombre para la leyenda
        label = player_names.get(player_id, player_id) if player_names else player_id
        
        # Graficar línea
        ax.plot(player_data['date'], player_data['win_ratio'], '-o', 
               label=label, color=colors[i % len(colors)], 
               markersize=4, alpha=0.8)
    
    # Configurar gráfico
    ax.set_title(f'Tendencias de forma (ventana de {window_size} partidos)', fontsize=15)
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Ratio de victorias', fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Formatear eje X de fechas
    plt.xticks(rotation=45)
    
    # Añadir líneas de referencia
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(df['date'].min(), 0.51, '50%', color='gray', alpha=0.7)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def compare_ranking_evolution(processor, player_ids: List[str],
                             time_points: List[datetime] = None,
                             player_names: Optional[Dict[str, str]] = None,
                             save_path: Optional[str] = None, 
                             figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Compara la evolución del ranking de los jugadores en puntos específicos del tiempo.
    
    Args:
        processor: Instancia del procesador ELO que contiene datos de ratings
        player_ids: Lista de IDs de jugadores a comparar
        time_points: Lista de fechas para las cuales obtener ratings (si es None, usa fechas automáticas)
        player_names: Diccionario de mapeo de IDs a nombres
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Validar que hay suficientes datos
    if not hasattr(processor, 'rating_history') or not processor.rating_history:
        logger.warning("No hay datos de historial disponibles para comparar evolución")
        return plt.figure()
    
    # Convertir historial a DataFrame si es necesario
    if isinstance(processor.rating_history, list):
        history_df = pd.DataFrame(processor.rating_history)
    else:
        history_df = processor.rating_history
    
    # Asegurar que la fecha es datetime
    if 'date' in history_df.columns and not pd.api.types.is_datetime64_any_dtype(history_df['date']):
        history_df['date'] = pd.to_datetime(history_df['date'])
    
    # Si no se proporcionan fechas, crearlas automáticamente
    if not time_points:
        if 'date' in history_df.columns:
            # Tomar fechas equidistantes entre min y max
            min_date = history_df['date'].min()
            max_date = history_df['date'].max()
            
            # Crear 6 fechas equidistantes
            delta = (max_date - min_date) / 5
            time_points = [min_date + delta * i for i in range(6)]
        else:
            logger.warning("No se pueden generar fechas automáticas")
            return plt.figure()
    
    # Preparar datos para el gráfico
    rankings = []
    
    for date_point in time_points:
        # Filtrar registros anteriores a esta fecha
        hist_until_date = history_df[history_df['date'] <= date_point]
        
        if hist_until_date.empty:
            continue
        
        # Para cada jugador, obtener su último rating antes de esta fecha
        for player_id in player_ids:
            # Ratings cuando ganó
            win_ratings = hist_until_date[hist_until_date['winner_id'] == player_id]
            # Ratings cuando perdió
            lose_ratings = hist_until_date[hist_until_date['loser_id'] == player_id]
            
            if not win_ratings.empty:
                # Tomar el más reciente
                latest_win = win_ratings.sort_values('date').iloc[-1]
                win_elo = latest_win['winner_rating_after']
                win_date = latest_win['date']
                
                if not lose_ratings.empty:
                    latest_lose = lose_ratings.sort_values('date').iloc[-1]
                    lose_date = latest_lose['date']
                    
                    # Usar el más reciente entre ambos
                    if lose_date > win_date:
                        elo = latest_lose['loser_rating_after']
                        actual_date = lose_date
                    else:
                        elo = win_elo
                        actual_date = win_date
                else:
                    elo = win_elo
                    actual_date = win_date
            elif not lose_ratings.empty:
                # Solo hay ratings de derrotas
                latest_lose = lose_ratings.sort_values('date').iloc[-1]
                elo = latest_lose['loser_rating_after']
                actual_date = latest_lose['date']
            else:
                # No hay ratings para este jugador antes de esta fecha
                continue
            
            # Añadir a la lista
            player_name = player_names.get(player_id, player_id) if player_names else player_id
            
            rankings.append({
                'player_id': player_id,
                'player_name': player_name,
                'date_point': date_point,
                'actual_date': actual_date,
                'elo': elo
            })
    
    if not rankings:
        logger.warning("No se pudieron obtener datos de evolución de ranking")
        return plt.figure()
    
    # Convertir a DataFrame
    rankings_df = pd.DataFrame(rankings)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # Ordenar jugadores por ELO final (último punto de tiempo)
    final_rankings = rankings_df[rankings_df['date_point'] == time_points[-1]]
    player_order = final_rankings.sort_values('elo', ascending=False)['player_id']
    
    # Paleta de colores
    colors = plt.cm.tab10.colors
    
    # Crear un diccionario para asignar colores consistentes
    color_dict = {player_id: colors[i % len(colors)] for i, player_id in enumerate(player_order)}
    
    # Graficar líneas de evolución para cada jugador
    for player_id in player_ids:
        player_data = rankings_df[rankings_df['player_id'] == player_id]
        
        if player_data.empty:
            continue
            
        # Ordenar por fecha
        player_data = player_data.sort_values('date_point')
        
        # Nombre para la leyenda
        label = player_names.get(player_id, player_id) if player_names else player_id
        
        # Color consistente para este jugador
        color = color_dict.get(player_id, colors[player_ids.index(player_id) % len(colors)])
        
        # Graficar línea con marcadores
        line = ax.plot(player_data['date_point'], player_data['elo'], '-o', 
                     label=label, color=color, 
                     markersize=8, linewidth=2)
        
        # Añadir etiquetas con valor de ELO
        for _, row in player_data.iterrows():
            ax.text(row['date_point'], row['elo'] + 5, 
                  f"{row['elo']:.0f}", 
                  ha='center', va='bottom', color=color, fontsize=9)
    
    # Configurar gráfico
    ax.set_title('Evolución del Rating ELO', fontsize=16)
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Rating ELO', fontsize=12)
    
    # Formatear eje X de fechas
    date_formatter = ticker.DateFormatter('%b %Y')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    
    # Ajustar rangos de ejes para mejor visualización
    elo_min = rankings_df['elo'].min() - 20
    elo_max = rankings_df['elo'].max() + 20
    
    ax.set_ylim(elo_min, elo_max)
    
    # Añadir leyenda ordenada por ELO final
    handles, labels = ax.get_legend_handles_labels()
    if player_order.size > 0:
        # Crear mapeo de etiquetas a índices
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        # Reordenar handles y labels según player_order
        ordered_indices = []
        for player_id in player_order:
            player_label = player_names.get(player_id, player_id) if player_names else player_id
            if player_label in label_to_idx:
                ordered_indices.append(label_to_idx[player_label])
        
        # Si tenemos índices válidos, reordenar
        if ordered_indices:
            handles = [handles[i] for i in ordered_indices]
            labels = [labels[i] for i in ordered_indices]
    
    ax.legend(handles, labels, loc='best', fontsize=10)
    
    # Añadir grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def compare_player_attributes(player_data: Dict[str, Dict], 
                            attributes: List[str],
                            player_names: Optional[Dict[str, str]] = None,
                            normalize: bool = True,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Compara atributos específicos entre jugadores usando gráfico de barras.
    
    Args:
        player_data: Diccionario con datos por jugador {player_id: data_dict}
        attributes: Lista de atributos a comparar
        player_names: Diccionario de mapeo de IDs a nombres
        normalize: Si se deben normalizar los valores para comparar en la misma escala
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    if not player_data or not attributes:
        logger.warning("No se proporcionaron datos o atributos para comparar")
        return plt.figure()
    
    # Extraer valores para cada atributo y jugador
    comparison_data = []
    
    for player_id, data in player_data.items():
        player_name = player_names.get(player_id, player_id) if player_names else player_id
        
        for attr in attributes:
            # Extraer valor del atributo con navegación de diccionario anidado
            value = None
            current_dict = data
            
            # Dividir la ruta del atributo si contiene puntos
            attr_parts = attr.split('.')
            
            for part in attr_parts:
                if isinstance(current_dict, dict) and part in current_dict:
                    current_dict = current_dict[part]
                else:
                    break
            
            # Si llegamos al final, el valor es lo que tenemos
            if attr_parts[-1] in current_dict or (len(attr_parts) == 1 and isinstance(current_dict, (int, float))):
                value = current_dict
            
            # Solo añadir si es un valor numérico
            if isinstance(value, (int, float)) and not pd.isna(value):
                comparison_data.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'attribute': attr,
                    'value': value
                })
    
    if not comparison_data:
        logger.warning("No se pudieron extraer atributos numéricos para comparar")
        return plt.figure()
    
    # Convertir a DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Normalizar valores si se solicita
    if normalize:
        # Agrupar por atributo y calcular min/max
        attr_stats = df.groupby('attribute').agg({'value': ['min', 'max']})
        attr_stats.columns = ['min', 'max']
        
        # Unir con DataFrame original
        df = df.join(attr_stats, on='attribute')
        
        # Calcular valor normalizado
        df['range'] = df['max'] - df['min']
        df['norm_value'] = df.apply(
            lambda row: (row['value'] - row['min']) / row['range'] if row['range'] > 0 else 0.5, 
            axis=1
        )
        
        # Usar valor normalizado para graficación
        value_col = 'norm_value'
    else:
        value_col = 'value'
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Identificar jugadores únicos
    players = df['player_id'].unique()
    attributes = df['attribute'].unique()
    
    # Configurar posiciones de barras
    bar_width = 0.8 / len(players)
    positions = np.arange(len(attributes))
    
    # Paleta de colores
    colors = plt.cm.tab10.colors
    
    # Graficar barras para cada jugador
    for i, player_id in enumerate(players):
        player_name = player_names.get(player_id, player_id) if player_names else player_id
        player_df = df[df['player_id'] == player_id]
        
        # Preparar valores para cada atributo
        values = []
        for attr in attributes:
            attr_row = player_df[player_df['attribute'] == attr]
            value = attr_row[value_col].values[0] if not attr_row.empty else 0
            values.append(value)
        
        # Posiciones de las barras
        bar_positions = positions + (i * bar_width) - (bar_width * (len(players) - 1) / 2)
        
        # Graficar barras
        bars = ax.bar(bar_positions, values, bar_width, label=player_name, 
                    color=colors[i % len(colors)])
        
        # Añadir etiquetas con valores originales
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                # Obtener valor original
                attr = attributes[j]
                orig_value = player_df[player_df['attribute'] == attr]['value'].values[0]
                
                # Formatear según magnitud
                if orig_value >= 1000:
                    label = f"{orig_value:.0f}"
                elif orig_value >= 100:
                    label = f"{orig_value:.1f}"
                else:
                    label = f"{orig_value:.2f}"
                
                ax.text(bar.get_x() + bar.get_width()/2, height,
                      label, ha='center', va='bottom', 
                      fontsize=9, color='black')
    
    # Configurar gráfico
    ax.set_title('Comparación de atributos de jugadores', fontsize=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(attributes, rotation=45, ha='right')
    
    if normalize:
        ax.set_ylabel('Valor normalizado')
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel('Valor')
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def compare_win_rate_by_surface(df: pd.DataFrame, 
                               player_names: Optional[Dict[str, str]] = None,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Compara el win rate por superficie entre jugadores.
    
    Args:
        df: DataFrame con datos de jugadores por superficie
        player_names: Diccionario de mapeo de IDs a nombres
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    if df.empty:
        logger.warning("No se proporcionaron datos para comparar")
        return plt.figure()
        
    surfaces = df['surface'].unique()
    players = df['player_id'].unique()
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 2])
    
    bar_width = 0.8 / len(players)
    positions = np.arange(len(surfaces))
    
    # Paleta de colores
    colors = plt.cm.tab10.colors
    
    # Graficar win rate por jugador y superficie
    for i, player_id in enumerate(players):
        player_name = player_names.get(player_id, player_id) if player_names else player_id
        player_data = df[df['player_id'] == player_id]
        
        # Preparar datos para cada superficie
        win_rates = []
        for surface in surfaces:
            surface_row = player_data[player_data['surface'] == surface]
            win_rate = surface_row['win_rate'].values[0] if not surface_row.empty else 0
            win_rates.append(win_rate)
        
        # Posición de las barras
        bar_positions = positions + (i * bar_width) - (bar_width * (len(players) - 1) / 2)
        
        # Graficar barras
        bars = ax1.bar(bar_positions, win_rates, bar_width, label=player_name, 
                     color=colors[i % len(colors)])
        
        # Añadir etiquetas de porcentaje
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, height,
                       f"{height*100:.0f}%", ha='center', va='bottom', 
                       fontsize=9, color='black')
    
    # Configurar primer gráfico
    ax1.set_ylabel('Porcentaje de victorias')
    ax1.set_title('Comparación de rendimiento por superficie', fontsize=15)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([s.capitalize() for s in surfaces])
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Gráfico de puntos para ELO por superficie
    if 'elo' in df.columns:
        # Graficar ELO por jugador y superficie
        for i, player_id in enumerate(players):
            player_name = player_names.get(player_id, player_id) if player_names else player_id
            player_data = df[df['player_id'] == player_id]
            
            # Preparar datos para cada superficie
            elos = []
            for surface in surfaces:
                surface_row = player_data[player_data['surface'] == surface]
                elo = surface_row['elo'].values[0] if not surface_row.empty and 'elo' in surface_row else 1500
                elos.append(elo)
            
            # Graficar líneas y puntos
            ax2.plot(positions, elos, 'o-', label=player_name, 
                   color=colors[i % len(colors)], markersize=8)
            
            # Añadir etiquetas con valor de ELO
            for j, pos in enumerate(positions):
                ax2.text(pos, elos[j] + 10, f"{elos[j]:.0f}", 
                       ha='center', va='bottom', fontsize=8, 
                       color=colors[i % len(colors)])
        
        # Configurar segundo gráfico
        ax2.set_ylabel('Rating ELO')
        ax2.set_xticks(positions)
        ax2.set_xticklabels([s.capitalize() for s in surfaces])
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Ajustar rango del eje Y para mejor visualización
        elo_min = df['elo'].min() if 'elo' in df.columns else 1400
        elo_max = df['elo'].max() if 'elo' in df.columns else 1600
        
        # Añadir margen
        elo_min = max(1000, elo_min - 50)
        elo_max = min(2500, elo_max + 50)
        
        ax2.set_ylim(elo_min, elo_max)
    else:
        # Si no hay datos de ELO, ajustar el subplot
        ax2.set_visible(False)
        plt.subplots_adjust(bottom=0.15, top=0.9)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def compare_players_historical(processor, player_ids: List[str], 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            surface: Optional[str] = None,
                            metric: str = 'elo',
                            player_names: Optional[Dict[str, str]] = None,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Compara la evolución histórica de una métrica entre varios jugadores.
    
    Args:
        processor: Instancia del procesador ELO que contiene el historial
        player_ids: Lista de IDs de jugadores a comparar
        start_date: Fecha de inicio para el análisis (opcional)
        end_date: Fecha final para el análisis (opcional)
        surface: Superficie específica para análisis (opcional)
        metric: Métrica a comparar ('elo', 'form', etc.)
        player_names: Diccionario de mapeo de IDs a nombres {player_id: name}
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Validar que hay datos de historial
    if not hasattr(processor, 'rating_history') or not processor.rating_history:
        logger.warning("No hay datos de historial disponibles para comparar")
        return plt.figure()
    
    # Convertir historial a DataFrame si es necesario
    if isinstance(processor.rating_history, list):
        history_df = pd.DataFrame(processor.rating_history)
    else:
        history_df = processor.rating_history
    
    # Asegurar que la fecha es datetime
    if 'date' in history_df.columns and not pd.api.types.is_datetime64_any_dtype(history_df['date']):
        history_df['date'] = pd.to_datetime(history_df['date'])
    
    # Seleccionar columnas según métrica y superficie
    if metric == 'elo':
        if surface:
            win_col = 'winner_surface_rating_after'
            lose_col = 'loser_surface_rating_after'
            # Filtrar por superficie
            history_df = history_df[history_df['surface'] == surface]
        else:
            win_col = 'winner_rating_after'
            lose_col = 'loser_rating_after'
    else:
        logger.warning(f"Métrica '{metric}' no soportada, usando 'elo' por defecto")
        win_col = 'winner_rating_after'
        lose_col = 'loser_rating_after'
    
    # Filtrar por fechas si se especifican
    if start_date:
        history_df = history_df[history_df['date'] >= start_date]
    if end_date:
        history_df = history_df[history_df['date'] <= end_date]
    
    if history_df.empty:
        logger.warning("No hay datos para el rango de fechas o superficie especificado")
        return plt.figure()
    
    # Preparar datos para graficación
    plot_data = []
    
    for player_id in player_ids:
        # Obtener registros donde el jugador ganó
        win_data = history_df[history_df['winner_id'] == player_id].copy()
        win_data['value'] = win_data[win_col]
        win_data['player_id'] = player_id
        
        # Obtener registros donde el jugador perdió
        lose_data = history_df[history_df['loser_id'] == player_id].copy()
        lose_data['value'] = lose_data[lose_col]
        lose_data['player_id'] = player_id
        
        # Combinar los datos
        player_data = pd.concat([
            win_data[['date', 'player_id', 'value']],
            lose_data[['date', 'player_id', 'value']]
        ])
        
        # Ordenar por fecha
        player_data = player_data.sort_values('date')
        
        if not player_data.empty:
            plot_data.append(player_data)
    
    if not plot_data:
        logger.warning(f"No se encontraron datos para los jugadores {player_ids}")
        return plt.figure()
    
    # Combinar todos los datos
    all_data = pd.concat(plot_data, ignore_index=True)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # Paleta de colores
    colors = plt.cm.tab10.colors
    
    # Graficar evolución para cada jugador
    for i, player_id in enumerate(player_ids):
        player_data = all_data[all_data['player_id'] == player_id]
        
        if player_data.empty:
            continue
            
        # Nombre para la leyenda
        label = player_names.get(player_id, player_id) if player_names else player_id
        
        # Graficar línea principal
        ax.plot(player_data['date'], player_data['value'], '-o', 
               label=label, color=colors[i % len(colors)], 
               markersize=4, alpha=0.8)
        
        # Añadir línea de tendencia suavizada
        if len(player_data) >= 5:
            # Usar rolling mean para suavizar
            window = min(5, len(player_data) // 2)
            if window > 1:
                player_data['smooth'] = player_data['value'].rolling(window=window, center=True).mean()
                ax.plot(player_data['date'], player_data['smooth'], '-', 
                       color=colors[i % len(colors)], alpha=0.5, linewidth=2)
    
    # Configurar título y etiquetas
    if surface:
        title = f'Evolución de {metric.upper()} en superficie {surface.upper()}'
    else:
        title = f'Evolución de {metric.upper()} general'
    
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Fecha', fontsize=12)
    
    if metric == 'elo':
        ax.set_ylabel('Rating ELO', fontsize=12)
    else:
        ax.set_ylabel(metric.capitalize(), fontsize=12)
    
    # Añadir leyenda
    ax.legend(fontsize=10)
    
    # Formatear eje X de fechas
    plt.xticks(rotation=45)
    
    # Añadir grid
    ax.grid(True, alpha=0.3)
    
    # Ajustar rangos de ejes para mejor visualización
    values = all_data['value']
    y_min = max(1000, values.min() - 50) if metric == 'elo' else values.min() - 0.1
    y_max = min(2500, values.max() + 50) if metric == 'elo' else values.max() + 0.1
    
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def compare_tournament_performance(player_stats: Dict[str, Dict], 
                                 player_names: Optional[Dict[str, str]] = None, 
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Compara el rendimiento de múltiples jugadores por nivel de torneo.
    
    Args:
        player_stats: Diccionario con estadísticas por jugador {player_id: stats_dict}
        player_names: Diccionario de mapeo de IDs a nombres {player_id: name}
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    if not player_stats:
        logger.warning("No se proporcionaron estadísticas de jugadores para comparar")
        return plt.figure()
    
    # Preparar datos para el gráfico
    tourney_data = []
    
    for player_id, stats in player_stats.items():
        player_name = player_names.get(player_id, player_id) if player_names else player_id
        
        # Extraer estadísticas por nivel de torneo si están disponibles
        if 'stats' in stats and 'by_tourney_level' in stats['stats']:
            tourney_stats = stats['stats']['by_tourney_level']
            
            for level, level_data in tourney_stats.items():
                if isinstance(level_data, dict):
                    matches = level_data.get('matches', 0)
                    win_rate = level_data.get('win_rate', 0)
                    
                    # Solo incluir si hay partidos
                    if matches > 0:
                        tourney_data.append({
                            'player_id': player_id,
                            'player_name': player_name,
                            'tourney_level': level,
                            'matches': matches,
                            'win_rate': win_rate,
                            'wins': level_data.get('wins', 0),
                            'losses': level_data.get('losses', 0)
                        })
    
    if not tourney_data:
        logger.warning("No se encontraron datos de rendimiento por nivel de torneo")
        return plt.figure()
    
    # Convertir a DataFrame
    df = pd.DataFrame(tourney_data)
    
    # Mapear códigos de torneo a nombres más descriptivos
    tourney_name_mapping = {
        'G': 'Grand Slam',
        'M': 'Masters 1000',
        'A': 'ATP 500',
        'D': 'ATP 250',
        'F': 'Tour Finals',
        'C': 'Challenger',
        'S': 'Satellite/ITF',
        'O': 'Other'
    }
    
    # Añadir nombres descriptivos
    df['tourney_name'] = df['tourney_level'].map(lambda x: tourney_name_mapping.get(x, x))
    
    # Ordenar niveles por importancia
    level_order = ['G', 'F', 'M', 'A', 'D', 'C', 'S', 'O']
    
    # Asignar valor numérico para ordenar
    df['level_rank'] = df['tourney_level'].map(lambda x: level_order.index(x) if x in level_order else 999)
    df = df.sort_values('level_rank')
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 2])
    
    # 1. Gráfico de barras para