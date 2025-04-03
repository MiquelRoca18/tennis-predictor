"""
visualization/evolution_plots.py

Módulo para crear visualizaciones de evolución temporal de ratings ELO y otros
parámetros de jugadores de tenis a lo largo del tiempo. Permite analizar
tendencias, comparar trayectorias y visualizar patrones históricos.
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

def plot_top_players_history(player_ids: List[str], 
                            rating_history: Union[List[Dict], pd.DataFrame],
                            start_date: Optional[datetime] = None, 
                            end_date: Optional[datetime] = None, 
                            surface: Optional[str] = None,
                            save_path: Optional[str] = None, 
                            player_names: Optional[Dict[str, str]] = None,
                            title: Optional[str] = None, 
                            figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Genera un gráfico con la evolución de ratings de los jugadores seleccionados.
    
    Args:
        player_ids: Lista de IDs de jugadores a incluir
        rating_history: Historial de ratings (DataFrame o lista de diccionarios)
        start_date: Fecha de inicio para el gráfico
        end_date: Fecha final para el gráfico
        surface: Superficie específica (opcional)
        save_path: Ruta para guardar el gráfico (opcional)
        player_names: Diccionario de {player_id: nombre} para etiquetas (opcional)
        title: Título personalizado para el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Convertir historial a DataFrame si no lo está ya
    if isinstance(rating_history, list):
        history_df = pd.DataFrame(rating_history)
    else:
        history_df = rating_history
            
    if history_df.empty:
        logger.warning("No hay datos de historial ELO disponibles para graficar")
        return plt.figure()
    
    # Asegurar que la fecha es datetime
    if 'date' in history_df.columns and not pd.api.types.is_datetime64_any_dtype(history_df['date']):
        history_df['date'] = pd.to_datetime(history_df['date'])
    
    # Columnas a usar según superficie
    if surface:
        winner_rating_col = 'winner_surface_rating_after'
        loser_rating_col = 'loser_surface_rating_after'
        
        # Filtrar por superficie
        history_df = history_df[history_df['surface'] == surface]
    else:
        winner_rating_col = 'winner_rating_after'
        loser_rating_col = 'loser_rating_after'
    
    # Preparar DataFrame para el gráfico
    plot_data = []
    
    for player_id in player_ids:
        player_id = str(player_id)
        
        # Datos cuando el jugador ganó
        winner_data = history_df[history_df['winner_id'] == player_id].copy()
        if not winner_data.empty:
            winner_data['rating'] = winner_data[winner_rating_col]
            winner_data['player_id'] = player_id
            plot_data.append(winner_data[['date', 'player_id', 'rating']])
        
        # Datos cuando el jugador perdió
        loser_data = history_df[history_df['loser_id'] == player_id].copy()
        if not loser_data.empty:
            loser_data['rating'] = loser_data[loser_rating_col]
            loser_data['player_id'] = player_id
            plot_data.append(loser_data[['date', 'player_id', 'rating']])
    
    if not plot_data:
        logger.warning(f"No hay datos disponibles para los jugadores {player_ids}")
        return plt.figure()
    
    # Concatenar todos los datos
    all_data = pd.concat(plot_data, ignore_index=True)
    
    # Filtrar por fechas si se especifican
    if start_date:
        all_data = all_data[all_data['date'] >= start_date]
    if end_date:
        all_data = all_data[all_data['date'] <= end_date]
    
    if all_data.empty:
        logger.warning("No hay datos para el rango de fechas especificado")
        return plt.figure()
            
    # Ordenar por fecha
    all_data = all_data.sort_values('date')
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # Paleta de colores más distintiva
    colors = plt.cm.tab10.colors
    
    # Graficar una línea por jugador
    for i, player_id in enumerate(player_ids):
        player_id = str(player_id)
        player_data = all_data[all_data['player_id'] == player_id]
        
        if not player_data.empty:
            # Nombre para la leyenda
            label = player_names.get(player_id, f'Jugador {player_id}') if player_names else f'Jugador {player_id}'
            
            # Añadir línea principal
            ax.plot(player_data['date'], player_data['rating'], '-o', 
                   label=label, color=colors[i % len(colors)], 
                   markersize=4, alpha=0.8)
            
            # Añadir línea de tendencia suavizada
            if len(player_data) >= 5:
                # Ordenar por fecha
                player_data = player_data.sort_values('date')
                
                # Usar rolling mean para suavizar
                window = min(5, len(player_data) // 2)
                if window > 1:
                    player_data['smooth_rating'] = player_data['rating'].rolling(window=window, center=True).mean()
                    ax.plot(player_data['date'], player_data['smooth_rating'], '-', 
                           color=colors[i % len(colors)], alpha=0.5, linewidth=2)
    
    # Añadir título y etiquetas
    if title:
        ax.set_title(title, fontsize=16)
    elif surface:
        ax.set_title(f'Evolución del rating ELO en superficie {surface.upper()}', fontsize=16)
    else:
        ax.set_title('Evolución del rating ELO general', fontsize=16)
    
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Rating ELO', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Ajustar eje Y para mostrar un rango razonable
    ratings = all_data['rating']
    min_rating = max(ratings.min() - 50, 1000)
    max_rating = min(ratings.max() + 50, 2500)
    ax.set_ylim(min_rating, max_rating)
    
    # Rotar etiquetas de fecha
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def plot_ranking_evolution(player_ids: List[str], player_ratings: Dict,
                         time_points: List[datetime],
                         player_names: Optional[Dict[str, str]] = None,
                         save_path: Optional[str] = None, 
                         figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Compara la evolución del ranking de los jugadores en puntos específicos del tiempo.
    
    Args:
        player_ids: Lista de IDs de jugadores a comparar
        player_ratings: Diccionario de ratings por tiempo {(player_id, time): rating}
        time_points: Lista de fechas para las cuales mostrar ratings
        player_names: Diccionario de mapeo de IDs a nombres
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Validar datos
    if not player_ids or not time_points or not player_ratings:
        logger.warning("Datos insuficientes para crear el gráfico de evolución de ranking")
        return plt.figure()
    
    # Preparar datos para el gráfico
    rankings = []
    
    for date_point in time_points:
        for player_id in player_ids:
            player_id = str(player_id)
            # Buscar el rating para este jugador y fecha
            rating = player_ratings.get((player_id, date_point))
            
            if rating is not None:
                # Añadir a la lista
                player_name = player_names.get(player_id, player_id) if player_names else player_id
                
                rankings.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'date_point': date_point,
                    'rating': rating
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
    
    # Ordenar jugadores por rating final (último punto de tiempo)
    final_rankings = rankings_df[rankings_df['date_point'] == time_points[-1]]
    player_order = final_rankings.sort_values('rating', ascending=False)['player_id']
    
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
        line = ax.plot(player_data['date_point'], player_data['rating'], '-o', 
                     label=label, color=color, 
                     markersize=8, linewidth=2)
        
        # Añadir etiquetas con valor de ELO
        for _, row in player_data.iterrows():
            ax.text(row['date_point'], row['rating'] + 5, 
                  f"{row['rating']:.0f}", 
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
    elo_min = rankings_df['rating'].min() - 20
    elo_max = rankings_df['rating'].max() + 20
    
    ax.set_ylim(elo_min, elo_max)
    
    # Añadir leyenda
    ax.legend(loc='best', fontsize=10)
    
    # Añadir grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    return fig

def plot_player_form_trends(player_ids: List[str], 
                          match_history: Dict[str, List[Dict]],
                          window_size: int = 10,
                          player_names: Optional[Dict[str, str]] = None,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Compara las tendencias de forma (rachas) entre varios jugadores.
    
    Args:
        player_ids: Lista de IDs de jugadores a comparar
        match_history: Diccionario con historial de partidos por jugador
        window_size: Tamaño de la ventana para calcular tendencias
        player_names: Diccionario de mapeo de IDs a nombres
        save_path: Ruta para guardar el gráfico (opcional)
        figsize: Tamaño de la figura (ancho, alto)
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Verificar que hay datos de partidos por jugador
    if not match_history:
        logger.warning("No hay datos de historial de partidos disponibles")
        return plt.figure()
    
    # Preparar datos para el gráfico
    form_data = []
    
    for player_id in player_ids:
        matches = match_history.get(player_id, [])
        
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

def plot_players_historical(player_ids: List[str], 
                         rating_history: Union[List[Dict], pd.DataFrame],
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
        player_ids: Lista de IDs de jugadores a comparar
        rating_history: Historial de ratings (DataFrame o lista de diccionarios)
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
    if not rating_history:
        logger.warning("No hay datos de historial disponibles para comparar")
        return plt.figure()
    
    # Convertir historial a DataFrame si es necesario
    if isinstance(rating_history, list):
        history_df = pd.DataFrame(rating_history)
    else:
        history_df = rating_history
    
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