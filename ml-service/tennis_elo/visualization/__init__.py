"""
visualization/__init__.py

Módulo para la visualización de datos relacionados con el sistema ELO de tenis.
Proporciona funciones para generar gráficos de distribución, evolución y comparación
entre jugadores.
"""

# Importar funciones de visualización de distribuciones
from .distribution_plots import (
    plot_rating_distribution,
    analyze_ratings_distribution,
    plot_rating_distributions_comparison,
    plot_rating_evolution_by_date
)

# Importar funciones de visualización de evolución
from .evolution_plots import (
    plot_top_players_history,
    plot_ranking_evolution,
    plot_player_form_trends,
    plot_players_historical
)

# Importar funciones de comparación entre jugadores
from .player_comparison import (
    compare_players_radar,
    plot_elo_vs_win_rate,
    compare_h2h_stats,
    compare_tournament_performance,
    compare_player_form_trends,
    compare_ranking_evolution,
    compare_player_attributes,
    compare_win_rate_by_surface,
    compare_players_historical
)

# Definir qué funciones están disponibles al importar con *
__all__ = [
    # Distribuciones
    'plot_rating_distribution',
    'analyze_ratings_distribution',
    'plot_rating_distributions_comparison',
    'plot_rating_evolution_by_date',
    
    # Evolución
    'plot_top_players_history',
    'plot_ranking_evolution',
    'plot_player_form_trends',
    'plot_players_historical',
    
    # Comparación entre jugadores
    'compare_players_radar',
    'plot_elo_vs_win_rate',
    'compare_h2h_stats',
    'compare_tournament_performance',
    'compare_player_form_trends',
    'compare_ranking_evolution',
    'compare_player_attributes',
    'compare_win_rate_by_surface',
    'compare_players_historical'
]