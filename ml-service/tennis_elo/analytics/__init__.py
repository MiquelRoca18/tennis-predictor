"""
analytics

Paquete para el análisis estadístico y evaluación del sistema ELO de tenis.
Proporciona funciones para evaluar la calidad predictiva, generar características
para modelos de machine learning, y realizar análisis estadísticos detallados.

Módulos:
- evaluation: Evaluación de la capacidad predictiva del sistema ELO
- feature_engineering: Creación y preparación de características para modelos ML
- statistics: Análisis estadísticos detallados del sistema ELO
"""

# Importar funciones principales para facilitar el acceso
from .evaluation import (
    evaluate_predictive_power,
    calculate_calibration_curve,
    analyze_prediction_errors
)

from .feature_engineering import (
    create_features_for_model,
    create_balanced_dataset,
    feature_selection,
    generate_player_features
)

from .statistics import (
    analyze_ratings_distribution,
    get_elo_statistics_summary,
    analyze_surface_specialists,
    analyze_rating_trends,
    analyze_player_consistency,
    analyze_upset_patterns,
    analyze_h2h_dominance
)

# Versión del paquete
__version__ = '0.1.0'