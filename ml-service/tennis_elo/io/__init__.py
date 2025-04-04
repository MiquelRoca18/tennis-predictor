"""
io/__init__.py

Módulo de entrada/salida para el sistema ELO de tenis.
Proporciona funcionalidades para:
- Guardar y cargar datos del sistema
- Generar informes y reportes
"""

from .reporting import (
    EloReportGenerator,
    TextReportGenerator,
    generate_tournament_performance_report,
    generate_surface_comparison_report
)

from .persistence import (
    EloDataManager,
    create_backup,
    load_config,
    save_config
)

__all__ = [
    # Reporting
    'EloReportGenerator',
    'TextReportGenerator',
    'generate_tournament_performance_report',
    'generate_surface_comparison_report',
    
    # Persistence
    'EloDataManager',
    'create_backup',
    'load_config',
    'save_config'
]