"""
Tennis ELO Rating System Package
================================

This package provides a modular implementation of an ELO rating system
specifically designed for tennis matches analysis and prediction.

The system is organized into several modules:
- core: Main ELO processing and rating calculation
- utils: Utility functions for data handling
- visualization: Data visualization components
- analytics: Advanced analytics and evaluation
- io: Input/output operations for data persistence

For usage examples, see the documentation or the tennis_elo_main.py file.
"""

__version__ = '1.0.0'
__author__ = 'Tennis Analytics Team'

# Import main components for easier access
from .core.processor import EnhancedTennisEloProcessor