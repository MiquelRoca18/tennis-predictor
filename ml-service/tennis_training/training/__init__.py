"""
Tennis Prediction Training Package

This package provides comprehensive tools for training, evaluating, and 
maintaining tennis match prediction models with advanced features for 
domain-specific analysis and model explainability.
"""

# Import components for cross-validation
from .cross_validation import TemporalCrossValidator

# Import hyperparameter tuning components
from .hyperparameter_tuning import (
    HyperparameterTuner,
    XGBoostTuner,
    NeuralNetTuner,
    EnsembleTuner
)

# Import model training components
from .model_trainer import ModelTrainer

# Import data processing components
from .tennis_data_processor import TennisDataProcessor

# Import data synchronization components
from .tennis_data_sync import TennisDataSynchronizer

# Import domain analysis components
from .tennis_domain_analyzer import TennisDomainAnalyzer

# Import model explainability components
from .tennis_model_explainer import TennisModelExplainer

# Define package metadata
__version__ = "1.0.0"
__author__ = "Tennis Prediction Team"
__all__ = [
    # Cross-validation
    'TemporalCrossValidator',
    
    # Hyperparameter tuning
    'HyperparameterTuner',
    'XGBoostTuner',
    'NeuralNetTuner',
    'EnsembleTuner',
    
    # Model training
    'ModelTrainer',
    
    # Data processing
    'TennisDataProcessor',
    
    # Data synchronization
    'TennisDataSynchronizer',
    
    # Domain analysis
    'TennisDomainAnalyzer',
    
    # Model explainability
    'TennisModelExplainer'
]