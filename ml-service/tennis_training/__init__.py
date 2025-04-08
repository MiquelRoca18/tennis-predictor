#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tennis Match Prediction ML Training Package

This package contains a comprehensive set of tools for training
machine learning models to predict tennis match outcomes.

The package architecture is designed to support the entire ML workflow:
- Data loading and preprocessing
- Feature engineering and selection
- Model training and hyperparameter optimization
- Evaluation and analysis
- Model serialization and deployment

Module Structure:
- config: Configuration files for training and models
- features: Feature engineering and transformations
- models: ML model implementations
- training: Training pipeline components
- evaluation: Model evaluation tools
- utils: Utilities for data handling, logging, etc.

Usage:
    from ml_training import train_models
    train_models.main()  # Run the complete training pipeline

    # Or import specific components
    from ml_training.models import TennisEnsembleModel
    from ml_training.features import TennisFeatureExtractor
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'
__license__ = 'Proprietary'
__status__ = 'Production'

# Define public API
from .train_models import main as run_training

# Import main components for easier access
from .models.base_model import TennisModel
from .models.ensemble_model import TennisEnsembleModel
from .models.xgboost_model import TennisXGBoostModel
from .models.neural_net_model import TennisNeuralNetModel

from .features.feature_extractor import TennisFeatureExtractor
from .features.feature_transformer import FeatureTransformer
from .features.feature_selector import FeatureSelector

from .training.model_trainer import ModelTrainer
from .training.cross_validation import TemporalCrossValidator

from .evaluation.metrics import TennisMetrics
from .evaluation.results_analyzer import ResultsAnalyzer

# Define what gets imported with "from ml_training import *"
__all__ = [
    'run_training',
    'TennisModel',
    'TennisEnsembleModel',
    'TennisXGBoostModel',
    'TennisNeuralNetModel',
    'TennisFeatureExtractor',
    'FeatureTransformer',
    'FeatureSelector',
    'ModelTrainer',
    'TemporalCrossValidator',
    'TennisMetrics',
    'ResultsAnalyzer'
]