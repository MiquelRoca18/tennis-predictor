# models/__init__.py
"""
Tennis prediction models package.

This package contains implementations of various machine learning models
specifically designed for tennis match prediction, ranging from base models
to advanced ensembles with specialized capabilities.
"""

# Import all model classes for easy access
from .base_model import TennisModel
from .ensemble_model import TennisEnsembleModel
from .xgboost_model import TennisXGBoostModel
from .neural_net_model import TennisNeuralNetModel
from .balanced_ensemble_model import BalancedTennisEnsembleModel
from .calibrated_model import CalibratedTennisModel
from .explainable_model import ExplainableTennisModel
from .bayesian_optimization_model import BayesianOptimizationModel
from .advanced_ensemble_model import DiverseEnsembleModel

# Version information
__version__ = '1.0.0'

# Dictionary mapping model names to their classes for easy instantiation
MODEL_REGISTRY = {
    'ensemble': TennisEnsembleModel,
    'xgboost': TennisXGBoostModel,
    'neural_net': TennisNeuralNetModel,
    'balanced_ensemble': BalancedTennisEnsembleModel,
    'calibrated': CalibratedTennisModel,
    'explainable': ExplainableTennisModel,
    'bayesian': BayesianOptimizationModel,
    'diverse_ensemble': DiverseEnsembleModel
}

# Model categories for better organization
MODEL_CATEGORIES = {
    'base_models': ['xgboost', 'neural_net'],
    'ensemble_models': ['ensemble', 'balanced_ensemble', 'diverse_ensemble'],
    'enhancement_models': ['calibrated', 'explainable', 'bayesian']
}

def get_model(model_type, **kwargs):
    """
    Factory function to create a model instance of the specified type.
    
    Args:
        model_type (str): Type of model to create
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        TennisModel: An instance of the requested model type
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Available types: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type](**kwargs)

def get_available_models():
    """
    Get a list of available model types.
    
    Returns:
        list: Names of available model types
    """
    return list(MODEL_REGISTRY.keys())

def get_models_by_category(category=None):
    """
    Get available models by category.
    
    Args:
        category (str): Category name or None for all categories
        
    Returns:
        dict: Models organized by category
    """
    if category is None:
        return MODEL_CATEGORIES
        
    if category not in MODEL_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. "
                         f"Available categories: {list(MODEL_CATEGORIES.keys())}")
                         
    return {category: MODEL_CATEGORIES[category]}

def create_ensemble_pipeline(base_model_type, calibrate=True, explain=True):
    """
    Create a complete model pipeline with calibration and explainability.
    
    This is a convenience function to create a model that combines multiple
    enhancements like calibration and explainability.
    
    Args:
        base_model_type (str): Base model type to use
        calibrate (bool): Whether to include probability calibration
        explain (bool): Whether to include explainability features
        
    Returns:
        TennisModel: Enhanced model pipeline
    """
    # Create base model
    base_model = get_model(base_model_type)
    
    # Apply enhancements
    model = base_model
    
    if calibrate:
        model = CalibratedTennisModel(model)
        
    if explain:
        model = ExplainableTennisModel(model)
        
    return model