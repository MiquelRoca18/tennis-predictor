"""
Model Trainer for Tennis Match Prediction

This module provides a unified interface for training, validating, and
evaluating various types of tennis prediction models. It handles the
complete training pipeline from data loading to model serialization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import json
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import copy
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training tennis prediction models.
    
    This class handles the entire training process including:
    - Loading and preprocessing data
    - Feature extraction and selection
    - Model training and hyperparameter tuning
    - Evaluation on test data
    - Model serialization
    
    Attributes:
        config_path (str): Path to configuration file
        models_dir (str): Directory to save trained models
        results_dir (str): Directory to save evaluation results
        config (dict): Configuration parameters
        data_loader (object): Data loader object
        feature_extractor (object): Feature extractor object
        feature_selector (object): Feature selector object
        cv_splitter (object): Cross-validation splitter
        model (object): Trained model
    """
    
    def __init__(
        self,
        config_path: str,
        models_dir: str = 'models',
        results_dir: str = 'results',
        hyperparams_dir: str = 'hyperparams',
        logs_dir: str = 'logs'
    ):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to configuration file
            models_dir: Directory to save trained models
            results_dir: Directory to save evaluation results
            hyperparams_dir: Directory to save hyperparameter tuning results
            logs_dir: Directory to save training logs
        """
        self.config_path = config_path
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.hyperparams_dir = hyperparams_dir
        self.logs_dir = logs_dir
        
        # Create directories if they don't exist
        for directory in [models_dir, results_dir, hyperparams_dir, logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Load configuration
        self._load_config()
        
        # Initialize components
        self.data_loader = None
        self.feature_extractor = None
        self.feature_selector = None
        self.cv_splitter = None
        
        # Placeholder for trained model
        self.model = None
        
        # Set up logging to file
        self._setup_file_logging()
        
        logger.info(f"Initialized ModelTrainer with config from {config_path}")
    
    def _setup_file_logging(self):
        """Set up logging to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.logs_dir, f"training_{timestamp}.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_file}")
    
    def _load_config(self):
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Validate configuration
        required_keys = [
            'data_sources', 'features', 'model_params',
            'training_params', 'evaluation_params'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        logger.info("Configuration loaded successfully")
    
    def initialize_components(
        self,
        data_loader_class: Any,
        feature_extractor_class: Any,
        feature_selector_class: Optional[Any] = None,
        cv_splitter_class: Optional[Any] = None
    ):
        """
        Initialize components for the training pipeline.
        
        Args:
            data_loader_class: Class for loading data
            feature_extractor_class: Class for extracting features
            feature_selector_class: Optional class for selecting features
            cv_splitter_class: Optional class for cross-validation splitting
        """
        # Initialize data loader
        self.data_loader = data_loader_class(**self.config.get('data_loader_params', {}))
        
        # Initialize feature extractor
        self.feature_extractor = feature_extractor_class(**self.config.get('feature_extractor_params', {}))
        
        # Initialize feature selector if provided
        if feature_selector_class:
            self.feature_selector = feature_selector_class(**self.config.get('feature_selector_params', {}))
        
        # Initialize cross-validation splitter if provided
        if cv_splitter_class:
            self.cv_splitter = cv_splitter_class(**self.config.get('cv_splitter_params', {}))
        
        logger.info("Components initialized successfully")
    
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
        """
        Load and prepare data for training.
        
        Returns:
            Tuple of (features, target, dates)
        """
        if self.data_loader is None:
            raise ValueError("Data loader not initialized. Call initialize_components() first.")
        
        # Get data sources from config
        data_sources = self.config['data_sources']
        
        # Load data using data loader
        data = self.data_loader.load_data(**data_sources)
        
        # Apply date filters if specified
        if 'start_date' in self.config['training_params']:
            start_date = self.config['training_params']['start_date']
            data = self.data_loader.filter_by_date(data, start_date=start_date)
        
        if 'end_date' in self.config['training_params']:
            end_date = self.config['training_params']['end_date']
            data = self.data_loader.filter_by_date(data, end_date=end_date)
        
        # Get feature specifications from config
        feature_specs = self.config['features']
        
        # Extract features
        X, y, dates = self.feature_extractor.extract_features(
            data, feature_specs
        )
        
        logger.info(f"Data loaded and processed: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, dates
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        dates: pd.Series,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        temporal_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Split data into training and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            dates: Dates for each sample
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            temporal_split: Whether to split data temporally
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, dates_train, dates_test)
        """
        if temporal_split:
            # Sort data by date
            sorted_indices = dates.sort_values().index
            X_sorted = X.loc[sorted_indices]
            y_sorted = y[sorted_indices]
            dates_sorted = dates.loc[sorted_indices]
            
            # Calculate split point
            split_idx = int((1 - test_size) * len(X_sorted))
            
            # Split data
            X_train = X_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_train = y_sorted[:split_idx]
            y_test = y_sorted[split_idx:]
            dates_train = dates_sorted.iloc[:split_idx]
            dates_test = dates_sorted.iloc[split_idx:]
            
            logger.info(f"Data split temporally: train={X_train.shape[0]}, test={X_test.shape[0]}")
            logger.info(f"Training data range: {dates_train.min()} to {dates_train.max()}")
            logger.info(f"Test data range: {dates_test.min()} to {dates_test.max()}")
        else:
            # Random split
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X, y, np.arange(len(X)), test_size=test_size, random_state=random_state
            )
            dates_train = dates.iloc[train_idx]
            dates_test = dates.iloc[test_idx]
            
            logger.info(f"Data split randomly: train={X_train.shape[0]}, test={X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test, dates_train, dates_test
    
    def select_features(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select important features if feature selector is available.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_test: Test feature matrix
            
        Returns:
            Tuple of (X_train_selected, X_test_selected)
        """
        if self.feature_selector is None:
            logger.info("No feature selector provided. Using all features.")
            return X_train, X_test
        
        # Select features
        X_train_selected = self.feature_selector.select_features(X_train, y_train)
        
        # Transform test data using the same features
        X_test_selected = self.feature_selector.transform(X_test)
        
        logger.info(f"Feature selection: reduced from {X_train.shape[1]} to {X_train_selected.shape[1]} features")
        
        return X_train_selected, X_test_selected
    
    def tune_hyperparameters(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        dates: pd.Series,
        param_tuner_class: Any,
        param_space: Optional[Dict[str, Any]] = None,
        tuning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for model.
        
        Args:
            model_class: Model class to tune
            X: Feature matrix
            y: Target vector
            dates: Dates for temporal cross-validation
            param_tuner_class: Hyperparameter tuner class
            param_space: Parameter space for tuning
            tuning_params: Additional parameters for tuning
            
        Returns:
            Dictionary with best parameters
        """
        # Default tuning parameters
        default_tuning_params = {
            'method': 'random',
            'metric': 'roc_auc',
            'n_iter': 20,
            'n_jobs': -1,
            'results_dir': self.hyperparams_dir
        }
        
        # Update with provided parameters
        if tuning_params:
            default_tuning_params.update(tuning_params)
        
        # Get tuning parameters from config if not provided
        if not tuning_params and 'tuning_params' in self.config:
            default_tuning_params.update(self.config['tuning_params'])
        
        # Initialize tuner
        tuner = param_tuner_class(
            method=default_tuning_params['method'],
            metric=default_tuning_params['metric'],
            cv_splitter=self.cv_splitter,
            n_jobs=default_tuning_params['n_jobs'],
            results_dir=default_tuning_params['results_dir']
        )
        
        # Get timestamp for experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_class.__name__
        experiment_name = f"{model_name}_{timestamp}"
        
        logger.info(f"Starting hyperparameter tuning: {experiment_name}")
        
        # Run tuning
        tuning_results = tuner.optimize(
            model_class=model_class,
            param_space=param_space,
            X=X,
            y=y,
            dates=dates,
            n_iter=default_tuning_params['n_iter'],
            experiment_name=experiment_name
        )
        
        # Get best parameters
        best_params = tuning_results['best_params']
        
        logger.info(f"Hyperparameter tuning completed. Best score: {tuning_results['best_score']:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def train_model(
        self,
        model_class: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None,
        tune_hyperparams: bool = False,
        param_tuner_class: Optional[Any] = None,
        param_space: Optional[Dict[str, Any]] = None,
        dates_train: Optional[pd.Series] = None,
        tuning_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Train a model on the provided data.
        
        Args:
            model_class: Model class to train
            X_train: Training feature matrix
            y_train: Training target vector
            model_params: Parameters for model initialization
            tune_hyperparams: Whether to tune hyperparameters
            param_tuner_class: Hyperparameter tuner class (required if tune_hyperparams is True)
            param_space: Parameter space for tuning (required if tune_hyperparams is True)
            dates_train: Training dates (required if tune_hyperparams is True)
            tuning_params: Additional parameters for tuning
            
        Returns:
            Trained model
        """
        start_time = time.time()
        
        # Get model parameters from config if not provided
        if not model_params:
            model_params = self.config['model_params'].get(
                model_class.__name__, {}
            )
        
        # Tune hyperparameters if requested
        if tune_hyperparams:
            if param_tuner_class is None or param_space is None or dates_train is None:
                raise ValueError(
                    "param_tuner_class, param_space, and dates_train must be provided "
                    "when tune_hyperparams is True"
                )
            
            # Tune hyperparameters
            best_params = self.tune_hyperparameters(
                model_class=model_class,
                X=X_train,
                y=y_train,
                dates=dates_train,
                param_tuner_class=param_tuner_class,
                param_space=param_space,
                tuning_params=tuning_params
            )
            
            # Update model parameters with best parameters
            model_params.update(best_params)
        
        logger.info(f"Training {model_class.__name__} with parameters: {model_params}")
        
        # Initialize and train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Store trained model
        self.model = model
        
        elapsed_time = time.time() - start_time
        logger.info(f"Model training completed in {elapsed_time:.2f} seconds")
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        dates_test: Optional[pd.Series] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            model: Trained model
            X_test: Test feature matrix
            y_test: Test target vector
            dates_test: Test dates
            custom_metrics: Dictionary of custom metric functions
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, log_loss
        )
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Generate probability predictions if available
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            has_proba = True
        except (AttributeError, IndexError):
            has_proba = False
        
        # Calculate standard metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if has_proba:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['log_loss'] = log_loss(y_test, y_pred_proba)
        
        # Calculate custom metrics if provided
        if custom_metrics:
            for metric_name, metric_func in custom_metrics.items():
                if has_proba:
                    metrics[metric_name] = metric_func(y_test, y_pred_proba)
                else:
                    metrics[metric_name] = metric_func(y_test, y_pred)
        
        # Log metrics
        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # If dates are provided, evaluate by time periods
        if dates_test is not None:
            time_metrics = self._evaluate_by_time_periods(
                model, X_test, y_test, dates_test
            )
            metrics['time_metrics'] = time_metrics
        
        return metrics
    
    def _evaluate_by_time_periods(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        dates_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance by time periods.
        
        Args:
            model: Trained model
            X_test: Test feature matrix
            y_test: Test target vector
            dates_test: Test dates
            
        Returns:
            Dictionary with metrics by time period
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        # Define time periods (quarters)
        dates_test = pd.to_datetime(dates_test)
        X_test_with_dates = X_test.copy()
        X_test_with_dates['date'] = dates_test
        X_test_with_dates['year'] = X_test_with_dates['date'].dt.year
        X_test_with_dates['quarter'] = X_test_with_dates['date'].dt.quarter
        
        # Group by year and quarter
        time_metrics = {}
        
        for (year, quarter), group in X_test_with_dates.groupby(['year', 'quarter']):
            period_key = f"{year}Q{quarter}"
            
            # Get data for this period
            period_indices = group.index
            X_period = X_test.loc[period_indices]
            y_period = y_test[period_indices]
            
            # Skip if too few samples
            if len(y_period) < 10:
                continue
            
            # Make predictions
            y_pred = model.predict(X_period)
            
            # Calculate metrics
            period_metrics = {
                'accuracy': accuracy_score(y_period, y_pred),
                'f1': f1_score(y_period, y_pred, zero_division=0),
                'samples': len(y_period)
            }
            
            time_metrics[period_key] = period_metrics
        
        return time_metrics
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name for the model
            metadata: Additional metadata to save with the model
            
        Returns:
            Path to saved model
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add basic metadata
        metadata.update({
            'model_name': model_name,
            'timestamp': timestamp,
            'python_class': model.__class__.__name__,
            'created_at': datetime.now().isoformat()
        })
        
        # Save metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save model
        model_path = os.path.join(model_dir, 'model.joblib')
        joblib.dump(model, model_path)
        
        logger.info(f"Model saved to {model_dir}")
        
        return model_dir
    
    def load_model(self, model_dir: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model from disk.
        
        Args:
            model_dir: Directory containing the model
            
        Returns:
            Tuple of (model, metadata)
        """
        # Load metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)
        
        logger.info(f"Model loaded from {model_dir}")
        
        return model, metadata
    
    def save_results(
        self,
        metrics: Dict[str, float],
        model_name: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save evaluation results to disk.
        
        Args:
            metrics: Evaluation metrics
            model_name: Name of the model
            additional_info: Additional information to save
            
        Returns:
            Path to saved results
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results file path
        results_path = os.path.join(
            self.results_dir, f"{model_name}_results_{timestamp}.json"
        )
        
        # Create results dictionary
        results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        # Add additional info if provided
        if additional_info:
            results.update(additional_info)
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return results_path
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot feature importance if available.
        
        Args:
            model: Trained model
            feature_names: Names of features
            top_n: Number of top features to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # Check if model is a pipeline with a final estimator that has feature_importances_
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('estimator', None), 'feature_importances_'):
            importances = model.named_steps['estimator'].feature_importances_
        # Check if model has coef_ attribute (linear models)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        # Check if model is a pipeline with a final estimator that has coef_
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('estimator', None), 'coef_'):
            importances = np.abs(model.named_steps['estimator'].coef_)
        else:
            logger.warning("Model does not have feature_importances_ or coef_ attribute")
            return None
        
        # Ensure importances is 1D
        if importances.ndim > 1:
            importances = importances.mean(axis=0)
        
        # Create DataFrame with feature names and importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Take top N features
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        sns.barplot(
            x='importance',
            y='feature',
            data=importance_df,
            ax=ax
        )
        
        # Set title and labels
        ax.set_title(f"Top {len(importance_df)} Feature Importances")
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_performance_over_time(
        self,
        metrics: Dict[str, Dict[str, float]],
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Plot model performance over time.
        
        Args:
            metrics: Dictionary with metrics by time period
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if 'time_metrics' not in metrics:
            logger.warning("No time-based metrics available")
            return None
        
        time_metrics = metrics['time_metrics']
        
        # Create DataFrame from time metrics
        periods = list(time_metrics.keys())
        accuracies = [m['accuracy'] for m in time_metrics.values()]
        f1_scores = [m['f1'] for m in time_metrics.values()]
        sample_counts = [m['samples'] for m in time_metrics.values()]
        
        df = pd.DataFrame({
            'period': periods,
            'accuracy': accuracies,
            'f1': f1_scores,
            'samples': sample_counts
        })
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot metrics
        ax1.plot(df['period'], df['accuracy'], 'o-', color='blue', label='Accuracy')
        ax1.plot(df['period'], df['f1'], 'o-', color='green', label='F1 Score')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Metric Value')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim([0, 1])
        
        # Create second y-axis for sample counts
        ax2 = ax1.twinx()
        ax2.bar(df['period'], df['samples'], alpha=0.2, color='gray', label='Samples')
        ax2.set_ylabel('Sample Count')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Set title
        plt.title('Model Performance Over Time')
        
        # Rotate x-tick labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def run_training_pipeline(
        self,
        model_class: Any,
        model_name: Optional[str] = None,
        tune_hyperparams: bool = False,
        param_tuner_class: Optional[Any] = None,
        param_space: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        tuning_params: Optional[Dict[str, Any]] = None,
        save_results: bool = True,
        custom_metrics: Optional[Dict[str, Callable]] = None
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Run complete training pipeline.
        
        Args:
            model_class: Model class to train
            model_name: Name for the model (defaults to class name)
            tune_hyperparams: Whether to tune hyperparameters
            param_tuner_class: Hyperparameter tuner class
            param_space: Parameter space for tuning
            model_params: Parameters for model initialization
            tuning_params: Additional parameters for tuning
            save_results: Whether to save results
            custom_metrics: Dictionary of custom metric functions
            
        Returns:
            Tuple of (model, metrics)
        """
        # Set default model name if not provided
        if model_name is None:
            model_name = model_class.__name__
        
        # Get training parameters from config
        training_params = self.config['training_params']
        
        # Load data
        X, y, dates = self.load_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test, dates_train, dates_test = self.prepare_data(
            X, y, dates, 
            test_size=training_params.get('test_size', 0.2),
            random_state=training_params.get('random_state', 42),
            temporal_split=training_params.get('temporal_split', True)
        )
        
        # Select features if feature selector is available
        if self.feature_selector is not None:
            X_train, X_test = self.select_features(X_train, y_train, X_test)
        
        # Train model
        model = self.train_model(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            model_params=model_params,
            tune_hyperparams=tune_hyperparams,
            param_tuner_class=param_tuner_class,
            param_space=param_space,
            dates_train=dates_train,
            tuning_params=tuning_params
        )
        
        # Evaluate model
        metrics = self.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            dates_test=dates_test,
            custom_metrics=custom_metrics
        )
        
        # Save model and results if requested
        if save_results:
            # Save model
            model_dir = self.save_model(
                model=model,
                model_name=model_name,
                metadata={
                    'metrics': metrics,
                    'model_params': model_params if model_params else {},
                    'feature_count': X_train.shape[1],
                    'training_count': X_train.shape[0],
                    'test_count': X_test.shape[0],
                    'features': list(X_train.columns)
                }
            )
            
            # Save feature importance plot if available
            try:
                fig = self.plot_feature_importance(
                    model=model,
                    feature_names=list(X_train.columns)
                )
                if fig is not None:
                    plt.savefig(os.path.join(model_dir, 'feature_importance.png'))
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not generate feature importance plot: {e}")
            
            # Save time performance plot if available
            try:
                fig = self.plot_performance_over_time(metrics)
                if fig is not None:
                    plt.savefig(os.path.join(model_dir, 'time_performance.png'))
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not generate time performance plot: {e}")
            
            # Save results
            self.save_results(
                metrics=metrics,
                model_name=model_name,
                additional_info={
                    'model_dir': model_dir,
                    'feature_count': X_train.shape[1],
                    'training_count': X_train.shape[0],
                    'test_count': X_test.shape[0]
                }
            )
        
        return model, metrics
    
    def run_multiple_models(
        self,
        model_configs: List[Dict[str, Any]],
        save_results: bool = True,
        custom_metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """
        Train and evaluate multiple models.
        
        Args:
            model_configs: List of model configurations, each containing:
                - model_class: Model class to train
                - model_name: Optional name for the model
                - tune_hyperparams: Whether to tune hyperparameters
                - param_tuner_class: Hyperparameter tuner class (if tuning)
                - param_space: Parameter space for tuning (if tuning)
                - model_params: Parameters for model initialization
            save_results: Whether to save results
            custom_metrics: Dictionary of custom metric functions
            
        Returns:
            Dictionary mapping model names to (model, metrics) tuples
        """
        results = {}
        
        # Train each model
        for config in model_configs:
            model_class = config['model_class']
            model_name = config.get('model_name', model_class.__name__)
            
            logger.info(f"Training model: {model_name}")
            
            # Run training pipeline
            model, metrics = self.run_training_pipeline(
                model_class=model_class,
                model_name=model_name,
                tune_hyperparams=config.get('tune_hyperparams', False),
                param_tuner_class=config.get('param_tuner_class'),
                param_space=config.get('param_space'),
                model_params=config.get('model_params'),
                tuning_params=config.get('tuning_params'),
                save_results=save_results,
                custom_metrics=custom_metrics
            )
            
            # Store results
            results[model_name] = (model, metrics)
        
        return results
    
    def compare_models(
        self,
        models_metrics: Dict[str, Tuple[Any, Dict[str, float]]],
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Compare multiple models visually.
        
        Args:
            models_metrics: Dictionary mapping model names to (model, metrics) tuples
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract metrics for each model
        model_names = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        roc_aucs = []
        
        for model_name, (_, metrics) in models_metrics.items():
            model_names.append(model_name)
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            roc_aucs.append(metrics.get('roc_auc', 0))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1_scores,
            'ROC AUC': roc_aucs
        })
        
        # Melt DataFrame for easier plotting
        df_melted = pd.melt(
            df, 
            id_vars=['Model'],
            var_name='Metric',
            value_name='Value'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot grouped bar chart
        sns.barplot(
            x='Model',
            y='Value',
            hue='Metric',
            data=df_melted,
            ax=ax
        )
        
        # Set title and labels
        ax.set_title('Model Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Metric Value')
        
        # Set y-axis limits
        ax.set_ylim([0, 1])
        
        # Rotate x-tick labels if many models
        if len(model_names) > 3:
            plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def save_comparison_results(
        self,
        models_metrics: Dict[str, Tuple[Any, Dict[str, float]]]
    ) -> str:
        """
        Save comparison results to disk.
        
        Args:
            models_metrics: Dictionary mapping model names to (model, metrics) tuples
            
        Returns:
            Path to saved results
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results file path
        results_path = os.path.join(
            self.results_dir, f"model_comparison_{timestamp}.json"
        )
        
        # Extract metrics for each model
        comparison = {}
        
        for model_name, (_, metrics) in models_metrics.items():
            # Filter out non-serializable metrics
            serializable_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, str, bool)) or k == 'time_metrics'
            }
            comparison[model_name] = serializable_metrics
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison results saved to {results_path}")
        
        # Create comparison plot
        fig = self.compare_models(models_metrics)
        
        # Save plot
        plot_path = os.path.join(
            self.results_dir, f"model_comparison_{timestamp}.png"
        )
        plt.savefig(plot_path)
        plt.close(fig)
        
        logger.info(f"Comparison plot saved to {plot_path}")
        
        return results_path
    
    def calibrate_probabilities(
                            self,
                            model: Any,
                            X_train: pd.DataFrame,
                            y_train: np.ndarray,
                            X_test: pd.DataFrame,
                            method: str = 'isotonic',
                            cv: int = 5
                        ) -> Any:
        """
        Calibrate probability predictions for better estimation.
        
        Args:
            model: Trained model
            X_train: Training feature matrix
            y_train: Training target vector
            X_test: Test feature matrix
            method: Calibration method ('isotonic' or 'sigmoid')
            cv: Number of cross-validation folds
            
        Returns:
            Calibrated model
        """
        from sklearn.calibration import CalibratedClassifierCV
        
        # Check if model already has predict_proba
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not have predict_proba method. Cannot calibrate.")
            return model
        
        # Fit calibrated model
        calibrated_model = CalibratedClassifierCV(
            model, method=method, cv=cv
        )
        
        # Train on calibration data
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate calibration
        from sklearn.calibration import calibration_curve
        
        # Get uncalibrated probabilities
        if hasattr(model, 'predict_proba'):
            y_prob_uncal = model.predict_proba(X_test)[:, 1]
        else:
            y_prob_uncal = model.decision_function(X_test)
            
        # Get calibrated probabilities
        y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate calibration curves
        prob_true_uncal, prob_pred_uncal = calibration_curve(
            y_test, y_prob_uncal, n_bins=10
        )
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_test, y_prob_cal, n_bins=10
        )
        
        # Calculate Brier scores
        from sklearn.metrics import brier_score_loss
        
        brier_uncal = brier_score_loss(y_test, y_prob_uncal)
        brier_cal = brier_score_loss(y_test, y_prob_cal)
        
        logger.info(f"Brier score (uncalibrated): {brier_uncal:.6f}")
        logger.info(f"Brier score (calibrated): {brier_cal:.6f}")
        
        # Create calibration plot
        plt.figure(figsize=(10, 6))
        
        # Plot perfectly calibrated line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Plot calibration curves
        plt.plot(
            prob_pred_uncal, prob_true_uncal, 
            marker='o', markersize=8, linestyle='-', 
            label=f'Uncalibrated (Brier: {brier_uncal:.3f})'
        )
        plt.plot(
            prob_pred_cal, prob_true_cal, 
            marker='s', markersize=8, linestyle='-', 
            label=f'Calibrated {method} (Brier: {brier_cal:.3f})'
        )
        
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(self.results_dir, f"calibration_curve_{method}.png"))
        plt.close()
        
        return calibrated_model
    
    def monitor_model_degradation(
                                self,
                                model: Any,
                                X: pd.DataFrame,
                                y: np.ndarray,
                                dates: pd.Series,
                                metric: str = 'roc_auc',
                                window_size: int = 30,
                                alert_threshold: float = 0.05,
                                rolling_windows: int = 5
                            ) -> Dict[str, Any]:
        """
        Monitor model performance over time to detect degradation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            dates: Match dates
            metric: Metric to monitor
            window_size: Size of each time window in days
            alert_threshold: Threshold for performance drop to trigger alert
            rolling_windows: Number of windows to use for trend detection
            
        Returns:
            Dictionary with monitoring results
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        # Convert dates to datetime
        dates = pd.to_datetime(dates)
        
        # Sort by date
        sorted_indices = dates.sort_values().index
        X_sorted = X.loc[sorted_indices]
        y_sorted = y[sorted_indices]
        dates_sorted = dates.loc[sorted_indices]
        
        # Create time windows
        min_date = dates_sorted.min()
        max_date = dates_sorted.max()
        
        window_start = min_date
        windows = []
        
        while window_start <= max_date:
            window_end = window_start + pd.Timedelta(days=window_size)
            windows.append((window_start, window_end))
            window_start = window_end
        
        # Calculate performance for each window
        window_metrics = []
        
        for window_start, window_end in windows:
            # Get data for this window
            window_mask = (dates_sorted >= window_start) & (dates_sorted < window_end)
            
            # Skip if window has too few samples
            if np.sum(window_mask) < 10:
                continue
            
            X_window = X_sorted.loc[window_mask]
            y_window = y_sorted[window_mask]
            
            # Make predictions
            y_pred = model.predict(X_window)
            
            # Calculate metrics
            if metric == 'accuracy':
                score = accuracy_score(y_window, y_pred)
            elif metric == 'precision':
                score = precision_score(y_window, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_window, y_pred, zero_division=0)
            elif metric == 'f1':
                score = f1_score(y_window, y_pred, zero_division=0)
            elif metric == 'roc_auc':
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_window)[:, 1]
                    score = roc_auc_score(y_window, y_pred_proba)
                else:
                    score = roc_auc_score(y_window, y_pred)
            
            window_metrics.append({
                'start_date': window_start,
                'end_date': window_end,
                'metric': metric,
                'score': score,
                'samples': np.sum(window_mask)
            })
        
        # Calculate rolling average and detect trend
        if len(window_metrics) >= rolling_windows:
            rolling_avg = []
            
            for i in range(len(window_metrics) - rolling_windows + 1):
                window_subset = window_metrics[i:i+rolling_windows]
                avg_score = np.mean([w['score'] for w in window_subset])
                rolling_avg.append({
                    'end_date': window_subset[-1]['end_date'],
                    'score': avg_score
                })
            
            # Check if performance is decreasing
            if len(rolling_avg) >= 2:
                start_score = rolling_avg[0]['score']
                end_score = rolling_avg[-1]['score']
                
                perf_change = end_score - start_score
                perf_change_pct = perf_change / start_score if start_score > 0 else 0
                
                # Generate alert if performance drops significantly
                alert = perf_change_pct < -alert_threshold
                
                # Create monitoring result
                monitoring_result = {
                    'windows': window_metrics,
                    'rolling_avg': rolling_avg,
                    'initial_score': start_score,
                    'current_score': end_score,
                    'change': perf_change,
                    'change_pct': perf_change_pct,
                    'alert': alert
                }
                
                # Log alert if necessary
                if alert:
                    logger.warning(
                        f"Model degradation detected! {metric} dropped by "
                        f"{abs(perf_change_pct)*100:.2f}% (threshold: {alert_threshold*100:.2f}%)"
                    )
                
                # Create plot
                plt.figure(figsize=(12, 6))
                
                # Plot window metrics
                dates = [w['end_date'] for w in window_metrics]
                scores = [w['score'] for w in window_metrics]
                
                plt.plot(dates, scores, 'o-', color='blue', alpha=0.5, label='Window metrics')
                
                # Plot rolling average
                roll_dates = [r['end_date'] for r in rolling_avg]
                roll_scores = [r['score'] for r in rolling_avg]
                
                plt.plot(roll_dates, roll_scores, 'r-', linewidth=2, label=f'{rolling_windows}-window rolling avg')
                
                # Add alert threshold
                threshold_value = start_score * (1 - alert_threshold)
                plt.axhline(y=threshold_value, color='red', linestyle='--', 
                            label=f'Alert threshold ({alert_threshold*100:.1f}% drop)')
                
                plt.xlabel('Time')
                plt.ylabel(f'{metric} Score')
                plt.title('Model Performance Over Time')
                plt.legend()
                plt.grid(True)
                
                # Save plot
                plt.savefig(os.path.join(self.results_dir, "model_degradation_monitor.png"))
                plt.close()
                
                return monitoring_result
            else:
                return {'windows': window_metrics, 'alert': False, 
                        'message': 'Not enough data for trend detection'}
        else:
            return {'windows': window_metrics, 'alert': False, 
                    'message': 'Not enough windows for monitoring'}
        
    def analyze_errors(
                    self,
                    model: Any,
                    X_test: pd.DataFrame,
                    y_test: np.ndarray,
                    dates_test: pd.Series = None,
                    player_data: pd.DataFrame = None,
                    match_data: pd.DataFrame = None,
                    n_errors: int = 20
                ) -> Dict[str, Any]:
        """
        Analyze misclassified matches to understand patterns in model errors.
        
        Args:
            model: Trained model
            X_test: Test feature matrix
            y_test: Test target vector
            dates_test: Test dates
            player_data: Player information (optional)
            match_data: Match information (optional)
            n_errors: Number of worst errors to analyze
            
        Returns:
            Dictionary with error analysis
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            proba_available = True
        else:
            y_proba = np.zeros_like(y_test)
            proba_available = False
        
        # Find errors
        errors = (y_pred != y_test)
        
        if not np.any(errors):
            return {'message': 'No errors found in the test set!'}
        
        # Calculate error severity
        if proba_available:
            # Severity based on confidence of wrong prediction
            error_severity = np.abs(y_proba - 0.5) * 2
            error_severity = error_severity[errors]
        else:
            # Can't calculate severity without probabilities
            error_severity = np.ones(np.sum(errors))
        
        # Get indices of errors, sorted by severity
        error_indices = np.where(errors)[0]
        sorted_indices = error_indices[np.argsort(-error_severity)]
        
        # Limit to requested number of errors
        n_errors = min(n_errors, len(sorted_indices))
        worst_indices = sorted_indices[:n_errors]
        
        # Create dataframe with error information
        error_df = pd.DataFrame({
            'true_label': y_test[worst_indices],
            'pred_label': y_pred[worst_indices],
            'confidence': y_proba[worst_indices] if proba_available else None
        })
        
        # Add date information if available
        if dates_test is not None:
            error_df['date'] = dates_test.iloc[worst_indices].values
        
        # Add feature values
        for col in X_test.columns:
            error_df[f'feature_{col}'] = X_test.iloc[worst_indices][col].values
        
        # Add match information if available
        if match_data is not None and 'match_id' in X_test.columns:
            match_ids = X_test.iloc[worst_indices]['match_id'].values
            
            # Try to join with match data
            matches = match_data[match_data['match_id'].isin(match_ids)]
            
            if not matches.empty:
                # Extract relevant match info
                match_info = matches[['match_id', 'player1', 'player2', 'tournament', 'surface']].copy()
                
                # Join with error dataframe
                error_df = error_df.merge(match_info, on='match_id', how='left')
        
        # Analyze patterns in errors
        error_patterns = {}
        
        # Check for common features among errors
        for col in X_test.columns:
            # Skip non-numeric columns
            if not np.issubdtype(X_test[col].dtype, np.number):
                continue
                
            # Calculate mean values for errors vs correct predictions
            mean_error = X_test.loc[errors, col].mean()
            mean_correct = X_test.loc[~errors, col].mean()
            
            # Calculate difference and ratio
            diff = mean_error - mean_correct
            ratio = mean_error / mean_correct if mean_correct != 0 else 0
            
            # Store if significant difference
            if abs(ratio - 1) > 0.2:  # 20% difference
                error_patterns[col] = {
                    'mean_error': mean_error,
                    'mean_correct': mean_correct,
                    'diff': diff,
                    'ratio': ratio
                }
        
        # Sort patterns by absolute difference ratio
        sorted_patterns = sorted(
            error_patterns.items(),
            key=lambda x: abs(x[1]['ratio'] - 1),
            reverse=True
        )
        
        # Create visualization of errors
        if proba_available:
            plt.figure(figsize=(10, 6))
            
            # Create scatter plot of prediction confidence
            plt.scatter(
                np.arange(len(y_test)),
                y_proba,
                c=['red' if e else 'green' for e in errors],
                alpha=0.6,
                s=30
            )
            
            # Highlight worst errors
            plt.scatter(
                worst_indices,
                y_proba[worst_indices],
                c='red',
                s=100,
                edgecolors='black'
            )
            
            plt.axhline(y=0.5, color='black', linestyle='--')
            plt.xlabel('Test Sample Index')
            plt.ylabel('Predicted Probability')
            plt.title('Model Errors Analysis')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                    markersize=10, label='Correct Predictions'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                    markersize=10, label='Errors'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                    markeredgecolor='black', markersize=15, label='Worst Errors')
            ]
            plt.legend(handles=legend_elements)
            
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(self.results_dir, "error_analysis.png"))
            plt.close()
        
        # Return comprehensive error analysis
        return {
            'error_count': np.sum(errors),
            'error_rate': np.mean(errors),
            'worst_errors': error_df.to_dict(orient='records'),
            'error_patterns': sorted_patterns[:10],  # Top 10 patterns
            'figure_path': os.path.join(self.results_dir, "error_analysis.png") if proba_available else None
        }
    
    def evaluate_model_by_segment(
                                self,
                                model: Any,
                                X_test: pd.DataFrame,
                                y_test: np.ndarray,
                                segment_column: str,
                                segment_values: Dict[str, Any] = None,
                                custom_metrics: Optional[Dict[str, Callable]] = None
                            ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance across different segments (surface, tournament, etc.).
        
        Args:
            model: Trained model
            X_test: Test feature matrix
            y_test: Test target vector
            segment_column: Column to segment by
            segment_values: Dictionary mapping segment values to human-readable names
            custom_metrics: Dictionary of custom metric functions
            
        Returns:
            Dictionary with evaluation metrics by segment
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        if segment_column not in X_test.columns:
            raise ValueError(f"Segment column '{segment_column}' not found in X_test")
        
        # Get unique segment values
        unique_segments = X_test[segment_column].unique()
        
        # Initialize results dictionary
        results = {}
        
        # Calculate metrics for each segment
        for segment in unique_segments:
            # Get indices for this segment
            segment_mask = X_test[segment_column] == segment
            
            # Skip if too few samples
            if np.sum(segment_mask) < 10:
                continue
            
            X_segment = X_test.loc[segment_mask]
            y_segment = y_test[segment_mask]
            
            # Make predictions
            y_pred = model.predict(X_segment)
            
            # Calculate probabilities if possible
            try:
                y_pred_proba = model.predict_proba(X_segment)[:, 1]
                has_proba = True
            except (AttributeError, IndexError):
                has_proba = False
            
            # Calculate standard metrics
            segment_metrics = {
                'accuracy': accuracy_score(y_segment, y_pred),
                'precision': precision_score(y_segment, y_pred, zero_division=0),
                'recall': recall_score(y_segment, y_pred, zero_division=0),
                'f1': f1_score(y_segment, y_pred, zero_division=0),
                'samples': int(np.sum(segment_mask))
            }
            
            if has_proba:
                segment_metrics['roc_auc'] = roc_auc_score(y_segment, y_pred_proba)
            
            # Calculate custom metrics if provided
            if custom_metrics:
                for metric_name, metric_func in custom_metrics.items():
                    if has_proba:
                        segment_metrics[metric_name] = metric_func(y_segment, y_pred_proba)
                    else:
                        segment_metrics[metric_name] = metric_func(y_segment, y_pred)
            
            # Use segment value mapping if provided
            segment_name = segment_values.get(segment, segment) if segment_values else segment
            results[segment_name] = segment_metrics
        
        # Create visualization
        if len(results) > 1:
            # Prepare data for plotting
            segments = list(results.keys())
            accuracies = [results[s]['accuracy'] for s in segments]
            if all('roc_auc' in results[s] for s in segments):
                roc_aucs = [results[s]['roc_auc'] for s in segments]
            else:
                roc_aucs = None
            
            counts = [results[s]['samples'] for s in segments]
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot counts as bars
            ax1 = plt.gca()
            ax1.bar(segments, counts, alpha=0.3, color='gray')
            ax1.set_ylabel('Sample Count', color='gray')
            ax1.tick_params(axis='y', labelcolor='gray')
            
            # Create second y-axis for metrics
            ax2 = ax1.twinx()
            
            # Plot metrics as lines
            ax2.plot(segments, accuracies, 'o-', color='blue', label='Accuracy')
            if roc_aucs:
                ax2.plot(segments, roc_aucs, 's-', color='red', label='ROC AUC')
            
            ax2.set_ylim([0, 1])
            ax2.set_ylabel('Metric Value')
            ax2.tick_params(axis='y')
            
            # Add legend
            ax2.legend()
            
            # Set title
            plt.title(f'Model Performance by {segment_column}')
            
            # Rotate x labels if needed
            plt.xticks(rotation=45 if len(segments) > 4 else 0)
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(self.results_dir, f"performance_by_{segment_column}.png"))
            plt.close()
        
        return results
    
    def calculate_betting_metrics(
                                self,
                                model: Any,
                                X_test: pd.DataFrame,
                                y_test: np.ndarray,
                                odds_column: Optional[str] = None,
                                odds_source: str = 'implied_probability', 
                                kelly_fraction: float = 0.25,
                                initial_bankroll: float = 1000.0,
                                bet_sizing: str = 'flat'
                            ) -> Dict[str, Any]:
        """
        Calculate betting-specific metrics for model evaluation.
        
        Args:
            model: Trained model
            X_test: Test feature matrix
            y_test: Test target vector
            odds_column: Column containing odds information
            odds_source: Source of odds ('implied_probability', 'decimal', 'american')
            kelly_fraction: Fraction of Kelly criterion to use (conservative approach)
            initial_bankroll: Initial bankroll for simulation
            bet_sizing: Strategy for bet sizing ('flat', 'kelly', 'percentage')
            
        Returns:
            Dictionary with betting metrics
        """
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model doesn't provide probabilities. Betting metrics require probabilities.")
            return {'error': 'Model does not provide probabilities'}
        
        # Get model probability predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Handle odds
        if odds_column and odds_column in X_test.columns:
            # Get odds from data
            odds = X_test[odds_column].values
            
            # Convert odds to probabilities depending on format
            if odds_source == 'decimal':
                implied_probabilities = 1 / odds
            elif odds_source == 'american':
                implied_probabilities = np.zeros_like(odds)
                # Convert positive American odds
                positive_mask = odds > 0
                implied_probabilities[positive_mask] = 100 / (odds[positive_mask] + 100)
                # Convert negative American odds
                negative_mask = odds < 0
                implied_probabilities[negative_mask] = abs(odds[negative_mask]) / (abs(odds[negative_mask]) + 100)
            else:  # already in probability form
                implied_probabilities = odds
        else:
            # No odds provided, can't calculate betting metrics
            logger.warning("No odds column provided. Some betting metrics will not be available.")
            implied_probabilities = None
        
        # Calculate basic counts
        y_pred = (y_proba > 0.5).astype(int)
        n_bets = len(y_test)
        n_wins = np.sum((y_pred == 1) & (y_test == 1))
        n_losses = np.sum((y_pred == 1) & (y_test == 0))
        
        # Calculate hit rate
        hit_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0
        
        # Initialize betting metrics
        metrics = {
            'hit_rate': hit_rate,
            'bets_placed': n_wins + n_losses,
            'wins': n_wins,
            'losses': n_losses
        }
        
        # Calculate ROI and other metrics if odds are available
        if implied_probabilities is not None:
            # Calculate expected value for each bet
            expected_values = y_proba - implied_probabilities
            
            # Calculate theoretical edge
            positive_ev_mask = expected_values > 0
            avg_edge = np.mean(expected_values[positive_ev_mask]) if np.any(positive_ev_mask) else 0
            
            # Simulate betting results
            bankroll = initial_bankroll
            bankroll_history = [bankroll]
            bet_amounts = []
            
            for i in range(len(y_test)):
                # Skip bets with negative expected value
                if expected_values[i] <= 0:
                    bet_amounts.append(0)
                    bankroll_history.append(bankroll)
                    continue
                
                # Calculate decimal odds from implied probability
                decimal_odds = 1 / implied_probabilities[i]
                
                # Determine bet size
                if bet_sizing == 'flat':
                    # Flat betting (1% of initial bankroll)
                    bet_amount = initial_bankroll * 0.01
                elif bet_sizing == 'kelly':
                    # Kelly criterion: (bp - q) / b
                    # where b = odds - 1, p = probability of winning, q = probability of losing
                    b = decimal_odds - 1
                    p = y_proba[i]
                    q = 1 - p
                    kelly_bet = (b * p - q) / b
                    
                    # Apply Kelly fraction for more conservative approach
                    kelly_bet = max(0, kelly_bet * kelly_fraction)
                    
                    # Bet size as percentage of current bankroll
                    bet_amount = bankroll * kelly_bet
                else:  # percentage
                    # Percentage of current bankroll (1%)
                    bet_amount = bankroll * 0.01
                
                bet_amounts.append(bet_amount)
                
                # Update bankroll based on bet outcome
                if y_test[i] == 1:  # Win
                    bankroll += bet_amount * (decimal_odds - 1)
                else:  # Loss
                    bankroll -= bet_amount
                
                bankroll_history.append(bankroll)
            
            # Calculate ROI
            total_investment = sum(bet_amounts)
            roi = (bankroll - initial_bankroll) / total_investment if total_investment > 0 else 0
            
            # Add betting metrics
            metrics.update({
                'avg_edge': avg_edge,
                'roi': roi,
                'profit': bankroll - initial_bankroll,
                'final_bankroll': bankroll,
                'bankroll_growth': (bankroll / initial_bankroll) - 1,
                'max_drawdown': 1 - min(bankroll_history) / max(bankroll_history[:bankroll_history.index(min(bankroll_history))+1])
            })
            
            # Create visualization of bankroll over time
            plt.figure(figsize=(12, 6))
            plt.plot(bankroll_history, 'b-')
            plt.axhline(y=initial_bankroll, color='r', linestyle='--', label='Initial bankroll')
            plt.xlabel('Bet Number')
            plt.ylabel('Bankroll')
            plt.title(f'Bankroll Evolution (ROI: {roi:.2%})')
            plt.grid(True)
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(self.results_dir, "betting_simulation.png"))
            plt.close()
        
        return metrics