"""
Hyperparameter Tuning for Tennis Prediction Models

This module provides tools for optimizing hyperparameters of tennis prediction models
using various strategies including Grid Search, Random Search, and Bayesian Optimization.
It's designed to work with the temporal nature of tennis match data and can handle
different types of models (ensemble, XGBoost, neural networks).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Any, Optional, Callable
import logging
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import copy

# For Bayesian optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    
# Configure logger
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Class for hyperparameter tuning of tennis prediction models.
    
    Supports:
    - Grid Search
    - Random Search
    - Bayesian Optimization (if skopt is available)
    
    Designed to work with temporal cross-validation to respect the time-series
    nature of tennis match data.
    
    Attributes:
        method (str): Tuning method ('grid', 'random', or 'bayesian')
        metric (str): Metric to optimize
        cv_splitter: Cross-validation splitter
        n_jobs (int): Number of parallel jobs
        verbose (int): Verbosity level
        random_state (int): Random state for reproducibility
        results (dict): Results of hyperparameter tuning
    """
    
    def __init__(
        self,
        method: str = 'random',
        metric: str = 'roc_auc',
        cv_splitter: Any = None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42,
        results_dir: str = 'hyperparameter_results'
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            method: Tuning method ('grid', 'random', or 'bayesian')
            metric: Metric to optimize
            cv_splitter: Cross-validation splitter object
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random state for reproducibility
            results_dir: Directory to save results
        """
        self.method = method.lower()
        if self.method not in ['grid', 'random', 'bayesian']:
            raise ValueError("Method must be one of 'grid', 'random', or 'bayesian'")
        
        if self.method == 'bayesian' and not SKOPT_AVAILABLE:
            logger.warning("Scikit-optimize not available. Falling back to random search.")
            self.method = 'random'
            
        self.metric = metric
        self.cv_splitter = cv_splitter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.results = {}
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Set scoring function based on metric
        self._set_scoring_function()
        
        logger.info(f"Initialized HyperparameterTuner with method '{method}'")
    
    def _set_scoring_function(self):
        """Set scoring function based on specified metric."""
        if self.metric == 'accuracy':
            self.scoring_function = accuracy_score
        elif self.metric == 'f1':
            self.scoring_function = f1_score
        elif self.metric == 'roc_auc':
            self.scoring_function = roc_auc_score
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def _evaluate_params(
        self, 
        model_class: Any, 
        params: Dict[str, Any],
        X: pd.DataFrame, 
        y: np.ndarray, 
        dates: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate a single set of hyperparameters using cross-validation.
        
        Args:
            model_class: Model class to instantiate
            params: Hyperparameters to evaluate
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            
        Returns:
            Dictionary with evaluation results
        """
        # Create model with current parameters
        model = model_class(**params)
        
        # Perform cross-validation
        if self.cv_splitter is not None:
            splits = self.cv_splitter.split(X, dates=dates, expanding_window=True)
            scores = []
            
            for train_idx, val_idx in splits:
                # Train model
                model_clone = copy.deepcopy(model)
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model_clone.fit(X_train, y_train)
                
                # Get predictions
                if hasattr(model_clone, 'predict_proba') and self.metric == 'roc_auc':
                    y_pred = model_clone.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model_clone.predict(X_val)
                
                # Calculate score
                score = self.scoring_function(y_val, y_pred)
                scores.append(score)
            
            # Calculate mean score
            mean_score = np.mean(scores)
            std_score = np.std(scores)
        else:
            # If no CV splitter, just fit on full data and return placeholder score
            # This is mainly for testing purposes - not recommended for actual use
            model.fit(X, y)
            mean_score = 0.0
            std_score = 0.0
            
        return {
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score
        }
    
    def _grid_search(
        self, 
        model_class: Any, 
        param_grid: Dict[str, List[Any]],
        X: pd.DataFrame, 
        y: np.ndarray, 
        dates: pd.Series,
        n_iter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            model_class: Model class to tune
            param_grid: Grid of hyperparameters to search
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            n_iter: Not used for grid search
            
        Returns:
            List of dictionaries with evaluation results
        """
        grid = list(ParameterGrid(param_grid))
        logger.info(f"Grid search with {len(grid)} parameter combinations")
        
        # Evaluate each parameter combination
        if self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._evaluate_params)(
                    model_class, params, X, y, dates
                ) for params in grid
            )
        else:
            results = [
                self._evaluate_params(model_class, params, X, y, dates)
                for params in grid
            ]
        
        return results
    
    def _random_search(
        self, 
        model_class: Any, 
        param_distributions: Dict[str, Any],
        X: pd.DataFrame, 
        y: np.ndarray, 
        dates: pd.Series,
        n_iter: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform random search for hyperparameter tuning.
        
        Args:
            model_class: Model class to tune
            param_distributions: Distributions of hyperparameters to sample
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            n_iter: Number of parameter combinations to try
            
        Returns:
            List of dictionaries with evaluation results
        """
        sampler = ParameterSampler(
            param_distributions,
            n_iter=n_iter,
            random_state=self.random_state
        )
        param_list = list(sampler)
        logger.info(f"Random search with {len(param_list)} parameter combinations")
        
        # Evaluate each parameter combination
        if self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._evaluate_params)(
                    model_class, params, X, y, dates
                ) for params in param_list
            )
        else:
            results = [
                self._evaluate_params(model_class, params, X, y, dates)
                for params in param_list
            ]
        
        return results
    
    def _bayesian_search(
        self, 
        model_class: Any, 
        param_distributions: Dict[str, Any],
        X: pd.DataFrame, 
        y: np.ndarray, 
        dates: pd.Series,
        n_iter: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform Bayesian optimization for hyperparameter tuning.
        
        Args:
            model_class: Model class to tune
            param_distributions: Distributions of hyperparameters to sample
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            n_iter: Number of parameter combinations to try
            
        Returns:
            List of dictionaries with evaluation results
        """
        if not SKOPT_AVAILABLE:
            logger.warning("Scikit-optimize not available. Using random search instead.")
            return self._random_search(model_class, param_distributions, X, y, dates, n_iter)
        
        # Convert param_distributions to skopt space
        space = []
        for param_name, param_value in param_distributions.items():
            if isinstance(param_value, list):
                space.append(Categorical(param_value, name=param_name))
            elif isinstance(param_value, tuple) and len(param_value) == 3:
                if isinstance(param_value[0], int) and isinstance(param_value[1], int):
                    space.append(Integer(param_value[0], param_value[1], name=param_name))
                else:
                    space.append(Real(param_value[0], param_value[1], prior=param_value[2], name=param_name))
            else:
                raise ValueError(f"Unsupported parameter distribution for {param_name}")
        
        # Define custom objective function for Bayesian optimization
        def objective(params):
            # Convert list of params to dict
            params_dict = {param.name: value for param, value in zip(space, params)}
            
            # Evaluate parameters
            result = self._evaluate_params(model_class, params_dict, X, y, dates)
            
            # Return negative score (since optimizers minimize)
            return -result['mean_score']
        
        # Run Bayesian optimization
        from skopt import gp_minimize
        
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iter,
            random_state=self.random_state,
            verbose=self.verbose > 0
        )
        
        # Convert skopt result to our format
        results = []
        for i, params in enumerate(result.x_iters):
            params_dict = {param.name: value for param, value in zip(space, params)}
            results.append({
                'params': params_dict,
                'mean_score': -result.func_vals[i],  # Convert back to positive score
                'std_score': 0.0  # Bayesian optimization doesn't provide std
            })
        
        return results
    
    def optimize(
        self, 
        model_class: Any, 
        param_space: Dict[str, Any],
        X: pd.DataFrame, 
        y: np.ndarray, 
        dates: pd.Series,
        n_iter: Optional[int] = None,
        experiment_name: str = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using the specified method.
        
        Args:
            model_class: Model class to tune
            param_space: Hyperparameter space to search
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            n_iter: Number of parameter combinations for random/bayesian search
            experiment_name: Name of this tuning experiment
            
        Returns:
            Dictionary with best parameters and scores
        """
        start_time = time.time()
        logger.info(f"Starting hyperparameter optimization with method '{self.method}'")
        
        # Set default n_iter for random and bayesian search
        if n_iter is None and self.method in ['random', 'bayesian']:
            n_iter = 10
        
        # Generate timestamp for experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            model_name = model_class.__name__ if hasattr(model_class, '__name__') else 'unknown_model'
            experiment_name = f"{model_name}_{self.method}_{timestamp}"
        
        # Run optimization method
        if self.method == 'grid':
            results = self._grid_search(model_class, param_space, X, y, dates)
        elif self.method == 'random':
            results = self._random_search(model_class, param_space, X, y, dates, n_iter)
        elif self.method == 'bayesian':
            results = self._bayesian_search(model_class, param_space, X, y, dates, n_iter)
        
        # Find best parameters
        best_idx = np.argmax([result['mean_score'] for result in results])
        best_result = results[best_idx]
        
        # Store results
        elapsed_time = time.time() - start_time
        self.results[experiment_name] = {
            'method': self.method,
            'metric': self.metric,
            'best_params': best_result['params'],
            'best_score': best_result['mean_score'],
            'all_results': results,
            'n_evaluations': len(results),
            'elapsed_time': elapsed_time,
            'timestamp': timestamp
        }
        
        # Log results
        logger.info(f"Hyperparameter optimization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best {self.metric}: {best_result['mean_score']:.4f}")
        logger.info(f"Best parameters: {best_result['params']}")
        
        # Save results
        self._save_results(experiment_name)
        
        return self.results[experiment_name]
    
    def _save_results(self, experiment_name: str):
        """
        Save optimization results to disk.
        
        Args:
            experiment_name: Name of the experiment to save
        """
        if experiment_name not in self.results:
            raise ValueError(f"No results found for experiment '{experiment_name}'")
        
        # Create file path
        file_path = os.path.join(self.results_dir, f"{experiment_name}.json")
        
        # Create a copy of results for serialization
        results_copy = copy.deepcopy(self.results[experiment_name])
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj
        
        results_copy = convert_to_serializable(results_copy)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Results saved to {file_path}")
    
    def load_results(self, experiment_name: str) -> Dict[str, Any]:
        """
        Load optimization results from disk.
        
        Args:
            experiment_name: Name of the experiment to load
            
        Returns:
            Dictionary with optimization results
        """
        # Create file path
        file_path = os.path.join(self.results_dir, f"{experiment_name}.json")
        
        # Load from file
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        # Store in results dict
        self.results[experiment_name] = results
        
        return results
    
    def plot_optimization_results(
        self, 
        experiment_name: str,
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot optimization results.
        
        Args:
            experiment_name: Name of the experiment to plot
            top_n: Number of top parameter combinations to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if experiment_name not in self.results:
            # Try to load results
            try:
                self.load_results(experiment_name)
            except FileNotFoundError:
                raise ValueError(f"No results found for experiment '{experiment_name}'")
        
        # Get results
        results = self.results[experiment_name]
        all_results = results['all_results']
        
        # Sort results by score
        sorted_results = sorted(
            all_results, 
            key=lambda x: x['mean_score'], 
            reverse=True
        )
        
        # Limit to top_n
        sorted_results = sorted_results[:top_n]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract scores for plotting
        labels = [f"Combo {i+1}" for i in range(len(sorted_results))]
        scores = [result['mean_score'] for result in sorted_results]
        std_scores = [result['std_score'] for result in sorted_results]
        
        # Create bar plot
        bars = ax.bar(labels, scores, yerr=std_scores, capsize=5)
        
        # Add score values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f"{scores[i]:.4f}",
                ha='center',
                va='bottom',
                rotation=0
            )
        
        # Set title and labels
        ax.set_title(f"Top {top_n} Parameter Combinations for {experiment_name}")
        ax.set_xlabel("Parameter Combination")
        ax.set_ylabel(f"{self.metric.upper()}")
        
        # Add table with parameters
        param_names = sorted(sorted_results[0]['params'].keys())
        param_values = []
        for i, result in enumerate(sorted_results):
            row = [f"Combo {i+1}"]
            for param in param_names:
                row.append(str(result['params'].get(param, '')))
            param_values.append(row)
        
        # Create a separate figure for the parameter table
        table_fig, table_ax = plt.subplots(figsize=(12, len(sorted_results) * 0.5))
        table_ax.axis('tight')
        table_ax.axis('off')
        
        table = table_ax.table(
            cellText=param_values,
            colLabels=["Combination"] + param_names,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        table_fig.tight_layout()
        
        return fig, table_fig
    
    def generate_report(self, experiment_name: str) -> str:
        """
        Generate a text report of optimization results.
        
        Args:
            experiment_name: Name of the experiment to report
            
        Returns:
            String with report text
        """
        if experiment_name not in self.results:
            # Try to load results
            try:
                self.load_results(experiment_name)
            except FileNotFoundError:
                raise ValueError(f"No results found for experiment '{experiment_name}'")
        
        # Get results
        results = self.results[experiment_name]
        
        # Generate report
        report = [
            f"Hyperparameter Optimization Report: {experiment_name}",
            f"=" * 50,
            f"Method: {results['method']}",
            f"Metric: {results['metric']}",
            f"Timestamp: {results['timestamp']}",
            f"Elapsed Time: {results['elapsed_time']:.2f} seconds",
            f"Number of Evaluations: {results['n_evaluations']}",
            f"",
            f"Best Score ({results['metric']}): {results['best_score']:.6f}",
            f"",
            f"Best Parameters:",
        ]
        
        # Add best parameters
        for param, value in results['best_score'].items():
            report.append(f"  {param}: {value}")
        
        report.append(f"")
        report.append(f"Top 5 Parameter Combinations:")
        
        # Sort and add top 5 combinations
        sorted_results = sorted(
            results['all_results'], 
            key=lambda x: x['mean_score'], 
            reverse=True
        )[:5]
        
        for i, result in enumerate(sorted_results):
            report.append(f"")
            report.append(f"Combination {i+1}: Score = {result['mean_score']:.6f}")
            for param, value in result['params'].items():
                report.append(f"  {param}: {value}")
        
        return "\n".join(report)


class XGBoostTuner(HyperparameterTuner):
    """
    Specialized hyperparameter tuner for XGBoost models.
    
    Includes specific parameter spaces and optimization strategies for XGBoost.
    """
    
    def __init__(
        self,
        method: str = 'random',
        metric: str = 'roc_auc',
        cv_splitter: Any = None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42,
        results_dir: str = 'hyperparameter_results'
    ):
        """
        Initialize XGBoost tuner.
        
        Args:
            method: Tuning method ('grid', 'random', or 'bayesian')
            metric: Metric to optimize
            cv_splitter: Cross-validation splitter object
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random state for reproducibility
            results_dir: Directory to save results
        """
        super().__init__(
            method=method,
            metric=metric,
            cv_splitter=cv_splitter,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            results_dir=results_dir
        )
    
    def get_default_param_space(self) -> Dict[str, Any]:
        """
        Get default parameter space for XGBoost tuning.
        
        Returns:
            Dictionary with parameter space
        """
        if self.method == 'grid':
            # Smaller space for grid search
            return {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1],
                'min_child_weight': [1, 3]
            }
        else:
            # Larger space for random/bayesian search
            return {
                'max_depth': (3, 12),  # Integer
                'learning_rate': (0.01, 0.3, 'log-uniform'),  # Real, log-uniform prior
                'n_estimators': (50, 500),  # Integer
                'subsample': (0.6, 1.0, 'uniform'),  # Real, uniform prior
                'colsample_bytree': (0.6, 1.0, 'uniform'),  # Real, uniform prior
                'gamma': (0, 0.5, 'uniform'),  # Real, uniform prior
                'min_child_weight': (1, 10),  # Integer
                'reg_alpha': (0, 10, 'log-uniform'),  # Real, log-uniform prior
                'reg_lambda': (1, 10, 'log-uniform')  # Real, log-uniform prior
            }
    
    def optimize_xgboost(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        dates: pd.Series,
        param_space: Optional[Dict[str, Any]] = None,
        n_iter: Optional[int] = None,
        experiment_name: str = None
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.
        
        Args:
            model_class: XGBoost model class
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            param_space: Custom parameter space (if None, use default)
            n_iter: Number of parameter combinations for random/bayesian search
            experiment_name: Name of this tuning experiment
            
        Returns:
            Dictionary with best parameters and scores
        """
        if param_space is None:
            param_space = self.get_default_param_space()
        
        return self.optimize(
            model_class=model_class,
            param_space=param_space,
            X=X,
            y=y,
            dates=dates,
            n_iter=n_iter,
            experiment_name=experiment_name
        )


class NeuralNetTuner(HyperparameterTuner):
    """
    Specialized hyperparameter tuner for neural network models.
    
    Includes specific parameter spaces and optimization strategies for neural networks.
    """
    
    def __init__(
        self,
        method: str = 'random',
        metric: str = 'roc_auc',
        cv_splitter: Any = None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42,
        results_dir: str = 'hyperparameter_results'
    ):
        """
        Initialize neural network tuner.
        
        Args:
            method: Tuning method ('grid', 'random', or 'bayesian')
            metric: Metric to optimize
            cv_splitter: Cross-validation splitter object
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random state for reproducibility
            results_dir: Directory to save results
        """
        super().__init__(
            method=method,
            metric=metric,
            cv_splitter=cv_splitter,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            results_dir=results_dir
        )
    
    def get_default_param_space(self) -> Dict[str, Any]:
        """
        Get default parameter space for neural network tuning.
        
        Returns:
            Dictionary with parameter space
        """
        if self.method == 'grid':
            # Smaller space for grid search
            return {
                'hidden_layers': [[64, 32], [128, 64, 32]],
                'learning_rate': [0.001, 0.01],
                'batch_size': [32, 64],
                'epochs': [50, 100],
                'dropout_rate': [0.2, 0.5],
                'l2_lambda': [0.0001, 0.001],
                'activation': ['relu', 'elu']
            }
        else:
            # Larger space for random/bayesian search
            return {
                'hidden_layers': [
                    [32], [64], [128],
                    [32, 16], [64, 32], [128, 64],
                    [64, 32, 16], [128, 64, 32]
                ],
                'learning_rate': (0.0001, 0.01, 'log-uniform'),
                'batch_size': [16, 32, 64, 128],
                'epochs': (30, 200),
                'dropout_rate': (0.1, 0.7, 'uniform'),
                'l2_lambda': (0.00001, 0.01, 'log-uniform'),
                'activation': ['relu', 'elu', 'selu'],
                'optimizer': ['adam', 'rmsprop']
            }
    
    def optimize_neural_net(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        dates: pd.Series,
        param_space: Optional[Dict[str, Any]] = None,
        n_iter: Optional[int] = None,
        experiment_name: str = None
    ) -> Dict[str, Any]:
        """
        Optimize neural network hyperparameters.
        
        Args:
            model_class: Neural network model class
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            param_space: Custom parameter space (if None, use default)
            n_iter: Number of parameter combinations for random/bayesian search
            experiment_name: Name of this tuning experiment
            
        Returns:
            Dictionary with best parameters and scores
        """
        if param_space is None:
            param_space = self.get_default_param_space()
        
        return self.optimize(
            model_class=model_class,
            param_space=param_space,
            X=X,
            y=y,
            dates=dates,
            n_iter=n_iter,
            experiment_name=experiment_name
        )


class EnsembleTuner(HyperparameterTuner):
    """
    Specialized hyperparameter tuner for ensemble models.
    
    Includes specific parameter spaces and optimization strategies for ensemble models
    that combine multiple base learners.
    """
    
    def __init__(
        self,
        method: str = 'random',
        metric: str = 'roc_auc',
        cv_splitter: Any = None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42,
        results_dir: str = 'hyperparameter_results'
    ):
        """
        Initialize ensemble tuner.
        
        Args:
            method: Tuning method ('grid', 'random', or 'bayesian')
            metric: Metric to optimize
            cv_splitter: Cross-validation splitter object
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random state for reproducibility
            results_dir: Directory to save results
        """
        super().__init__(
            method=method,
            metric=metric,
            cv_splitter=cv_splitter,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            results_dir=results_dir
        )
    
    def get_default_param_space(self) -> Dict[str, Any]:
        """
        Get default parameter space for ensemble model tuning.
        
        Returns:
            Dictionary with parameter space
        """
        if self.method == 'grid':
            # Smaller space for grid search
            return {
                # Random Forest parameters
                'rf_n_estimators': [100, 200],
                'rf_max_depth': [None, 10, 20],
                'rf_min_samples_split': [2, 5],
                
                # Gradient Boosting parameters
                'gb_n_estimators': [100, 200],
                'gb_learning_rate': [0.05, 0.1],
                'gb_max_depth': [3, 5],
                
                # SVM parameters
                'svm_C': [0.1, 1.0],
                'svm_kernel': ['linear', 'rbf'],
                
                # XGBoost parameters
                'xgb_n_estimators': [100, 200],
                'xgb_max_depth': [3, 5],
                'xgb_learning_rate': [0.05, 0.1],
                
                # Meta-learner parameters
                'meta_C': [0.1, 1.0],
                'meta_solver': ['liblinear', 'lbfgs']
            }
        else:
            # Larger space for random/bayesian search
            return {
                # Random Forest parameters
                'rf_n_estimators': (50, 300),
                'rf_max_depth': [None, 5, 10, 15, 20, 25],
                'rf_min_samples_split': (2, 10),
                'rf_min_samples_leaf': (1, 5),
                
                # Gradient Boosting parameters
                'gb_n_estimators': (50, 300),
                'gb_learning_rate': (0.01, 0.2, 'log-uniform'),
                'gb_max_depth': (2, 8),
                'gb_min_samples_split': (2, 10),
                
                # SVM parameters
                'svm_C': (0.01, 10.0, 'log-uniform'),
                'svm_kernel': ['linear', 'rbf', 'poly'],
                'svm_gamma': ['scale', 'auto', 0.1, 0.01],
                
                # XGBoost parameters
                'xgb_n_estimators': (50, 300),
                'xgb_max_depth': (2, 8),
                'xgb_learning_rate': (0.01, 0.2, 'log-uniform'),
                'xgb_subsample': (0.7, 1.0, 'uniform'),
                'xgb_colsample_bytree': (0.7, 1.0, 'uniform'),
                
                # Meta-learner parameters
                'meta_C': (0.01, 10.0, 'log-uniform'),
                'meta_solver': ['liblinear', 'lbfgs', 'newton-cg'],
                'meta_class_weight': [None, 'balanced']
            }
    
    def optimize_ensemble(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        dates: pd.Series,
        param_space: Optional[Dict[str, Any]] = None,
        n_iter: Optional[int] = None,
        experiment_name: str = None
    ) -> Dict[str, Any]:
        """
        Optimize ensemble model hyperparameters.
        
        Args:
            model_class: Ensemble model class
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            param_space: Custom parameter space (if None, use default)
            n_iter: Number of parameter combinations for random/bayesian search
            experiment_name: Name of this tuning experiment
            
        Returns:
            Dictionary with best parameters and scores
        """
        if param_space is None:
            param_space = self.get_default_param_space()
        
        return self.optimize(
            model_class=model_class,
            param_space=param_space,
            X=X,
            y=y,
            dates=dates,
            n_iter=n_iter,
            experiment_name=experiment_name
        )
    
    def get_specific_param_space(self, model_type: str) -> Dict[str, Any]:
        """
        Get parameter space for a specific model type within the ensemble.
        
        Args:
            model_type: Type of model ('rf', 'gb', 'svm', 'xgb', 'nn', 'meta')
                
        Returns:
            Dictionary with parameter space for the specific model
        """
        if model_type == 'rf':
            # Random Forest parameters
            if self.method == 'grid':
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': [None, 'balanced']
                }
            else:
                return {
                    'n_estimators': (50, 300),
                    'max_depth': [None, 5, 10, 15, 20, 25],
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'class_weight': [None, 'balanced', 'balanced_subsample'],
                    'max_features': ['sqrt', 'log2', None]
                }
        
        elif model_type == 'svm':
            # SVM parameters
            if self.method == 'grid':
                return {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'probability': [True],
                    'class_weight': [None, 'balanced']
                }
            else:
                return {
                    'C': (0.01, 100.0, 'log-uniform'),
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 6)),
                    'degree': [2, 3, 4] if 'poly' in self.param_distributions.get('kernel', []) else [3],
                    'probability': [True],
                    'class_weight': [None, 'balanced']
                }
        # Add other model types as needed

    def optimize_multi_objective(
                                self, 
                                model_class: Any, 
                                param_space: Dict[str, Any],
                                X: pd.DataFrame, 
                                y: np.ndarray, 
                                dates: pd.Series,
                                objectives: List[str] = ['roc_auc', 'inference_time'],
                                weights: List[float] = [0.8, 0.2],
                                n_iter: int = 20,
                                experiment_name: str = None
                            ) -> Dict[str, Any]:
        """
        Multi-objective hyperparameter optimization.
        
        Args:
            model_class: Model class to tune
            param_space: Parameter space for tuning
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            objectives: List of objectives to optimize
                Supported: 'roc_auc', 'accuracy', 'f1', 'inference_time', 'memory_usage'
            weights: Weights for each objective (must sum to 1)
            n_iter: Number of iterations
            experiment_name: Name of this tuning experiment
                
        Returns:
            Dictionary with best parameters and Pareto front
        """
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        
        if len(objectives) != len(weights):
            raise ValueError("Must provide a weight for each objective")
        
        # Generate parameter combinations
        if self.method == 'grid':
            param_combinations = list(ParameterGrid(param_space))
            if n_iter is not None and n_iter < len(param_combinations):
                np.random.seed(self.random_state)
                param_combinations = np.random.choice(param_combinations, n_iter, replace=False)
        else:
            sampler = ParameterSampler(
                param_space, n_iter=n_iter, random_state=self.random_state
            )
            param_combinations = list(sampler)
        
        # Evaluate each parameter combination
        results = []
        
        for params in param_combinations:
            # Initialize model
            model = model_class(**params)
            
            # Evaluate primary metric (always uses cross-validation)
            primary_scores = []
            
            for train_idx, val_idx in self.cv_splitter.split(X, dates=dates):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model_clone = copy.deepcopy(model)
                model_clone.fit(X_train, y_train)
                
                # Get predictions
                if hasattr(model_clone, 'predict_proba') and objectives[0] == 'roc_auc':
                    y_pred = model_clone.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred)
                else:
                    y_pred = model_clone.predict(X_val)
                    if objectives[0] == 'accuracy':
                        score = accuracy_score(y_val, y_pred)
                    elif objectives[0] == 'f1':
                        score = f1_score(y_val, y_pred)
                
                primary_scores.append(score)
            
            # Calculate other objectives
            secondary_scores = []
            
            for obj in objectives[1:]:
                if obj == 'inference_time':
                    # Measure inference time
                    import time
                    model.fit(X.iloc[:1000], y[:1000])  # Fit on subset for speed
                    start_time = time.time()
                    model.predict(X.iloc[:1000])
                    end_time = time.time()
                    # Normalize: lower is better, so use negative
                    score = -(end_time - start_time)
                
                elif obj == 'memory_usage':
                    # Estimate memory usage
                    import sys
                    model.fit(X.iloc[:1000], y[:1000])  # Fit on subset for speed
                    # Normalize: lower is better, so use negative
                    score = -sys.getsizeof(model) / 1024 / 1024  # MB
                
                secondary_scores.append(score)
            
            # Calculate weighted score
            mean_primary = np.mean(primary_scores)
            weighted_score = weights[0] * mean_primary
            
            for i, score in enumerate(secondary_scores):
                weighted_score += weights[i+1] * score
            
            # Store result
            results.append({
                'params': params,
                'primary_score': mean_primary,
                'secondary_scores': secondary_scores,
                'weighted_score': weighted_score
            })
        
        # Find Pareto optimal solutions
        pareto_optimal = []
        
        for i, res_i in enumerate(results):
            dominated = False
            
            for j, res_j in enumerate(results):
                if i != j:
                    # Check if res_j dominates res_i
                    if res_j['primary_score'] >= res_i['primary_score']:
                        all_better = True
                        
                        for k in range(len(res_i['secondary_scores'])):
                            if res_j['secondary_scores'][k] <= res_i['secondary_scores'][k]:
                                all_better = False
                                break
                        
                        if all_better:
                            dominated = True
                            break
            
            if not dominated:
                pareto_optimal.append(res_i)
        
        # Get best solution by weighted score
        best_idx = np.argmax([res['weighted_score'] for res in results])
        best_result = results[best_idx]
        
        # Store results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            model_name = model_class.__name__
            experiment_name = f"{model_name}_multi_{timestamp}"
        
        multi_results = {
            'method': self.method,
            'objectives': objectives,
            'weights': weights,
            'best_params': best_result['params'],
            'best_score': best_result['weighted_score'],
            'all_results': results,
            'pareto_front': pareto_optimal,
            'n_evaluations': len(results),
            'timestamp': timestamp
        }
        
        self.results[experiment_name] = multi_results
        
        # Save results
        self._save_results(experiment_name)
        
        return self.results[experiment_name]
    
    def _genetic_algorithm_search(
                                self, 
                                model_class: Any, 
                                param_space: Dict[str, Any],
                                X: pd.DataFrame, 
                                y: np.ndarray, 
                                dates: pd.Series,
                                n_iter: int = 10,
                                population_size: int = 20,
                                mutation_rate: float = 0.1,
                                elite_size: int = 2
                            ) -> List[Dict[str, Any]]:
        """
        Perform genetic algorithm search for hyperparameter tuning.
        
        Args:
            model_class: Model class to tune
            param_space: Parameter space
            X: Feature matrix
            y: Target vector
            dates: Match dates for temporal cross-validation
            n_iter: Number of generations
            population_size: Size of population
            mutation_rate: Probability of mutation
            elite_size: Number of best individuals to keep
                
        Returns:
            List of dictionaries with evaluation results
        """
        # Helper functions for genetic algorithm
        def create_individual():
            """Create a random individual from parameter space."""
            individual = {}
            for param, values in param_space.items():
                if isinstance(values, list):
                    individual[param] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) >= 2:
                    if isinstance(values[0], int) and isinstance(values[1], int):
                        individual[param] = np.random.randint(values[0], values[1]+1)
                    else:
                        if len(values) >= 3 and values[2] == 'log-uniform':
                            # Log-uniform sampling
                            log_min = np.log(values[0])
                            log_max = np.log(values[1])
                            individual[param] = np.exp(np.random.uniform(log_min, log_max))
                        else:
                            # Uniform sampling
                            individual[param] = np.random.uniform(values[0], values[1])
            return individual
        
        def crossover(parent1, parent2):
            """Create child by combining parameters from parents."""
            child = {}
            for param in parent1:
                if np.random.random() < 0.5:
                    child[param] = parent1[param]
                else:
                    child[param] = parent2[param]
            return child
        
        def mutate(individual):
            """Randomly mutate some parameters."""
            mutated = individual.copy()
            for param in mutated:
                if np.random.random() < mutation_rate:
                    values = param_space[param]
                    if isinstance(values, list):
                        mutated[param] = np.random.choice(values)
                    elif isinstance(values, tuple) and len(values) >= 2:
                        if isinstance(values[0], int) and isinstance(values[1], int):
                            mutated[param] = np.random.randint(values[0], values[1]+1)
                        else:
                            if len(values) >= 3 and values[2] == 'log-uniform':
                                log_min = np.log(values[0])
                                log_max = np.log(values[1])
                                mutated[param] = np.exp(np.random.uniform(log_min, log_max))
                            else:
                                mutated[param] = np.random.uniform(values[0], values[1])
            return mutated
        
        # Create initial population
        population = [create_individual() for _ in range(population_size)]
        
        # Evaluate initial population
        evaluated_population = [
            {
                'params': ind,
                'score': self._evaluate_params(model_class, ind, X, y, dates)['mean_score']
            }
            for ind in population
        ]
        
        # Sort by score
        evaluated_population.sort(key=lambda x: x['score'], reverse=True)
        
        # Store all evaluated individuals
        all_evaluated = evaluated_population.copy()
        
        # Run generations
        for generation in range(n_iter):
            # Select parents (tournament selection)
            def select_parent():
                # Select random individuals for tournament
                tournament_size = 3
                tournament = np.random.choice(range(len(evaluated_population)), tournament_size, replace=False)
                tournament = [evaluated_population[i] for i in tournament]
                # Return the best
                return max(tournament, key=lambda x: x['score'])
            
            # Create new population
            new_population = []
            
            # Keep elite individuals
            new_population.extend(evaluated_population[:elite_size])
            
            # Create children
            while len(new_population) < population_size:
                parent1 = select_parent()['params']
                parent2 = select_parent()['params']
                
                child = crossover(parent1, parent2)
                child = mutate(child)
                
                new_population.append(child)
            
            # Evaluate new population
            evaluated_population = []
            for ind in new_population:
                # Check if already evaluated
                existing = next((x for x in all_evaluated if x['params'] == ind), None)
                if existing:
                    evaluated_population.append(existing)
                else:
                    result = {
                        'params': ind,
                        'score': self._evaluate_params(model_class, ind, X, y, dates)['mean_score']
                    }
                    evaluated_population.append(result)
                    all_evaluated.append(result)
            
            # Sort by score
            evaluated_population.sort(key=lambda x: x['score'], reverse=True)
        
        # Format results for return
        return [
            {
                'params': res['params'],
                'mean_score': res['score'],
                'std_score': 0.0  # Not available in genetic algorithm
            }
            for res in all_evaluated
        ]