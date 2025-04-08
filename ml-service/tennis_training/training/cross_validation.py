"""
Temporal Cross Validation for Tennis Match Prediction

This module provides a specialized cross-validation approach that respects
the temporal nature of tennis match data. Traditional k-fold cross-validation
randomly splits data, which is inappropriate for time-series data as it would
allow using future information to predict past events.

The implementation ensures that models are always trained on past data and
validated on future data, mimicking the real-world prediction scenario.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, log_loss
)
from typing import List, Dict, Tuple, Callable, Union, Optional, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configure logger
logger = logging.getLogger(__name__)

class TemporalCrossValidator:
    """
    Implements temporal cross-validation for tennis match prediction models.
    
    This class creates time-based folds for training and validation, ensuring
    that models are always trained on past data and validated on future data.
    
    Attributes:
        n_splits (int): Number of cross-validation splits
        initial_train_ratio (float): Initial ratio of data to use for first training
        gap_days (int): Gap in days between training and validation periods
        validation_window_days (int): Number of days to include in each validation window
        metrics (dict): Dictionary to store evaluation metrics for each fold
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        initial_train_ratio: float = 0.6,
        gap_days: int = 0,
        validation_window_days: int = 90
    ):
        """
        Initialize the temporal cross-validator.
        
        Args:
            n_splits: Number of cross-validation splits
            initial_train_ratio: Ratio of data to use for first training
            gap_days: Gap in days between training and validation periods
            validation_window_days: Number of days to include in each validation window
        """
        self.n_splits = n_splits
        self.initial_train_ratio = initial_train_ratio
        self.gap_days = gap_days
        self.validation_window_days = validation_window_days
        self.metrics = {}
        
        if not 0 < initial_train_ratio < 1:
            raise ValueError("initial_train_ratio must be between 0 and 1")
        
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
            
        logger.info(f"Initialized TemporalCrossValidator with {n_splits} splits")
    
    def _generate_date_splits(
        self, 
        dates: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for temporal cross-validation splits based on dates.
        
        Args:
            dates: Series of datetime objects representing match dates
            
        Returns:
            List of tuples (train_indices, val_indices) for each fold
        """
        # Sort dates and get min/max
        sorted_dates = dates.sort_values().reset_index(drop=True)
        min_date = sorted_dates.min()
        max_date = sorted_dates.max()
        
        # Calculate initial training cutoff date
        total_days = (max_date - min_date).days
        initial_train_days = int(total_days * self.initial_train_ratio)
        initial_train_end = min_date + timedelta(days=initial_train_days)
        
        # Calculate remaining time for validation windows
        remaining_days = total_days - initial_train_days
        
        # If validation_window_days is not set, calculate based on remaining time
        if self.validation_window_days is None:
            # Distribute remaining days across n_splits
            self.validation_window_days = remaining_days // self.n_splits
        
        # Generate splits
        splits = []
        current_train_end = initial_train_end
        
        for i in range(self.n_splits):
            # Define validation window
            val_start = current_train_end + timedelta(days=self.gap_days)
            val_end = val_start + timedelta(days=self.validation_window_days)
            
            # Get indices for this split
            train_indices = np.where(dates < current_train_end)[0]
            val_indices = np.where((dates >= val_start) & (dates < val_end))[0]
            
            # Add split if validation set is not empty
            if len(val_indices) > 0:
                splits.append((train_indices, val_indices))
            
            # Update for next split
            current_train_end = val_end
        
        logger.info(f"Generated {len(splits)} temporal splits")
        return splits
    
    def _generate_expanding_window_splits(
        self, 
        dates: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for expanding window temporal cross-validation.
        
        In this approach, the training window expands to include all past data
        for each new validation period.
        
        Args:
            dates: Series of datetime objects representing match dates
            
        Returns:
            List of tuples (train_indices, val_indices) for each fold
        """
        # Sort dates and get min/max
        sorted_dates = dates.sort_values().reset_index(drop=True)
        min_date = sorted_dates.min()
        max_date = sorted_dates.max()
        
        # Calculate initial training cutoff date
        total_days = (max_date - min_date).days
        initial_train_days = int(total_days * self.initial_train_ratio)
        initial_train_end = min_date + timedelta(days=initial_train_days)
        
        # Calculate validation window size if not specified
        if self.validation_window_days is None:
            remaining_days = total_days - initial_train_days
            self.validation_window_days = remaining_days // self.n_splits
        
        # Generate splits
        splits = []
        current_train_end = initial_train_end
        
        for i in range(self.n_splits):
            # Define validation window
            val_start = current_train_end + timedelta(days=self.gap_days)
            val_end = val_start + timedelta(days=self.validation_window_days)
            
            # Get indices for this split (all past data for training)
            train_indices = np.where(dates < current_train_end)[0]
            val_indices = np.where((dates >= val_start) & (dates < val_end))[0]
            
            # Add split if validation set is not empty
            if len(val_indices) > 0:
                splits.append((train_indices, val_indices))
            
            # Update for next split (only update validation end for expanding window)
            current_train_end = val_end
        
        logger.info(f"Generated {len(splits)} expanding window splits")
        return splits
    
    def split(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        dates: pd.Series,
        expanding_window: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for temporal splits.
        
        Args:
            X: Feature matrix
            dates: Series of datetime objects representing match dates
            expanding_window: Whether to use expanding window approach
            
        Returns:
            List of tuples (train_indices, val_indices) for each fold
        """
        if len(X) != len(dates):
            raise ValueError("X and dates must have the same length")
        
        if expanding_window:
            return self._generate_expanding_window_splits(dates)
        else:
            return self._generate_date_splits(dates)
    
    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        dates: pd.Series,
        expanding_window: bool = True,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        return_models: bool = False
    ) -> Dict[str, List[float]]:
        """
        Evaluate a model using temporal cross-validation.
        
        Args:
            model: Scikit-learn compatible model with fit and predict_proba methods
            X: Feature matrix
            y: Target vector
            dates: Series of datetime objects representing match dates
            expanding_window: Whether to use expanding window approach
            custom_metrics: Dictionary of custom metric functions
            return_models: Whether to return trained models for each fold
            
        Returns:
            Dictionary of evaluation metrics for each fold
        """
        splits = self.split(X, dates, expanding_window)
        
        # Initialize metrics dictionary
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'log_loss': [],
            'train_size': [],
            'val_size': [],
            'train_start_date': [],
            'train_end_date': [],
            'val_start_date': [],
            'val_end_date': []
        }
        
        # Add custom metrics if provided
        if custom_metrics:
            for metric_name in custom_metrics:
                metrics[metric_name] = []
        
        # Store trained models if requested
        trained_models = []
        
        # Evaluate for each split
        for i, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Evaluating fold {i+1}/{len(splits)}")
            
            # Get train/validation sets
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Record date ranges
            train_dates = dates.iloc[train_idx]
            val_dates = dates.iloc[val_idx]
            metrics['train_start_date'].append(train_dates.min())
            metrics['train_end_date'].append(train_dates.max())
            metrics['val_start_date'].append(val_dates.min())
            metrics['val_end_date'].append(val_dates.max())
            
            # Record sizes
            metrics['train_size'].append(len(X_train))
            metrics['val_size'].append(len(X_val))
            
            # Clone and train model
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Store trained model if requested
            if return_models:
                trained_models.append(fold_model)
            
            # Make predictions
            y_pred = fold_model.predict(X_val)
            try:
                y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
                has_proba = True
            except (AttributeError, IndexError):
                has_proba = False
            
            # Calculate standard metrics
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            
            if has_proba:
                metrics['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
                metrics['log_loss'].append(log_loss(y_val, y_pred_proba))
            else:
                metrics['roc_auc'].append(np.nan)
                metrics['log_loss'].append(np.nan)
            
            # Calculate custom metrics if provided
            if custom_metrics:
                for metric_name, metric_func in custom_metrics.items():
                    if has_proba:
                        metrics[metric_name].append(metric_func(y_val, y_pred_proba))
                    else:
                        metrics[metric_name].append(metric_func(y_val, y_pred))
        
        # Store metrics
        self.metrics = metrics
        
        # Return metrics (and models if requested)
        if return_models:
            return metrics, trained_models
        return metrics
    
    def plot_metrics(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot evaluation metrics across folds.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot standard metrics
        standard_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for i, metric in enumerate(standard_metrics):
            if metric in self.metrics:
                axes[i].plot(range(1, len(self.metrics[metric]) + 1), self.metrics[metric], 'o-')
                axes[i].set_title(f'{metric.capitalize()} by Fold')
                axes[i].set_xlabel('Fold')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_temporal_performance(self, figsize: Tuple[int, int] = (14, 7)) -> plt.Figure:
        """
        Plot model performance over time.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get date midpoints for each validation period
        val_midpoints = [
            start + (end - start) / 2 
            for start, end in zip(
                self.metrics['val_start_date'], 
                self.metrics['val_end_date']
            )
        ]
        
        # Plot metrics over time
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics_to_plot:
            if metric in self.metrics:
                ax.plot(val_midpoints, self.metrics[metric], 'o-', label=metric.capitalize())
        
        # Add validation period indicators
        for i, (start, end) in enumerate(zip(
            self.metrics['val_start_date'], 
            self.metrics['val_end_date']
        )):
            ax.axvspan(start, end, alpha=0.1, color='gray')
            
        ax.set_title('Model Performance Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis as dates
        fig.autofmt_xdate()
        
        return fig
    
    def summary(self) -> pd.DataFrame:
        """
        Generate a summary of cross-validation results.
        
        Returns:
            DataFrame with summary statistics for each metric
        """
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        # Select numerical metrics
        numerical_metrics = [
            k for k, v in self.metrics.items() 
            if isinstance(v, list) and len(v) > 0 and 
            isinstance(v[0], (int, float)) and 'date' not in k and 'size' not in k
        ]
        
        # Create summary dataframe
        summary = pd.DataFrame({
            'Metric': numerical_metrics,
            'Mean': [np.mean(self.metrics[m]) for m in numerical_metrics],
            'Std': [np.std(self.metrics[m]) for m in numerical_metrics],
            'Min': [np.min(self.metrics[m]) for m in numerical_metrics],
            'Max': [np.max(self.metrics[m]) for m in numerical_metrics],
            'Trend': [
                'Improving' if self.metrics[m][-1] > self.metrics[m][0] else
                'Degrading' if self.metrics[m][-1] < self.metrics[m][0] else
                'Stable'
                for m in numerical_metrics
            ]
        })
        
        return summary
    
    def _create_balanced_indices(self, y, indices):
        """
        Create balanced indices for training to handle class imbalance.
        
        Args:
            y: Target vector
            indices: Original indices
            
        Returns:
            Balanced indices
        """
        y_subset = y[indices]
        
        # Count samples per class
        class_counts = np.bincount(y_subset)
        min_class_count = np.min(class_counts)
        
        balanced_indices = []
        
        # Sample equal number of instances from each class
        for class_value in range(len(class_counts)):
            class_indices = indices[y_subset == class_value]
            
            # If we have more samples than needed, undersample
            if len(class_indices) > min_class_count:
                np.random.shuffle(class_indices)
                class_indices = class_indices[:min_class_count]
            
            balanced_indices.extend(class_indices)
        
        return np.array(balanced_indices)

    def evaluate_with_balancing(
                            self,
                            model: Any,
                            X: pd.DataFrame,
                            y: np.ndarray,
                            dates: pd.Series,
                            expanding_window: bool = True,
                            custom_metrics: Optional[Dict[str, Callable]] = None,
                            balance_classes: bool = True,
                            return_models: bool = False
                        ) -> Dict[str, List[float]]:
        """
        Evaluate a model using temporal cross-validation with optional class balancing.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            dates: Series of datetime objects representing match dates
            expanding_window: Whether to use expanding window approach
            custom_metrics: Dictionary of custom metric functions
            balance_classes: Whether to balance classes in training data
            return_models: Whether to return trained models for each fold
            
        Returns:
            Dictionary of evaluation metrics for each fold
        """
        # Get splits
        splits = self.split(X, dates, expanding_window)
        
        # Initialize metrics dictionary
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'log_loss': [],
            'train_size': [],
            'val_size': [],
            'train_start_date': [],
            'train_end_date': [],
            'val_start_date': [],
            'val_end_date': [],
            'class_balance': []  # Added to track class balance
        }
        
        # Add custom metrics if provided
        if custom_metrics:
            for metric_name in custom_metrics:
                metrics[metric_name] = []
        
        # Store trained models if requested
        trained_models = []
        
        # Evaluate for each split
        for i, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Evaluating fold {i+1}/{len(splits)}")
            
            # Balance training indices if requested
            if balance_classes:
                original_train_size = len(train_idx)
                train_idx = self._create_balanced_indices(y, train_idx)
                logger.info(f"Balanced training set from {original_train_size} to {len(train_idx)} samples")
                
                # Record class balance information
                train_class_counts = np.bincount(y[train_idx])
                metrics['class_balance'].append(train_class_counts[1] / len(train_idx) if len(train_idx) > 0 else 0)
            else:
                # Record original class balance
                train_class_counts = np.bincount(y[train_idx])
                metrics['class_balance'].append(train_class_counts[1] / len(train_idx) if len(train_idx) > 0 else 0)
            
            # Get train/validation sets
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Record date ranges
            train_dates = dates.iloc[train_idx]
            val_dates = dates.iloc[val_idx]
            metrics['train_start_date'].append(train_dates.min())
            metrics['train_end_date'].append(train_dates.max())
            metrics['val_start_date'].append(val_dates.min())
            metrics['val_end_date'].append(val_dates.max())
            
            # Record sizes
            metrics['train_size'].append(len(X_train))
            metrics['val_size'].append(len(X_val))
            
            # Clone and train model
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Store trained model if requested
            if return_models:
                trained_models.append(fold_model)
            
            # Make predictions
            y_pred = fold_model.predict(X_val)
            try:
                y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
                has_proba = True
            except (AttributeError, IndexError):
                has_proba = False
            
            # Calculate standard metrics
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            
            if has_proba:
                metrics['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
                metrics['log_loss'].append(log_loss(y_val, y_pred_proba))
            else:
                metrics['roc_auc'].append(np.nan)
                metrics['log_loss'].append(np.nan)
            
            # Calculate custom metrics if provided
            if custom_metrics:
                for metric_name, metric_func in custom_metrics.items():
                    if has_proba:
                        metrics[metric_name].append(metric_func(y_val, y_pred_proba))
                    else:
                        metrics[metric_name].append(metric_func(y_val, y_pred))
        
        # Store metrics
        self.metrics = metrics
        
        # Return metrics (and models if requested)
        if return_models:
            return metrics, trained_models
        return metrics

    def forecast_performance_trend(self, return_coefficients: bool = False):
        """
        Estimate future performance trend based on historical performance.
        
        Args:
            return_coefficients: Whether to return regression coefficients
            
        Returns:
            Estimated future performance or regression results
        """
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        # Get key metrics and dates
        metric_values = np.array(self.metrics['accuracy'])  # Or other metrics
        val_midpoints = np.array([
            (start + (end - start) / 2).timestamp() 
            for start, end in zip(
                self.metrics['val_start_date'], 
                self.metrics['val_end_date']
            )
        ])
        
        # Fit linear regression to detect trend
        from sklearn.linear_model import LinearRegression
        X_time = val_midpoints.reshape(-1, 1)
        model = LinearRegression().fit(X_time, metric_values)
        
        # Calculate future points (next 3 periods)
        time_delta = np.mean(np.diff(val_midpoints))
        future_points = np.array([
            val_midpoints[-1] + time_delta * (i+1) 
            for i in range(3)
        ]).reshape(-1, 1)
        
        # Predict future performance
        future_performance = model.predict(future_points)
        
        # Create result dict
        result = {
            'current_performance': metric_values[-1],
            'trend_slope': model.coef_[0],
            'forecast': future_performance,
            'forecast_dates': [
                datetime.fromtimestamp(ts[0]) 
                for ts in future_points
            ]
        }
        
        if return_coefficients:
            result['model'] = model
        
        return result
    
    def _generate_stratified_temporal_splits(
                                            self, 
                                            dates: pd.Series,
                                            y: np.ndarray,
                                            n_bins: int = 5
                                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate temporally stratified splits to maintain similar class distributions.
        
        Args:
            dates: Series of datetime objects representing match dates
            y: Target vector
            n_bins: Number of bins to stratify within
            
        Returns:
            List of tuples (train_indices, val_indices) for each fold
        """
        # Sort dates and get min/max
        sorted_indices = dates.sort_values().index
        sorted_dates = dates.loc[sorted_indices]
        sorted_y = y[sorted_indices]
        
        min_date = sorted_dates.min()
        max_date = sorted_dates.max()
        
        # Calculate initial training cutoff date
        total_days = (max_date - min_date).days
        initial_train_days = int(total_days * self.initial_train_ratio)
        initial_train_end = min_date + timedelta(days=initial_train_days)
        
        # Generate time bins
        time_bins = pd.cut(sorted_dates, bins=n_bins)
        bin_indices = {bin_name: [] for bin_name in time_bins.unique()}
        
        # Group indices by time bin and class
        for i, (bin_name, y_value) in enumerate(zip(time_bins, sorted_y)):
            if bin_name in bin_indices:
                bin_indices[bin_name].append((sorted_indices[i], y_value))
        
        # Generate splits similar to expanding window but with stratification
        splits = []
        current_train_end = initial_train_end
        
        for i in range(self.n_splits):
            # Define validation window
            val_start = current_train_end + timedelta(days=self.gap_days)
            val_end = val_start + timedelta(days=self.validation_window_days)
            
            # Get indices for this split with basic temporal logic
            train_mask = sorted_dates < current_train_end
            val_mask = (sorted_dates >= val_start) & (sorted_dates < val_end)
            
            train_indices = sorted_indices[train_mask]
            val_indices = sorted_indices[val_mask]
            
            # Add split if validation set is not empty
            if len(val_indices) > 0:
                splits.append((train_indices, val_indices))
            
            # Update for next split
            current_train_end = val_end
        
        return splits

    def split_stratified(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: np.ndarray,
        dates: pd.Series,
        n_bins: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for temporally stratified splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            dates: Series of datetime objects representing match dates
            n_bins: Number of bins to stratify within
            
        Returns:
            List of tuples (train_indices, val_indices) for each fold
        """
        if len(X) != len(dates) or len(X) != len(y):
            raise ValueError("X, y, and dates must have the same length")
        
        return self._generate_stratified_temporal_splits(dates, y, n_bins)