# models/base_model.py
from abc import ABC, abstractmethod
import json
import os
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt

class TennisModel(ABC):
    """
    Abstract base class for tennis prediction models.
    Defines a common interface that all model implementations must follow.
    """
    
    def __init__(self, name, version="1.0.0"):
        """
        Initialize the base model with a name and version.
        
        Args:
            name (str): Name of the model
            version (str): Version string in semantic versioning format
        """
        self.name = name
        self.version = version
        self.model = None
        self.feature_names = None
        self.metrics = {
            'training': {},
            'validation': {},
            'test': {}
        }
        self.training_history = []
        self.trained = False
        self.training_date = None
        self.model_params = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        
    @abstractmethod
    def fit(self, X, y, validation_data=None, **kwargs):
        """
        Train the model on the provided data.
        
        Args:
            X: Features data
            y: Target data
            validation_data: Optional tuple of (X_val, y_val) for validation during training
            **kwargs: Additional parameters for model training
            
        Returns:
            self: The trained model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make binary predictions with the trained model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Make probability predictions with the trained model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Probability predictions (between 0 and 1)
        """
        pass
    
    def evaluate(self, X, y, dataset_name="test"):
        """
        Evaluate the model on the provided data and store metrics.
        
        Args:
            X: Features data
            y: True target values
            dataset_name (str): Name of the dataset for storing metrics
                               (training, validation, or test)
        
        Returns:
            dict: Dictionary containing the evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'log_loss': log_loss(y, y_proba)
        }
        
        # Add betting metrics
        metrics['kelly_criterion'] = self._calculate_kelly_criterion(y, y_proba)
        metrics['calibration_error'] = self._calibration_error(y, y_proba)
        
        # Store metrics
        self.metrics[dataset_name] = metrics
        
        return metrics
    
    def _calculate_kelly_criterion(self, y_true, y_pred, odds=2.0):
        """
        Calculate expected value using Kelly Criterion for tennis betting.
        
        Args:
            y_true: True outcomes
            y_pred: Predicted probabilities
            odds: Fixed odds to use for calculation (default: 2.0)
            
        Returns:
            float: Average Kelly stake
        """
        # For each prediction, calculate optimal Kelly stake
        kelly_stakes = []
        
        for p, y in zip(y_pred, y_true):
            # Kelly formula: f* = (bp - q) / b
            # where b = odds - 1, p = probability of winning, q = 1 - p
            b = odds - 1
            stake = (b * p - (1 - p)) / b
            
            # Limit to positive stakes only (don't bet when negative)
            stake = max(0, stake)
            
            # Store result
            kelly_stakes.append(stake)
            
        return np.mean(kelly_stakes)
    
    def _calibration_error(self, y_true, y_pred, bins=10):
        """
        Calculate calibration error (difference between predicted probabilities
        and observed frequencies).
        
        Args:
            y_true: True outcomes
            y_pred: Predicted probabilities
            bins: Number of bins for grouping probabilities
            
        Returns:
            float: Mean absolute calibration error
        """
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(y_pred, bin_edges) - 1
        
        # Clip to ensure we stay within valid bin range
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        
        errors = []
        for bin_idx in range(bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 0:  # Ensure we have samples in this bin
                mean_pred_prob = np.mean(y_pred[mask])
                observed_freq = np.mean(y_true[mask])
                errors.append(abs(mean_pred_prob - observed_freq))
        
        return np.mean(errors) if errors else 0.0
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot feature importance if the model supports it.
        
        Args:
            top_n (int): Number of top features to display
            save_path (str): Path to save the plot image
            
        Returns:
            None
        """
        try:
            importances = self.feature_importances()
            if self.feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            else:
                feature_names = self.feature_names
                
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Select top_n features
            indices = indices[:top_n]
            top_importances = importances[indices]
            top_names = [feature_names[i] for i in indices]
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.title(f"Top {top_n} Feature Importances - {self.name}")
            plt.barh(range(len(top_importances)), top_importances, align='center')
            plt.yticks(range(len(top_importances)), top_names)
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()  # Show highest importance at the top
            
            if save_path:
                plt.tight_layout()
                plt.savefig(save_path)
                self.logger.info(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()
            
        except (AttributeError, NotImplementedError):
            self.logger.warning("Feature importance plotting not supported for this model")
    
    @abstractmethod
    def feature_importances(self):
        """
        Get the feature importances from the model if available.
        
        Returns:
            numpy.ndarray: Feature importance scores
        """
        pass
    
    def save(self, directory, include_history=True):
        """
        Save the model and its metadata to the specified directory.
        
        Args:
            directory (str): Directory path to save to
            include_history (bool): Whether to include training history
            
        Returns:
            str: Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Create metadata
        metadata = {
            'name': self.name,
            'version': self.version,
            'model_type': self.__class__.__name__,
            'trained': self.trained,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'metrics': self.metrics,
            'model_params': self.model_params,
            'feature_names': self.feature_names
        }
        
        if include_history:
            metadata['training_history'] = self.training_history
        
        # Save metadata
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log the save operation
        self.logger.info(f"Model metadata saved to {metadata_path}")
        
        # Let subclasses implement their own model saving
        model_path = self._save_model(directory)
        
        return model_path
    
    @abstractmethod
    def _save_model(self, directory):
        """
        Save the model implementation to a file.
        To be implemented by subclasses.
        
        Args:
            directory (str): Directory to save the model to
            
        Returns:
            str: Path to the saved model file
        """
        pass
    
    @abstractmethod
    def load(self, directory):
        """
        Load a previously saved model.
        
        Args:
            directory (str): Directory containing the saved model
            
        Returns:
            self: The loaded model instance
        """
        pass
    
    def summary(self):
        """
        Return a summary of the model and its performance.
        
        Returns:
            str: Text summary of the model
        """
        if not self.trained:
            return f"Model {self.name} (v{self.version}) - Not trained"
        
        summary_text = [
            f"Model: {self.name} (v{self.version})",
            f"Type: {self.__class__.__name__}",
            f"Trained: {self.trained}",
            f"Training Date: {self.training_date}",
            "\nPerformance Metrics:"
        ]
        
        for dataset_name, metrics in self.metrics.items():
            if metrics:
                summary_text.append(f"\n{dataset_name.capitalize()} set:")
                for metric_name, value in metrics.items():
                    summary_text.append(f"  {metric_name}: {value:.4f}")
        
        return "\n".join(summary_text)