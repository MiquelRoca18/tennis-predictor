# models/calibrated_model.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from .base_model import TennisModel

class CalibratedTennisModel(TennisModel):
    """
    A wrapper model that calibrates the probability outputs of any tennis
    prediction model to ensure well-calibrated probabilities.
    
    Tennis prediction models often output probabilities that are not well-calibrated,
    meaning that events predicted with probability p don't occur with frequency p.
    This is critical for betting applications where accurate probability estimation
    (not just correct classification) is important.
    """
    
    def __init__(self, base_model, name=None, version="1.0.0", method='isotonic', 
                 cv=5, random_state=42):
        """
        Initialize the calibrated model wrapper.
        
        Args:
            base_model (TennisModel): Base model to calibrate
            name (str): Name of the model (defaults to "calibrated_{base_model.name}")
            version (str): Version string
            method (str): Calibration method - 'sigmoid' (Platt scaling) or 'isotonic'
            cv (int): Number of cross-validation folds for calibration
            random_state (int): Random seed for reproducibility
        """
        if name is None:
            name = f"calibrated_{base_model.name}"
            
        super().__init__(name, version)
        
        self.base_model = base_model
        self.method = method
        self.cv = cv
        self.random_state = random_state
        
        # Initialize calibration model
        self.calibrator = None
        
        # Set model parameters
        self.model_params = {
            'method': method,
            'cv': cv,
            'random_state': random_state,
            'base_model_name': base_model.name,
            'base_model_type': base_model.__class__.__name__
        }
        
        # Setup logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def fit(self, X, y, validation_data=None, calibration_fraction=0.2, **kwargs):
        """
        Train the base model and calibrate its probability predictions.
        
        Args:
            X: Features data
            y: Target data
            validation_data: Optional tuple of (X_val, y_val) for validation
            calibration_fraction (float): Fraction of training data to use for calibration
            **kwargs: Additional parameters passed to base model's fit method
            
        Returns:
            self: The trained model instance
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            
        # Store training start time
        start_time = datetime.now()
        
        # If the base model is already trained, we'll use it directly
        if not self.base_model.trained:
            self.logger.info("Training base model")
            self.base_model.fit(X, y, validation_data=validation_data, **kwargs)
        else:
            self.logger.info("Using pre-trained base model")
        
        # We need raw predictions for calibration, not post-processed ones
        self.logger.info(f"Calibrating probabilities using {self.method} method")
        
        # Split data for calibration if no validation data is provided
        if validation_data is not None:
            X_calib, y_calib = validation_data
            if isinstance(X_calib, pd.DataFrame):
                X_calib = X_calib.values
        elif calibration_fraction > 0:
            # Split off some data for calibration
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_array, y, 
                test_size=calibration_fraction,
                random_state=self.random_state,
                stratify=y
            )
            
            # Retrain base model on reduced training set if it wasn't pre-trained
            if not getattr(self.base_model, '_was_pretrained', False):
                self.base_model.fit(X_train, y_train, **kwargs)
        else:
            # Use cross-validation for calibration
            X_calib, y_calib = X_array, y
        
        # Get base model predictions for calibration
        y_pred_proba = self.base_model.predict_proba(X_calib)
        
        # Train the calibrator based on the method
        if self.method == 'sigmoid':
            # Platt scaling (logistic regression)
            self.calibrator = LogisticRegression(C=1.0, solver='lbfgs')
            self.calibrator.fit(y_pred_proba.reshape(-1, 1), y_calib)
        elif self.method == 'isotonic':
            # Isotonic regression (non-parametric monotonic mapping)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_calib)
        elif self.method == 'sklearn':
            # Use scikit-learn's CalibratedClassifierCV
            # This is more complex but potentially more robust
            # We create a wrapper for the base model to match sklearn's API
            class BaseModelWrapper:
                def __init__(self, model):
                    self.model = model
                
                def predict_proba(self, X):
                    probs = self.model.predict_proba(X)
                    # Convert to 2D array with [P(y=0), P(y=1)]
                    return np.vstack([1-probs, probs]).T
                
                def fit(self, X, y):
                    # This should not be called by CalibratedClassifierCV
                    # when cv='prefit'
                    pass
            
            wrapper = BaseModelWrapper(self.base_model)
            self.calibrator = CalibratedClassifierCV(
                wrapper, cv='prefit', method='isotonic' if self.method == 'isotonic_cv' else 'sigmoid'
            )
            self.calibrator.fit(X_calib, y_calib)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        # Evaluate calibration performance
        calibration_metrics = self._evaluate_calibration(X_calib, y_calib)
        self.logger.info(f"Calibration metrics: {calibration_metrics}")
        
        # Update model metadata
        self.trained = True
        self.training_date = datetime.now()
        
        # Calculate training duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Record training run in history
        training_record = {
            'date': self.training_date.isoformat(),
            'duration_seconds': duration,
            'calibration_method': self.method,
            'calibration_metrics': calibration_metrics,
            'samples': len(X),
            'base_model_name': self.base_model.name,
            'base_model_type': self.base_model.__class__.__name__
        }
        
        self.training_history.append(training_record)
        
        self.logger.info(f"Probability calibration completed in {duration:.2f} seconds")
        
        return self
    
    def _evaluate_calibration(self, X, y, n_bins=10):
        """
        Evaluate how well calibrated the model probabilities are.
        
        Args:
            X: Features data
            y: True labels
            n_bins (int): Number of bins for calibration curve
            
        Returns:
            dict: Calibration metrics
        """
        # Get raw base model predictions
        y_base_proba = self.base_model.predict_proba(X)
        
        # Get calibrated predictions
        y_cal_proba = self.predict_proba(X)
        
        # Compute calibration curves
        base_prob_true, base_prob_pred = calibration_curve(y, y_base_proba, n_bins=n_bins)
        cal_prob_true, cal_prob_pred = calibration_curve(y, y_cal_proba, n_bins=n_bins)
        
        # Calculate calibration error (Mean Absolute Error between
        # predicted probabilities and observed frequencies)
        base_cal_error = np.mean(np.abs(base_prob_true - base_prob_pred))
        cal_cal_error = np.mean(np.abs(cal_prob_true - cal_prob_pred))
        
        # Calculate Brier score (mean squared error between
        # predicted probabilities and actual outcomes)
        base_brier = np.mean((y_base_proba - y) ** 2)
        cal_brier = np.mean((y_cal_proba - y) ** 2)
        
        # Return metrics
        return {
            'base_calibration_error': base_cal_error,
            'calibrated_calibration_error': cal_cal_error,
            'calibration_improvement': base_cal_error - cal_cal_error,
            'base_brier_score': base_brier,
            'calibrated_brier_score': cal_brier,
            'brier_improvement': base_brier - cal_brier
        }
    
    def predict(self, X):
        """
        Make binary predictions with the calibrated model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        # Get calibrated probabilities
        proba = self.predict_proba(X)
        
        # Convert to binary predictions
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Make calibrated probability predictions.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Calibrated probability predictions
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Get base model raw predictions
        base_proba = self.base_model.predict_proba(X)
        
        # Apply calibration based on the method
        if self.method in ['sigmoid', 'isotonic']:
            # Apply the trained calibrator
            calibrated_proba = self.calibrator.predict(
                base_proba.reshape(-1, 1) if self.method == 'sigmoid' else base_proba
            )
            
            return calibrated_proba
        elif self.method == 'sklearn':
            # For scikit-learn's CalibratedClassifierCV
            return self.calibrator.predict_proba(X)[:, 1]
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
    
    def feature_importances(self):
        """
        Get feature importances from the base model.
        
        Returns:
            numpy.ndarray: Feature importance scores
        """
        # Delegated to base model
        return self.base_model.feature_importances()
    
    def plot_calibration_curve(self, X, y, n_bins=10, save_path=None):
        """
        Plot the calibration curve before and after calibration.
        
        Args:
            X: Features data
            y: True labels
            n_bins (int): Number of bins for calibration curve
            save_path (str): Path to save the plot image
            
        Returns:
            None
        """
        if not self.trained:
            raise ValueError("Model must be trained before plotting calibration curve")
            
        # Get predictions
        y_base_proba = self.base_model.predict_proba(X)
        y_cal_proba = self.predict_proba(X)
        
        # Compute calibration curves
        base_prob_true, base_prob_pred = calibration_curve(y, y_base_proba, n_bins=n_bins)
        cal_prob_true, cal_prob_pred = calibration_curve(y, y_cal_proba, n_bins=n_bins)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Plot perfectly calibrated line
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        
        # Plot base model calibration
        plt.plot(base_prob_pred, base_prob_true, marker='o', linewidth=1, label=f'Base model ({self.base_model.name})')
        
        # Plot calibrated model
        plt.plot(cal_prob_pred, cal_prob_true, marker='s', linewidth=1, label=f'Calibrated ({self.method})')
        
        # Add labels and legend
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve')
        plt.legend(loc='best')
        
        # Add grid for readability
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Calibration curve saved to {save_path}")
        else:
            plt.show()
    
    def _save_model(self, directory):
        """
        Save the model implementation to files.
        
        Args:
            directory (str): Directory to save the model to
            
        Returns:
            str: Path to the saved model directory
        """
        # Create a model-specific directory
        model_dir = os.path.join(directory, self.name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the base model
        base_model_dir = os.path.join(model_dir, "base_model")
        os.makedirs(base_model_dir, exist_ok=True)
        self.base_model.save(base_model_dir)
        
        # Save the calibrator
        calibrator_path = os.path.join(model_dir, "calibrator.joblib")
        dump(self.calibrator, calibrator_path)
        
        # Save additional parameters
        params_path = os.path.join(model_dir, "params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'cv': self.cv,
                'random_state': self.random_state,
                'feature_names': self.feature_names
            }, f)
            
        self.logger.info(f"Calibrated model saved to {model_dir}")
        
        return model_dir
    
    def load(self, directory):
        """
        Load a previously saved model.
        
        Args:
            directory (str): Directory containing the saved model
            
        Returns:
            self: The loaded model instance
        """
        # If directory is a specific model directory
        if os.path.basename(directory) == self.name:
            model_dir = directory
        else:
            # Otherwise, it's the parent directory
            model_dir = os.path.join(directory, self.name)
            
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist")
            
        # Load parameters
        params_path = os.path.join(model_dir, "params.pkl")
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            
        self.method = params['method']
        self.cv = params['cv']
        self.random_state = params['random_state']
        self.feature_names = params['feature_names']
        
        # Load calibrator
        calibrator_path = os.path.join(model_dir, "calibrator.joblib")
        self.calibrator = load(calibrator_path)
        
        # Load base model
        base_model_dir = os.path.join(model_dir, "base_model")
        
        # We need to know the base model type to load it
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            base_model_type = metadata['model_params']['base_model_type']
            base_model_name = metadata['model_params']['base_model_name']
            
            # Import the base model class dynamically
            from . import MODEL_REGISTRY
            
            # Create an instance of the base model
            for model_key, model_class in MODEL_REGISTRY.items():
                if model_class.__name__ == base_model_type:
                    self.base_model = model_class(name=base_model_name)
                    break
            else:
                raise ValueError(f"Base model type {base_model_type} not found in registry")
                
            # Load the base model
            self.base_model.load(base_model_dir)
            
            # Load own metadata
            self.version = metadata['version']
            self.trained = metadata['trained']
            self.training_date = datetime.fromisoformat(metadata['training_date']) if metadata['training_date'] else None
            self.metrics = metadata['metrics']
            
            if 'training_history' in metadata:
                self.training_history = metadata['training_history']
                
        else:
            raise ValueError(f"Metadata file for {self.name} not found")
            
        self.logger.info(f"Calibrated model loaded from {model_dir}")
        
        return self