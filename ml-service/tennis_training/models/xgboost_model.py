# models/xgboost_model.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from joblib import dump, load

from .base_model import TennisModel

class TennisXGBoostModel(TennisModel):
    """
    XGBoost model for tennis prediction with automated hyperparameter 
    optimization and temporal validation.
    """
    
    def __init__(self, name="tennis_xgboost", version="1.0.0", random_state=42, 
                 params=None, optimize_hyperparams=False):
        """
        Initialize the XGBoost model with configurable parameters.
        
        Args:
            name (str): Name of the model
            version (str): Version string
            random_state (int): Random seed for reproducibility
            params (dict): XGBoost parameters
            optimize_hyperparams (bool): Whether to perform hyperparameter optimization
        """
        super().__init__(name, version)
        self.random_state = random_state
        self.optimize_hyperparams = optimize_hyperparams
        
        # Set default params if none provided
        if params is None:
            self.params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 5,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'scale_pos_weight': 1,
                'seed': random_state,
                'use_label_encoder': False,
                'verbosity': 0
            }
        else:
            self.params = params
            
        # Set the model_params attribute
        self.model_params = {
            'params': self.params,
            'random_state': random_state,
            'optimize_hyperparams': optimize_hyperparams
        }
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        
        # Setup logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        
        # Hyperparameter search spaces
        self.hyper_param_spaces = {
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'scale_pos_weight': [1, 3, 5, 7]
        }
    
    def fit(self, X, y, validation_data=None, early_stopping_rounds=20,
            optimize_hyperparams=None, cv=5, **kwargs):
        """
        Train the XGBoost model on the provided data.
        
        Args:
            X: Features data
            y: Target data
            validation_data: Optional tuple of (X_val, y_val) for validation
            early_stopping_rounds (int): Stops training when validation score 
                                        doesn't improve for this many rounds
            optimize_hyperparams (bool): Whether to perform hyperparameter optimization
            cv (int): Number of cross-validation folds for hyperparameter optimization
            **kwargs: Additional parameters passed to XGBoost's fit method
            
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
        
        # Decide whether to optimize hyperparameters
        if optimize_hyperparams is not None:
            self.optimize_hyperparams = optimize_hyperparams
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Perform hyperparameter optimization if requested
        if self.optimize_hyperparams:
            self.logger.info("Starting hyperparameter optimization")
            self._hyperparameter_optimization(X_scaled, y, cv=cv)
            
        # Create and train the model with the selected parameters
        self.model = xgb.XGBClassifier(**self.params)
        
        # If validation data is provided, use it for early stopping
        if validation_data is not None:
            X_val, y_val = validation_data
            
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
                
            X_val_scaled = self.scaler.transform(X_val)
            
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_scaled, y), (X_val_scaled, y_val)],
                eval_metric='logloss',
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
                **kwargs
            )
            
            # Extract validation metrics
            evals_result = self.model.evals_result()
            train_logloss = evals_result['validation_0']['logloss'][-1]
            val_logloss = evals_result['validation_1']['logloss'][-1]
            
            self.metrics['training']['log_loss'] = train_logloss
            self.metrics['validation']['log_loss'] = val_logloss
            
            self.logger.info(f"Training log_loss: {train_logloss:.4f}, "
                            f"Validation log_loss: {val_logloss:.4f}")
        else:
            # Train without validation data
            self.model.fit(X_scaled, y, **kwargs)
        
        # Update model metadata
        self.trained = True
        self.training_date = datetime.now()
        
        # Calculate training duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Record training run in history
        training_record = {
            'date': self.training_date.isoformat(),
            'duration_seconds': duration,
            'params': self.params.copy(),
            'samples': len(X),
            'optimization_performed': self.optimize_hyperparams
        }
        
        if validation_data is not None:
            training_record['validation_log_loss'] = val_logloss
            
        self.training_history.append(training_record)
        
        self.logger.info(f"XGBoost model training completed in {duration:.2f} seconds")
        
        return self
    
    def _hyperparameter_optimization(self, X, y, cv=5):
        """
        Perform hyperparameter optimization using grid search with temporal validation.
        
        Args:
            X: Features data
            y: Target data
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters found
        """
        # Initialize results storage
        best_params = self.params.copy()
        best_score = float('inf')  # Lower is better for log loss
        
        # Log the start of optimization
        self.logger.info("Starting hyperparameter grid search")
        
        # Outer optimization loop - we will optimize parameters one by one
        # This is more efficient than trying all combinations
        optimization_stages = [
            # Stage 1: tree-based params
            {
                'max_depth': self.hyper_param_spaces['max_depth'],
                'min_child_weight': self.hyper_param_spaces['min_child_weight']
            },
            # Stage 2: regularization params
            {
                'gamma': self.hyper_param_spaces['gamma']
            },
            # Stage 3: sampling params
            {
                'subsample': self.hyper_param_spaces['subsample'],
                'colsample_bytree': self.hyper_param_spaces['colsample_bytree']
            },
            # Stage 4: learning rate and estimators
            {
                'learning_rate': self.hyper_param_spaces['learning_rate']
            },
            # Stage 5: class balance
            {
                'scale_pos_weight': self.hyper_param_spaces['scale_pos_weight']
            }
        ]
        
        # Temporal cross-validation split
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Loop through each optimization stage
        for stage_idx, param_space in enumerate(optimization_stages):
            self.logger.info(f"Optimization stage {stage_idx+1}/{len(optimization_stages)}")
            
            # Generate parameter grid for this stage
            param_grid = list(ParameterGrid(param_space))
            self.logger.info(f"Testing {len(param_grid)} parameter combinations")
            
            # Track best parameters for this stage
            stage_best_params = {}
            stage_best_score = float('inf')
            
            # Test each parameter combination
            for param_idx, params in enumerate(param_grid):
                # Update parameters with current combination
                current_params = best_params.copy()
                current_params.update(params)
                
                # Track scores for this parameter set
                cv_scores = []
                
                # Cross-validation
                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    # Train model with current parameters
                    model = xgb.XGBClassifier(**current_params)
                    model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        eval_metric='logloss',
                        early_stopping_rounds=10,
                        verbose=False
                    )
                    
                    # Get the best validation score
                    val_score = min(model.evals_result()['validation_0']['logloss'])
                    cv_scores.append(val_score)
                
                # Calculate average score across folds
                avg_score = np.mean(cv_scores)
                
                # Check if this is the best score for this stage
                if avg_score < stage_best_score:
                    stage_best_score = avg_score
                    stage_best_params = params
                    
                # Log progress
                if (param_idx+1) % 5 == 0 or param_idx+1 == len(param_grid):
                    self.logger.info(f"Tested {param_idx+1}/{len(param_grid)} combinations, "
                                    f"best score so far: {stage_best_score:.4f}")
            
            # Update best parameters with the best from this stage
            best_params.update(stage_best_params)
            best_score = stage_best_score
            
            self.logger.info(f"Stage {stage_idx+1} best parameters: {stage_best_params}")
            self.logger.info(f"Current best score: {best_score:.4f}")
        
        # Update the model parameters with the best found
        self.params.update(best_params)
        
        # Log the final best parameters
        self.logger.info(f"Hyperparameter optimization completed. Best parameters: {best_params}")
        self.logger.info(f"Best cross-validation score (log loss): {best_score:.4f}")
        
        # Record optimization results
        self.model_params['optimization_results'] = {
            'best_score': best_score,
            'best_params': best_params
        }
        
        return best_params
    
    def predict(self, X):
        """
        Make binary predictions with the trained XGBoost model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Make probability predictions with the trained XGBoost model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Probability predictions (between 0 and 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make probability predictions
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def feature_importances(self):
        """
        Get feature importances from the trained XGBoost model.
        
        Returns:
            numpy.ndarray: Feature importance scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importances")
            
        return self.model.feature_importances_
    
    def feature_importance(self, plot=True, top_n=20, save_path=None):
        """
        Get and optionally plot feature importances.
        
        Args:
            plot (bool): Whether to generate a plot
            top_n (int): Number of top features to display
            save_path (str): Path to save the plot image
            
        Returns:
            dict: Feature importances mapped to feature names
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importances")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is not None:
            # Create a dictionary mapping feature names to importance scores
            feature_importance_dict = dict(zip(self.feature_names, importances))
            
            # Sort by importance
            sorted_importances = sorted(
                feature_importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Generate plot if requested
            if plot:
                # Select top_n features
                if top_n is not None and top_n < len(sorted_importances):
                    plot_importances = sorted_importances[:top_n]
                else:
                    plot_importances = sorted_importances
                
                # Extract names and values
                names = [item[0] for item in plot_importances]
                values = [item[1] for item in plot_importances]
                
                # Create plot
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(names)), values, align='center')
                plt.yticks(range(len(names)), names)
                plt.xlabel('Importance')
                plt.title(f'XGBoost Feature Importance - {self.name}')
                plt.gca().invert_yaxis()  # Show highest importance at the top
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path)
                    self.logger.info(f"Feature importance plot saved to {save_path}")
                else:
                    plt.show()
                
            return dict(sorted_importances)
        else:
            # If feature names are not available, return a list of importance scores
            return importances
    
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
        
        # Save XGBoost model
        model_path = os.path.join(model_dir, "xgb_model.json")
        self.model.save_model(model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        dump(self.scaler, scaler_path)
        
        # Save additional model parameters
        params_path = os.path.join(model_dir, "params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'random_state': self.random_state,
                'optimize_hyperparams': self.optimize_hyperparams,
                'feature_names': self.feature_names
            }, f)
            
        self.logger.info(f"XGBoost model saved to {model_dir}")
        
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
            
        # Load model parameters
        params_path = os.path.join(model_dir, "params.pkl")
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            
        self.params = params['params']
        self.random_state = params['random_state']
        self.optimize_hyperparams = params['optimize_hyperparams']
        self.feature_names = params['feature_names']
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.scaler = load(scaler_path)
        
        # Load XGBoost model
        model_path = os.path.join(model_dir, "xgb_model.json")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self.version = metadata['version']
            self.trained = metadata['trained']
            self.training_date = datetime.fromisoformat(metadata['training_date']) if metadata['training_date'] else None
            self.metrics = metadata['metrics']
            self.model_params = metadata['model_params']
            
            if 'training_history' in metadata:
                self.training_history = metadata['training_history']
                
        self.logger.info(f"XGBoost model loaded from {model_dir}")
        
        return self