# models/bayesian_optimization_model.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# For Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from .base_model import TennisModel
from .xgboost_model import TennisXGBoostModel
from .neural_net_model import TennisNeuralNetModel

class BayesianOptimizationModel(TennisModel):
    """
    A model that uses Bayesian Optimization to find optimal hyperparameters
    for tennis prediction models, resulting in better performance than
    traditional grid search or random search methods.
    
    This approach is more efficient for exploring high-dimensional parameter
    spaces and can find better configurations with fewer evaluations.
    """
    
    def __init__(self, model_type="xgboost", name=None, version="1.0.0", 
                 random_state=42, n_iter=50, cv=5, custom_param_space=None):
        """
        Initialize the Bayesian optimization model.
        
        Args:
            model_type (str): Type of model to optimize ('xgboost' or 'neural_net')
            name (str): Name of the model (defaults to "bayesian_{model_type}")
            version (str): Version string
            random_state (int): Random seed for reproducibility
            n_iter (int): Number of iterations for Bayesian optimization
            cv (int): Number of cross-validation folds
            custom_param_space (dict): Custom parameter space definition
        """
        if name is None:
            name = f"bayesian_{model_type}"
            
        super().__init__(name, version)
        
        self.model_type = model_type
        self.random_state = random_state
        self.n_iter = n_iter
        self.cv = cv
        self.custom_param_space = custom_param_space
        
        # Initialize optimization results
        self.optimization_results = None
        self.best_params = None
        self.inner_model = None
        self.param_importances = None
        
        # Set model parameters
        self.model_params = {
            'model_type': model_type,
            'random_state': random_state,
            'n_iter': n_iter,
            'cv': cv
        }
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Setup logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def _get_default_param_space(self):
        """
        Get the default parameter space for the selected model type.
        
        Returns:
            dict: Parameter space definition
        """
        if self.model_type == 'xgboost':
            return {
                'max_depth': Integer(3, 10),
                'min_child_weight': Real(0.1, 10, prior='log-uniform'),
                'gamma': Real(0, 5, prior='uniform'),
                'subsample': Real(0.5, 1.0, prior='uniform'),
                'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'n_estimators': Integer(50, 500),
                'scale_pos_weight': Real(0.1, 10, prior='log-uniform')
            }
        elif self.model_type == 'neural_net':
            return {
                'hidden_layer_sizes': Categorical([
                    (50,), (100,), (200,),
                    (50, 25), (100, 50), (200, 100),
                    (50, 25, 10), (100, 50, 25)
                ]),
                'activation': Categorical(['relu', 'tanh']),
                'dropout_rate': Real(0.1, 0.5, prior='uniform'),
                'batch_size': Categorical([16, 32, 64]),
                'learning_rate': Real(0.0001, 0.01, prior='log-uniform'),
                'l2_lambda': Real(0.0001, 0.1, prior='log-uniform')
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_inner_model(self, params=None):
        """
        Create an instance of the inner model with given parameters.
        
        Args:
            params (dict): Model parameters
            
        Returns:
            TennisModel: Instance of the inner model
        """
        if self.model_type == 'xgboost':
            if params is None:
                return TennisXGBoostModel(
                    name=f"{self.name}_inner",
                    random_state=self.random_state
                )
            else:
                # Convert params to the format expected by XGBoostModel
                xgb_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'seed': self.random_state,
                    **params
                }
                return TennisXGBoostModel(
                    name=f"{self.name}_inner",
                    random_state=self.random_state,
                    params=xgb_params
                )
        elif self.model_type == 'neural_net':
            if params is None:
                return TennisNeuralNetModel(
                    name=f"{self.name}_inner",
                    random_state=self.random_state
                )
            else:
                # Convert params to the format expected by NeuralNetModel
                hidden_layer_sizes = params.pop('hidden_layer_sizes', (100, 50))
                activation = params.pop('activation', 'relu')
                dropout_rate = params.pop('dropout_rate', 0.3)
                batch_size = params.pop('batch_size', 32)
                learning_rate = params.pop('learning_rate', 0.001)
                l2_lambda = params.pop('l2_lambda', 0.001)
                
                # Create model_params dictionary
                nn_params = {
                    'architecture': {
                        'hidden_layers': list(hidden_layer_sizes),
                        'dropout_rates': [dropout_rate] * len(hidden_layer_sizes),
                        'activation': activation,
                        'output_activation': 'sigmoid',
                        'use_batch_norm': True
                    },
                    'regularization': {
                        'l2_lambda': l2_lambda,
                    },
                    'training': {
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'epochs': 100,
                        'patience': 10,
                        'validation_split': 0.2
                    }
                }
                
                return TennisNeuralNetModel(
                    name=f"{self.name}_inner",
                    random_state=self.random_state,
                    model_params=nn_params
                )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y, validation_data=None, **kwargs):
        """
        Perform Bayesian hyperparameter optimization and train the best model.
        
        Args:
            X: Features data
            y: Target data
            validation_data: Optional tuple of (X_val, y_val) for validation
            **kwargs: Additional parameters passed to inner model's fit method
            
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
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Get parameter space
        param_space = self.custom_param_space or self._get_default_param_space()
        
        self.logger.info(f"Starting Bayesian optimization with {self.n_iter} iterations")
        self.logger.info(f"Parameter space: {param_space}")
        
        # Define the objective function
        @use_named_args(dimensions=list(param_space.items()))
        def objective_func(**params):
            # Create model with these parameters
            model = self._create_inner_model(params)
            
            # Use cross-validation to evaluate model
            kfold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            
            cv_scores = []
            for train_idx, val_idx in kfold.split(X_scaled, y):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Train model
                model.fit(X_train_fold, y_train_fold, **kwargs)
                
                # Evaluate
                y_pred = model.predict_proba(X_val_fold)
                
                # Use log loss as objective
                score = log_loss(y_val_fold, y_pred)
                cv_scores.append(score)
                
            mean_score = np.mean(cv_scores)
            
            self.logger.debug(f"Parameters: {params}, Score: {mean_score:.5f}")
            
            return mean_score
        
        # Perform Bayesian optimization
        from skopt import gp_minimize
        
        optimization_result = gp_minimize(
            objective_func,
            list(param_space.values()),
            n_calls=self.n_iter,
            random_state=self.random_state,
            verbose=True,
            n_jobs=-1
        )
        
        # Extract best parameters
        self.optimization_results = optimization_result
        self.best_params = dict(zip(param_space.keys(), optimization_result.x))
        
        self.logger.info(f"Optimization complete. Best parameters: {self.best_params}")
        self.logger.info(f"Best score: {optimization_result.fun:.5f}")
        
        # Calculate parameter importance
        try:
            from skopt.plots import plot_objective
            parameter_importance = dict(zip(
                param_space.keys(),
                optimization_result.space.point_to_dict(optimization_result.x_iters).values()
            ))
            self.param_importances = parameter_importance
        except:
            self.logger.warning("Could not calculate parameter importance")
        
        # Train final model with best parameters
        self.logger.info("Training final model with best parameters")
        self.inner_model = self._create_inner_model(self.best_params)
        
        # For NN, we want to use the full dataset and validation data
        if validation_data is not None and self.model_type == 'neural_net':
            X_val, y_val = validation_data
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            X_val_scaled = self.scaler.transform(X_val)
            self.inner_model.fit(X_scaled, y, validation_data=(X_val_scaled, y_val), **kwargs)
        else:
            self.inner_model.fit(X_scaled, y, **kwargs)
        
        # If validation data is provided, evaluate
        if validation_data is not None:
            X_val, y_val = validation_data
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            X_val_scaled = self.scaler.transform(X_val)
            
            # Evaluate on validation data
            y_pred = self.inner_model.predict(X_val_scaled)
            acc = accuracy_score(y_val, y_pred)
            
            # Get probability predictions
            y_pred_proba = self.inner_model.predict_proba(X_val_scaled)
            loss = log_loss(y_val, y_pred_proba)
            
            self.logger.info(f"Validation accuracy: {acc:.4f}, log loss: {loss:.4f}")
            
            # Store metrics
            self.metrics['validation'] = {
                'accuracy': acc,
                'log_loss': loss
            }
        
        # Update model metadata
        self.trained = True
        self.training_date = datetime.now()
        
        # Calculate training duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Record training run in history
        training_record = {
            'date': self.training_date.isoformat(),
            'duration_seconds': duration,
            'n_iter': self.n_iter,
            'best_params': self.best_params,
            'best_score': float(optimization_result.fun),
            'model_type': self.model_type,
            'inner_model_name': self.inner_model.name
        }
        
        self.training_history.append(training_record)
        
        self.logger.info(f"Model training completed in {duration:.2f} seconds")
        
        return self
    
    def predict(self, X):
        """
        Make binary predictions with the optimized model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.inner_model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Make probability predictions with the optimized model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Probability predictions
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make probability predictions
        return self.inner_model.predict_proba(X_scaled)
    
    def feature_importances(self):
        """
        Get feature importances from the inner model.
        
        Returns:
            numpy.ndarray: Feature importance scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importances")
            
        # Delegate to inner model
        return self.inner_model.feature_importances()
    
    def plot_optimization_results(self, save_path=None):
        """
        Plot the optimization results.
        
        Args:
            save_path (str): Path to save the plot image
            
        Returns:
            None
        """
        if not self.trained or self.optimization_results is None:
            raise ValueError("Model must be trained with optimization results available")
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot convergence
        from skopt.plots import plot_convergence
        plot_convergence(self.optimization_results, ax=ax1)
        
        # Plot parameter importance if available
        if self.param_importances:
            # Sort by importance
            params = list(self.param_importances.keys())
            importances = list(self.param_importances.values())
            
            # Plot
            y_pos = np.arange(len(params))
            ax2.barh(y_pos, importances, align='center')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(params)
            ax2.set_xlabel('Importance')
            ax2.set_title('Parameter Importance')
            ax2.invert_yaxis()  # Show highest importance at the top
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Optimization results plot saved to {save_path}")
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
        
        # Save the inner model
        inner_model_dir = os.path.join(model_dir, "inner_model")
        os.makedirs(inner_model_dir, exist_ok=True)
        self.inner_model.save(inner_model_dir)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        dump(self.scaler, scaler_path)
        
        # Save optimization results if available
        if self.optimization_results is not None:
            opt_results_path = os.path.join(model_dir, "optimization_results.pkl")
            with open(opt_results_path, 'wb') as f:
                pickle.dump(self.optimization_results, f)
        
        # Save additional parameters
        params_path = os.path.join(model_dir, "params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'random_state': self.random_state,
                'n_iter': self.n_iter,
                'cv': self.cv,
                'best_params': self.best_params,
                'param_importances': self.param_importances,
                'feature_names': self.feature_names
            }, f)
            
        self.logger.info(f"Bayesian optimization model saved to {model_dir}")
        
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
            
        self.model_type = params['model_type']
        self.random_state = params['random_state']
        self.n_iter = params['n_iter']
        self.cv = params['cv']
        self.best_params = params['best_params']
        self.param_importances = params['param_importances']
        self.feature_names = params['feature_names']
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.scaler = load(scaler_path)
        
        # Load optimization results if available
        opt_results_path = os.path.join(model_dir, "optimization_results.pkl")
        if os.path.exists(opt_results_path):
            with open(opt_results_path, 'rb') as f:
                self.optimization_results = pickle.load(f)
        
        # Load inner model
        inner_model_dir = os.path.join(model_dir, "inner_model")
        
        # Create inner model instance
        self.inner_model = self._create_inner_model(self.best_params)
        
        # Load inner model
        self.inner_model.load(inner_model_dir)
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self.version = metadata['version']
            self.trained = metadata['trained']
            self.training_date = datetime.fromisoformat(metadata['training_date']) if metadata['training_date'] else None
            self.metrics = metadata['metrics']
            
            if 'training_history' in metadata:
                self.training_history = metadata['training_history']
                
        self.logger.info(f"Bayesian optimization model loaded from {model_dir}")
        
        return self