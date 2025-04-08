# models/ensemble_model.py
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from joblib import dump, load

from .base_model import TennisModel

class TennisEnsembleModel(TennisModel):
    """
    Ensemble model for tennis prediction that combines multiple base models
    using a stacking approach.
    
    The model combines:
    - Random Forest
    - Gradient Boosting
    - SVM
    - XGBoost
    - Neural Network (MLP)
    """
    
    def __init__(self, name="tennis_ensemble", version="1.0.0", random_state=42, 
                 base_model_params=None, meta_model_params=None):
        """
        Initialize the ensemble model with configurable base models and meta-model.
        
        Args:
            name (str): Name of the model
            version (str): Version string
            random_state (int): Random seed for reproducibility
            base_model_params (dict): Parameters for base models
            meta_model_params (dict): Parameters for meta-model
        """
        super().__init__(name, version)
        self.random_state = random_state
        
        # Set default params if none provided
        if base_model_params is None:
            self.base_model_params = {
                'rf': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'n_jobs': -1,
                    'random_state': random_state
                },
                'gb': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': random_state
                },
                'svm': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'probability': True,
                    'random_state': random_state
                },
                'xgb': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'random_state': random_state
                },
                'mlp': {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'batch_size': 'auto',
                    'learning_rate': 'adaptive',
                    'max_iter': 200,
                    'early_stopping': True,
                    'random_state': random_state
                }
            }
        else:
            self.base_model_params = base_model_params
            
        if meta_model_params is None:
            self.meta_model_params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': random_state
            }
        else:
            self.meta_model_params = meta_model_params
        
        # Initialize base models
        self._init_base_models()
        
        # Initialize meta model
        self.meta_model = LogisticRegression(**self.meta_model_params)
        
        # Store all parameters
        self.model_params = {
            'base_model_params': self.base_model_params,
            'meta_model_params': self.meta_model_params,
            'random_state': random_state
        }
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def _init_base_models(self):
        """Initialize all base models with specified parameters."""
        self.base_models = {
            'rf': Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(**self.base_model_params['rf']))
            ]),
            'gb': Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(**self.base_model_params['gb']))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(**self.base_model_params['svm']))
            ]),
            'xgb': Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBClassifier(**self.base_model_params['xgb']))
            ]),
            'mlp': Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPClassifier(**self.base_model_params['mlp']))
            ])
        }
    
    def fit(self, X, y, validation_data=None, cv=5, early_stopping=False, 
            early_stopping_rounds=10, **kwargs):
        """
        Train the ensemble model using a stacking approach.
        
        Args:
            X: Features data
            y: Target data
            validation_data: Optional tuple of (X_val, y_val) for validation
            cv (int): Number of cross-validation folds
            early_stopping (bool): Whether to use early stopping for base models
            early_stopping_rounds (int): Number of rounds for early stopping
            **kwargs: Additional parameters
            
        Returns:
            self: The trained model instance
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Store training start time
        start_time = datetime.now()
        
        # Record parameters for this training run
        train_params = {
            'cv': cv,
            'early_stopping': early_stopping,
            'early_stopping_rounds': early_stopping_rounds,
            **kwargs
        }
        
        self.logger.info(f"Training ensemble model with {len(X)} samples")
        
        # Train base models and generate predictions for meta-model training
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        # Use cross-validation to generate meta-features
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for i, model_name in enumerate(self.base_models.keys()):
            self.logger.info(f"Training base model: {model_name}")
            
            # Store model-specific metrics
            model_metrics = []
            
            # Cross-validation for stacking
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Create a fresh instance of the base model for each fold
                model = self._create_base_model(model_name)
                
                # Train the model
                if model_name == 'xgb' and early_stopping:
                    model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                else:
                    model.fit(X_train_fold, y_train_fold)
                
                # Generate predictions for this fold's validation data
                if hasattr(model, 'predict_proba'):
                    fold_preds = model.predict_proba(X_val_fold)[:, 1]
                else:
                    fold_preds = model.predict(X_val_fold)
                
                # Store predictions in meta_features array
                meta_features[val_idx, i] = fold_preds
                
                # Calculate and store metrics
                acc = np.mean(np.round(fold_preds) == y_val_fold)
                model_metrics.append({'fold': fold, 'accuracy': acc})
                
            # Log average metrics for this model
            avg_acc = np.mean([m['accuracy'] for m in model_metrics])
            self.logger.info(f"{model_name} average CV accuracy: {avg_acc:.4f}")
            
            # Now train the model on the full dataset
            self.base_models[model_name].fit(X, y)
        
        # Train meta-model on the meta-features
        self.logger.info("Training meta-model")
        self.meta_model.fit(meta_features, y)
        
        # Generate final predictions for training set
        meta_features_final = self._generate_meta_features(X)
        train_preds = self.meta_model.predict_proba(meta_features_final)[:, 1]
        train_acc = np.mean(np.round(train_preds) == y)
        
        self.logger.info(f"Ensemble training accuracy: {train_acc:.4f}")
        
        # If validation data is provided, evaluate on it
        if validation_data is not None:
            X_val, y_val = validation_data
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
                
            val_meta_features = self._generate_meta_features(X_val)
            val_preds = self.meta_model.predict_proba(val_meta_features)[:, 1]
            val_acc = np.mean(np.round(val_preds) == y_val)
            
            self.logger.info(f"Ensemble validation accuracy: {val_acc:.4f}")
            
            # Store validation metrics
            self.metrics['validation'] = {
                'accuracy': val_acc,
                # Other metrics would be calculated by the evaluate() method
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
            'params': train_params,
            'samples': len(X),
            'training_accuracy': train_acc,
            'validation_accuracy': val_acc if validation_data is not None else None
        }
        
        self.training_history.append(training_record)
        
        self.logger.info(f"Ensemble model training completed in {duration:.2f} seconds")
        
        return self
    
    def _create_base_model(self, model_name):
        """Create a fresh instance of a base model with its parameters."""
        if model_name == 'rf':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(**self.base_model_params['rf']))
            ])
        elif model_name == 'gb':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(**self.base_model_params['gb']))
            ])
        elif model_name == 'svm':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(**self.base_model_params['svm']))
            ])
        elif model_name == 'xgb':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBClassifier(**self.base_model_params['xgb']))
            ])
        elif model_name == 'mlp':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPClassifier(**self.base_model_params['mlp']))
            ])
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def _generate_meta_features(self, X):
        """Generate meta-features by getting predictions from all base models."""
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model_name in enumerate(self.base_models.keys()):
            model = self.base_models[model_name]
            
            if hasattr(model, 'predict_proba'):
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model.predict(X)
                
        return meta_features
    
    def predict(self, X):
        """
        Make binary predictions with the trained ensemble model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Use meta-model to make predictions
        return np.round(self.meta_model.predict_proba(meta_features)[:, 1]).astype(int)
    
    def predict_proba(self, X):
        """
        Make probability predictions with the trained ensemble model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Probability predictions (between 0 and 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Use meta-model to make probability predictions
        return self.meta_model.predict_proba(meta_features)[:, 1]
    
    def feature_importances(self):
        """
        Get the combined feature importances from base models.
        
        Returns:
            numpy.ndarray: Feature importance scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importances")
            
        if self.feature_names is None:
            raise ValueError("Feature names not available")
            
        # Initialize feature importances array
        importances = np.zeros(len(self.feature_names))
        
        # Collect feature importances from models that support it
        models_with_importances = 0
        
        for model_name, model in self.base_models.items():
            try:
                # Get the actual model from the pipeline
                if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                    estimator = model.named_steps['model']
                else:
                    estimator = model
                
                # Get feature importances
                if hasattr(estimator, 'feature_importances_'):
                    importances += estimator.feature_importances_
                    models_with_importances += 1
                elif model_name == 'svm' and hasattr(estimator, 'coef_'):
                    # For linear SVM
                    importances += np.abs(estimator.coef_[0])
                    models_with_importances += 1
            except (AttributeError, IndexError):
                self.logger.warning(f"Could not get feature importances for {model_name}")
        
        # Average the importances
        if models_with_importances > 0:
            importances /= models_with_importances
        
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
        
        # Save each base model
        for model_name, model in self.base_models.items():
            model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
            dump(model, model_path)
            
        # Save meta model
        meta_model_path = os.path.join(model_dir, "meta_model.joblib")
        dump(self.meta_model, meta_model_path)
        
        # Save model parameters
        params_path = os.path.join(model_dir, "params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump({
                'base_model_params': self.base_model_params,
                'meta_model_params': self.meta_model_params,
                'random_state': self.random_state,
                'feature_names': self.feature_names
            }, f)
            
        self.logger.info(f"Ensemble model saved to {model_dir}")
        
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
            
        self.base_model_params = params['base_model_params']
        self.meta_model_params = params['meta_model_params']
        self.random_state = params['random_state']
        self.feature_names = params['feature_names']
        
        # Initialize base models with loaded parameters
        self._init_base_models()
        
        # Load each base model
        for model_name in self.base_models.keys():
            model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
            self.base_models[model_name] = load(model_path)
            
        # Load meta model
        meta_model_path = os.path.join(model_dir, "meta_model.joblib")
        self.meta_model = load(meta_model_path)
        
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
                
        self.logger.info(f"Ensemble model loaded from {model_dir}")
        
        return self