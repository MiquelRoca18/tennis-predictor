# models/balanced_ensemble_model.py
import os
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
from sklearn.utils import class_weight
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

from .ensemble_model import TennisEnsembleModel

class BalancedTennisEnsembleModel(TennisEnsembleModel):
    """
    Enhanced ensemble model for tennis prediction with advanced handling
    of imbalanced datasets using various resampling techniques.
    
    This model extends the standard ensemble model with options for:
    - Class weighting
    - Oversampling (SMOTE, ADASYN)
    - Undersampling (Random, TomekLinks)
    - Hybrid approaches (SMOTETomek, SMOTEENN)
    """
    
    def __init__(self, name="balanced_tennis_ensemble", version="1.0.0", random_state=42, 
                 base_model_params=None, meta_model_params=None, 
                 balance_strategy='auto', sampling_strategy='auto'):
        """
        Initialize the balanced ensemble model.
        
        Args:
            name (str): Name of the model
            version (str): Version string
            random_state (int): Random seed for reproducibility
            base_model_params (dict): Parameters for base models
            meta_model_params (dict): Parameters for meta-model
            balance_strategy (str): Strategy to handle class imbalance
                Options: 'none', 'class_weight', 'smote', 'adasyn', 
                'random_under', 'tomek', 'smote_tomek', 'smote_enn', 'auto'
            sampling_strategy (str or float): Strategy for resampling
                'auto': Automatic determination based on the algorithm
                float: Ratio of minority to majority class
        """
        super().__init__(name, version, random_state, base_model_params, meta_model_params)
        
        # Balance strategy options
        self.balance_strategy = balance_strategy
        self.sampling_strategy = sampling_strategy
        
        # Update model params
        self.model_params.update({
            'balance_strategy': balance_strategy,
            'sampling_strategy': sampling_strategy
        })
        
        # Setup logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def _get_class_weights(self, y):
        """
        Calculate class weights based on the training data.
        
        Args:
            y: Target labels
            
        Returns:
            dict: Class weights
        """
        return dict(enumerate(class_weight.compute_class_weight(
            'balanced', classes=np.unique(y), y=y)))
    
    def _apply_resampling(self, X, y):
        """
        Apply the selected resampling strategy to balance the dataset.
        
        Args:
            X: Features data
            y: Target labels
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        # Handle case when no resampling is needed
        if self.balance_strategy == 'none':
            return X, y
        
        # Configuration for all resamplers
        config = {
            'random_state': self.random_state,
            'sampling_strategy': self.sampling_strategy
        }
        
        # Select and apply resampling technique
        if self.balance_strategy == 'smote':
            self.logger.info(f"Applying SMOTE resampling with strategy: {self.sampling_strategy}")
            resampler = SMOTE(**config)
        elif self.balance_strategy == 'adasyn':
            self.logger.info(f"Applying ADASYN resampling with strategy: {self.sampling_strategy}")
            resampler = ADASYN(**config)
        elif self.balance_strategy == 'random_under':
            self.logger.info(f"Applying RandomUnderSampler with strategy: {self.sampling_strategy}")
            resampler = RandomUnderSampler(**config)
        elif self.balance_strategy == 'tomek':
            self.logger.info("Applying TomekLinks cleaning")
            resampler = TomekLinks(**config)
        elif self.balance_strategy == 'smote_tomek':
            self.logger.info(f"Applying SMOTETomek with strategy: {self.sampling_strategy}")
            resampler = SMOTETomek(**config)
        elif self.balance_strategy == 'smote_enn':
            self.logger.info(f"Applying SMOTEENN with strategy: {self.sampling_strategy}")
            resampler = SMOTEENN(**config)
        elif self.balance_strategy == 'auto':
            # Auto selection based on imbalance ratio
            class_counts = pd.Series(y).value_counts()
            imbalance_ratio = class_counts.min() / class_counts.max()
            
            if imbalance_ratio < 0.2:  # Severe imbalance
                self.logger.info("Auto-selecting SMOTETomek for severe imbalance")
                resampler = SMOTETomek(**config)
            elif imbalance_ratio < 0.5:  # Moderate imbalance
                self.logger.info("Auto-selecting SMOTE for moderate imbalance")
                resampler = SMOTE(**config)
            else:  # Mild imbalance
                self.logger.info("Auto-selecting TomekLinks for mild imbalance")
                resampler = TomekLinks(**config)
        else:
            self.logger.warning(f"Unknown balance strategy: {self.balance_strategy}. No resampling applied.")
            return X, y
        
        # Apply resampling
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        
        # Log resampling results
        original_class_dist = pd.Series(y).value_counts().to_dict()
        resampled_class_dist = pd.Series(y_resampled).value_counts().to_dict()
        
        self.logger.info(f"Original class distribution: {original_class_dist}")
        self.logger.info(f"Resampled class distribution: {resampled_class_dist}")
        self.logger.info(f"Resampling changed dataset from {len(y)} to {len(y_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def fit(self, X, y, validation_data=None, cv=5, early_stopping=False, 
            early_stopping_rounds=10, **kwargs):
        """
        Train the balanced ensemble model with imbalance handling.
        
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
            X_array = X.values
        else:
            X_array = X
        
        # Store training start time
        start_time = datetime.now()
        
        # Record parameters for this training run
        train_params = {
            'cv': cv,
            'early_stopping': early_stopping,
            'early_stopping_rounds': early_stopping_rounds,
            'balance_strategy': self.balance_strategy,
            'sampling_strategy': self.sampling_strategy,
            **kwargs
        }
        
        # Balance the dataset if needed
        if self.balance_strategy != 'class_weight':
            X_balanced, y_balanced = self._apply_resampling(X_array, y)
        else:
            X_balanced, y_balanced = X_array, y
            self.logger.info("Using class weights instead of resampling")
        
        self.logger.info(f"Training balanced ensemble model with {len(X_balanced)} samples")
        
        # Train base models and generate predictions for meta-model training
        meta_features = np.zeros((len(X_balanced), len(self.base_models)))
        
        # Use cross-validation to generate meta-features
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for i, model_name in enumerate(self.base_models.keys()):
            self.logger.info(f"Training base model: {model_name}")
            
            # Store model-specific metrics
            model_metrics = []
            
            # Cross-validation for stacking
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_balanced, y_balanced)):
                X_train_fold, X_val_fold = X_balanced[train_idx], X_balanced[val_idx]
                y_train_fold, y_val_fold = y_balanced[train_idx], y_balanced[val_idx]
                
                # Create a fresh instance of the base model for each fold
                model = self._create_base_model(model_name)
                
                # Apply class weights if that's the chosen strategy
                if self.balance_strategy == 'class_weight':
                    # Different APIs for different models
                    if model_name == 'xgb':
                        # For XGBoost
                        if 'model' in model.named_steps:
                            weights = np.ones(len(y_train_fold))
                            minority_idx = (y_train_fold == np.min(y_train_fold))
                            weights[minority_idx] = (len(y_train_fold) - sum(minority_idx)) / sum(minority_idx)
                            model.named_steps['model'].fit(
                                X_train_fold, y_train_fold,
                                sample_weight=weights,
                                eval_set=[(X_val_fold, y_val_fold)] if early_stopping else None,
                                early_stopping_rounds=early_stopping_rounds if early_stopping else None,
                                verbose=False
                            )
                        else:
                            model.fit(X_train_fold, y_train_fold)
                    elif model_name in ['rf', 'gb']:
                        # For tree-based models
                        class_weights = self._get_class_weights(y_train_fold)
                        if 'model' in model.named_steps:
                            model.named_steps['model'].class_weight = class_weights
                        model.fit(X_train_fold, y_train_fold)
                    elif model_name == 'svm':
                        # For SVM
                        class_weights = self._get_class_weights(y_train_fold)
                        if 'model' in model.named_steps:
                            model.named_steps['model'].class_weight = class_weights
                        model.fit(X_train_fold, y_train_fold)
                    else:
                        # Default for models without class_weight
                        model.fit(X_train_fold, y_train_fold)
                else:
                    # No class weights, just normal training
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
            
            # Now train the model on the full balanced dataset
            model = self._create_base_model(model_name)
            
            # Apply class weights if that's the chosen strategy
            if self.balance_strategy == 'class_weight':
                if model_name in ['rf', 'gb', 'svm']:
                    class_weights = self._get_class_weights(y_balanced)
                    if 'model' in model.named_steps:
                        model.named_steps['model'].class_weight = class_weights
            
            model.fit(X_balanced, y_balanced)
            self.base_models[model_name] = model
        
        # Train meta-model on the meta-features
        self.logger.info("Training meta-model")
        
        # Apply class weights to meta-model if using that strategy
        if self.balance_strategy == 'class_weight':
            class_weights = self._get_class_weights(y_balanced)
            self.meta_model = LogisticRegression(**self.meta_model_params, class_weight=class_weights)
        
        self.meta_model.fit(meta_features, y_balanced)
        
        # Generate final predictions for training set
        meta_features_final = self._generate_meta_features(X_balanced)
        train_preds = self.meta_model.predict_proba(meta_features_final)[:, 1]
        train_acc = np.mean(np.round(train_preds) == y_balanced)
        
        self.logger.info(f"Balanced ensemble training accuracy: {train_acc:.4f}")
        
        # If validation data is provided, evaluate on it
        if validation_data is not None:
            X_val, y_val = validation_data
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
                
            val_meta_features = self._generate_meta_features(X_val)
            val_preds = self.meta_model.predict_proba(val_meta_features)[:, 1]
            val_acc = np.mean(np.round(val_preds) == y_val)
            
            self.logger.info(f"Balanced ensemble validation accuracy: {val_acc:.4f}")
            
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
            'samples': {
                'original': len(X),
                'balanced': len(X_balanced)
            },
            'training_accuracy': train_acc,
            'validation_accuracy': val_acc if validation_data is not None else None,
            'balance_strategy': self.balance_strategy,
            'sampling_strategy': self.sampling_strategy
        }
        
        self.training_history.append(training_record)
        
        self.logger.info(f"Balanced ensemble model training completed in {duration:.2f} seconds")
        
        return self
    
    def _create_base_model(self, model_name):
        """Create a fresh instance of a base model with its parameters."""
        # Inherit from parent class but with potential modifications for balanced models
        base_model = super()._create_base_model(model_name)
        
        return base_model