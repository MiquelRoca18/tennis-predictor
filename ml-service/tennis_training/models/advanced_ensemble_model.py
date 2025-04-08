# models/advanced_ensemble_model.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.base import clone
from joblib import dump, load

from .base_model import TennisModel
from . import get_model, get_available_models

class DiverseEnsembleModel(TennisModel):
    """
    Advanced ensemble model for tennis prediction that combines multiple diverse
    models using stacking with soft voting and model diversity optimization.
    
    This model creates an ensemble that is more robust and accurate than any 
    individual model by:
    1. Using different types of base models (diversity in algorithm)
    2. Training each model on different subsets of features (diversity in representation)
    3. Optimizing model weights based on correlation of errors (minimizing redundancy)
    4. Using confidence-weighted soft voting for final predictions
    """
    
    def __init__(self, name="diverse_ensemble", version="1.0.0", random_state=42,
                 base_models=None, feature_groups=None, meta_model_type='logistic'):
        """
        Initialize the diverse ensemble model.
        
        Args:
            name (str): Name of the model
            version (str): Version string
            random_state (int): Random seed for reproducibility
            base_models (list): List of base model configurations or instances
            feature_groups (dict): Groups of features for diversity
            meta_model_type (str): Type of meta-model ('logistic', 'xgboost', 'neural_net')
        """
        super().__init__(name, version)
        
        self.random_state = random_state
        self.meta_model_type = meta_model_type
        
        # Initialize base models if provided, otherwise use defaults
        self.base_models = base_models or self._get_default_base_models()
        
        # Initialize feature groups if provided, otherwise all models use all features
        self.feature_groups = feature_groups
        
        # Placeholders for trained models and meta-model
        self.models = []
        self.model_weights = None
        self.meta_model = None
        self.meta_features = None
        
        # Standard scaler for input features
        self.scaler = StandardScaler()
        
        # Set model parameters
        self.model_params = {
            'random_state': random_state,
            'base_models_config': self.base_models,
            'feature_groups': self.feature_groups,
            'meta_model_type': meta_model_type
        }
        
        # Setup logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def _get_default_base_models(self):
        """
        Get default configurations for base models.
        
        Returns:
            list: List of base model configurations
        """
        return [
            # XGBoost with different parameters
            {
                'type': 'xgboost',
                'name': 'xgb_default',
                'params': {}  # Default parameters
            },
            {
                'type': 'xgboost',
                'name': 'xgb_depth',
                'params': {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 200}
            },
            # Neural network models
            {
                'type': 'neural_net',
                'name': 'nn_small',
                'params': {
                    'architecture': {
                        'hidden_layers': [50, 25],
                        'dropout_rates': [0.3, 0.3],
                        'activation': 'relu'
                    }
                }
            },
            {
                'type': 'neural_net',
                'name': 'nn_large',
                'params': {
                    'architecture': {
                        'hidden_layers': [200, 100, 50],
                        'dropout_rates': [0.4, 0.4, 0.4],
                        'activation': 'relu'
                    }
                }
            },
            # Ensemble model
            {
                'type': 'ensemble',
                'name': 'ensemble_default',
                'params': {}  # Default parameters
            }
        ]
    
    def _create_model_instance(self, model_config):
        """
        Create a model instance from configuration.
        
        Args:
            model_config (dict or TennisModel): Model configuration or instance
            
        Returns:
            TennisModel: Model instance
        """
        if isinstance(model_config, TennisModel):
            # It's already a model instance
            return model_config
        
        # Create new instance from configuration
        model_type = model_config['type']
        model_name = model_config.get('name', f"{model_type}_{len(self.models)}")
        model_params = model_config.get('params', {})
        
        # Add random state to params
        if 'random_state' not in model_params:
            model_params['random_state'] = self.random_state
            
        # Create model instance
        return get_model(model_type, name=model_name, **model_params)
    
    def _get_feature_subset(self, X, group_name):
        """
        Get a subset of features based on group name.
        
        Args:
            X: Feature data (DataFrame or array)
            group_name (str): Name of feature group
            
        Returns:
            array or DataFrame: Subset of features
        """
        if self.feature_groups is None or group_name is None:
            # Use all features
            return X
        
        if group_name not in self.feature_groups:
            raise ValueError(f"Unknown feature group: {group_name}")
            
        # Get feature indices or names
        feature_subset = self.feature_groups[group_name]
        
        if isinstance(X, pd.DataFrame):
            # DataFrame - use column names
            return X[feature_subset]
        else:
            # Array - use column indices
            return X[:, feature_subset]
    
    def fit(self, X, y, validation_data=None, cv=5, **kwargs):
        """
        Train the diverse ensemble model.
        
        Args:
            X: Features data
            y: Target data
            validation_data: Optional tuple of (X_val, y_val) for validation
            cv (int): Number of cross-validation folds for stacking
            **kwargs: Additional parameters passed to base models' fit method
            
        Returns:
            self: The trained model instance
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X.copy()
            
        # Store training start time
        start_time = datetime.now()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Convert back to DataFrame if original was DataFrame
        if isinstance(X, pd.DataFrame):
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        else:
            X_scaled_df = X_scaled
        
        # Initialize models list
        self.models = []
        
        # Lists to store cross-validation predictions
        cv_meta_features = np.zeros((len(X_scaled), len(self.base_models)))
        
        # Cross-validation for stacking
        self.logger.info(f"Training {len(self.base_models)} base models with {cv}-fold CV stacking")
        
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Dictionary to track model metrics
        model_metrics = {model_config.get('name', f"model_{i}"): [] 
                         for i, model_config in enumerate(self.base_models)}
        
        # Train each base model with cross-validation for stacking
        for i, model_config in enumerate(self.base_models):
            model_name = model_config.get('name', f"model_{i}")
            group_name = model_config.get('feature_group', None)
            
            self.logger.info(f"Training base model {i+1}/{len(self.base_models)}: {model_name}")
            
            # Get feature subset for this model
            X_model = self._get_feature_subset(X_scaled_df, group_name)
            
            # For CV-based stacking
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
                # Split data
                if isinstance(X_model, pd.DataFrame):
                    X_train_fold = X_model.iloc[train_idx]
                    X_val_fold = X_model.iloc[val_idx]
                else:
                    X_train_fold = X_model[train_idx]
                    X_val_fold = X_model[val_idx]
                    
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Create and train model
                model = self._create_model_instance(model_config)
                model.fit(X_train_fold, y_train_fold, **kwargs)
                
                # Generate predictions
                fold_preds = model.predict_proba(X_val_fold)
                cv_meta_features[val_idx, i] = fold_preds
                
                # Calculate metrics
                fold_acc = accuracy_score(y_val_fold, np.round(fold_preds))
                fold_ll = log_loss(y_val_fold, fold_preds)
                fold_auc = roc_auc_score(y_val_fold, fold_preds)
                
                model_metrics[model_name].append({
                    'fold': fold,
                    'accuracy': fold_acc,
                    'log_loss': fold_ll,
                    'roc_auc': fold_auc
                })
                
            # Train final model on all data
            model = self._create_model_instance(model_config)
            X_model_full = self._get_feature_subset(X_scaled_df, group_name)
            model.fit(X_model_full, y, **kwargs)
            
            # Store trained model
            self.models.append({
                'model': model,
                'name': model_name,
                'feature_group': group_name,
                'metrics': model_metrics[model_name]
            })
            
            # Log average metrics
            avg_acc = np.mean([m['accuracy'] for m in model_metrics[model_name]])
            avg_ll = np.mean([m['log_loss'] for m in model_metrics[model_name]])
            avg_auc = np.mean([m['roc_auc'] for m in model_metrics[model_name]])
            
            self.logger.info(f"  {model_name} CV metrics - Accuracy: {avg_acc:.4f}, "
                            f"Log Loss: {avg_ll:.4f}, AUC: {avg_auc:.4f}")
        
        # Calculate model correlations and determine optimal weights
        self.logger.info("Calculating model correlations and weights")
        self._calculate_model_weights(cv_meta_features, y)
        
        # Train meta-model
        self.logger.info(f"Training meta-model using '{self.meta_model_type}'")
        
        if self.meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(C=1.0, solver='lbfgs', random_state=self.random_state)
            self.meta_model.fit(cv_meta_features, y)
        elif self.meta_model_type == 'xgboost':
            from xgboost import XGBClassifier
            self.meta_model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=self.random_state
            )
            self.meta_model.fit(cv_meta_features, y)
        elif self.meta_model_type == 'neural_net':
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            
            self.meta_model = Sequential([
                Dense(32, activation='relu', input_shape=(cv_meta_features.shape[1],)),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            self.meta_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.meta_model.fit(
                cv_meta_features, y,
                epochs=50,
                batch_size=32,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown meta-model type: {self.meta_model_type}")
            
        # Store meta-features for reference
        self.meta_features = cv_meta_features
        
        # Evaluate on validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            self._evaluate_ensemble(X_val, y_val)
            
        # Update model metadata
        self.trained = True
        self.training_date = datetime.now()
        
        # Calculate training duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Record training run in history
        training_record = {
            'date': self.training_date.isoformat(),
            'duration_seconds': duration,
            'num_base_models': len(self.models),
            'meta_model_type': self.meta_model_type,
            'model_weights': self.model_weights.tolist() if self.model_weights is not None else None,
            'base_model_names': [m['name'] for m in self.models]
        }
        
        self.training_history.append(training_record)
        
        self.logger.info(f"Diverse ensemble model training completed in {duration:.2f} seconds")
        
        return self
    
    def _calculate_model_weights(self, meta_features, y):
        """
        Calculate optimal weights for models based on performance and correlation.
        
        Args:
            meta_features: Model predictions for meta-model
            y: True labels
            
        Returns:
            None (sets self.model_weights)
        """
        # Calculate accuracy for each model
        accuracies = []
        for i in range(meta_features.shape[1]):
            acc = accuracy_score(y, np.round(meta_features[:, i]))
            accuracies.append(acc)
            
        # Calculate correlation matrix of errors
        errors = np.zeros_like(meta_features)
        for i in range(meta_features.shape[1]):
            errors[:, i] = np.abs(np.round(meta_features[:, i]) - y)
            
        corr_matrix = np.corrcoef(errors.T)
        
        # Basic weight allocation based on accuracy and diversity
        raw_weights = np.array(accuracies)
        
        # Adjust weights to account for correlation
        # Models with low correlation (high diversity) get higher weights
        diversity_bonus = np.zeros(len(accuracies))
        
        for i in range(len(accuracies)):
            # Average correlation with other models (excluding self-correlation)
            other_corrs = [corr_matrix[i, j] for j in range(len(accuracies)) if j != i]
            avg_corr = np.mean(other_corrs) if other_corrs else 0
            
            # Diversity bonus is inversely proportional to correlation
            diversity_bonus[i] = 1.0 - avg_corr
            
        # Combine accuracy and diversity (weighted sum)
        combined_weights = (0.7 * raw_weights) + (0.3 * diversity_bonus)
        
        # Normalize weights to sum to 1
        self.model_weights = combined_weights / combined_weights.sum()
        
        # Log weights
        model_names = [m.get('name', f"model_{i}") for i, m in enumerate(self.base_models)]
        for name, weight in zip(model_names, self.model_weights):
            self.logger.info(f"  Model '{name}' weight: {weight:.4f}")
    
    def _evaluate_ensemble(self, X_val, y_val):
        """
        Evaluate the ensemble model on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            dict: Validation metrics
        """
        # Get predictions
        val_preds = self.predict_proba(X_val)
        val_acc = accuracy_score(y_val, np.round(val_preds))
        val_ll = log_loss(y_val, val_preds)
        val_auc = roc_auc_score(y_val, val_preds)
        
        # Store metrics
        self.metrics['validation'] = {
            'accuracy': val_acc,
            'log_loss': val_ll,
            'roc_auc': val_auc
        }
        
        self.logger.info(f"Ensemble validation metrics - Accuracy: {val_acc:.4f}, "
                         f"Log Loss: {val_ll:.4f}, AUC: {val_auc:.4f}")
        
        # Compare to best base model
        base_model_metrics = []
        for model_info in self.models:
            model = model_info['model']
            group_name = model_info['feature_group']
            
            # Get feature subset for this model
            X_val_model = self._get_feature_subset(X_val, group_name)
            
            # Get predictions
            model_preds = model.predict_proba(X_val_model)
            model_acc = accuracy_score(y_val, np.round(model_preds))
            model_ll = log_loss(y_val, model_preds)
            model_auc = roc_auc_score(y_val, model_preds)
            
            base_model_metrics.append({
                'name': model_info['name'],
                'accuracy': model_acc,
                'log_loss': model_ll,
                'roc_auc': model_auc
            })
            
        # Find best base model
        best_model = max(base_model_metrics, key=lambda x: x['accuracy'])
        self.logger.info(f"Best base model: {best_model['name']} - "
                         f"Accuracy: {best_model['accuracy']:.4f}, "
                         f"Log Loss: {best_model['log_loss']:.4f}, "
                         f"AUC: {best_model['roc_auc']:.4f}")
        
        # Log improvement over best base model
        acc_improvement = val_acc - best_model['accuracy']
        self.logger.info(f"Ensemble improvement over best base model: {acc_improvement:.4f}")
        
        return self.metrics['validation']
    
    def predict(self, X):
        """
        Make binary predictions with the ensemble model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        # Convert probabilities to binary predictions
        proba = self.predict_proba(X)
        return np.round(proba).astype(int)
    
    def predict_proba(self, X):
        """
        Make probability predictions with the ensemble model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Probability predictions
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        # Prepare input
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
            
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Convert back to DataFrame if original was DataFrame
        if isinstance(X, pd.DataFrame):
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        else:
            X_scaled_df = X_scaled
            
        # Generate predictions from each base model
        meta_features = np.zeros((len(X_scaled), len(self.models)))
        
        for i, model_info in enumerate(self.models):
            model = model_info['model']
            group_name = model_info['feature_group']
            
            # Get feature subset for this model
            X_model = self._get_feature_subset(X_scaled_df, group_name)
            
            # Get predictions
            model_preds = model.predict_proba(X_model)
            meta_features[:, i] = model_preds
        
        # Apply meta-model or weighted average
        if self.meta_model_type in ['logistic', 'xgboost']:
            final_preds = self.meta_model.predict_proba(meta_features)[:, 1]
        elif self.meta_model_type == 'neural_net':
            final_preds = self.meta_model.predict(meta_features).flatten()
        else:
            # Fallback to weighted average
            final_preds = np.average(meta_features, axis=1, weights=self.model_weights)
            
        return final_preds
    
    def feature_importances(self):
        """
        Get aggregated feature importances from all base models.
        
        Returns:
            numpy.ndarray: Feature importance scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importances")
            
        # Initialize importance array
        if self.feature_names:
            importances = np.zeros(len(self.feature_names))
        else:
            # Get the first model with all features
            for model_info in self.models:
                if model_info['feature_group'] is None:
                    model = model_info['model']
                    if hasattr(model, 'feature_importances'):
                        return model.feature_importances()
            
            raise ValueError("Cannot determine feature importances")
            
        # Aggregate importances from all models, weighted by model weights
        for i, model_info in enumerate(self.models):
            model = model_info['model']
            group_name = model_info['feature_group']
            weight = self.model_weights[i]
            
            try:
                model_importances = model.feature_importances()
                
                if group_name is None:
                    # Model uses all features
                    importances += weight * model_importances
                else:
                    # Model uses subset of features
                    feature_subset = self.feature_groups[group_name]
                    for j, feat_idx in enumerate(feature_subset):
                        importances[feat_idx] += weight * model_importances[j]
            except (AttributeError, NotImplementedError):
                # Skip models that don't support feature importances
                continue
                
        return importances
    
    def get_model_contributions(self, X, y=None):
        """
        Analyze how each model contributes to the ensemble predictions.
        
        Args:
            X: Features data
            y: Optional true labels for performance evaluation
            
        Returns:
            dict: Model contribution analysis
        """
        if not self.trained:
            raise ValueError("Model must be trained before analyzing contributions")
            
        # Prepare input
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
            
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Convert back to DataFrame if original was DataFrame
        if isinstance(X, pd.DataFrame):
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        else:
            X_scaled_df = X_scaled
            
        # Get predictions from each model
        model_predictions = []
        
        for model_info in self.models:
            model = model_info['model']
            name = model_info['name']
            group_name = model_info['feature_group']
            
            # Get feature subset for this model
            X_model = self._get_feature_subset(X_scaled_df, group_name)
            
            # Get predictions
            preds = model.predict_proba(X_model)
            
            model_predictions.append({
                'name': name,
                'predictions': preds,
                'feature_group': group_name
            })
            
        # Get ensemble predictions
        ensemble_preds = self.predict_proba(X)
        
        # Analysis dictionary
        analysis = {
            'model_predictions': model_predictions,
            'ensemble_predictions': ensemble_preds,
            'model_weights': self.model_weights.tolist(),
            'model_names': [m['name'] for m in self.models]
        }
        
        # Calculate agreement statistics
        agreement_matrix = np.zeros((len(self.models), len(self.models)))
        
        for i in range(len(self.models)):
            for j in range(len(self.models)):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # Calculate agreement as 1 - mean absolute difference in predictions
                    preds_i = model_predictions[i]['predictions']
                    preds_j = model_predictions[j]['predictions']
                    agreement_matrix[i, j] = 1.0 - np.mean(np.abs(preds_i - preds_j))
                    
        analysis['model_agreement_matrix'] = agreement_matrix.tolist()
        
        # If true labels are provided, calculate model performance
        if y is not None:
            model_performance = []
            
            for i, model_pred in enumerate(model_predictions):
                preds = model_pred['predictions']
                acc = accuracy_score(y, np.round(preds))
                ll = log_loss(y, preds)
                auc = roc_auc_score(y, preds)
                
                model_performance.append({
                    'name': model_pred['name'],
                    'accuracy': float(acc),
                    'log_loss': float(ll),
                    'roc_auc': float(auc)
                })
                
            # Ensemble performance
            ens_acc = accuracy_score(y, np.round(ensemble_preds))
            ens_ll = log_loss(y, ensemble_preds)
            ens_auc = roc_auc_score(y, ensemble_preds)
            
            ensemble_performance = {
                'accuracy': float(ens_acc),
                'log_loss': float(ens_ll),
                'roc_auc': float(ens_auc)
            }
            
            analysis['model_performance'] = model_performance
            analysis['ensemble_performance'] = ensemble_performance
            
        return analysis
    
    def plot_model_contributions(self, X, y=None, save_path=None):
        """
        Plot contribution analysis showing how models contribute to the ensemble.
        
        Args:
            X: Features data
            y: Optional true labels for performance evaluation
            save_path (str): Path to save the plot image
            
        Returns:
            None
        """
        # Get contribution analysis
        analysis = self.get_model_contributions(X, y)
        
        # Create figure with multiple subplots
        if y is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
        # Plot model weights
        model_names = analysis['model_names']
        weights = analysis['model_weights']
        
        ax1.bar(model_names, weights)
        ax1.set_title('Model Weights in Ensemble')
        ax1.set_ylabel('Weight')
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot agreement matrix
        agreement = np.array(analysis['model_agreement_matrix'])
        
        im = ax2.imshow(agreement, cmap='viridis')
        ax2.set_title('Model Agreement Matrix')
        ax2.set_xticks(np.arange(len(model_names)))
        ax2.set_yticks(np.arange(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_yticklabels(model_names)
        
        # Add colorbar
        plt.colorbar(im, ax=ax2)
        
        # Plot performance comparison if labels are provided
        if y is not None:
            model_perf = analysis['model_performance']
            ens_perf = analysis['ensemble_performance']
            
            # Extract accuracy values
            acc_values = [mp['accuracy'] for mp in model_perf]
            acc_values.append(ens_perf['accuracy'])
            
            labels = model_names + ['Ensemble']
            
            ax3.bar(labels, acc_values)
            ax3.set_title('Model Accuracy Comparison')
            ax3.set_ylabel('Accuracy')
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            
            # Highlight ensemble bar
            bars = ax3.patches
            bars[-1].set_color('red')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Model contribution plot saved to {save_path}")
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
        
        # Save each base model
        for i, model_info in enumerate(self.models):
            model = model_info['model']
            model_name = model_info['name']
            
            model_subdir = os.path.join(model_dir, f"model_{i}_{model_name}")
            os.makedirs(model_subdir, exist_ok=True)
            
            model.save(model_subdir)
            
        # Save meta-model
        if self.meta_model_type in ['logistic', 'xgboost']:
            meta_model_path = os.path.join(model_dir, "meta_model.joblib")
            dump(self.meta_model, meta_model_path)
        elif self.meta_model_type == 'neural_net':
            meta_model_path = os.path.join(model_dir, "meta_model")
            self.meta_model.save(meta_model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        dump(self.scaler, scaler_path)
        
        # Save model configuration and metadata
        for i, model_info in enumerate(self.models):
            # Remove actual model object for serialization
            model_info.pop('model', None)
            
        config_path = os.path.join(model_dir, "ensemble_config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump({
                'model_weights': self.model_weights,
                'meta_model_type': self.meta_model_type,
                'random_state': self.random_state,
                'feature_groups': self.feature_groups,
                'models_info': self.models,
                'feature_names': self.feature_names
            }, f)
            
        self.logger.info(f"Diverse ensemble model saved to {model_dir}")
        
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
            
        # Load configuration
        config_path = os.path.join(model_dir, "ensemble_config.pkl")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
            
        self.model_weights = config['model_weights']
        self.meta_model_type = config['meta_model_type']
        self.random_state = config['random_state']
        self.feature_groups = config['feature_groups']
        self.feature_names = config['feature_names']
        
        models_info = config['models_info']
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.scaler = load(scaler_path)
        
        # Load base models
        self.models = []
        for i, model_info in enumerate(models_info):
            model_name = model_info['name']
            model_subdir = os.path.join(model_dir, f"model_{i}_{model_name}")
            
            # Determine model type
            metadata_path = os.path.join(model_subdir, f"{model_name}_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            model_type = metadata['model_type']
            
            # Import the model class dynamically
            from . import MODEL_REGISTRY
            
            # Create and load the model
            for model_key, model_class in MODEL_REGISTRY.items():
                if model_class.__name__ == model_type:
                    model = model_class(name=model_name)
                    break
            else:
                raise ValueError(f"Model type {model_type} not found in registry")
                
            model.load(model_subdir)
            
            # Add model to list
            model_info['model'] = model
            self.models.append(model_info)
            
        # Load meta-model
        if self.meta_model_type in ['logistic', 'xgboost']:
            meta_model_path = os.path.join(model_dir, "meta_model.joblib")
            self.meta_model = load(meta_model_path)
        elif self.meta_model_type == 'neural_net':
            from tensorflow.keras.models import load_model
            meta_model_path = os.path.join(model_dir, "meta_model")
            self.meta_model = load_model(meta_model_path)
            
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
                
        self.logger.info(f"Diverse ensemble model loaded from {model_dir}")
        
        return self