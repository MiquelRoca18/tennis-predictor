# models/explainable_model.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import shap
from joblib import dump, load

from .base_model import TennisModel

class ExplainableTennisModel(TennisModel):
    """
    A wrapper model that adds advanced interpretability features to any tennis
    prediction model using SHAP (SHapley Additive exPlanations) values.
    
    This allows detailed explanation of individual predictions, showing which
    features contributed most to a specific match prediction, which is critical
    for understanding model behavior and building trust in the predictions.
    """
    
    def __init__(self, base_model, name=None, version="1.0.0", 
                 background_samples=100, random_state=42):
        """
        Initialize the explainable model wrapper.
        
        Args:
            base_model (TennisModel): Base model to explain
            name (str): Name of the model (defaults to "explainable_{base_model.name}")
            version (str): Version string
            background_samples (int): Number of background samples for SHAP explainer
            random_state (int): Random seed for reproducibility
        """
        if name is None:
            name = f"explainable_{base_model.name}"
            
        super().__init__(name, version)
        
        self.base_model = base_model
        self.background_samples = background_samples
        self.random_state = random_state
        
        # Initialize SHAP explainer
        self.explainer = None
        self.background_data = None
        
        # Set model parameters
        self.model_params = {
            'background_samples': background_samples,
            'random_state': random_state,
            'base_model_name': base_model.name,
            'base_model_type': base_model.__class__.__name__
        }
        
        # Setup logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def fit(self, X, y, validation_data=None, **kwargs):
        """
        Train the base model and create the SHAP explainer.
        
        Args:
            X: Features data
            y: Target data
            validation_data: Optional tuple of (X_val, y_val) for validation
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
        
        # Create background data for SHAP explainer
        self.logger.info(f"Creating SHAP explainer with {self.background_samples} background samples")
        
        np.random.seed(self.random_state)
        
        if len(X_array) > self.background_samples:
            # Randomly sample background data
            bg_indices = np.random.choice(
                len(X_array), self.background_samples, replace=False
            )
            if isinstance(X, pd.DataFrame):
                self.background_data = X.iloc[bg_indices]
            else:
                self.background_data = X_array[bg_indices]
        else:
            # Use all data as background
            self.background_data = X
            
        # Create appropriate SHAP explainer based on model type
        try:
            # Try to determine the best explainer type
            model_type = self.base_model.__class__.__name__.lower()
            
            if 'xgboost' in model_type or hasattr(self.base_model.model, 'feature_importances_'):
                # For tree-based models
                self.logger.info("Using TreeExplainer for SHAP")
                
                # For XGBoost, we need the underlying model
                if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'get_booster'):
                    tree_model = self.base_model.model.get_booster()
                    self.explainer = shap.TreeExplainer(tree_model)
                else:
                    # For other tree-based models
                    tree_model = self.base_model.model
                    self.explainer = shap.TreeExplainer(tree_model)
            elif 'neural' in model_type:
                # For neural networks
                self.logger.info("Using DeepExplainer for SHAP")
                
                # We need a function that returns model output
                def model_output(x):
                    if isinstance(x, pd.DataFrame):
                        return self.base_model.predict_proba(x)
                    else:
                        return self.base_model.predict_proba(x)
                    
                self.explainer = shap.KernelExplainer(model_output, self.background_data)
            else:
                # Default to KernelExplainer which works with any model
                self.logger.info("Using KernelExplainer for SHAP")
                
                def model_output(x):
                    if isinstance(x, pd.DataFrame):
                        return self.base_model.predict_proba(x)
                    else:
                        return self.base_model.predict_proba(x)
                    
                self.explainer = shap.KernelExplainer(model_output, self.background_data)
                
        except Exception as e:
            self.logger.warning(f"Error creating SHAP explainer: {str(e)}")
            self.logger.warning("Falling back to KernelExplainer")
            
            # Fall back to KernelExplainer which works with any model
            def model_output(x):
                if isinstance(x, pd.DataFrame):
                    return self.base_model.predict_proba(x)
                else:
                    return self.base_model.predict_proba(x)
                
            self.explainer = shap.KernelExplainer(model_output, self.background_data)
        
        # Update model metadata
        self.trained = True
        self.training_date = datetime.now()
        
        # Calculate training duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Record training run in history
        training_record = {
            'date': self.training_date.isoformat(),
            'duration_seconds': duration,
            'background_samples': self.background_samples,
            'explainer_type': self.explainer.__class__.__name__,
            'base_model_name': self.base_model.name,
            'base_model_type': self.base_model.__class__.__name__
        }
        
        self.training_history.append(training_record)
        
        self.logger.info(f"SHAP explainer created in {duration:.2f} seconds")
        
        return self
    
    def predict(self, X):
        """
        Make binary predictions with the base model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions with the base model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Probability predictions
        """
        return self.base_model.predict_proba(X)
    
    def explain_prediction(self, X, player1=None, player2=None, match_details=None, plot=True, save_path=None):
        """
        Explain a specific prediction using SHAP values.
        
        Args:
            X: Features data for prediction (single sample)
            player1 (str): Name of player 1 for display
            player2 (str): Name of player 2 for display
            match_details (dict): Additional match details for display
            plot (bool): Whether to generate visualization
            save_path (str): Path to save the visualization
            
        Returns:
            dict: Explanation with SHAP values and feature impacts
        """
        if not self.trained:
            raise ValueError("Model must be trained before explanation")
            
        # Convert to correct format
        if isinstance(X, pd.DataFrame):
            if len(X) > 1:
                self.logger.warning("More than one sample provided. Using only the first one.")
                X_single = X.iloc[[0]]
            else:
                X_single = X
        else:
            if X.ndim > 1 and X.shape[0] > 1:
                self.logger.warning("More than one sample provided. Using only the first one.")
                X_single = X[[0]]
            else:
                X_single = X.reshape(1, -1)
                
        # Get prediction
        prediction = self.predict_proba(X_single)[0]
        predicted_winner = "Player 1" if prediction > 0.5 else "Player 2"
        
        # If player names are provided, use them
        if player1 is not None and player2 is not None:
            predicted_winner = player1 if prediction > 0.5 else player2
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_single)
        
        # Handle different formats of SHAP values
        if isinstance(shap_values, list):
            # For classification, we typically get a list of arrays (one per class)
            if len(shap_values) == 2:
                # Binary classification case
                shap_values = shap_values[1]  # Take values for positive class
        
        # Flatten if needed
        if shap_values.ndim > 2:
            shap_values = shap_values.squeeze()
        
        # Create explanation
        if self.feature_names and len(self.feature_names) == shap_values.shape[1]:
            feature_names = self.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
            
        # Map SHAP values to features
        feature_impacts = dict(zip(feature_names, shap_values[0]))
        
        # Sort features by absolute impact
        sorted_impacts = sorted(
            feature_impacts.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Create explanation object
        explanation = {
            'prediction': float(prediction),
            'predicted_winner': predicted_winner,
            'predicted_winner_probability': float(prediction) if prediction > 0.5 else 1 - float(prediction),
            'top_positive_features': [],
            'top_negative_features': [],
            'all_feature_impacts': sorted_impacts
        }
        
        # Extract top positive and negative features
        for feature, impact in sorted_impacts:
            if impact > 0 and len(explanation['top_positive_features']) < 5:
                explanation['top_positive_features'].append((feature, float(impact)))
            elif impact < 0 and len(explanation['top_negative_features']) < 5:
                explanation['top_negative_features'].append((feature, float(impact)))
                
        # Generate visualization if requested
        if plot:
            plt.figure(figsize=(10, 6))
            
            # Create title
            if player1 and player2:
                title = f"Prediction: {player1} vs {player2}"
                if match_details:
                    if 'tournament' in match_details:
                        title += f" - {match_details['tournament']}"
                    if 'surface' in match_details:
                        title += f" ({match_details['surface']})"
            else:
                title = "Match Prediction Explanation"
                
            subtitle = f"Prediction: {predicted_winner} to win ({prediction:.1%} probability)"
            
            # Plot SHAP values
            shap.summary_plot(
                shap_values, 
                X_single,
                feature_names=feature_names,
                show=False,
                plot_size=(10, 6)
            )
            
            # Add title and subtitle
            plt.suptitle(title, fontsize=14, y=1.02)
            plt.title(subtitle, fontsize=12)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                self.logger.info(f"Explanation plot saved to {save_path}")
            else:
                plt.tight_layout()
                plt.show()
                
        return explanation
    
    def generate_match_report(self, X, player1, player2, match_details=None, save_path=None):
        """
        Generate a detailed match report with prediction and explanation.
        
        Args:
            X: Features data for prediction (single sample)
            player1 (str): Name of player 1
            player2 (str): Name of player 2
            match_details (dict): Additional match details
            save_path (str): Path to save the report
            
        Returns:
            str: Markdown formatted match report
        """
        # Get prediction and explanation
        explanation = self.explain_prediction(
            X, player1, player2, match_details, plot=False
        )
        
        # Format prediction probability
        prob = explanation['prediction']
        prob_str = f"{prob:.1%}" if prob > 0.5 else f"{1-prob:.1%}"
        
        # Create report
        report = [
            f"# Match Prediction Report: {player1} vs {player2}",
            ""
        ]
        
        # Add match details if provided
        if match_details:
            report.extend([
                "## Match Details",
                ""
            ])
            
            for key, value in match_details.items():
                report.append(f"- **{key.title()}**: {value}")
                
            report.append("")
        
        # Add prediction
        report.extend([
            "## Prediction",
            "",
            f"**Predicted Winner: {explanation['predicted_winner']}** with {prob_str} probability",
            "",
            "## Key Factors",
            "",
            "### Factors favoring the predicted winner:",
            ""
        ])
        
        # Add positive factors
        for feature, impact in explanation['top_positive_features']:
            # Format feature name for readability
            readable_feature = feature.replace('_', ' ').title()
            report.append(f"- **{readable_feature}**: +{impact:.4f}")
            
        report.extend([
            "",
            "### Factors against the predicted winner:",
            ""
        ])
        
        # Add negative factors
        for feature, impact in explanation['top_negative_features']:
            # Format feature name for readability
            readable_feature = feature.replace('_', ' ').title()
            report.append(f"- **{readable_feature}**: {impact:.4f}")
            
        # Join the report
        report_text = "\n".join(report)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Match report saved to {save_path}")
            
        return report_text
    
    def feature_importances(self):
        """
        Get global feature importances using SHAP values.
        
        Returns:
            numpy.ndarray: Feature importance scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importances")
            
        # We'll try to return feature importances from the base model first
        try:
            return self.base_model.feature_importances()
        except (AttributeError, NotImplementedError):
            self.logger.warning(
                "Base model does not support feature_importances(). "
                "Using SHAP mean absolute values instead."
            )
            
            # If we have background data, we can use it to calculate SHAP-based importances
            if self.background_data is not None:
                # Calculate SHAP values for background data
                shap_values = self.explainer.shap_values(self.background_data)
                
                # Handle different formats of SHAP values
                if isinstance(shap_values, list):
                    # For classification, we typically get a list of arrays (one per class)
                    if len(shap_values) == 2:
                        # Binary classification case
                        shap_values = shap_values[1]  # Take values for positive class
                
                # Calculate mean absolute SHAP value for each feature
                return np.mean(np.abs(shap_values), axis=0)
            else:
                raise NotImplementedError(
                    "Cannot calculate feature importances: "
                    "Base model does not support it and no background data available for SHAP"
                )
    
    def plot_global_importance(self, top_n=20, plot_type='bar', save_path=None):
        """
        Plot global feature importance using SHAP values.
        
        Args:
            top_n (int): Number of top features to display
            plot_type (str): Type of plot - 'bar', 'beeswarm', or 'summary'
            save_path (str): Path to save the plot image
            
        Returns:
            None
        """
        if not self.trained:
            raise ValueError("Model must be trained before plotting")
            
        if not hasattr(self, 'background_data') or self.background_data is None:
            raise ValueError("No background data available for SHAP visualization")
            
        # Calculate SHAP values for background data
        shap_values = self.explainer.shap_values(self.background_data)
        
        # Handle different formats of SHAP values
        if isinstance(shap_values, list):
            # For classification, we typically get a list of arrays (one per class)
            if len(shap_values) == 2:
                # Binary classification case
                shap_values = shap_values[1]  # Take values for positive class
        
        # Get feature names
        if self.feature_names and len(self.feature_names) == shap_values.shape[1]:
            feature_names = self.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
            
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'bar':
            # Calculate mean absolute SHAP value for each feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Sort features by importance
            indices = np.argsort(mean_abs_shap)
            indices = indices[-top_n:]  # Get top N features
            
            # Create bar plot
            plt.barh(
                y=np.array(feature_names)[indices],
                width=mean_abs_shap[indices]
            )
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance (SHAP Values)')
            
        elif plot_type == 'beeswarm':
            # Use SHAP's built-in beeswarm plot
            shap.summary_plot(
                shap_values, 
                self.background_data,
                feature_names=feature_names,
                max_display=top_n,
                show=False,
                plot_type='dot'
            )
            
        elif plot_type == 'summary':
            # Use SHAP's built-in summary plot
            shap.summary_plot(
                shap_values, 
                self.background_data,
                feature_names=feature_names,
                max_display=top_n,
                show=False
            )
            
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            self.logger.info(f"Global importance plot saved to {save_path}")
        else:
            plt.tight_layout()
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
        
        # Save the explainer if possible (SHAP explainers might not be serializable)
        try:
            explainer_path = os.path.join(model_dir, "explainer.joblib")
            dump(self.explainer, explainer_path)
        except:
            self.logger.warning("Could not serialize SHAP explainer. It will be recreated on load.")
        
        # Save background data
        if isinstance(self.background_data, pd.DataFrame):
            bg_data_path = os.path.join(model_dir, "background_data.csv")
            self.background_data.to_csv(bg_data_path, index=False)
        else:
            bg_data_path = os.path.join(model_dir, "background_data.npy")
            np.save(bg_data_path, self.background_data)
        
        # Save additional parameters
        params_path = os.path.join(model_dir, "params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump({
                'background_samples': self.background_samples,
                'random_state': self.random_state,
                'feature_names': self.feature_names,
                'background_data_path': os.path.basename(bg_data_path),
                'background_data_type': 'dataframe' if isinstance(self.background_data, pd.DataFrame) else 'ndarray'
            }, f)
            
        self.logger.info(f"Explainable model saved to {model_dir}")
        
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
            
        self.background_samples = params['background_samples']
        self.random_state = params['random_state']
        self.feature_names = params['feature_names']
        
        # Load background data
        bg_data_path = os.path.join(model_dir, params['background_data_path'])
        if params['background_data_type'] == 'dataframe':
            self.background_data = pd.read_csv(bg_data_path)
        else:
            self.background_data = np.load(bg_data_path)
        
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
            
            # Try to load explainer if available
            explainer_path = os.path.join(model_dir, "explainer.joblib")
            if os.path.exists(explainer_path):
                try:
                    self.explainer = load(explainer_path)
                except:
                    self.logger.warning("Could not load SHAP explainer. Recreating it.")
                    self._recreate_explainer()
            else:
                self._recreate_explainer()
            
            # Load own metadata
            self.version = metadata['version']
            self.trained = metadata['trained']
            self.training_date = datetime.fromisoformat(metadata['training_date']) if metadata['training_date'] else None
            self.metrics = metadata['metrics']
            
            if 'training_history' in metadata:
                self.training_history = metadata['training_history']
                
        else:
            raise ValueError(f"Metadata file for {self.name} not found")
            
        self.logger.info(f"Explainable model loaded from {model_dir}")
        
        return self
    
    def _recreate_explainer(self):
        """Recreate the SHAP explainer using loaded base model and background data."""
        # Create appropriate SHAP explainer based on model type
        try:
            # Try to determine the best explainer type
            model_type = self.base_model.__class__.__name__.lower()
            
            if 'xgboost' in model_type or hasattr(self.base_model.model, 'feature_importances_'):
                # For tree-based models
                self.logger.info("Using TreeExplainer for SHAP")
                
                # For XGBoost, we need the underlying model
                if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'get_booster'):
                    tree_model = self.base_model.model.get_booster()
                    self.explainer = shap.TreeExplainer(tree_model)
                else:
                    # For other tree-based models
                    tree_model = self.base_model.model
                    self.explainer = shap.TreeExplainer(tree_model)
            elif 'neural' in model_type:
                # For neural networks
                self.logger.info("Using DeepExplainer for SHAP")
                
                # We need a function that returns model output
                def model_output(x):
                    if isinstance(x, pd.DataFrame):
                        return self.base_model.predict_proba(x)
                    else:
                        return self.base_model.predict_proba(x)
                    
                self.explainer = shap.KernelExplainer(model_output, self.background_data)
            else:
                # Default to KernelExplainer which works with any model
                self.logger.info("Using KernelExplainer for SHAP")
                
                def model_output(x):
                    if isinstance(x, pd.DataFrame):
                        return self.base_model.predict_proba(x)
                    else:
                        return self.base_model.predict_proba(x)
                    
                self.explainer = shap.KernelExplainer(model_output, self.background_data)
                
        except Exception as e:
            self.logger.warning(f"Error creating SHAP explainer: {str(e)}")
            self.logger.warning("Falling back to KernelExplainer")
            
            # Fall back to KernelExplainer which works with any model
            def model_output(x):
                if isinstance(x, pd.DataFrame):
                    return self.base_model.predict_proba(x)
                else:
                    return self.base_model.predict_proba(x)
                
            self.explainer = shap.KernelExplainer(model_output, self.background_data)