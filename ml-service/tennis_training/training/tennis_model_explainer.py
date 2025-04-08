"""
Tennis Model Explainer

This module provides advanced techniques for interpreting and explaining
tennis prediction models, including SHAP value analysis and match-specific
explanations for why a player is favored.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Try to import SHAP, but handle case where it's not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)

class TennisModelExplainer:
    """
    Class for explaining and interpreting tennis prediction models.
    
    This class provides tools for:
    - Calculating SHAP values to understand feature importance
    - Generating natural language explanations for match predictions
    - Visualizing key factors affecting match outcomes
    - Explaining why a specific player is favored in a match
    
    Attributes:
        model: Trained machine learning model
        feature_names: List of feature names
        results_dir: Directory for storing explanation results
        explainer: SHAP explainer object (if SHAP is available)
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        results_dir: str = 'explanation_results',
        model_type: str = 'tree'
    ):
        """
        Initialize the tennis model explainer.
        
        Args:
            model: Trained machine learning model
            feature_names: List of feature names
            results_dir: Directory for storing explanation results
            model_type: Type of model ('tree', 'linear', or 'kernel')
        """
        self.model = model
        self.feature_names = feature_names
        self.results_dir = results_dir
        self.model_type = model_type
        self.explainer = None
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            self._init_shap_explainer()
        else:
            logger.warning("SHAP not available. Install with: pip install shap")
        
        logger.info("Initialized TennisModelExplainer")
    
    def _init_shap_explainer(self) -> None:
        """Initialize SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            return
        
        try:
            if self.model_type == 'tree':
                if hasattr(self.model, 'predict_proba'):
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    logger.warning("Model does not have predict_proba method")
            elif self.model_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, shap.sample(np.zeros((1, len(self.feature_names)))))
            elif self.model_type == 'kernel':
                # Kernel explainer needs a background dataset, which we'll create as needed
                self.explainer = None
            else:
                logger.warning(f"Unsupported model type: {self.model_type}")
            
            if self.explainer:
                logger.info(f"Initialized SHAP {self.model_type} explainer")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            self.explainer = None
    
    def calculate_shap_values(
        self,
        X: pd.DataFrame,
        background_data: Optional[pd.DataFrame] = None,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Calculate SHAP values for input data.
        
        Args:
            X: Input data for explanation
            background_data: Background data for kernel explainer
            n_samples: Number of samples to use for kernel explainer
            
        Returns:
            Array of SHAP values
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Cannot calculate SHAP values.")
            return np.zeros((len(X), len(self.feature_names)))
        
        if self.explainer is None and self.model_type == 'kernel':
            # Initialize kernel explainer with background data
            if background_data is None:
                logger.warning("No background data provided for kernel explainer. Using input data.")
                background_data = X
            
            # Sample background data if needed
            if len(background_data) > n_samples:
                background_sample = shap.sample(background_data, n_samples)
            else:
                background_sample = background_data
            
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                background_sample
            )
            logger.info(f"Initialized SHAP kernel explainer with {len(background_sample)} background samples")
        
        try:
            if self.explainer is None:
                raise ValueError("SHAP explainer not initialized")
            
            # Use feature names from input data if available
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = self.feature_names
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # For classifiers, SHAP returns a list of arrays (one per class)
            # We're interested in the second class (index 1) for binary classification
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
            
            logger.info(f"Calculated SHAP values for {len(X)} samples with {len(feature_names)} features")
            return shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return np.zeros((len(X), len(self.feature_names)))
    
    def plot_shap_summary(
        self,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 20,
        plot_type: str = "bar",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create SHAP summary plot for feature importance.
        
        Args:
            X: Input data for explanation
            shap_values: Pre-calculated SHAP values (or None to calculate)
            max_display: Maximum number of features to display
            plot_type: Type of plot ("bar", "dot", or "violin")
            save_path: Path to save the plot (or None to use default)
            
        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Cannot create summary plot.")
            return None
        
        try:
            # Calculate SHAP values if not provided
            if shap_values is None:
                shap_values = self.calculate_shap_values(X)
            
            # Create figure
            plt.figure(figsize=(10, 10))
            
            # Create SHAP summary plot
            if plot_type == "bar":
                shap.summary_plot(
                    shap_values, 
                    X, 
                    plot_type="bar", 
                    max_display=max_display,
                    show=False
                )
            elif plot_type == "dot":
                shap.summary_plot(
                    shap_values, 
                    X, 
                    plot_type="dot", 
                    max_display=max_display,
                    show=False
                )
            elif plot_type == "violin":
                shap.summary_plot(
                    shap_values, 
                    X, 
                    max_display=max_display,
                    show=False
                )
            else:
                logger.warning(f"Unsupported plot type: {plot_type}. Using 'bar'.")
                shap.summary_plot(
                    shap_values, 
                    X, 
                    plot_type="bar", 
                    max_display=max_display,
                    show=False
                )
            
            # Get current figure
            fig = plt.gcf()
            
            # Save plot if path provided
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                save_path = os.path.join(self.results_dir, f"shap_summary_{plot_type}_{timestamp}.png")
            
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot to {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            return None
    
    def plot_shap_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create SHAP dependence plot to show relationship between a feature and its SHAP values.
        
        Args:
            X: Input data for explanation
            feature: Feature to plot
            interaction_feature: Feature to use for coloring (or None for auto)
            shap_values: Pre-calculated SHAP values (or None to calculate)
            save_path: Path to save the plot (or None to use default)
            
        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Cannot create dependence plot.")
            return None
        
        try:
            # Check if feature exists
            if feature not in X.columns:
                logger.warning(f"Feature '{feature}' not found in input data")
                return None
            
            # Calculate SHAP values if not provided
            if shap_values is None:
                shap_values = self.calculate_shap_values(X)
            
            # Create figure
            plt.figure(figsize=(10, 7))
            
            # Create SHAP dependence plot
            if interaction_feature is not None and interaction_feature in X.columns:
                shap.dependence_plot(
                    feature, 
                    shap_values, 
                    X,
                    interaction_index=interaction_feature,
                    show=False
                )
            else:
                shap.dependence_plot(
                    feature, 
                    shap_values, 
                    X,
                    show=False
                )
            
            # Get current figure
            fig = plt.gcf()
            
            # Save plot if path provided
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                interaction_suffix = f"_x_{interaction_feature}" if interaction_feature else ""
                save_path = os.path.join(self.results_dir, f"shap_dependence_{feature}{interaction_suffix}_{timestamp}.png")
            
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved SHAP dependence plot to {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {e}")
            return None
    
    def plot_shap_waterfall(
        self,
        X: pd.DataFrame,
        instance_idx: int = 0,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create SHAP waterfall plot for a specific prediction.
        
        Args:
            X: Input data for explanation
            instance_idx: Index of the instance to explain
            shap_values: Pre-calculated SHAP values (or None to calculate)
            max_display: Maximum number of features to display
            save_path: Path to save the plot (or None to use default)
            
        Returns:
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Cannot create waterfall plot.")
            return None
        
        try:
            # Calculate SHAP values if not provided
            if shap_values is None:
                shap_values = self.calculate_shap_values(X)
            
            # Check instance index
            if instance_idx >= len(X):
                logger.warning(f"Instance index {instance_idx} out of range")
                return None
            
            # Get instance data
            instance = X.iloc[instance_idx:instance_idx+1]
            instance_shap = shap_values[instance_idx]
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Create SHAP waterfall plot
            try:
                # For newer SHAP versions
                shap.plots.waterfall(
                    shap.Explanation(
                        values=instance_shap,
                        base_values=self.explainer.expected_value,
                        data=instance.values,
                        feature_names=X.columns.tolist()
                    ),
                    max_display=max_display,
                    show=False
                )
            except (AttributeError, TypeError):
                # For older SHAP versions
                from shap.plots import _waterfall
                _waterfall.waterfall_legacy(
                    self.explainer.expected_value,
                    instance_shap,
                    features=instance.values[0],
                    feature_names=X.columns.tolist(),
                    max_display=max_display,
                    show=False
                )
            
            # Get current figure
            fig = plt.gcf()
            
            # Save plot if path provided
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                save_path = os.path.join(self.results_dir, f"shap_waterfall_instance{instance_idx}_{timestamp}.png")
            
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved SHAP waterfall plot to {save_path}")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating SHAP waterfall plot: {e}")
            return None
    
    def explain_match_prediction(
        self,
        match_features: pd.DataFrame,
        player1_name: str,
        player2_name: str,
        match_meta: Optional[Dict[str, Any]] = None,
        n_factors: int = 5,
        generate_nl_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a prediction for a specific tennis match.
        
        Args:
            match_features: Features for the match
            player1_name: Name of player 1
            player2_name: Name of player 2
            match_meta: Additional match metadata
            n_factors: Number of top factors to include
            generate_nl_explanation: Whether to generate natural language explanation
            
        Returns:
            Dictionary with explanation details
        """
        try:
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(match_features)[0, 1]
            else:
                prediction = self.model.predict(match_features)[0]
                prediction = float(prediction)
            
            # Predict winner
            predicted_winner = player1_name if prediction >= 0.5 else player2_name
            win_probability = prediction if prediction >= 0.5 else 1 - prediction
            
            # Calculate SHAP values
            shap_values = self.calculate_shap_values(match_features)
            
            # Get feature values
            feature_values = match_features.iloc[0].to_dict()
            
            # Get feature contributions
            feature_contributions = {}
            for i, feature in enumerate(match_features.columns):
                feature_contributions[feature] = float(shap_values[0, i])
            
            # Sort features by absolute contribution
            sorted_contributions = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Get top N factors
            top_factors = []
            for feature, contribution in sorted_contributions[:n_factors]:
                # Determine if factor favors player1 or player2
                favors_player1 = contribution > 0
                factor = {
                    'feature': feature,
                    'contribution': contribution,
                    'value': feature_values[feature],
                    'favors': player1_name if favors_player1 else player2_name
                }
                top_factors.append(factor)
            
            # Create visualization
            self._plot_match_explanation(
                top_factors, 
                player1_name, 
                player2_name, 
                prediction
            )
            
            # Generate explanation
            explanation = {
                'player1': player1_name,
                'player2': player2_name,
                'prediction': prediction,
                'predicted_winner': predicted_winner,
                'win_probability': win_probability,
                'top_factors': top_factors,
                'shap_values': shap_values[0].tolist()
            }
            
            # Add metadata if provided
            if match_meta:
                explanation['match_meta'] = match_meta
            
            # Generate natural language explanation if requested
            if generate_nl_explanation:
                explanation['nl_explanation'] = self._generate_nl_explanation(
                    player1_name,
                    player2_name,
                    prediction,
                    top_factors,
                    match_meta
                )
            
            return explanation
        except Exception as e:
            logger.error(f"Error explaining match prediction: {e}")
            return {
                'error': str(e),
                'player1': player1_name,
                'player2': player2_name
            }
    
    def _plot_match_explanation(
        self,
        top_factors: List[Dict[str, Any]],
        player1_name: str,
        player2_name: str,
        prediction: float,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visualization of match explanation.
        
        Args:
            top_factors: List of top contributing factors
            player1_name: Name of player 1
            player2_name: Name of player 2
            prediction: Predicted probability for player 1
            save_path: Path to save the plot (or None to use default)
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up factor visualization
        features = [self._format_feature_name(f['feature']) for f in top_factors]
        values = [f['contribution'] for f in top_factors]
        colors = ['green' if f['favors'] == player1_name else 'red' for f in top_factors]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color=colors)
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        
        # Add legend
        player1_patch = plt.Rectangle((0, 0), 1, 1, fc="green", edgecolor='none')
        player2_patch = plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='none')
        ax.legend([player1_patch, player2_patch], [f"Favors {player1_name}", f"Favors {player2_name}"])
        
        # Add title
        predicted_winner = player1_name if prediction >= 0.5 else player2_name
        win_prob = prediction if prediction >= 0.5 else 1 - prediction
        ax.set_title(f"Match Prediction: {predicted_winner} to win ({win_prob:.1%} probability)\n{player1_name} vs {player2_name}")
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels for each bar showing the actual feature value
        for i, factor in enumerate(top_factors):
            value_text = self._format_feature_value(factor['feature'], factor['value'])
            ax.text(
                0,
                i,
                f" {value_text}",
                va='center',
                ha='center',
                fontsize=9,
                fontweight='bold',
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
            )
        
        # Save plot if path provided
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            player1_short = player1_name.split()[-1]
            player2_short = player2_name.split()[-1]
            save_path = os.path.join(self.results_dir, f"match_explanation_{player1_short}_vs_{player2_short}_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved match explanation plot to {save_path}")
        
        return fig
    
    def _format_feature_name(self, feature_name: str) -> str:
        """Format feature name for display."""
        # Replace underscores with spaces
        name = feature_name.replace('_', ' ')
        
        # Capitalize words
        name = ' '.join(word.capitalize() for word in name.split())
        
        # Handle special cases
        name = name.replace('Elo', 'ELO')
        name = name.replace('H2h', 'H2H')
        
        return name
    
    def _format_feature_value(self, feature_name: str, value: Any) -> str:
        """Format feature value for display."""
        # Handle different feature types
        if 'elo' in feature_name.lower() or 'rating' in feature_name.lower():
            return f"{value:.0f}"
        elif 'pct' in feature_name.lower() or 'percentage' in feature_name.lower() or 'rate' in feature_name.lower():
            return f"{value:.1%}"
        elif isinstance(value, float):
            return f"{value:.2f}"
        else:
            return str(value)
    
    def _generate_nl_explanation(
        self,
        player1_name: str,
        player2_name: str,
        prediction: float,
        top_factors: List[Dict[str, Any]],
        match_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate natural language explanation for match prediction.
        
        Args:
            player1_name: Name of player 1
            player2_name: Name of player 2
            prediction: Predicted probability for player 1
            top_factors: List of top contributing factors
            match_meta: Additional match metadata
            
        Returns:
            Natural language explanation
        """
        # Determine predicted winner and probability
        predicted_winner = player1_name if prediction >= 0.5 else player2_name
        predicted_loser = player2_name if prediction >= 0.5 else player1_name
        win_probability = prediction if prediction >= 0.5 else 1 - prediction
        
        # Start with summary
        if win_probability > 0.9:
            strength = "heavily favored"
        elif win_probability > 0.75:
            strength = "strongly favored"
        elif win_probability > 0.6:
            strength = "favored"
        else:
            strength = "slightly favored"
        
        explanation = [
            f"{predicted_winner} is {strength} to win against {predicted_loser}",
            f"with a {win_probability:.1%} probability."
        ]
        
        # Add match details if available
        if match_meta:
            if 'tournament' in match_meta:
                explanation.append(f" This {match_meta['tournament']} match")
                
                if 'round' in match_meta:
                    explanation[-1] += f" in the {match_meta['round']}"
                    
                if 'surface' in match_meta:
                    explanation[-1] += f" on {match_meta['surface']}"
                
                explanation[-1] += "."
            elif 'surface' in match_meta:
                explanation.append(f" This match on {match_meta['surface']}.")
        
        explanation.append("\n\nKey factors influencing this prediction:")
        
        # Add factors favoring winner
        winner_factors = [f for f in top_factors if f['favors'] == predicted_winner]
        if winner_factors:
            explanation.append(f"\nFactors favoring {predicted_winner}:")
            for factor in winner_factors:
                feature_name = self._format_feature_name(factor['feature'])
                feature_value = self._format_feature_value(factor['feature'], factor['value'])
                
                factor_text = self._get_factor_description(
                    factor['feature'],
                    feature_value,
                    factor['contribution'],
                    predicted_winner,
                    predicted_loser
                )
                
                if factor_text:
                    explanation.append(f"- {factor_text}")
                else:
                    explanation.append(f"- {feature_name}: {feature_value}")
        
        # Add factors favoring loser
        loser_factors = [f for f in top_factors if f['favors'] == predicted_loser]
        if loser_factors:
            explanation.append(f"\nFactors favoring {predicted_loser}:")
            for factor in loser_factors:
                feature_name = self._format_feature_name(factor['feature'])
                feature_value = self._format_feature_value(factor['feature'], factor['value'])
                
                factor_text = self._get_factor_description(
                    factor['feature'],
                    feature_value,
                    factor['contribution'],
                    predicted_loser,
                    predicted_winner
                )
                
                if factor_text:
                    explanation.append(f"- {factor_text}")
                else:
                    explanation.append(f"- {feature_name}: {feature_value}")
        
        # Join all parts
        return " ".join(explanation[:2]) + "".join(explanation[2:])
    
    def _get_factor_description(
        self,
        feature: str,
        value: str,
        contribution: float,
        favored_player: str,
        other_player: str
    ) -> str:
        """Generate a description for a specific factor."""
        feature_lower = feature.lower()
        abs_contribution = abs(contribution)
        
        # Customize based on feature type
        if 'elo' in feature_lower:
            if 'difference' in feature_lower:
                return f"Higher ELO rating ({value} points difference)"
            else:
                return f"Strong ELO rating of {value}"
        
        elif 'h2h' in feature_lower:
            if 'win' in feature_lower:
                return f"Favorable head-to-head record ({value})"
            else:
                return f"Head-to-head advantage ({value})"
        
        elif 'surface' in feature_lower:
            if abs_contribution > 0.05:
                return f"Strong performance on this surface ({value})"
            else:
                return f"Slight advantage on this surface ({value})"
        
        elif 'age' in feature_lower or 'experience' in feature_lower:
            return f"Age/experience factor ({value})"
        
        elif 'rank' in feature_lower:
            if 'difference' in feature_lower:
                return f"Better ranking ({value} positions difference)"
            else:
                return f"Higher ranking ({value})"
        
        elif 'win_rate' in feature_lower or 'winrate' in feature_lower:
            return f"Better recent win rate ({value})"
        
        elif 'rest' in feature_lower or 'fatigue' in feature_lower:
            return f"Better recovery/less fatigue ({value})"
        
        # Generic description for other features
        return ""
    
    def explain_feature_impact(
        self,
        feature_name: str,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Create detailed explanation of a specific feature's impact on predictions.
        
        Args:
            feature_name: Name of the feature to explain
            X: Input data
            shap_values: Pre-calculated SHAP values (or None to calculate)
            
        Returns:
            Dictionary with feature impact details
        """
        if feature_name not in X.columns:
            logger.warning(f"Feature '{feature_name}' not found in input data")
            return {'error': f"Feature '{feature_name}' not found"}
        
        try:
            # Calculate SHAP values if not provided
            if shap_values is None:
                shap_values = self.calculate_shap_values(X)
            
            # Get feature index
            feature_idx = list(X.columns).index(feature_name)
            
            # Get feature values and corresponding SHAP values
            feature_values = X[feature_name].values
            feature_shaps = shap_values[:, feature_idx]
            
            # Calculate basic statistics
            mean_impact = np.mean(feature_shaps)
            abs_mean_impact = np.mean(np.abs(feature_shaps))
            max_pos_impact = np.max(feature_shaps)
            max_neg_impact = np.min(feature_shaps)
            
            # Find samples with highest positive and negative impact
            max_pos_idx = np.argmax(feature_shaps)
            max_neg_idx = np.argmin(feature_shaps)
            
            max_pos_value = feature_values[max_pos_idx]
            max_neg_value = feature_values[max_neg_idx]
            
            # Calculate correlation between feature value and impact
            correlation = np.corrcoef(feature_values, feature_shaps)[0, 1]
            
            # Determine if impact is monotonic or complex
            increasing = correlation > 0.5
            decreasing = correlation < -0.5
            if abs(correlation) > 0.5:
                relationship = "strongly increasing" if increasing else "strongly decreasing"
            elif abs(correlation) > 0.3:
                relationship = "moderately increasing" if increasing else "moderately decreasing"
            elif abs(correlation) > 0.1:
                relationship = "weakly increasing" if increasing else "weakly decreasing"
            else:
                relationship = "complex/non-linear"
            
            # Create feature impact visualization
            fig = self.plot_shap_dependence(X, feature_name)
            
            # Determine optimal feature values (where SHAP is highest)
            if relationship == "complex/non-linear":
                # For complex relationships, bin the data and find the best bin
                bins = 10
                bin_edges = np.histogram_bin_edges(feature_values, bins=bins)
                bin_indices = np.digitize(feature_values, bin_edges)
                
                bin_mean_impacts = []
                for i in range(1, bins + 1):
                    bin_mask = bin_indices == i
                    if np.any(bin_mask):
                        bin_mean_impacts.append(np.mean(feature_shaps[bin_mask]))
                    else:
                        bin_mean_impacts.append(np.nan)
                
                best_bin = np.nanargmax(bin_mean_impacts) + 1
                worst_bin = np.nanargmin(bin_mean_impacts) + 1
                
                optimal_range = (bin_edges[best_bin-1], bin_edges[best_bin])
                suboptimal_range = (bin_edges[worst_bin-1], bin_edges[worst_bin])
            else:
                # For monotonic relationships, simply use the direction
                if increasing:
                    optimal_range = (np.percentile(feature_values, 90), np.max(feature_values))
                    suboptimal_range = (np.min(feature_values), np.percentile(feature_values, 10))
                else:
                    optimal_range = (np.min(feature_values), np.percentile(feature_values, 10))
                    suboptimal_range = (np.percentile(feature_values, 90), np.max(feature_values))
            
            # Format output
            impact_details = {
                'feature': feature_name,
                'formatted_name': self._format_feature_name(feature_name),
                'mean_impact': float(mean_impact),
                'abs_mean_impact': float(abs_mean_impact),
                'max_positive_impact': float(max_pos_impact),
                'max_negative_impact': float(max_neg_impact),
                'correlation_with_impact': float(correlation),
                'relationship': relationship,
                'optimal_range': (float(optimal_range[0]), float(optimal_range[1])),
                'suboptimal_range': (float(suboptimal_range[0]), float(suboptimal_range[1])),
                'max_positive_example': float(max_pos_value),
                'max_negative_example': float(max_neg_value),
                'value_distribution': {
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'mean': float(np.mean(feature_values)),
                    'p25': float(np.percentile(feature_values, 25)),
                    'p50': float(np.percentile(feature_values, 50)),
                    'p75': float(np.percentile(feature_values, 75))
                }
            }
            
            # Generate natural language explanation
            impact_details['explanation'] = self._generate_feature_explanation(impact_details)
            
            return impact_details
        except Exception as e:
            logger.error(f"Error explaining feature impact: {e}")
            return {'error': str(e), 'feature': feature_name}
    
    def _generate_feature_explanation(self, impact_details: Dict[str, Any]) -> str:
        """
        Generate natural language explanation of feature impact.
        
        Args:
            impact_details: Dictionary with feature impact details
            
        Returns:
            Natural language explanation
        """
        feature_name = impact_details['formatted_name']
        relationship = impact_details['relationship']
        mean_impact = impact_details['mean_impact']
        abs_mean_impact = impact_details['abs_mean_impact']
        
        # Start with overall impact
        if abs_mean_impact > 0.1:
            impact_strength = "substantial"
        elif abs_mean_impact > 0.05:
            impact_strength = "moderate"
        elif abs_mean_impact > 0.02:
            impact_strength = "modest"
        else:
            impact_strength = "minimal"
        
        explanation = [
            f"{feature_name} has a {impact_strength} impact on match predictions",
            f"with an average absolute SHAP value of {abs_mean_impact:.4f}."
        ]
        
        # Describe relationship
        if "increasing" in relationship:
            direction = "positive" if mean_impact > 0 else "negative"
            explanation.append(
                f" There is a {relationship} relationship between {feature_name} and match outcome,"
                f" meaning higher values generally have a {direction} effect on predictions."
            )
        elif "decreasing" in relationship:
            direction = "negative" if mean_impact > 0 else "positive"
            explanation.append(
                f" There is a {relationship} relationship between {feature_name} and match outcome,"
                f" meaning lower values generally have a {direction} effect on predictions."
            )
        else:
            explanation.append(
                f" The relationship between {feature_name} and match outcome is {relationship},"
                f" with different values having varying effects."
            )
        
        # Add optimal values
        optimal_min, optimal_max = impact_details['optimal_range']
        
        if "increasing" in relationship or "decreasing" in relationship:
            if optimal_min == optimal_max:
                explanation.append(
                    f" For best predictions, {feature_name} should ideally be around {self._format_feature_value(impact_details['feature'], optimal_min)}."
                )
            else:
                explanation.append(
                    f" For best predictions, {feature_name} should ideally be between {self._format_feature_value(impact_details['feature'], optimal_min)}"
                    f" and {self._format_feature_value(impact_details['feature'], optimal_max)}."
                )
        else:
            explanation.append(
                f" The most favorable values for {feature_name} are in the range of {self._format_feature_value(impact_details['feature'], optimal_min)}"
                f" to {self._format_feature_value(impact_details['feature'], optimal_max)}."
            )
        
        # Join all parts
        return "".join(explanation)
    
    def generate_match_comparison(
        self,
        player1_features: Dict[str, Any],
        player2_features: Dict[str, Any],
        player1_name: str,
        player2_name: str,
        match_features: Optional[pd.DataFrame] = None,
        feature_categories: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed comparison between two players for a match.
        
        Args:
            player1_features: Features for player 1
            player2_features: Features for player 2
            player1_name: Name of player 1
            player2_name: Name of player 2
            match_features: Features for the match prediction
            feature_categories: Dictionary mapping categories to feature lists
            
        Returns:
            Dictionary with comparison details
        """
        # Define default feature categories if not provided
        if feature_categories is None:
            feature_categories = {
                'Ranking & Rating': ['rank', 'ranking', 'elo', 'rating'],
                'Form & Performance': ['win_rate', 'form', 'streak'],
                'Surface Performance': ['surface', 'carpet', 'clay', 'grass', 'hard'],
                'Physical & Style': ['age', 'height', 'weight', 'style'],
                'Match Context': ['h2h', 'tournament', 'round', 'rest', 'fatigue']
            }
        
        # Initialize comparison
        comparison = {
            'player1_name': player1_name,
            'player2_name': player2_name,
            'categories': {},
            'advantages': {
                player1_name: [],
                player2_name: []
            }
        }
        
        # Add prediction if match features provided
        if match_features is not None and not match_features.empty:
            prediction = self.model.predict_proba(match_features)[0, 1]
            predicted_winner = player1_name if prediction >= 0.5 else player2_name
            win_probability = prediction if prediction >= 0.5 else 1 - prediction
            
            comparison['prediction'] = {
                'winner': predicted_winner,
                'probability': float(win_probability),
                'player1_probability': float(prediction)
            }
            
            # Calculate SHAP values
            shap_values = self.calculate_shap_values(match_features)[0]
            feature_influences = dict(zip(match_features.columns, shap_values))
            
            # Sort features by absolute influence
            sorted_influences = sorted(
                feature_influences.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Get top influencing features
            comparison['key_factors'] = [
                {
                    'feature': feature,
                    'influence': float(influence),
                    'favors': player1_name if influence > 0 else player2_name
                }
                for feature, influence in sorted_influences[:10]
            ]
        
        # Compare features by category
        all_features = set(player1_features.keys()) | set(player2_features.keys())
        categorized_features = {}
        
        # Assign features to categories
        for feature in all_features:
            assigned = False
            
            for category, patterns in feature_categories.items():
                if any(pattern in feature.lower() for pattern in patterns):
                    if category not in categorized_features:
                        categorized_features[category] = []
                    
                    categorized_features[category].append(feature)
                    assigned = True
                    break
            
            if not assigned:
                if 'Other' not in categorized_features:
                    categorized_features['Other'] = []
                
                categorized_features['Other'].append(feature)
        
        # Compare features within each category
        for category, features in categorized_features.items():
            comparison['categories'][category] = []
            
            for feature in sorted(features):
                p1_value = player1_features.get(feature, None)
                p2_value = player2_features.get(feature, None)
                
                # Skip if both values are missing
                if p1_value is None and p2_value is None:
                    continue
                
                # Determine advantage
                advantage = None
                advantage_size = 0
                
                if p1_value is not None and p2_value is not None:
                    # Check if higher is better (default assumption)
                    higher_is_better = not any(neg in feature.lower() for neg in ['error', 'fault', 'fatigue'])
                    
                    if higher_is_better:
                        if p1_value > p2_value:
                            advantage = player1_name
                            if isinstance(p1_value, (int, float)) and isinstance(p2_value, (int, float)):
                                advantage_size = (p1_value - p2_value) / max(abs(p1_value), abs(p2_value), 1e-10)
                        elif p2_value > p1_value:
                            advantage = player2_name
                            if isinstance(p1_value, (int, float)) and isinstance(p2_value, (int, float)):
                                advantage_size = (p2_value - p1_value) / max(abs(p1_value), abs(p2_value), 1e-10)
                    else:
                        if p1_value < p2_value:
                            advantage = player1_name
                            if isinstance(p1_value, (int, float)) and isinstance(p2_value, (int, float)):
                                advantage_size = (p2_value - p1_value) / max(abs(p1_value), abs(p2_value), 1e-10)
                        elif p2_value < p1_value:
                            advantage = player2_name
                            if isinstance(p1_value, (int, float)) and isinstance(p2_value, (int, float)):
                                advantage_size = (p1_value - p2_value) / max(abs(p1_value), abs(p2_value), 1e-10)
                
                # Add feature comparison
                comparison['categories'][category].append({
                    'feature': feature,
                    'formatted_name': self._format_feature_name(feature),
                    'player1_value': p1_value,
                    'player2_value': p2_value,
                    'advantage': advantage,
                    'advantage_size': float(advantage_size) if advantage else 0
                })
                
                # Track significant advantages
                if advantage and abs(advantage_size) > 0.1:
                    formatted_feature = self._format_feature_name(feature)
                    
                    if advantage == player1_name:
                        comparison['advantages'][player1_name].append({
                            'feature': feature,
                            'formatted_name': formatted_feature,
                            'advantage_size': float(advantage_size)
                        })
                    else:
                        comparison['advantages'][player2_name].append({
                            'feature': feature,
                            'formatted_name': formatted_feature,
                            'advantage_size': float(advantage_size)
                        })
        
        # Sort advantages by size
        for player in comparison['advantages']:
            comparison['advantages'][player] = sorted(
                comparison['advantages'][player],
                key=lambda x: abs(x['advantage_size']),
                reverse=True
            )
        
        # Generate natural language summary
        comparison['summary'] = self._generate_comparison_summary(comparison)
        
        # Create visualization
        self._plot_player_comparison(comparison)
        
        return comparison
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> str:
        """
        Generate a natural language summary of the player comparison.
        
        Args:
            comparison: Comparison data
            
        Returns:
            Natural language summary
        """
        player1 = comparison['player1_name']
        player2 = comparison['player2_name']
        
        # Start with prediction if available
        if 'prediction' in comparison:
            prediction = comparison['prediction']
            winner = prediction['winner']
            loser = player2 if winner == player1 else player1
            probability = prediction['probability']
            
            if probability > 0.9:
                strength = "heavily favored"
            elif probability > 0.75:
                strength = "strongly favored"
            elif probability > 0.6:
                strength = "favored"
            else:
                strength = "slightly favored"
            
            summary = [f"{winner} is {strength} over {loser} with a {probability:.1%} win probability."]
        else:
            summary = [f"Comparison between {player1} and {player2}:"]
        
        # Summarize key advantages for each player
        for player in [player1, player2]:
            advantages = comparison['advantages'][player]
            
            if not advantages:
                continue
                
            if len(advantages) == 1:
                adv = advantages[0]
                summary.append(f" {player}'s main advantage is in {adv['formatted_name']}.")
            elif len(advantages) == 2:
                adv1, adv2 = advantages[0], advantages[1]
                summary.append(
                    f" {player}'s advantages are in {adv1['formatted_name']} "
                    f"and {adv2['formatted_name']}."
                )
            else:
                top_advantages = [adv['formatted_name'] for adv in advantages[:3]]
                summary.append(
                    f" {player}'s key advantages include {', '.join(top_advantages[:-1])} "
                    f"and {top_advantages[-1]}."
                )
        
        # Add category summaries
        significant_categories = []
        
        for category, features in comparison['categories'].items():
            # Calculate cumulative advantage for the category
            p1_advantage = sum(
                f['advantage_size'] for f in features 
                if f['advantage'] == player1
            )
            p2_advantage = sum(
                f['advantage_size'] for f in features 
                if f['advantage'] == player2
            )
            
            net_advantage = p1_advantage - p2_advantage
            advantaged_player = player1 if net_advantage > 0 else player2
            
            if abs(net_advantage) > 0.2:
                significant_categories.append({
                    'category': category,
                    'advantage': abs(net_advantage),
                    'player': advantaged_player
                })
        
        # Include most significant category advantage
        if significant_categories:
            significant_categories.sort(key=lambda x: x['advantage'], reverse=True)
            top_category = significant_categories[0]
            
            summary.append(
                f" Overall, {top_category['player']} has a significant advantage "
                f"in the {top_category['category']} category."
            )
        
        # Return combined summary
        return " ".join(summary)
    
    def _plot_player_comparison(
        self,
        comparison: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visualization of player comparison.
        
        Args:
            comparison: Comparison data
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        player1 = comparison['player1_name']
        player2 = comparison['player2_name']
        
        # Count total features for sizing
        total_features = sum(len(features) for features in comparison['categories'].values())
        
        # Create figure with dynamic height
        fig_height = max(8, total_features * 0.4)
        fig, axes = plt.subplots(
            len(comparison['categories']), 
            1, 
            figsize=(12, fig_height),
            gridspec_kw={'hspace': 0.4}
        )
        
        # Handle case of single category
        if len(comparison['categories']) == 1:
            axes = [axes]
        
        # Plot each category
        for i, (category, features) in enumerate(comparison['categories'].items()):
            ax = axes[i]
            
            feature_names = [f['formatted_name'] for f in features]
            advantages = [
                f['advantage_size'] if f['advantage'] == player1 else -f['advantage_size']
                for f in features
            ]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(feature_names))
            bars = ax.barh(
                y_pos, 
                advantages,
                color=['green' if adv > 0 else 'red' for adv in advantages]
            )
            
            # Add labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_title(category)
            
            # Add vertical line at zero
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add player names on either side
            ax.text(
                max(advantages + [0.1]) * 1.05, 
                len(feature_names) - 1, 
                player1,
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='green'
            )
            
            ax.text(
                min(advantages + [-0.1]) * 1.05, 
                len(feature_names) - 1, 
                player2,
                ha='right',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='red'
            )
            
            # Add value labels
            for j, feature in enumerate(features):
                p1_value = feature['player1_value']
                p2_value = feature['player2_value']
                
                # Format values
                if p1_value is not None:
                    p1_text = self._format_feature_value(feature['feature'], p1_value)
                    ax.text(
                        0.02, 
                        j, 
                        p1_text,
                        ha='left',
                        va='center',
                        fontsize=8,
                        color='green',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1')
                    )
                
                if p2_value is not None:
                    p2_text = self._format_feature_value(feature['feature'], p2_value)
                    ax.text(
                        -0.02, 
                        j, 
                        p2_text,
                        ha='right',
                        va='center',
                        fontsize=8,
                        color='red',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1')
                    )
        
        # Add title with prediction if available
        if 'prediction' in comparison:
            prediction = comparison['prediction']
            winner = prediction['winner']
            probability = prediction['probability']
            
            plt.suptitle(
                f"Player Comparison: {player1} vs {player2}\n"
                f"Prediction: {winner} to win ({probability:.1%} probability)",
                fontsize=14
            )
        else:
            plt.suptitle(
                f"Player Comparison: {player1} vs {player2}",
                fontsize=14
            )
        
        # Save plot if path provided
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            player1_short = player1.split()[-1]
            player2_short = player2.split()[-1]
            save_path = os.path.join(
                self.results_dir, 
                f"player_comparison_{player1_short}_vs_{player2_short}_{timestamp}.png"
            )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Make room for title
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved player comparison plot to {save_path}")
        
        return fig