"""
Tennis Domain Analyzer

This module provides specialized analysis tools for tennis match prediction,
focusing on domain-specific factors like surface analysis, player styles,
tournament levels, and betting-specific metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configure logger
logger = logging.getLogger(__name__)

class TennisDomainAnalyzer:
    """
    Class for analyzing tennis-specific aspects of predictions and performance.
    
    This class provides tools for analyzing tennis prediction models in the context
    of domain-specific factors such as:
    - Surface-specific performance
    - Tournament level analysis
    - Player style matchups
    - Betting-specific metrics and strategies
    
    Attributes:
        results_dir (str): Directory for storing analysis results
        player_data (DataFrame): DataFrame with player information
        tournament_data (DataFrame): DataFrame with tournament information
    """
    
    def __init__(
        self,
        results_dir: str = 'analysis_results',
        player_data: Optional[pd.DataFrame] = None,
        tournament_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the tennis domain analyzer.
        
        Args:
            results_dir: Directory for storing analysis results
            player_data: DataFrame with player information
            tournament_data: DataFrame with tournament information
        """
        self.results_dir = results_dir
        self.player_data = player_data
        self.tournament_data = tournament_data
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info("Initialized TennisDomainAnalyzer")
    
    def analyze_by_surface(
        self,
        predictions_df: pd.DataFrame,
        true_outcomes: np.ndarray,
        surface_column: str = 'surface',
        prob_column: str = 'predicted_prob'
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze prediction performance by playing surface.
        
        Args:
            predictions_df: DataFrame with predictions
            true_outcomes: Array with true outcomes
            surface_column: Column with surface information
            prob_column: Column with predicted probabilities
            
        Returns:
            Dictionary with metrics by surface
        """
        if surface_column not in predictions_df.columns:
            logger.warning(f"Surface column '{surface_column}' not found")
            return {}
        
        # Check if predicted probabilities are available
        has_proba = prob_column in predictions_df.columns
        
        # Get unique surfaces
        surfaces = predictions_df[surface_column].dropna().unique()
        
        # Calculate overall metrics
        overall_metrics = {}
        
        if has_proba:
            # Convert probabilities to binary predictions
            y_pred = (predictions_df[prob_column] > 0.5).astype(int)
            
            # Calculate overall metrics
            overall_metrics = {
                'accuracy': accuracy_score(true_outcomes, y_pred),
                'roc_auc': roc_auc_score(true_outcomes, predictions_df[prob_column]),
                'brier_score': brier_score_loss(true_outcomes, predictions_df[prob_column]),
                'log_loss': self._calculate_log_loss(true_outcomes, predictions_df[prob_column])
            }
        else:
            # No probabilities, just use binary predictions
            y_pred = predictions_df.iloc[:, 0].values  # Assuming first column has predictions
            
            # Calculate overall metrics
            overall_metrics = {
                'accuracy': accuracy_score(true_outcomes, y_pred)
            }
        
        # Analyze each surface
        surface_metrics = {'overall': overall_metrics}
        
        for surface in surfaces:
            # Get matches on this surface
            surface_mask = predictions_df[surface_column] == surface
            surface_preds = predictions_df[surface_mask]
            surface_true = true_outcomes[surface_mask]
            
            # Skip if too few samples
            if len(surface_true) < 10:
                logger.info(f"Too few samples for surface '{surface}': {len(surface_true)}")
                continue
            
            # Calculate metrics
            metrics = {}
            
            if has_proba:
                # Convert probabilities to binary predictions
                surface_y_pred = (surface_preds[prob_column] > 0.5).astype(int)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(surface_true, surface_y_pred),
                    'roc_auc': roc_auc_score(surface_true, surface_preds[prob_column]),
                    'brier_score': brier_score_loss(surface_true, surface_preds[prob_column]),
                    'log_loss': self._calculate_log_loss(surface_true, surface_preds[prob_column]),
                    'sample_count': len(surface_true)
                }
                
                # Calculate relative to overall
                for metric in ['accuracy', 'roc_auc']:
                    if metric in overall_metrics:
                        metrics[f'{metric}_vs_overall'] = metrics[metric] / overall_metrics[metric]
            else:
                # No probabilities, just use binary predictions
                surface_y_pred = surface_preds.iloc[:, 0].values
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(surface_true, surface_y_pred),
                    'sample_count': len(surface_true)
                }
                
                # Calculate relative to overall
                metrics['accuracy_vs_overall'] = metrics['accuracy'] / overall_metrics['accuracy']
            
            # Store metrics
            surface_metrics[surface] = metrics
        
        # Create visualization
        self._plot_surface_performance(surface_metrics)
        
        return surface_metrics
    
    def _calculate_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
        """
        Calculate log loss (cross-entropy).
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            eps: Small value to avoid log(0)
            
        Returns:
            Log loss value
        """
        # Clip probabilities to avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Calculate log loss
        losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(losses)
    
    def _plot_surface_performance(
        self,
        surface_metrics: Dict[str, Dict[str, float]],
        metric: str = 'accuracy'
    ) -> None:
        """
        Create visualization of performance by surface.
        
        Args:
            surface_metrics: Dictionary with metrics by surface
            metric: Metric to visualize
        """
        # Remove 'overall' from surfaces
        surfaces = [s for s in surface_metrics.keys() if s != 'overall']
        
        if not surfaces:
            logger.warning("No surface data available for plotting")
            return
        
        # Prepare data for plotting
        metric_values = [surface_metrics[s].get(metric, 0) for s in surfaces]
        sample_counts = [surface_metrics[s].get('sample_count', 0) for s in surfaces]
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot metric values
        bars = ax1.bar(surfaces, metric_values, alpha=0.7)
        
        # Add metric values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f"{metric_values[i]:.3f}",
                ha='center',
                va='bottom',
                rotation=0
            )
        
        # Add horizontal line for overall metric
        if 'overall' in surface_metrics and metric in surface_metrics['overall']:
            overall_value = surface_metrics['overall'][metric]
            ax1.axhline(y=overall_value, color='r', linestyle='--', 
                      label=f'Overall: {overall_value:.3f}')
        
        # Create second y-axis for sample counts
        ax2 = ax1.twinx()
        ax2.plot(surfaces, sample_counts, 'o-', color='green', label='Sample count')
        ax2.set_ylabel('Sample count', color='green')
        
        # Set labels and title
        ax1.set_xlabel('Surface')
        ax1.set_ylabel(f'{metric.capitalize()}', color='blue')
        plt.title(f'{metric.capitalize()} by Surface')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"performance_by_surface_{metric}.png"))
        plt.close(fig)
        
        logger.info(f"Saved surface performance plot to {os.path.join(self.results_dir, f'performance_by_surface_{metric}.png')}")
    
    def analyze_by_tournament_level(
        self,
        predictions_df: pd.DataFrame,
        true_outcomes: np.ndarray,
        tournament_level_column: str = 'tournament_level',
        prob_column: str = 'predicted_prob'
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze prediction performance by tournament level.
        
        Args:
            predictions_df: DataFrame with predictions
            true_outcomes: Array with true outcomes
            tournament_level_column: Column with tournament level information
            prob_column: Column with predicted probabilities
            
        Returns:
            Dictionary with metrics by tournament level
        """
        if tournament_level_column not in predictions_df.columns:
            logger.warning(f"Tournament level column '{tournament_level_column}' not found")
            return {}
        
        # Check if predicted probabilities are available
        has_proba = prob_column in predictions_df.columns
        
        # Get unique tournament levels
        tournament_levels = predictions_df[tournament_level_column].dropna().unique()
        
        # Calculate overall metrics
        overall_metrics = {}
        
        if has_proba:
            # Convert probabilities to binary predictions
            y_pred = (predictions_df[prob_column] > 0.5).astype(int)
            
            # Calculate overall metrics
            overall_metrics = {
                'accuracy': accuracy_score(true_outcomes, y_pred),
                'roc_auc': roc_auc_score(true_outcomes, predictions_df[prob_column]),
                'brier_score': brier_score_loss(true_outcomes, predictions_df[prob_column]),
                'log_loss': self._calculate_log_loss(true_outcomes, predictions_df[prob_column])
            }
        else:
            # No probabilities, just use binary predictions
            y_pred = predictions_df.iloc[:, 0].values  # Assuming first column has predictions
            
            # Calculate overall metrics
            overall_metrics = {
                'accuracy': accuracy_score(true_outcomes, y_pred)
            }
        
        # Analyze each tournament level
        level_metrics = {'overall': overall_metrics}
        
        for level in tournament_levels:
            # Get matches at this tournament level
            level_mask = predictions_df[tournament_level_column] == level
            level_preds = predictions_df[level_mask]
            level_true = true_outcomes[level_mask]
            
            # Skip if too few samples
            if len(level_true) < 10:
                logger.info(f"Too few samples for tournament level '{level}': {len(level_true)}")
                continue
            
            # Calculate metrics
            metrics = {}
            
            if has_proba:
                # Convert probabilities to binary predictions
                level_y_pred = (level_preds[prob_column] > 0.5).astype(int)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(level_true, level_y_pred),
                    'roc_auc': roc_auc_score(level_true, level_preds[prob_column]),
                    'brier_score': brier_score_loss(level_true, level_preds[prob_column]),
                    'log_loss': self._calculate_log_loss(level_true, level_preds[prob_column]),
                    'sample_count': len(level_true)
                }
                
                # Calculate relative to overall
                for metric in ['accuracy', 'roc_auc']:
                    if metric in overall_metrics:
                        metrics[f'{metric}_vs_overall'] = metrics[metric] / overall_metrics[metric]
            else:
                # No probabilities, just use binary predictions
                level_y_pred = level_preds.iloc[:, 0].values
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(level_true, level_y_pred),
                    'sample_count': len(level_true)
                }
                
                # Calculate relative to overall
                metrics['accuracy_vs_overall'] = metrics['accuracy'] / overall_metrics['accuracy']
            
            # Store metrics
            level_metrics[level] = metrics
        
        # Create visualization
        self._plot_tournament_level_performance(level_metrics)
        
        return level_metrics
    
    def _plot_tournament_level_performance(
        self,
        level_metrics: Dict[str, Dict[str, float]],
        metric: str = 'accuracy'
    ) -> None:
        """
        Create visualization of performance by tournament level.
        
        Args:
            level_metrics: Dictionary with metrics by tournament level
            metric: Metric to visualize
        """
        # Remove 'overall' from levels
        levels = [l for l in level_metrics.keys() if l != 'overall']
        
        if not levels:
            logger.warning("No tournament level data available for plotting")
            return
        
        # Sort levels by common hierarchy
        level_hierarchy = {
            'grand_slam': 0,
            'masters': 1,
            'atp1000': 1,
            'atp500': 2,
            'atp250': 3,
            'challenger': 4,
            'futures': 5,
            'itf': 6
        }
        
        levels.sort(key=lambda x: level_hierarchy.get(str(x).lower(), 99))
        
        # Prepare data for plotting
        metric_values = [level_metrics[l].get(metric, 0) for l in levels]
        sample_counts = [level_metrics[l].get('sample_count', 0) for l in levels]
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        # Plot metric values
        bars = ax1.bar(levels, metric_values, alpha=0.7)
        
        # Add metric values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f"{metric_values[i]:.3f}",
                ha='center',
                va='bottom',
                rotation=0
            )
        
        # Add horizontal line for overall metric
        if 'overall' in level_metrics and metric in level_metrics['overall']:
            overall_value = level_metrics['overall'][metric]
            ax1.axhline(y=overall_value, color='r', linestyle='--', 
                      label=f'Overall: {overall_value:.3f}')
        
        # Create second y-axis for sample counts
        ax2 = ax1.twinx()
        ax2.plot(levels, sample_counts, 'o-', color='green', label='Sample count')
        ax2.set_ylabel('Sample count', color='green')
        
        # Set labels and title
        ax1.set_xlabel('Tournament Level')
        ax1.set_ylabel(f'{metric.capitalize()}', color='blue')
        plt.title(f'{metric.capitalize()} by Tournament Level')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Rotate x-labels if too many levels
        plt.xticks(rotation=45 if len(levels) > 3 else 0)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"performance_by_tournament_level_{metric}.png"))
        plt.close(fig)
        
        logger.info(f"Saved tournament level performance plot to {os.path.join(self.results_dir, f'performance_by_tournament_level_{metric}.png')}")
    
    def identify_player_styles(
        self,
        player_stats_df: pd.DataFrame,
        n_clusters: int = 4,
        stats_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Identify distinct player styles using clustering on player statistics.
        
        Args:
            player_stats_df: DataFrame with player statistics
            n_clusters: Number of clusters/styles to identify
            stats_columns: Columns to use for clustering
            
        Returns:
            DataFrame with player style assignments
        """
        if not self.player_data is None and player_stats_df is None:
            player_stats_df = self.player_data
        
        if player_stats_df is None:
            logger.warning("No player statistics data provided")
            return pd.DataFrame()
        
        # Select columns for clustering
        if stats_columns is None:
            # Default columns (adapt to your data)
            potential_columns = [
                'ace_rate', 'df_rate', 'first_serve_pct', 'first_serve_win_pct',
                'second_serve_win_pct', 'bp_saved_pct', 'bp_converted_pct',
                'return_points_won_pct', 'height_cm', 'weight_kg'
            ]
            
            stats_columns = [col for col in potential_columns if col in player_stats_df.columns]
            
            if not stats_columns:
                logger.warning("No suitable statistics columns found for clustering")
                return player_stats_df
        
        # Check if all columns exist
        missing_columns = [col for col in stats_columns if col not in player_stats_df.columns]
        if missing_columns:
            logger.warning(f"Missing statistics columns: {missing_columns}")
            stats_columns = [col for col in stats_columns if col in player_stats_df.columns]
        
        # Create a copy to avoid modifying the original
        styled_df = player_stats_df.copy()
        
        # Extract player IDs
        player_ids = styled_df.index if styled_df.index.name else styled_df.iloc[:, 0]
        
        # Extract statistics for clustering
        X = styled_df[stats_columns].copy()
        
        # Remove rows with missing values
        valid_mask = ~X.isna().any(axis=1)
        X_valid = X[valid_mask]
        
        if len(X_valid) < n_clusters:
            logger.warning(f"Too few valid samples ({len(X_valid)}) for {n_clusters} clusters")
            return styled_df
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster assignments to valid rows
        valid_indices = X_valid.index
        styled_df.loc[valid_indices, 'style_cluster'] = clusters
        
        # Analyze cluster characteristics
        cluster_profiles = self._analyze_cluster_profiles(X_valid, clusters, stats_columns)
        
        # Assign style labels based on profiles
        style_labels = self._assign_style_labels(cluster_profiles)
        style_map = {i: label for i, label in enumerate(style_labels)}
        
        # Map cluster numbers to style labels
        styled_df['player_style'] = styled_df['style_cluster'].map(style_map)
        
        # Visualize clusters
        self._plot_player_styles(X_valid, clusters, stats_columns, style_labels)
        
        logger.info(f"Identified {n_clusters} distinct player styles")
        return styled_df
    
    def _analyze_cluster_profiles(
        self,
        X: pd.DataFrame,
        clusters: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Analyze the characteristics of each player style cluster.
        
        Args:
            X: DataFrame with player statistics
            clusters: Cluster assignments
            feature_names: Names of statistics features
            
        Returns:
            DataFrame with cluster profiles
        """
        # Calculate cluster centers and other statistics
        profiles = []
        
        for cluster_id in range(max(clusters) + 1):
            # Get players in this cluster
            cluster_mask = clusters == cluster_id
            cluster_data = X[cluster_mask]
            
            # Skip empty clusters
            if len(cluster_data) == 0:
                continue
            
            # Calculate statistics
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X) * 100
            }
            
            # Add feature means
            for i, feature in enumerate(feature_names):
                profile[f'{feature}_mean'] = cluster_data[feature].mean()
                
                # Calculate relative position (z-score from overall mean)
                overall_mean = X[feature].mean()
                overall_std = X[feature].std()
                if overall_std > 0:
                    profile[f'{feature}_z'] = (profile[f'{feature}_mean'] - overall_mean) / overall_std
                else:
                    profile[f'{feature}_z'] = 0
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def _assign_style_labels(self, cluster_profiles: pd.DataFrame) -> List[str]:
        """
        Assign tennis style labels to clusters based on their characteristics.
        
        Args:
            cluster_profiles: DataFrame with cluster statistics
            
        Returns:
            List of style labels for each cluster
        """
        labels = []
        
        # Get z-score columns
        z_columns = [col for col in cluster_profiles.columns if col.endswith('_z')]
        
        for _, profile in cluster_profiles.iterrows():
            # Get dominant attributes (highest z-scores)
            z_scores = {col.replace('_z', ''): profile[col] for col in z_columns}
            sorted_attributes = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Identify style based on top attributes
            if 'ace_rate_z' in z_columns and profile['ace_rate_z'] > 1.0:
                if 'first_serve_win_pct_z' in z_columns and profile['first_serve_win_pct_z'] > 0.5:
                    style = "Big Server"
                else:
                    style = "Aggressive Baseliner"
            elif 'return_points_won_pct_z' in z_columns and profile['return_points_won_pct_z'] > 0.7:
                style = "Counter-Puncher"
            elif 'bp_converted_pct_z' in z_columns and profile['bp_converted_pct_z'] > 0.5:
                style = "All-Court Player"
            else:
                # Default label based on top attributes
                top_attribute = sorted_attributes[0][0] if sorted_attributes else "Unknown"
                style = f"Style-{profile['cluster_id']} ({top_attribute})"
            
            labels.append(style)
        
        return labels
    
    def _plot_player_styles(
        self,
        X: pd.DataFrame,
        clusters: np.ndarray,
        feature_names: List[str],
        style_labels: List[str]
    ) -> None:
        """
        Create visualization of player style clusters.
        
        Args:
            X: DataFrame with player statistics
            clusters: Cluster assignments
            feature_names: Names of statistics features
            style_labels: Labels for each cluster
        """
        # Create a color map for clusters
        colors = plt.cm.tab10(np.linspace(0, 1, max(clusters) + 1))
        
        # Select two most important features for visualization
        if len(feature_names) >= 2:
            feature_x = feature_names[0]
            feature_y = feature_names[1]
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            
            for cluster_id in range(max(clusters) + 1):
                cluster_mask = clusters == cluster_id
                plt.scatter(
                    X.loc[cluster_mask, feature_x],
                    X.loc[cluster_mask, feature_y],
                    c=[colors[cluster_id]],
                    label=f"{style_labels[cluster_id]} (n={np.sum(cluster_mask)})",
                    alpha=0.7
                )
            
            plt.xlabel(feature_x)
            plt.ylabel(feature_y)
            plt.title('Player Styles Clustering')
            plt.legend()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "player_styles_scatter.png"))
            plt.close()
            
            logger.info(f"Saved player styles scatter plot to {os.path.join(self.results_dir, 'player_styles_scatter.png')}")
        
        # Create radar chart for each style
        self._plot_style_radar_chart(X, clusters, feature_names, style_labels)
    
    def _plot_style_radar_chart(
        self,
        X: pd.DataFrame,
        clusters: np.ndarray,
        feature_names: List[str],
        style_labels: List[str]
    ) -> None:
        """
        Create radar charts to visualize player style characteristics.
        
        Args:
            X: DataFrame with player statistics
            clusters: Cluster assignments
            feature_names: Names of statistics features
            style_labels: Labels for each cluster
        """
        # Skip if fewer than 3 features
        if len(feature_names) < 3:
            return
        
        # Calculate cluster means
        cluster_means = []
        for cluster_id in range(max(clusters) + 1):
            cluster_mask = clusters == cluster_id
            if np.sum(cluster_mask) > 0:
                means = X[cluster_mask].mean().values
                cluster_means.append(means)
        
        # Calculate feature ranges for normalization
        feature_mins = X.min().values
        feature_maxs = X.max().values
        feature_ranges = feature_maxs - feature_mins
        
        # Normalize cluster means
        normalized_means = []
        for means in cluster_means:
            normalized = np.zeros_like(means)
            for i, (mean, min_val, range_val) in enumerate(zip(means, feature_mins, feature_ranges)):
                if range_val > 0:
                    normalized[i] = (mean - min_val) / range_val
                else:
                    normalized[i] = 0.5
            normalized_means.append(normalized)
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(feature_names), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, (means, label) in enumerate(zip(normalized_means, style_labels)):
            # Close the loop for this cluster
            values = means.tolist()
            values += values[:1]
            
            # Plot cluster
            ax.plot(angles, values, linewidth=2, label=label)
            ax.fill(angles, values, alpha=0.1)
        
        # Set feature labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)
        
        # Configure chart
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.set_ylim(0, 1)
        
        plt.title('Player Style Profiles')
        plt.legend(loc='upper right')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "player_styles_radar.png"))
        plt.close(fig)
        
        logger.info(f"Saved player styles radar chart to {os.path.join(self.results_dir, 'player_styles_radar.png')}")
    
    def analyze_style_matchups(
        self,
        matches_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        true_outcomes: np.ndarray,
        player1_id_column: str = 'player1_id',
        player2_id_column: str = 'player2_id',
        player1_style_column: str = 'player1_style',
        player2_style_column: str = 'player2_style',
        prob_column: str = 'predicted_prob'
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Analyze prediction performance by player style matchups.
        
        Args:
            matches_df: DataFrame with match information
            predictions_df: DataFrame with predictions
            true_outcomes: Array with true outcomes
            player1_id_column: Column with player 1 ID
            player2_id_column: Column with player 2 ID
            player1_style_column: Column with player 1 style
            player2_style_column: Column with player 2 style
            prob_column: Column with predicted probabilities
            
        Returns:
            Dictionary with metrics by style matchup
        """
        # Check if style columns exist
        if player1_style_column not in matches_df.columns or player2_style_column not in matches_df.columns:
            logger.warning(f"Player style columns not found")
            return {}
        
        # Merge match data with predictions
        if 'match_id' in matches_df.columns and 'match_id' in predictions_df.columns:
            merged_df = pd.merge(matches_df, predictions_df, on='match_id')
        else:
            # Assume same order
            if len(matches_df) != len(predictions_df):
                logger.warning("Match data and predictions have different lengths")
                return {}
            
            merged_df = matches_df.copy()
            if prob_column in predictions_df.columns:
                merged_df[prob_column] = predictions_df[prob_column].values
        
        # Check if predicted probabilities are available
        has_proba = prob_column in merged_df.columns
        
        # Get unique player styles
        styles1 = matches_df[player1_style_column].dropna().unique()
        styles2 = matches_df[player2_style_column].dropna().unique()
        styles = np.unique(np.concatenate([styles1, styles2]))
        
        # Calculate overall metrics
        overall_metrics = {}
        
        if has_proba:
            # Convert probabilities to binary predictions
            y_pred = (merged_df[prob_column] > 0.5).astype(int)
            
            # Calculate overall metrics
            overall_metrics = {
                'accuracy': accuracy_score(true_outcomes, y_pred),
                'roc_auc': roc_auc_score(true_outcomes, merged_df[prob_column]),
                'brier_score': brier_score_loss(true_outcomes, merged_df[prob_column])
            }
        else:
            # No probabilities, just use binary predictions
            y_pred = merged_df.iloc[:, -1].values  # Assuming last column has predictions
            
            # Calculate overall metrics
            overall_metrics = {
                'accuracy': accuracy_score(true_outcomes, y_pred)
            }
        
        # Analyze each style matchup
        matchup_metrics = {'overall': overall_metrics}
        
        for style1 in styles:
            for style2 in styles:
                # Skip same-style matchups if too many styles
                if len(styles) > 4 and style1 == style2:
                    continue
                
                # Get matches with this style matchup
                matchup_mask = (
                    (merged_df[player1_style_column] == style1) & 
                    (merged_df[player2_style_column] == style2)
                )
                
                matchup_data = merged_df[matchup_mask]
                matchup_true = true_outcomes[matchup_mask]
                
                # Skip if too few samples
                if len(matchup_true) < 10:
                    logger.info(f"Too few samples for matchup '{style1}' vs '{style2}': {len(matchup_true)}")
                    continue
                
                # Calculate metrics
                metrics = {}
                
                if has_proba:
                    # Convert probabilities to binary predictions
                    matchup_y_pred = (matchup_data[prob_column] > 0.5).astype(int)
                    
                    # Calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(matchup_true, matchup_y_pred),
                        'sample_count': len(matchup_true)
                    }
                    
                    if len(np.unique(matchup_true)) > 1:  # Ensure both classes are present
                        metrics['roc_auc'] = roc_auc_score(matchup_true, matchup_data[prob_column])
                        metrics['brier_score'] = brier_score_loss(matchup_true, matchup_data[prob_column])
                    
                    # Calculate relative to overall
                    for metric in ['accuracy', 'roc_auc']:
                        if metric in metrics and metric in overall_metrics:
                            metrics[f'{metric}_vs_overall'] = metrics[metric] / overall_metrics[metric]
                else:
                    # No probabilities, just use binary predictions
                    matchup_y_pred = matchup_data.iloc[:, -1].values
                    
                    # Calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(matchup_true, matchup_y_pred),
                        'sample_count': len(matchup_true)
                    }
                    
                    # Calculate relative to overall
                    metrics['accuracy_vs_overall'] = metrics['accuracy'] / overall_metrics['accuracy']
                
                # Store metrics
                matchup_metrics[(style1, style2)] = metrics
        
        # Create visualization
        self._plot_style_matchup_performance(matchup_metrics, styles)
        
        return matchup_metrics
    
    def _plot_style_matchup_performance(
        self,
        matchup_metrics: Dict[Tuple[str, str], Dict[str, float]],
        styles: List[str],
        metric: str = 'accuracy'
    ) -> None:
        """
        Create heatmap visualization of performance by style matchup.
        
        Args:
            matchup_metrics: Dictionary with metrics by style matchup
            styles: List of player styles
            metric: Metric to visualize
        """
        # Remove 'overall' from matchups
        style_matchups = {k: v for k, v in matchup_metrics.items() if k != 'overall'}
        
        if not style_matchups:
            logger.warning("No style matchup data available for plotting")
            return
        
        # Prepare data for heatmap
        heatmap_data = np.zeros((len(styles), len(styles)))
        mask = np.ones((len(styles), len(styles)), dtype=bool)
        
        for (style1, style2), metrics in style_matchups.items():
            if style1 in styles and style2 in styles and metric in metrics:
                i = np.where(styles == style1)[0][0]
                j = np.where(styles == style2)[0][0]
                heatmap_data[i, j] = metrics[metric]
                mask[i, j] = False
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            xticklabels=styles,
            yticklabels=styles,
            mask=mask
        )
        
        plt.title(f'{metric.capitalize()} by Player Style Matchup')
        plt.xlabel('Player 2 Style')
        plt.ylabel('Player 1 Style')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"performance_by_style_matchup_{metric}.png"))
        plt.close()
        
        logger.info(f"Saved style matchup performance plot to {os.path.join(self.results_dir, f'performance_by_style_matchup_{metric}.png')}")
    
    def calculate_betting_metrics(
        self,
        predictions_df: pd.DataFrame,
        true_outcomes: np.ndarray,
        prob_column: str = 'predicted_prob',
        odds_column: Optional[str] = None,
        odds_type: str = 'decimal',
        min_prob_threshold: float = 0.0,
        max_prob_threshold: float = 1.0,
        kelly_fraction: float = 0.25,
        initial_bankroll: float = 1000.0
    ) -> Dict[str, float]:
        """
        Calculate betting-specific metrics for evaluating prediction performance.
        
        Args:
            predictions_df: DataFrame with predictions
            true_outcomes: Array with true outcomes
            prob_column: Column with predicted probabilities
            odds_column: Column with betting odds (if available)
            odds_type: Type of odds ('decimal', 'american', 'fractional')
            min_prob_threshold: Minimum probability threshold for placing bets
            max_prob_threshold: Maximum probability threshold for placing bets
            kelly_fraction: Fraction of Kelly criterion to use
            initial_bankroll: Initial bankroll for simulation
            
        Returns:
            Dictionary with betting metrics
        """
        # Check if predicted probabilities are available
        if prob_column not in predictions_df.columns:
            logger.warning(f"Probability column '{prob_column}' not found")
            return {}
        
        # Check if odds are available
        has_odds = odds_column is not None and odds_column in predictions_df.columns
        
        # Create a copy of predictions
        betting_df = predictions_df.copy()
        
        # Convert odds to decimal format if available
        if has_odds:
            if odds_type == 'american':
                # Convert American odds to decimal
                american_odds = betting_df[odds_column].values
                decimal_odds = np.zeros_like(american_odds, dtype=float)
                
                # Positive American odds
                positive_mask = american_odds > 0
                decimal_odds[positive_mask] = (american_odds[positive_mask] / 100) + 1
                
                # Negative American odds
                negative_mask = american_odds < 0
                decimal_odds[negative_mask] = (100 / abs(american_odds[negative_mask])) + 1
                
                betting_df['decimal_odds'] = decimal_odds
            
            elif odds_type == 'fractional':
                # Convert fractional odds to decimal
                fractional_odds = betting_df[odds_column].astype(str)
                decimal_odds = np.zeros(len(fractional_odds))
                
                for i, odds_str in enumerate(fractional_odds):
                    if '/' in odds_str:
                        try:
                            numerator, denominator = odds_str.split('/')
                            decimal_odds[i] = (float(numerator) / float(denominator)) + 1
                        except ValueError:
                            decimal_odds[i] = 1.0
                    else:
                        decimal_odds[i] = 1.0
                
                betting_df['decimal_odds'] = decimal_odds
            
            else:
                # Already in decimal format
                betting_df['decimal_odds'] = betting_df[odds_column]
        else:
            # No odds available, use fair odds based on predicted probabilities
            betting_df['decimal_odds'] = 1 / betting_df[prob_column]
        
        # Apply probability thresholds
        valid_bets = (
            (betting_df[prob_column] >= min_prob_threshold) &
            (betting_df[prob_column] <= max_prob_threshold)
        )
        
        # Simulated betting results
        bet_outcomes = []
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        
        for i, valid in enumerate(valid_bets):
            if not valid:
                # Skip invalid bets
                bet_outcomes.append({
                    'bet_placed': False,
                    'amount': 0,
                    'outcome': None,
                    'profit': 0,
                    'bankroll': bankroll
                })
                bankroll_history.append(bankroll)
                continue
            
            # Get predicted probability and odds
            predicted_prob = betting_df[prob_column].iloc[i]
            odds = betting_df['decimal_odds'].iloc[i]
            
            # Calculate edge
            fair_odds = 1 / predicted_prob
            edge = (odds / fair_odds) - 1
            
            # Skip negative edge bets
            if edge <= 0:
                bet_outcomes.append({
                    'bet_placed': False,
                    'amount': 0,
                    'outcome': None,
                    'profit': 0,
                    'bankroll': bankroll,
                    'edge': edge
                })
                bankroll_history.append(bankroll)
                continue
            
            # Calculate Kelly stake
            kelly_stake = (odds * predicted_prob - 1) / (odds - 1)
            kelly_stake = max(0, min(kelly_stake, 1))  # Clip to [0, 1]
            
            # Apply Kelly fraction
            kelly_stake *= kelly_fraction
            
            # Calculate bet amount
            bet_amount = bankroll * kelly_stake
            
            # Determine outcome
            won = true_outcomes[i] == 1
            
            # Calculate profit
            profit = bet_amount * (odds - 1) if won else -bet_amount
            
            # Update bankroll
            bankroll += profit
            
            # Record bet details
            bet_outcomes.append({
                'bet_placed': True,
                'amount': bet_amount,
                'outcome': won,
                'profit': profit,
                'bankroll': bankroll,
                'edge': edge,
                'kelly_stake': kelly_stake
            })
            
            bankroll_history.append(bankroll)
        
        # Calculate betting metrics
        bet_outcomes_df = pd.DataFrame(bet_outcomes)
        placed_bets = bet_outcomes_df[bet_outcomes_df['bet_placed']]
        
        if len(placed_bets) == 0:
            logger.warning("No valid bets placed")
            return {
                'bets_placed': 0,
                'final_bankroll': initial_bankroll,
                'roi': 0.0
            }
        
        # Calculate metrics
        total_bets = len(placed_bets)
        winning_bets = placed_bets[placed_bets['outcome'] == True].shape[0]
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        total_staked = placed_bets['amount'].sum()
        total_profit = placed_bets['profit'].sum()
        roi = total_profit / total_staked if total_staked > 0 else 0
        
        final_bankroll = bankroll_history[-1]
        bankroll_growth = (final_bankroll / initial_bankroll) - 1
        
        # Calculate max drawdown
        peak = initial_bankroll
        drawdowns = []
        
        for balance in bankroll_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        
        # Calculate average edge
        avg_edge = placed_bets['edge'].mean()
        
        # Store metrics
        metrics = {
            'bets_placed': total_bets,
            'win_rate': win_rate,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': roi,
            'initial_bankroll': initial_bankroll,
            'final_bankroll': final_bankroll,
            'bankroll_growth': bankroll_growth,
            'max_drawdown': max_drawdown,
            'avg_edge': avg_edge
        }
        
        # Create visualization
        self._plot_betting_simulation(bankroll_history, metrics)
        
        return metrics
    
    def _plot_betting_simulation(
        self,
        bankroll_history: List[float],
        metrics: Dict[str, float]
    ) -> None:
        """
        Create visualization of betting simulation results.
        
        Args:
            bankroll_history: List of bankroll values over time
            metrics: Dictionary with betting metrics
        """
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot bankroll history
        plt.plot(bankroll_history, 'b-')
        
        # Add initial bankroll line
        plt.axhline(y=metrics['initial_bankroll'], color='r', linestyle='--', 
                    label=f"Initial bankroll: ${metrics['initial_bankroll']:.2f}")
        
        # Add metrics to plot
        plt.title('Betting Simulation Results')
        plt.xlabel('Bet Number')
        plt.ylabel('Bankroll ($)')
        
        # Add text box with metrics
        textstr = '\n'.join([
            f"Bets placed: {metrics['bets_placed']}",
            f"Win rate: {metrics['win_rate']:.2%}",
            f"ROI: {metrics['roi']:.2%}",
            f"Final bankroll: ${metrics['final_bankroll']:.2f}",
            f"Growth: {metrics['bankroll_growth']:.2%}",
            f"Max drawdown: {metrics['max_drawdown']:.2%}"
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "betting_simulation.png"))
        plt.close()
        
        logger.info(f"Saved betting simulation plot to {os.path.join(self.results_dir, 'betting_simulation.png')}")
    
    def analyze_probability_calibration(
        self,
        predictions_df: pd.DataFrame,
        true_outcomes: np.ndarray,
        prob_column: str = 'predicted_prob',
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze the calibration of predicted probabilities.
        
        Args:
            predictions_df: DataFrame with predictions
            true_outcomes: Array with true outcomes
            prob_column: Column with predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics and bin data
        """
        # Check if predicted probabilities are available
        if prob_column not in predictions_df.columns:
            logger.warning(f"Probability column '{prob_column}' not found")
            return {}
        
        # Get predicted probabilities
        y_pred = predictions_df[prob_column].values
        
        # Calculate calibration curve
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.searchsorted(bin_edges, y_pred)
        bin_indices = np.clip(bin_indices, 1, n_bins) - 1  # Clip and shift to get correct bin index
        
        # Initialize bin statistics
        bin_stats = []
        
        for i in range(n_bins):
            # Get samples in this bin
            bin_mask = bin_indices == i
            bin_count = np.sum(bin_mask)
            
            if bin_count > 0:
                # Calculate statistics
                bin_pred_prob = np.mean(y_pred[bin_mask])
                bin_true_prob = np.mean(true_outcomes[bin_mask])
                bin_error = bin_true_prob - bin_pred_prob
                
                bin_stats.append({
                    'bin_index': i,
                    'bin_start': bin_edges[i],
                    'bin_end': bin_edges[i+1],
                    'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                    'count': bin_count,
                    'pred_prob': bin_pred_prob,
                    'true_prob': bin_true_prob,
                    'error': bin_error,
                    'abs_error': abs(bin_error)
                })
        
        # Convert to DataFrame
        bins_df = pd.DataFrame(bin_stats)
        
        # Calculate overall calibration metrics
        if len(bins_df) > 0:
            # Weighted mean absolute error
            total_samples = np.sum(bins_df['count'])
            wmae = np.sum(bins_df['abs_error'] * bins_df['count']) / total_samples
            
            # Brier score
            brier = brier_score_loss(true_outcomes, y_pred)
            
            # Calculate expected calibration error (ECE)
            ece = np.sum(bins_df['abs_error'] * bins_df['count']) / total_samples
            
            # Calculate maximum calibration error (MCE)
            mce = np.max(bins_df['abs_error']) if len(bins_df) > 0 else 0
            
            calibration_metrics = {
                'brier_score': brier,
                'expected_calibration_error': ece,
                'maximum_calibration_error': mce,
                'weighted_mean_absolute_error': wmae,
                'bin_data': bins_df.to_dict(orient='records')
            }
        else:
            calibration_metrics = {
                'bin_data': []
            }
        
        # Create visualization
        self._plot_calibration_curve(bins_df)
        
        return calibration_metrics
    
    def _plot_calibration_curve(self, bins_df: pd.DataFrame) -> None:
        """
        Create visualization of probability calibration.
        
        Args:
            bins_df: DataFrame with calibration bin data
        """
        if len(bins_df) == 0:
            logger.warning("No calibration data available for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Plot calibration curve
        ax.plot(bins_df['pred_prob'], bins_df['true_prob'], 'o-', label='Model')
        
        # Plot histogram of predicted probabilities
        ax2 = ax.twinx()
        ax2.hist(bins_df['pred_prob'], weights=bins_df['count'], 
               bins=len(bins_df), alpha=0.3, label='Predictions')
        ax2.set_ylabel('Count')
        ax2.set_ylim(0, ax2.get_ylim()[1] * 1.2)  # Add some space at the top
        
        # Set labels and title
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('True probability (fraction of positives)')
        ax.set_title('Calibration Curve')
        
        # Add metrics to plot
        ece = np.sum(bins_df['abs_error'] * bins_df['count']) / np.sum(bins_df['count'])
        mce = np.max(bins_df['abs_error'])
        
        textstr = '\n'.join([
            f"Expected Calibration Error: {ece:.3f}",
            f"Maximum Calibration Error: {mce:.3f}"
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        
        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "probability_calibration.png"))
        plt.close(fig)
        
        logger.info(f"Saved probability calibration plot to {os.path.join(self.results_dir, 'probability_calibration.png')}")
    
    def analyze_probability_thresholds(
        self,
        predictions_df: pd.DataFrame,
        true_outcomes: np.ndarray,
        prob_column: str = 'predicted_prob',
        odds_column: Optional[str] = None,
        n_thresholds: int = 20
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Analyze the effect of different probability thresholds on prediction metrics.
        
        Args:
            predictions_df: DataFrame with predictions
            true_outcomes: Array with true outcomes
            prob_column: Column with predicted probabilities
            odds_column: Column with betting odds (if available)
            n_thresholds: Number of threshold values to evaluate
            
        Returns:
            Dictionary with metrics for each threshold
        """
        # Check if predicted probabilities are available
        if prob_column not in predictions_df.columns:
            logger.warning(f"Probability column '{prob_column}' not found")
            return {}
        
        # Get predicted probabilities
        y_pred = predictions_df[prob_column].values
        
        # Check if odds are available
        has_odds = odds_column is not None and odds_column in predictions_df.columns
        
        # Create threshold values
        thresholds = np.linspace(0.05, 0.95, n_thresholds)
        
        # Analyze each threshold
        threshold_metrics = []
        
        for threshold in thresholds:
            # Make binary predictions
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calculate classification metrics
            accuracy = accuracy_score(true_outcomes, y_pred_binary)
            
            # Calculate true positives, false positives, etc.
            tp = np.sum((y_pred_binary == 1) & (true_outcomes == 1))
            fp = np.sum((y_pred_binary == 1) & (true_outcomes == 0))
            tn = np.sum((y_pred_binary == 0) & (true_outcomes == 0))
            fn = np.sum((y_pred_binary == 0) & (true_outcomes == 1))
            
            # Calculate additional metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store metrics
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'predictions_made': int(tp + fp),
                'fraction_predicted': (tp + fp) / len(true_outcomes)
            }
            
            # Calculate betting metrics if odds are available
            if has_odds:
                # Get decimal odds for positive predictions
                positive_mask = y_pred >= threshold
                if np.any(positive_mask):
                    odds = predictions_df[odds_column].values[positive_mask]
                    positive_outcomes = true_outcomes[positive_mask]
                    
                    # Calculate profit
                    winnings = np.sum(odds[positive_outcomes == 1]) - np.sum(positive_outcomes == 1)
                    losses = np.sum(positive_outcomes == 0)
                    profit = winnings - losses
                    
                    # Calculate ROI
                    bets_placed = len(positive_outcomes)
                    roi = profit / bets_placed if bets_placed > 0 else 0
                    
                    # Add to metrics
                    metrics.update({
                        'bets_placed': bets_placed,
                        'profit': float(profit),
                        'roi': float(roi)
                    })
            
            threshold_metrics.append(metrics)
        
        # Create visualization
        self._plot_threshold_analysis(threshold_metrics, has_odds)
        
        return {'thresholds': threshold_metrics}
    
    def _plot_threshold_analysis(
        self,
        threshold_metrics: List[Dict[str, float]],
        plot_betting: bool = False
    ) -> None:
        """
        Create visualization of threshold analysis.
        
        Args:
            threshold_metrics: List of metrics for each threshold
            plot_betting: Whether to plot betting metrics
        """
        if not threshold_metrics:
            logger.warning("No threshold data available for plotting")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(threshold_metrics)
        
        # Create figure for classification metrics
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot classification metrics
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics_to_plot:
            if metric in df.columns:
                ax1.plot(df['threshold'], df[metric], '-o', label=metric.capitalize())
        
        # Set labels and title
        ax1.set_xlabel('Probability Threshold')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Classification Metrics by Probability Threshold')
        
        # Add legend
        ax1.legend(loc='upper left')
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        
        # Create second y-axis for predictions made
        ax2 = ax1.twinx()
        ax2.plot(df['threshold'], df['fraction_predicted'], '--', color='gray', 
               label='Fraction of predictions made')
        ax2.set_ylabel('Fraction of Predictions Made')
        
        # Add legend for second axis
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "threshold_analysis_classification.png"))
        plt.close(fig)
        
        logger.info(f"Saved threshold analysis (classification) plot to {os.path.join(self.results_dir, 'threshold_analysis_classification.png')}")
        
        # Create betting metrics plot if available
        if plot_betting and 'roi' in df.columns:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot ROI
            ax1.plot(df['threshold'], df['roi'], '-o', color='green', label='ROI')
            
            # Set labels and title
            ax1.set_xlabel('Probability Threshold')
            ax1.set_ylabel('ROI')
            ax1.set_title('Betting Metrics by Probability Threshold')
            
            # Create second y-axis for bets placed
            ax2 = ax1.twinx()
            ax2.plot(df['threshold'], df['bets_placed'], '--', color='blue', 
                   label='Bets placed')
            ax2.set_ylabel('Bets Placed')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "threshold_analysis_betting.png"))
            plt.close(fig)
            
            logger.info(f"Saved threshold analysis (betting) plot to {os.path.join(self.results_dir, 'threshold_analysis_betting.png')}")
    
    def generate_analysis_report(
        self,
        surface_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        tournament_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        style_metrics: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
        betting_metrics: Optional[Dict[str, float]] = None,
        calibration_metrics: Optional[Dict[str, Any]] = None,
        threshold_analysis: Optional[Dict[str, List[Dict[str, float]]]] = None
    ) -> str:
        """
        Generate a comprehensive analysis report in markdown format.
        
        Args:
            surface_metrics: Metrics by surface
            tournament_metrics: Metrics by tournament level
            style_metrics: Metrics by player style matchup
            betting_metrics: Betting simulation metrics
            calibration_metrics: Probability calibration metrics
            threshold_analysis: Analysis of different probability thresholds
            
        Returns:
            Markdown formatted report
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize report sections
        report_sections = [
            f"# Tennis Prediction Analysis Report\n\n",
            f"Generated on: {timestamp}\n\n"
        ]
        
        # Add surface analysis section
        if surface_metrics:
            surface_section = [
                "## Performance by Surface\n\n",
                "This section analyzes prediction performance across different playing surfaces.\n\n",
                "### Overall Metrics\n\n"
            ]
            
            if 'overall' in surface_metrics:
                overall = surface_metrics['overall']
                metrics_table = "| Metric | Value |\n|--------|-------|\n"
                
                for metric, value in overall.items():
                    if isinstance(value, (int, float)):
                        metrics_table += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
                
                surface_section.append(metrics_table + "\n\n")
            
            surface_section.append("### Surface-Specific Performance\n\n")
            
            surfaces = [s for s in surface_metrics.keys() if s != 'overall']
            if surfaces:
                surfaces_table = "| Surface | Accuracy | ROC AUC | Samples | vs. Overall |\n|---------|----------|---------|---------|------------|\n"
                
                for surface in surfaces:
                    metrics = surface_metrics[surface]
                    accuracy = metrics.get('accuracy', float('nan'))
                    roc_auc = metrics.get('roc_auc', float('nan'))
                    samples = metrics.get('sample_count', 0)
                    vs_overall = metrics.get('accuracy_vs_overall', float('nan'))
                    
                    surfaces_table += f"| {surface} | {accuracy:.4f} | {roc_auc:.4f} | {samples} | {vs_overall:.4f} |\n"
                
                surface_section.append(surfaces_table + "\n\n")
                surface_section.append(f"![Performance by Surface]({os.path.join(self.results_dir, 'performance_by_surface_accuracy.png')})\n\n")
            
            report_sections.append("".join(surface_section))
        
        # Add tournament level analysis section
        if tournament_metrics:
            tournament_section = [
                "## Performance by Tournament Level\n\n",
                "This section analyzes prediction performance across different tournament levels.\n\n"
            ]
            
            tournament_levels = [t for t in tournament_metrics.keys() if t != 'overall']
            if tournament_levels:
                levels_table = "| Tournament Level | Accuracy | ROC AUC | Samples | vs. Overall |\n|-----------------|----------|---------|---------|------------|\n"
                
                for level in tournament_levels:
                    metrics = tournament_metrics[level]
                    accuracy = metrics.get('accuracy', float('nan'))
                    roc_auc = metrics.get('roc_auc', float('nan'))
                    samples = metrics.get('sample_count', 0)
                    vs_overall = metrics.get('accuracy_vs_overall', float('nan'))
                    
                    levels_table += f"| {level} | {accuracy:.4f} | {roc_auc:.4f} | {samples} | {vs_overall:.4f} |\n"
                
                tournament_section.append(levels_table + "\n\n")
                tournament_section.append(f"![Performance by Tournament Level]({os.path.join(self.results_dir, 'performance_by_tournament_level_accuracy.png')})\n\n")
            
            report_sections.append("".join(tournament_section))
        
        # Add player style analysis section
        if style_metrics:
            style_section = [
                "## Performance by Player Style Matchup\n\n",
                "This section analyzes prediction performance across different player style matchups.\n\n"
            ]
            
            style_matchups = {k: v for k, v in style_metrics.items() if k != 'overall'}
            if style_matchups:
                matchup_table = "| Style 1 | Style 2 | Accuracy | Samples | vs. Overall |\n|---------|---------|----------|---------|------------|\n"
                
                for (style1, style2), metrics in style_matchups.items():
                    accuracy = metrics.get('accuracy', float('nan'))
                    samples = metrics.get('sample_count', 0)
                    vs_overall = metrics.get('accuracy_vs_overall', float('nan'))
                    
                    matchup_table += f"| {style1} | {style2} | {accuracy:.4f} | {samples} | {vs_overall:.4f} |\n"
                
                style_section.append(matchup_table + "\n\n")
                style_section.append(f"![Performance by Style Matchup]({os.path.join(self.results_dir, 'performance_by_style_matchup_accuracy.png')})\n\n")
            
            report_sections.append("".join(style_section))
        
        # Add betting analysis section
        if betting_metrics:
            betting_section = [
                "## Betting Performance Analysis\n\n",
                "This section analyzes the simulated betting performance using the prediction model.\n\n",
                "### Betting Metrics\n\n"
            ]
            
            metrics_table = "| Metric | Value |\n|--------|-------|\n"
            
            metrics_to_include = [
                ('bets_placed', 'Bets Placed'),
                ('win_rate', 'Win Rate'),
                ('roi', 'Return on Investment'),
                ('total_profit', 'Total Profit'),
                ('initial_bankroll', 'Initial Bankroll'),
                ('final_bankroll', 'Final Bankroll'),
                ('bankroll_growth', 'Bankroll Growth'),
                ('max_drawdown', 'Maximum Drawdown'),
                ('avg_edge', 'Average Edge')
            ]
            
            for key, label in metrics_to_include:
                if key in betting_metrics:
                    value = betting_metrics[key]
                    if key in ['win_rate', 'roi', 'bankroll_growth', 'max_drawdown']:
                        metrics_table += f"| {label} | {value:.2%} |\n"
                    elif key in ['total_profit', 'initial_bankroll', 'final_bankroll']:
                        metrics_table += f"| {label} | ${value:.2f} |\n"
                    else:
                        metrics_table += f"| {label} | {value} |\n"
            
            betting_section.append(metrics_table + "\n\n")
            betting_section.append(f"![Betting Simulation]({os.path.join(self.results_dir, 'betting_simulation.png')})\n\n")
            
            # Add threshold analysis if available
            if threshold_analysis and 'thresholds' in threshold_analysis:
                betting_section.append("### Threshold Analysis\n\n")
                betting_section.append("Analysis of how different probability thresholds affect betting performance:\n\n")
                betting_section.append(f"![Threshold Analysis (Betting)]({os.path.join(self.results_dir, 'threshold_analysis_betting.png')})\n\n")
            
            report_sections.append("".join(betting_section))
        
        # Add calibration analysis section
        if calibration_metrics:
            calibration_section = [
                "## Probability Calibration Analysis\n\n",
                "This section analyzes how well calibrated the predicted probabilities are.\n\n",
                "### Calibration Metrics\n\n"
            ]
            
            metrics_table = "| Metric | Value |\n|--------|-------|\n"
            
            metrics_to_include = [
                ('brier_score', 'Brier Score'),
                ('expected_calibration_error', 'Expected Calibration Error (ECE)'),
                ('maximum_calibration_error', 'Maximum Calibration Error (MCE)'),
                ('weighted_mean_absolute_error', 'Weighted Mean Absolute Error')
            ]
            
            for key, label in metrics_to_include:
                if key in calibration_metrics:
                    value = calibration_metrics[key]
                    metrics_table += f"| {label} | {value:.4f} |\n"
            
            calibration_section.append(metrics_table + "\n\n")
            calibration_section.append(f"![Probability Calibration]({os.path.join(self.results_dir, 'probability_calibration.png')})\n\n")
            
            report_sections.append("".join(calibration_section))
        
        # Add threshold analysis section
        if threshold_analysis and 'thresholds' in threshold_analysis and not betting_metrics:
            threshold_section = [
                "## Probability Threshold Analysis\n\n",
                "This section analyzes the effect of different probability thresholds on prediction performance.\n\n"
            ]
            
            threshold_section.append(f"![Threshold Analysis]({os.path.join(self.results_dir, 'threshold_analysis_classification.png')})\n\n")
            
            report_sections.append("".join(threshold_section))
        
        # Combine all sections
        report = "\n".join(report_sections)
        
        # Save report to file
        report_path = os.path.join(self.results_dir, "tennis_analysis_report.md")
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Generated analysis report at {report_path}")
        
        return report