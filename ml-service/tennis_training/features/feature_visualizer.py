"""
Feature visualizer module for tennis match prediction.

This module provides interactive visualization tools for exploring tennis features,
identifying patterns, and understanding feature relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import json
import logging
import os
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class TennisFeatureVisualizer:
    """
    Class for creating interactive visualizations of tennis features.
    
    This class provides methods to visualize feature importance, feature
    relationships, temporal trends, and player comparisons.
    """
    
    def __init__(self, features_df=None, player_features=None, feature_importance=None):
        """
        Initialize the TennisFeatureVisualizer.
        
        Args:
            features_df (pd.DataFrame, optional): DataFrame containing features
            player_features (dict, optional): Dictionary of player features
            feature_importance (dict, optional): Dictionary of feature importance scores
        """
        self.features_df = features_df
        self.player_features = player_features
        self.feature_importance = feature_importance
        
        # Output directory for saving visualizations
        self.output_dir = None
        
        # Figure style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        logger.info("TennisFeatureVisualizer initialized")
    
    def set_data(self, features_df=None, player_features=None, feature_importance=None):
        """
        Set or update data sources.
        
        Args:
            features_df (pd.DataFrame, optional): DataFrame containing features
            player_features (dict, optional): Dictionary of player features
            feature_importance (dict, optional): Dictionary of feature importance scores
        """
        if features_df is not None:
            self.features_df = features_df
        if player_features is not None:
            self.player_features = player_features
        if feature_importance is not None:
            self.feature_importance = feature_importance
            
        logger.info("Data sources updated")
    
    def set_output_directory(self, output_dir):
        """
        Set the output directory for saving visualizations.
        
        Args:
            output_dir (str): Path to output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.output_dir = output_dir
        logger.info(f"Output directory set to {output_dir}")
    
    def visualize_feature_importance(self, top_n=20, model_name="Model", figsize=(10, 8), 
                                    highlight_features=None, save=False):
        """
        Visualize feature importance.
        
        Args:
            top_n (int): Number of top features to show
            model_name (str): Name of the model for the title
            figsize (tuple): Figure size
            highlight_features (list, optional): List of features to highlight
            save (bool): Whether to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if not self.feature_importance:
            logger.error("Feature importance not available. Set feature_importance first.")
            return None
        
        # Convert feature importance to Series if it's a dict
        if isinstance(self.feature_importance, dict):
            importance = pd.Series(self.feature_importance)
        else:
            importance = self.feature_importance
        
        # Sort and get top features
        importance = importance.sort_values(ascending=False)
        top_features = importance.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(np.arange(len(top_features)), top_features.values, align='center')
        
        # Set labels and title
        ax.set_yticks(np.arange(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Features by Importance ({model_name})')
        
        # Highlight specific features if requested
        if highlight_features:
            for i, feature in enumerate(top_features.index):
                if feature in highlight_features:
                    bars[i].set_color('red')
        
        # Add importance values as text
        for i, v in enumerate(top_features.values):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # Format y-axis labels for better readability
        plt.tight_layout()
        
        # Save if requested
        if save and self.output_dir:
            filename = os.path.join(self.output_dir, f"feature_importance_{model_name.replace(' ', '_')}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance visualization saved to {filename}")
        
        return fig
    
    def visualize_feature_relationships(self, features, target=None, plot_type='scatter', 
                                       figsize=(12, 10), save=False):
        """
        Visualize relationships between selected features.
        
        Args:
            features (list): List of features to visualize
            target (str, optional): Target feature for coloring points
            plot_type (str): Type of plot ('scatter', 'pair', 'heatmap')
            figsize (tuple): Figure size
            save (bool): Whether to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Prepare data
        if self.features_df is not None:
            data = self.features_df
        elif self.player_features:
            # Convert player_features to DataFrame
            records = []
            for player_id, player_feats in self.player_features.items():
                record = {'player_id': player_id}
                record.update(player_feats)
                records.append(record)
            data = pd.DataFrame(records)
        else:
            logger.error("No feature data available")
            return None
        
        # Filter to selected features
        selected_features = [f for f in features if f in data.columns]
        if target and target in data.columns:
            selected_features.append(target)
        
        if len(selected_features) < 2:
            logger.error("Not enough valid features for visualization")
            return None
        
        subset_data = data[selected_features].copy()
        
        # Create visualization based on plot type
        if plot_type == 'scatter':
            # Create scatter plot matrix
            fig, axes = plt.subplots(nrows=len(features), ncols=len(features), figsize=figsize)
            
            # Iterate through feature pairs
            for i, feat1 in enumerate(features):
                for j, feat2 in enumerate(features):
                    if i == j:  # Diagonal - show distribution
                        if feat1 in subset_data.columns:
                            sns.histplot(subset_data[feat1].dropna(), ax=axes[i, j], kde=True)
                            axes[i, j].set_title(feat1)
                    elif feat1 in subset_data.columns and feat2 in subset_data.columns:
                        # Off-diagonal - show scatter plot
                        if target and target in subset_data.columns:
                            scatter = axes[i, j].scatter(
                                subset_data[feat2], 
                                subset_data[feat1],
                                c=subset_data[target], 
                                alpha=0.6, 
                                cmap='viridis',
                                s=30
                            )
                            if i == 0 and j == len(features) - 1:  # Add colorbar to one plot
                                cbar = plt.colorbar(scatter, ax=axes[i, j])
                                cbar.set_label(target)
                        else:
                            axes[i, j].scatter(subset_data[feat2], subset_data[feat1], alpha=0.6)
                        
                        # Set axis labels
                        if i == len(features) - 1:
                            axes[i, j].set_xlabel(feat2)
                        if j == 0:
                            axes[i, j].set_ylabel(feat1)
                    
                    # Hide unnecessary axis labels
                    if i < len(features) - 1:
                        axes[i, j].set_xlabel('')
                    if j > 0:
                        axes[i, j].set_ylabel('')
            
            plt.tight_layout()
            plt.suptitle('Feature Relationships', y=1.02, fontsize=16)
            
        elif plot_type == 'pair':
            # Use seaborn's pairplot
            if target and target in subset_data.columns:
                g = sns.pairplot(subset_data, hue=target, diag_kind='kde', plot_kws={'alpha': 0.6})
                g.fig.suptitle('Feature Relationships', y=1.02, fontsize=16)
                fig = g.fig
            else:
                g = sns.pairplot(subset_data, diag_kind='kde')
                g.fig.suptitle('Feature Relationships', y=1.02, fontsize=16)
                fig = g.fig
            
        elif plot_type == 'heatmap':
            # Create correlation heatmap
            plt.figure(figsize=figsize)
            corr = subset_data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            fig = plt.figure(figsize=figsize)
            sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                      square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
            
            plt.title('Feature Correlation Heatmap', fontsize=16)
            plt.tight_layout()
        
        else:
            logger.error(f"Unknown plot type: {plot_type}")
            return None
        
        # Save if requested
        if save and self.output_dir:
            filename = os.path.join(self.output_dir, f"feature_relationships_{plot_type}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Feature relationships visualization saved to {filename}")
        
        return fig
    
    def visualize_feature_distributions(self, features=None, n_features=12, by_group=None,
                                       figsize=(15, 10), save=False):
        """
        Visualize distributions of selected features.
        
        Args:
            features (list, optional): List of features to visualize
            n_features (int): Number of features to show if not specified
            by_group (str, optional): Column name to group by (e.g., 'surface')
            figsize (tuple): Figure size
            save (bool): Whether to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Prepare data
        if self.features_df is not None:
            data = self.features_df
        elif self.player_features:
            # Convert player_features to DataFrame
            records = []
            for player_id, player_feats in self.player_features.items():
                record = {'player_id': player_id}
                record.update(player_feats)
                records.append(record)
            data = pd.DataFrame(records)
        else:
            logger.error("No feature data available")
            return None
        
        # Filter to numeric features
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        # If features not specified, use top features by importance or most varying
        if features is None:
            if self.feature_importance:
                # Use top features by importance
                if isinstance(self.feature_importance, dict):
                    importance = pd.Series(self.feature_importance)
                else:
                    importance = self.feature_importance
                
                # Filter to numeric features
                importance = importance[importance.index.isin(numeric_cols)]
                
                # Sort and get top features
                top_features = importance.sort_values(ascending=False).head(n_features).index.tolist()
                features = [f for f in top_features if f in data.columns]
            else:
                # Use features with highest coefficient of variation
                cv = data[numeric_cols].std() / data[numeric_cols].mean().replace(0, float('inf'))
                features = cv.sort_values(ascending=False).head(n_features).index.tolist()
        else:
            # Filter to available features
            features = [f for f in features if f in data.columns]
        
        if not features:
            logger.error("No valid features for visualization")
            return None
        
        # Determine grid size
        n_cols = min(4, len(features))
        n_rows = (len(features) + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each feature
        for i, feature in enumerate(features):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            if by_group and by_group in data.columns:
                # Plot distributions by group
                groups = data[by_group].unique()
                for group in groups:
                    group_data = data[data[by_group] == group][feature].dropna()
                    if len(group_data) > 0:
                        sns.kdeplot(group_data, ax=ax, label=str(group))
                
                ax.legend()
                ax.set_title(f"{feature} by {by_group}")
            else:
                # Plot overall distribution
                sns.histplot(data[feature].dropna(), kde=True, ax=ax)
                ax.set_title(feature)
            
            # Add basic statistics
            stats_text = f"Mean: {data[feature].mean():.3f}\nStd: {data[feature].std():.3f}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Feature Distributions', y=1.02, fontsize=16)
        
        # Save if requested
        if save and self.output_dir:
            group_str = f"_by_{by_group}" if by_group else ""
            filename = os.path.join(self.output_dir, f"feature_distributions{group_str}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Feature distributions visualization saved to {filename}")
        
        return fig
    
    def visualize_player_comparison(self, player_ids, features=None, n_features=10, 
                                   figsize=(12, 8), save=False):
        """
        Visualize feature comparison between players.
        
        Args:
            player_ids (list): List of player IDs to compare
            features (list, optional): List of features to compare
            n_features (int): Number of features to show if not specified
            figsize (tuple): Figure size
            save (bool): Whether to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if not self.player_features:
            logger.error("Player features not available. Set player_features first.")
            return None
        
        # Convert player_ids to strings
        player_ids = [str(pid) for pid in player_ids]
        
        # Filter to available players
        available_players = [pid for pid in player_ids if pid in self.player_features]
        
        if not available_players:
            logger.error("None of the specified players are available")
            return None
        
        # Get feature set from first player
        first_player_features = self.player_features[available_players[0]]
        
        # If features not specified, use top features by importance or select common features
        if features is None:
            if self.feature_importance:
                # Use top features by importance
                if isinstance(self.feature_importance, dict):
                    importance = pd.Series(self.feature_importance)
                else:
                    importance = self.feature_importance
                
                # Sort and get top features
                top_features = importance.sort_values(ascending=False).head(n_features).index.tolist()
                
                # Filter to features available for all players
                features = []
                for feat in top_features:
                    if all(feat in self.player_features[pid] for pid in available_players):
                        features.append(feat)
                
                # Limit to specified number
                features = features[:n_features]
                
            else:
                # Get common features across players
                common_features = set(first_player_features.keys())
                
                for pid in available_players[1:]:
                    common_features = common_features.intersection(set(self.player_features[pid].keys()))
                
                # Filter to numeric features
                numeric_features = []
                for feat in common_features:
                    value = first_player_features[feat]
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        numeric_features.append(feat)
                
                # Limit to specified number
                features = numeric_features[:n_features]
        else:
            # Filter to features available for all players
            features = [f for f in features if all(f in self.player_features[pid] for pid in available_players)]
        
        if not features:
            logger.error("No valid common features for comparison")
            return None
        
        # Create data for visualization
        comparison_data = []
        
        for feat in features:
            feat_data = {'feature': feat}
            
            for pid in available_players:
                feat_data[pid] = self.player_features[pid].get(feat, None)
            
            comparison_data.append(feat_data)
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Melt for easier plotting
        melted_df = pd.melt(comparison_df, id_vars=['feature'], var_name='player', value_name='value')
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create grouped bar chart
        g = sns.catplot(
            data=melted_df,
            kind="bar",
            x="feature", y="value", hue="player",
            height=figsize[1]/2, aspect=figsize[0]/figsize[1]*2,
            legend_out=False
        )
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.title('Player Feature Comparison', fontsize=16)
        plt.tight_layout()
        
        # Save if requested
        if save and self.output_dir:
            player_str = "_".join(available_players)
            filename = os.path.join(self.output_dir, f"player_comparison_{player_str}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Player comparison visualization saved to {filename}")
        
        return g.fig
    
    def create_interactive_dashboard(self, output_file=None):
        """
        Create an interactive HTML dashboard of feature visualizations.
        
        Args:
            output_file (str, optional): Path to save the dashboard HTML
            
        Returns:
            str: HTML string of the dashboard
        """
        # Check if we have the necessary data
        if not self.player_features and not self.features_df:
            logger.error("No feature data available")
            return None
        
        # Prepare data for visualizations
        if self.features_df is not None:
            data = self.features_df
        else:
            # Convert player_features to DataFrame
            records = []
            for player_id, player_feats in self.player_features.items():
                record = {'player_id': player_id}
                record.update(player_feats)
                records.append(record)
            data = pd.DataFrame(records)
        
        # Select features for visualizations
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        if self.feature_importance:
            # Use top features by importance
            if isinstance(self.feature_importance, dict):
                importance = pd.Series(self.feature_importance)
            else:
                importance = self.feature_importance
            
            # Filter to numeric features and sort
            importance = importance[importance.index.isin(numeric_cols)]
            top_features = importance.sort_values(ascending=False).head(20).index.tolist()
        else:
            # Use features with highest coefficient of variation
            cv = data[numeric_cols].std() / data[numeric_cols].mean().replace(0, float('inf'))
            top_features = cv.sort_values(ascending=False).head(20).index.tolist()
        
        # Create visualizations
        # 1. Feature Importance
        if self.feature_importance:
            imp_fig = self.visualize_feature_importance(top_n=15, save=False)
            importance_html = self._fig_to_html(imp_fig)
        else:
            importance_html = "<p>Feature importance not available</p>"
        
        # 2. Feature Distributions
        dist_fig = self.visualize_feature_distributions(features=top_features[:12], save=False)
        distributions_html = self._fig_to_html(dist_fig)
        
        # 3. Feature Correlations
        corr_fig = self.visualize_feature_relationships(features=top_features[:8], plot_type='heatmap', save=False)
        correlations_html = self._fig_to_html(corr_fig)
        
        # 4. Feature Relationships
        pair_features = top_features[:5]  # Limit to avoid too many plots
        pair_fig = self.visualize_feature_relationships(features=pair_features, plot_type='pair', save=False)
        relationships_html = self._fig_to_html(pair_fig)
        
        # Create HTML dashboard
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tennis Feature Analysis Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .section {{
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .section h2 {{
                    color: #2c3e50;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                .viz {{
                    width: 100%;
                    overflow-x: auto;
                    text-align: center;
                }}
                .row {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: 0 -15px;
                }}
                .col {{
                    flex: 1;
                    padding: 0 15px;
                    min-width: 300px;
                }}
                footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding: 20px;
                    color: #777;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Tennis Feature Analysis Dashboard</h1>
                <p>Interactive visualization of tennis prediction features</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2>Feature Importance</h2>
                    <p>The most influential features for tennis match prediction</p>
                    <div class="viz">
                        {importance_html}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Feature Distributions</h2>
                    <p>Statistical distributions of key tennis features</p>
                    <div class="viz">
                        {distributions_html}
                    </div>
                </div>
                
                <div class="row">
                    <div class="col">
                        <div class="section">
                            <h2>Feature Correlations</h2>
                            <p>Heatmap showing relationships between features</p>
                            <div class="viz">
                                {correlations_html}
                            </div>
                        </div>
                    </div>
                    
                    <div class="col">
                        <div class="section">
                            <h2>Feature Relationships</h2>
                            <p>Pairwise relationships between key features</p>
                            <div class="viz">
                                {relationships_html}
                            </div>
                        </div>
                    </div>
                </div>
                
                <footer>
                    <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Tennis Feature Analysis Dashboard</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Interactive dashboard saved to {output_file}")
        
        return html
    
    def _fig_to_html(self, fig):
        """
        Convert matplotlib figure to HTML img tag.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            str: HTML img tag with embedded figure
        """
        if fig is None:
            return "<p>Visualization not available</p>"
            
        # Save figure to in-memory buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Encode as base64
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Create HTML
        img_html = f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;">'
        
        # Close figure to free memory
        plt.close(fig)
        
        return img_html
    
    def generate_feature_summary_report(self, features=None, n_features=20, output_file=None):
        """
        Generate a comprehensive feature summary report.
        
        Args:
            features (list, optional): List of features to include
            n_features (int): Number of features to include if not specified
            output_file (str, optional): Path to save the report
            
        Returns:
            str: HTML string of the report
        """
        # Check if we have the necessary data
        if not self.player_features and not self.features_df:
            logger.error("No feature data available")
            return None
        
        # Prepare data for analysis
        if self.features_df is not None:
            data = self.features_df
        else:
            # Convert player_features to DataFrame
            records = []
            for player_id, player_feats in self.player_features.items():
                record = {'player_id': player_id}
                record.update(player_feats)
                records.append(record)
            data = pd.DataFrame(records)
        
        # Select features for report
        if features is None:
            if self.feature_importance:
                # Use top features by importance
                if isinstance(self.feature_importance, dict):
                    importance = pd.Series(self.feature_importance)
                else:
                    importance = self.feature_importance
                
                # Sort and get top features
                features = importance.sort_values(ascending=False).head(n_features).index.tolist()
                features = [f for f in features if f in data.columns]
            else:
                # Use numeric features
                numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
                
                # Calculate coefficient of variation
                cv = data[numeric_cols].std() / data[numeric_cols].mean().replace(0, float('inf'))
                features = cv.sort_values(ascending=False).head(n_features).index.tolist()
        else:
            # Filter to available features
            features = [f for f in features if f in data.columns]
        
        # Generate feature summary
        feature_summaries = []
        
        for feature in features:
            # Skip non-numeric features
            if data[feature].dtype not in ['float64', 'int64']:
                continue
                
            # Calculate statistics
            stats = {
                'mean': data[feature].mean(),
                'median': data[feature].median(),
                'std': data[feature].std(),
                'min': data[feature].min(),
                'max': data[feature].max(),
                'missing': data[feature].isna().sum(),
                'missing_pct': data[feature].isna().mean() * 100
            }
            
            # Calculate percentiles
            percentiles = {}
            for p in [1, 5, 25, 50, 75, 95, 99]:
                percentiles[f'p{p}'] = data[feature].quantile(p/100)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Distribution plot
            sns.histplot(data[feature].dropna(), kde=True, ax=ax1)
            ax1.set_title(f"{feature} Distribution")
            
            # Box plot
            sns.boxplot(y=data[feature].dropna(), ax=ax2)
            ax2.set_title(f"{feature} Box Plot")
            
            plt.tight_layout()
            
            # Convert to HTML
            viz_html = self._fig_to_html(fig)
            
            # Feature importance
            importance = None
            if self.feature_importance:
                if isinstance(self.feature_importance, dict):
                    importance = self.feature_importance.get(feature, None)
                elif feature in self.feature_importance.index:
                    importance = self.feature_importance[feature]
            
            # Add to summaries
            feature_summaries.append({
                'feature': feature,
                'stats': stats,
                'percentiles': percentiles,
                'importance': importance,
                'visualization': viz_html
            })
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tennis Feature Summary Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .feature-card {{
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .feature-title {{
                    color: #2c3e50;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                .feature-importance {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .stats-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                .stats-table th, .stats-table td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .stats-table th {{
                    background-color: #f2f2f2;
                }}
                .viz {{
                    width: 100%;
                    text-align: center;
                    margin-top: 20px;
                }}
                footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding: 20px;
                    color: #777;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Tennis Feature Summary Report</h1>
                <p>Comprehensive analysis of tennis prediction features</p>
            </div>
            
            <div class="container">
                <h2>Feature Analysis ({len(feature_summaries)} features)</h2>
                
                {self._generate_feature_cards_html(feature_summaries)}
                
                <footer>
                    <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Tennis Feature Summary Report</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Feature summary report saved to {output_file}")
        
        return html
    
    def _generate_feature_cards_html(self, feature_summaries):
        """
        Generate HTML for feature cards in the summary report.
        
        Args:
            feature_summaries (list): List of feature summary dictionaries
            
        Returns:
            str: HTML string with feature cards
        """
        html = ""
        
        for summary in feature_summaries:
            feature = summary['feature']
            stats = summary['stats']
            percentiles = summary['percentiles']
            importance = summary['importance']
            viz_html = summary['visualization']
            
            # Create importance HTML
            importance_html = ""
            if importance is not None:
                importance_html = f'<p class="feature-importance">Importance: {importance:.6f}</p>'
            
            # Create card HTML
            card_html = f"""
            <div class="feature-card">
                <h3 class="feature-title">{feature}</h3>
                {importance_html}
                
                <table class="stats-table">
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                        <th>Percentile</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Mean</td>
                        <td>{stats['mean']:.4f}</td>
                        <td>1%</td>
                        <td>{percentiles['p1']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Median</td>
                        <td>{stats['median']:.4f}</td>
                        <td>5%</td>
                        <td>{percentiles['p5']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Std Dev</td>
                        <td>{stats['std']:.4f}</td>
                        <td>25%</td>
                        <td>{percentiles['p25']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Min</td>
                        <td>{stats['min']:.4f}</td>
                        <td>75%</td>
                        <td>{percentiles['p75']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Max</td>
                        <td>{stats['max']:.4f}</td>
                        <td>95%</td>
                        <td>{percentiles['p95']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Missing</td>
                        <td>{stats['missing']} ({stats['missing_pct']:.2f}%)</td>
                        <td>99%</td>
                        <td>{percentiles['p99']:.4f}</td>
                    </tr>
                </table>
                
                <div class="viz">
                    {viz_html}
                </div>
            </div>
            """
            
            html += card_html
        
        return html