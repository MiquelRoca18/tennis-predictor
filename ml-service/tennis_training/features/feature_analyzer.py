"""
Feature analyzer module for tennis match prediction.

This module provides functionality to analyze how player features
change over time and their impact on prediction accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
import os

logger = logging.getLogger(__name__)

class TennisFeatureAnalyzer:
    """
    Class for analyzing tennis player features over time.
    
    This class provides methods to track feature evolution over time,
    identify significant changes, and analyze the relationship between
    features and performance.
    """
    
    def __init__(self, matches_df=None, players_df=None, feature_extractor=None, 
                 feature_snapshots=None, snapshot_dir=None):
        """
        Initialize the TennisFeatureAnalyzer.
        
        Args:
            matches_df (pd.DataFrame, optional): DataFrame containing match data
            players_df (pd.DataFrame, optional): DataFrame containing player information
            feature_extractor (object, optional): Feature extractor object
            feature_snapshots (dict, optional): Dictionary of feature snapshots by date
            snapshot_dir (str, optional): Directory containing feature snapshot files
        """
        self.matches_df = matches_df
        self.players_df = players_df
        self.feature_extractor = feature_extractor
        self.feature_snapshots = feature_snapshots or {}
        self.snapshot_dir = snapshot_dir
        
        # Track feature evolution
        self.feature_history = defaultdict(dict)
        self.significant_changes = defaultdict(list)
        
        # Analysis cache for performance
        self._analysis_cache = {}
        
        logger.info("TennisFeatureAnalyzer initialized")
    
    def set_data(self, matches_df=None, players_df=None, feature_extractor=None):
        """
        Set or update data sources.
        
        Args:
            matches_df (pd.DataFrame, optional): DataFrame containing match data
            players_df (pd.DataFrame, optional): DataFrame containing player information
            feature_extractor (object, optional): Feature extractor object
        """
        if matches_df is not None:
            self.matches_df = matches_df
        if players_df is not None:
            self.players_df = players_df
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            
        # Clear cache when data changes
        self._analysis_cache = {}
        
        logger.info("Data sources updated")
    
    def load_snapshots_from_directory(self, directory=None):
        """
        Load feature snapshots from directory.
        
        Args:
            directory (str, optional): Directory containing snapshot files
            
        Returns:
            int: Number of snapshots loaded
        """
        dir_path = directory or self.snapshot_dir
        if not dir_path or not os.path.isdir(dir_path):
            logger.error(f"Invalid snapshot directory: {dir_path}")
            return 0
        
        loaded_count = 0
        
        # Find all snapshot files
        for filename in os.listdir(dir_path):
            if filename.startswith('player_features_') and filename.endswith('.json'):
                try:
                    # Extract date from filename format 'player_features_YYYYMMDD_HHMMSS.json'
                    date_str = filename.replace('player_features_', '').replace('.json', '')
                    snapshot_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                    
                    # Load snapshot
                    filepath = os.path.join(dir_path, filename)
                    with open(filepath, 'r') as f:
                        snapshot_data = json.load(f)
                    
                    # Extract features from loaded data
                    if 'features' in snapshot_data:
                        features = snapshot_data['features']
                    else:
                        features = snapshot_data  # Legacy format
                    
                    # Add to snapshots
                    self.feature_snapshots[snapshot_date.strftime('%Y-%m-%d')] = features
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Error loading snapshot file {filename}: {e}")
        
        logger.info(f"Loaded {loaded_count} feature snapshots from {dir_path}")
        return loaded_count
    
    def add_snapshot(self, date, features):
        """
        Add a feature snapshot for a specific date.
        
        Args:
            date (str or datetime): Date of the snapshot
            features (dict): Dictionary of player features
        """
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date
        
        self.feature_snapshots[date_str] = features
        logger.info(f"Added feature snapshot for {date_str}")
    
    def analyze_player_feature_evolution(self, player_id, features=None, 
                                         start_date=None, end_date=None):
        """
        Analyze how a player's features have evolved over time.
        
        Args:
            player_id (int or str): ID of the player
            features (list, optional): List of specific features to analyze
            start_date (str or datetime, optional): Start date for analysis
            end_date (str or datetime, optional): End date for analysis
            
        Returns:
            dict: Analysis of feature evolution
        """
        player_id = str(player_id)
        
        # Convert dates to string format if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Create cache key
        cache_key = f"evolution_{player_id}_{start_date}_{end_date}_{str(features)}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        # Check if we have snapshots
        if not self.feature_snapshots:
            logger.error("No feature snapshots available. Load or add snapshots first.")
            return {}
        
        # Sort snapshot dates
        sorted_dates = sorted(self.feature_snapshots.keys())
        
        # Filter by date range if specified
        if start_date:
            sorted_dates = [d for d in sorted_dates if d >= start_date]
        if end_date:
            sorted_dates = [d for d in sorted_dates if d <= end_date]
        
        if not sorted_dates:
            logger.warning(f"No snapshots found in the specified date range for player {player_id}")
            return {}
        
        # Initialize feature history
        feature_history = defaultdict(list)
        dates = []
        
        # Extract feature values over time
        for date in sorted_dates:
            snapshot = self.feature_snapshots[date]
            dates.append(date)
            
            if player_id in snapshot:
                player_features = snapshot[player_id]
                
                # Filter specific features if requested
                if features:
                    feature_set = {f: player_features.get(f) for f in features if f in player_features}
                else:
                    feature_set = player_features
                
                # Add feature values to history
                for feature, value in feature_set.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_history[feature].append(value)
                    # Ensure all features have the same length
                    elif feature in feature_history:
                        # Use previous value if available
                        prev_value = feature_history[feature][-1] if feature_history[feature] else 0
                        feature_history[feature].append(prev_value)
        
        # Calculate statistics for each feature
        feature_stats = {}
        for feature, values in feature_history.items():
            if len(values) >= 2:
                feature_stats[feature] = {
                    'first_value': values[0],
                    'last_value': values[-1],
                    'min_value': min(values),
                    'max_value': max(values),
                    'mean_value': np.mean(values),
                    'std_value': np.std(values),
                    'total_change': values[-1] - values[0],
                    'percent_change': ((values[-1] - values[0]) / abs(values[0])) * 100 if values[0] != 0 else float('inf'),
                    'values': values,
                    'dates': dates
                }
                
                # Identify significant changes (>= 20% or 2 standard deviations)
                significant = False
                if abs(feature_stats[feature]['percent_change']) >= 20:
                    significant = True
                elif len(values) >= 5:
                    std_dev = np.std(values)
                    if std_dev > 0 and abs(values[-1] - values[0]) >= 2 * std_dev:
                        significant = True
                
                feature_stats[feature]['significant_change'] = significant
                
                # Calculate trend using linear regression
                if len(values) >= 3:
                    x = np.arange(len(values))
                    slope, intercept = np.polyfit(x, values, 1)
                    feature_stats[feature]['trend_slope'] = slope
                    feature_stats[feature]['trend_direction'] = 'increasing' if slope > 0 else 'decreasing'
                    feature_stats[feature]['trend_strength'] = abs(slope) / np.mean(values) if np.mean(values) != 0 else 0
        
        # Cache results
        self._analysis_cache[cache_key] = {
            'player_id': player_id,
            'features': feature_stats,
            'start_date': dates[0] if dates else None,
            'end_date': dates[-1] if dates else None,
            'snapshot_count': len(sorted_dates)
        }
        
        return self._analysis_cache[cache_key]
    
    def identify_significant_changes(self, player_id, threshold_percent=20, threshold_std=2):
        """
        Identify significant changes in player features.
        
        Args:
            player_id (int or str): ID of the player
            threshold_percent (float): Percentage change threshold
            threshold_std (float): Standard deviation change threshold
            
        Returns:
            dict: Dictionary of significant feature changes
        """
        player_id = str(player_id)
        
        # Check if we have evolution data
        evolution = self.analyze_player_feature_evolution(player_id)
        if not evolution or 'features' not in evolution:
            logger.warning(f"No feature evolution data available for player {player_id}")
            return {}
        
        # Filter for significant changes
        significant_changes = {}
        
        for feature, stats in evolution['features'].items():
            if 'percent_change' not in stats or 'std_value' not in stats:
                continue
                
            percent_change = stats['percent_change']
            if np.isinf(percent_change):
                continue
                
            std_change = abs(stats['total_change']) / stats['std_value'] if stats['std_value'] > 0 else 0
            
            # Check if change is significant
            if abs(percent_change) >= threshold_percent or std_change >= threshold_std:
                significant_changes[feature] = {
                    'start_value': stats['first_value'],
                    'end_value': stats['last_value'],
                    'total_change': stats['total_change'],
                    'percent_change': percent_change,
                    'std_change': std_change,
                    'direction': 'increased' if stats['total_change'] > 0 else 'decreased'
                }
        
        return {
            'player_id': player_id,
            'significant_changes': significant_changes,
            'change_count': len(significant_changes),
            'start_date': evolution['start_date'],
            'end_date': evolution['end_date']
        }
    
    def analyze_performance_impact(self, player_id, start_date=None, end_date=None):
        """
        Analyze how feature changes have impacted player performance.
        
        Args:
            player_id (int or str): ID of the player
            start_date (str or datetime, optional): Start date for analysis
            end_date (str or datetime, optional): End date for analysis
            
        Returns:
            dict: Analysis of performance impact
        """
        player_id = str(player_id)
        
        # Create cache key
        cache_key = f"performance_{player_id}_{start_date}_{end_date}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        # Check if we have match data
        if self.matches_df is None:
            logger.error("Match data not set. Cannot analyze performance impact.")
            return {}
        
        # Convert dates to pandas datetime
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
        else:
            # Default to earliest snapshot date
            if self.feature_snapshots:
                start_date = pd.to_datetime(min(self.feature_snapshots.keys()))
        
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
        else:
            # Default to latest snapshot date
            if self.feature_snapshots:
                end_date = pd.to_datetime(max(self.feature_snapshots.keys()))
        
        # Get feature evolution
        evolution = self.analyze_player_feature_evolution(
            player_id, 
            start_date=start_date.strftime('%Y-%m-%d') if start_date else None,
            end_date=end_date.strftime('%Y-%m-%d') if end_date else None
        )
        
        if not evolution or 'features' not in evolution:
            logger.warning(f"No feature evolution data available for player {player_id}")
            return {}
        
        # Filter player matches in the time range
        player_matches = self.matches_df[
            ((self.matches_df['winner_id'] == int(player_id)) | 
             (self.matches_df['loser_id'] == int(player_id)))
        ]
        
        if start_date:
            player_matches = player_matches[player_matches['tournament_date'] >= start_date]
        if end_date:
            player_matches = player_matches[player_matches['tournament_date'] <= end_date]
        
        # Calculate performance metrics over time
        dates = []
        win_rates = []
        
        # Split time range into periods (e.g., monthly)
        if not player_matches.empty:
            # Calculate monthly win rates
            monthly_stats = []
            for name, group in player_matches.groupby(pd.Grouper(key='tournament_date', freq='M')):
                if not group.empty:
                    month_date = name.strftime('%Y-%m-%d')
                    total_matches = len(group)
                    wins = len(group[group['winner_id'] == int(player_id)])
                    win_rate = wins / total_matches
                    
                    monthly_stats.append({
                        'date': month_date,
                        'total_matches': total_matches,
                        'wins': wins,
                        'win_rate': win_rate
                    })
                    
                    dates.append(month_date)
                    win_rates.append(win_rate)
        
        # Correlate feature changes with performance changes
        feature_correlations = {}
        
        if len(win_rates) >= 3:  # Need at least 3 points for correlation
            for feature, stats in evolution['features'].items():
                # Extract feature values
                feature_values = stats.get('values', [])
                feature_dates = stats.get('dates', [])
                
                # Align feature values with performance dates
                aligned_feature_values = []
                
                for date in dates:
                    # Find closest feature date before or equal to performance date
                    closest_date = None
                    closest_value = None
                    
                    for i, fd in enumerate(feature_dates):
                        if fd <= date and (closest_date is None or fd > closest_date):
                            closest_date = fd
                            closest_value = feature_values[i]
                    
                    if closest_value is not None:
                        aligned_feature_values.append(closest_value)
                    else:
                        # If no prior feature value, use the first available
                        aligned_feature_values.append(feature_values[0] if feature_values else 0)
                
                # Calculate correlation if we have enough aligned data points
                if len(aligned_feature_values) >= 3 and len(aligned_feature_values) == len(win_rates):
                    correlation = np.corrcoef(aligned_feature_values, win_rates)[0, 1]
                    
                    feature_correlations[feature] = {
                        'correlation': correlation,
                        'impact': 'positive' if correlation > 0 else 'negative',
                        'strength': abs(correlation)
                    }
        
        # Identify top influential features
        top_features = sorted(
            feature_correlations.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )[:10]  # Top 10 most correlated features
        
        # Build result
        result = {
            'player_id': player_id,
            'start_date': start_date.strftime('%Y-%m-%d') if start_date else None,
            'end_date': end_date.strftime('%Y-%m-%d') if end_date else None,
            'match_count': len(player_matches),
            'win_rate_trend': {
                'dates': dates,
                'win_rates': win_rates
            },
            'monthly_performance': monthly_stats,
            'feature_correlations': feature_correlations,
            'top_influential_features': [
                {
                    'feature': feature,
                    'correlation': stats['correlation'],
                    'impact': stats['impact'],
                    'strength': stats['strength']
                } for feature, stats in top_features
            ]
        }
        
        # Cache results
        self._analysis_cache[cache_key] = result
        
        return result
    
    def compare_players_evolution(self, player_ids, features, start_date=None, end_date=None):
        """
        Compare feature evolution across multiple players.
        
        Args:
            player_ids (list): List of player IDs to compare
            features (list): List of features to compare
            start_date (str or datetime, optional): Start date for comparison
            end_date (str or datetime, optional): End date for comparison
            
        Returns:
            dict: Comparative analysis of player feature evolution
        """
        # Convert player IDs to strings
        player_ids = [str(pid) for pid in player_ids]
        
        # Create cache key
        cache_key = f"compare_{'_'.join(player_ids)}_{str(features)}_{start_date}_{end_date}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        # Get feature evolution for each player
        player_evolutions = {}
        
        for player_id in player_ids:
            evolution = self.analyze_player_feature_evolution(
                player_id, features=features, start_date=start_date, end_date=end_date
            )
            
            if evolution and 'features' in evolution:
                player_evolutions[player_id] = evolution
        
        if not player_evolutions:
            logger.warning("No feature evolution data available for the specified players")
            return {}
        
        # Compare feature trends
        feature_comparison = {}
        
        for feature in features:
            feature_comparison[feature] = {
                'values_by_player': {},
                'trend_slopes': {},
                'total_changes': {},
                'percent_changes': {}
            }
            
            # Extract data for each player
            for player_id, evolution in player_evolutions.items():
                if feature in evolution['features']:
                    stats = evolution['features'][feature]
                    
                    feature_comparison[feature]['values_by_player'][player_id] = stats.get('values', [])
                    feature_comparison[feature]['trend_slopes'][player_id] = stats.get('trend_slope', 0)
                    feature_comparison[feature]['total_changes'][player_id] = stats.get('total_change', 0)
                    feature_comparison[feature]['percent_changes'][player_id] = stats.get('percent_change', 0)
            
            # Find player with most improvement and decline
            if feature_comparison[feature]['percent_changes']:
                best_player = max(
                    feature_comparison[feature]['percent_changes'].items(),
                    key=lambda x: x[1]
                )
                worst_player = min(
                    feature_comparison[feature]['percent_changes'].items(),
                    key=lambda x: x[1]
                )
                
                feature_comparison[feature]['most_improved'] = {
                    'player_id': best_player[0],
                    'percent_change': best_player[1]
                }
                
                feature_comparison[feature]['most_declined'] = {
                    'player_id': worst_player[0],
                    'percent_change': worst_player[1]
                }
        
        # Calculate overall evolution ranking
        overall_ranking = {}
        
        for player_id in player_ids:
            if player_id not in player_evolutions:
                continue
                
            # Sum up absolute percent changes across features
            total_improvement = 0
            feature_count = 0
            
            for feature in features:
                if (feature in player_evolutions[player_id]['features'] and
                    'percent_change' in player_evolutions[player_id]['features'][feature]):
                    
                    percent_change = player_evolutions[player_id]['features'][feature]['percent_change']
                    if not np.isinf(percent_change):
                        total_improvement += percent_change
                        feature_count += 1
            
            if feature_count > 0:
                overall_ranking[player_id] = total_improvement / feature_count
        
        # Sort players by overall improvement
        sorted_ranking = sorted(
            overall_ranking.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build result
        result = {
            'player_ids': player_ids,
            'features': features,
            'start_date': player_evolutions[player_ids[0]]['start_date'] if player_ids else None,
            'end_date': player_evolutions[player_ids[0]]['end_date'] if player_ids else None,
            'feature_comparison': feature_comparison,
            'overall_ranking': [
                {'player_id': player_id, 'avg_improvement': improvement}
                for player_id, improvement in sorted_ranking
            ]
        }
        
        # Cache results
        self._analysis_cache[cache_key] = result
        
        return result
    
    def plot_feature_evolution(self, player_id, features, start_date=None, end_date=None, 
                              figsize=(12, 8), save_path=None):
        """
        Plot the evolution of selected features for a player.
        
        Args:
            player_id (int or str): ID of the player
            features (list): List of features to plot
            start_date (str or datetime, optional): Start date for analysis
            end_date (str or datetime, optional): End date for analysis
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get feature evolution
        evolution = self.analyze_player_feature_evolution(
            player_id, features=features, start_date=start_date, end_date=end_date
        )
        
        if not evolution or 'features' not in evolution:
            logger.warning(f"No feature evolution data available for player {player_id}")
            return None
        
        # Create figure
        fig, axes = plt.subplots(len(features), 1, figsize=figsize, sharex=True)
        if len(features) == 1:
            axes = [axes]
        
        # Get player name if available
        player_name = f"Player {player_id}"
        if self.players_df is not None and 'player_id' in self.players_df.columns and 'name' in self.players_df.columns:
            player_info = self.players_df[self.players_df['player_id'] == int(player_id)]
            if not player_info.empty:
                player_name = player_info.iloc[0]['name']
        
        # Plot each feature
        for i, feature in enumerate(features):
            if feature in evolution['features']:
                stats = evolution['features'][feature]
                values = stats.get('values', [])
                dates = stats.get('dates', [])
                
                if values and dates and len(values) == len(dates):
                    # Convert dates to datetime for plotting
                    plot_dates = [pd.to_datetime(d) for d in dates]
                    
                    # Plot feature values
                    axes[i].plot(plot_dates, values, marker='o', linestyle='-')
                    
                    # Add trend line if available
                    if 'trend_slope' in stats:
                        x = np.arange(len(values))
                        slope = stats['trend_slope']
                        intercept = values[0] - slope * 0
                        trend_line = slope * x + intercept
                        axes[i].plot(plot_dates, trend_line, linestyle='--', color='red', 
                                     label=f"Trend: {slope:.4f}")
                    
                    # Set axis labels and title
                    axes[i].set_ylabel(feature)
                    axes[i].set_title(f"{feature} Evolution")
                    axes[i].grid(True, linestyle='--', alpha=0.7)
                    
                    # Add percentage change annotation
                    if 'percent_change' in stats and not np.isinf(stats['percent_change']):
                        percent_change = stats['percent_change']
                        color = 'green' if percent_change > 0 else 'red'
                        axes[i].annotate(
                            f"{percent_change:.2f}% change",
                            xy=(0.02, 0.85),
                            xycoords='axes fraction',
                            color=color,
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8)
                        )
                    
                    # Add legend if we have a trend line
                    if 'trend_slope' in stats:
                        axes[i].legend()
        
        # Set common x-axis label and title
        fig.tight_layout()
        plt.xlabel('Date')
        plt.suptitle(f"Feature Evolution for {player_name}", fontsize=16, y=1.02)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_performance_correlation(self, player_id, top_n=5, start_date=None, end_date=None,
                                    figsize=(14, 10), save_path=None):
        """
        Plot correlation between features and performance for a player.
        
        Args:
            player_id (int or str): ID of the player
            top_n (int): Number of top correlated features to plot
            start_date (str or datetime, optional): Start date for analysis
            end_date (str or datetime, optional): End date for analysis
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get performance impact analysis
        analysis = self.analyze_performance_impact(player_id, start_date, end_date)
        
        if not analysis or 'feature_correlations' not in analysis:
            logger.warning(f"No performance correlation data available for player {player_id}")
            return None
        
        # Get player name if available
        player_name = f"Player {player_id}"
        if self.players_df is not None and 'player_id' in self.players_df.columns and 'name' in self.players_df.columns:
            player_info = self.players_df[self.players_df['player_id'] == int(player_id)]
            if not player_info.empty:
                player_name = player_info.iloc[0]['name']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2]})
        
        # Plot win rate trend
        win_rate_trend = analysis.get('win_rate_trend', {})
        dates = win_rate_trend.get('dates', [])
        win_rates = win_rate_trend.get('win_rates', [])
        
        if dates and win_rates and len(dates) == len(win_rates):
            # Convert dates to datetime for plotting
            plot_dates = [pd.to_datetime(d) for d in dates]
            
            # Plot win rate
            ax1.plot(plot_dates, win_rates, marker='o', linestyle='-', color='blue')
            ax1.set_ylabel('Win Rate')
            ax1.set_title(f"Win Rate Trend for {player_name}")
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add trend line
            if len(win_rates) >= 2:
                x = np.arange(len(win_rates))
                slope, intercept = np.polyfit(x, win_rates, 1)
                trend_line = slope * x + intercept
                ax1.plot(plot_dates, trend_line, linestyle='--', color='red', 
                         label=f"Trend: {slope:.4f}")
                ax1.legend()
        
        # Plot feature correlations
        feature_correlations = analysis.get('feature_correlations', {})
        
        if feature_correlations:
            # Extract top correlated features
            top_features = sorted(
                feature_correlations.items(),
                key=lambda x: abs(x[1]['correlation']),
                reverse=True
            )[:top_n]
            
            # Prepare data for bar plot
            features = [item[0] for item in top_features]
            correlations = [item[1]['correlation'] for item in top_features]
            
            # Create horizontal bar plot
            bars = ax2.barh(features, correlations)
            
            # Color bars based on correlation sign
            for i, bar in enumerate(bars):
                bar.set_color('green' if correlations[i] > 0 else 'red')
            
            # Add value labels
            for i, corr in enumerate(correlations):
                ax2.text(
                    corr + (0.02 if corr >= 0 else -0.08),
                    i,
                    f"{corr:.3f}",
                    va='center',
                    fontsize=9
                )
            
            # Set axis labels and title
            ax2.set_xlabel('Correlation with Win Rate')
            ax2.set_title(f"Top {top_n} Features Correlated with Performance")
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, linestyle='--', alpha=0.5, axis='x')
            
            # Add interpretation
            ax2.text(
                0.02, -0.15,
                "Positive values indicate features that improve with better performance.\n"
                "Negative values indicate features that decline with better performance.",
                transform=ax2.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        # Format figure
        fig.tight_layout()
        plt.suptitle(f"Feature-Performance Correlation for {player_name}", fontsize=16, y=0.98)
        
        # Format x-axis dates on the first plot
        fig.autofmt_xdate()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig