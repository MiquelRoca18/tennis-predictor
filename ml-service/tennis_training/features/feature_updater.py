"""
Feature updater module for tennis match prediction.

This module provides functionality for incremental updates to player features
after new matches, without needing to reprocess the entire dataset.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class TennisFeatureUpdater:
    """
    Class for incrementally updating tennis match prediction features.
    
    This class provides mechanisms to efficiently update player features
    after new matches, avoiding the need to reprocess the entire dataset.
    It manages feature state and applies delta updates based on new information.
    """
    
    def __init__(self, feature_store_path=None, matches_df=None, players_df=None, 
                 feature_extractor=None, initial_features=None):
        """
        Initialize the TennisFeatureUpdater.
        
        Args:
            feature_store_path (str, optional): Path to store feature snapshots
            matches_df (pd.DataFrame, optional): DataFrame containing match data
            players_df (pd.DataFrame, optional): DataFrame containing player information
            feature_extractor (TennisFeatureExtractor, optional): Feature extractor instance
            initial_features (dict, optional): Initial player features dictionary
        """
        self.feature_store_path = feature_store_path
        self.matches_df = matches_df
        self.players_df = players_df
        self.feature_extractor = feature_extractor
        
        # Initialize player features
        self.player_features = initial_features or {}
        
        # Track when features were last updated
        self.last_update = {}
        
        # Match history buffer for recent matches
        self.match_buffer = []
        
        # Feature update history for analysis
        self.update_history = {}
        
        logger.info("TennisFeatureUpdater initialized")
    
    def set_data_sources(self, matches_df=None, players_df=None, feature_extractor=None):
        """
        Set or update data sources.
        
        Args:
            matches_df (pd.DataFrame, optional): DataFrame containing match data
            players_df (pd.DataFrame, optional): DataFrame containing player information
            feature_extractor (TennisFeatureExtractor, optional): Feature extractor instance
        """
        if matches_df is not None:
            self.matches_df = matches_df
        if players_df is not None:
            self.players_df = players_df
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            
        logger.info("Data sources updated")
    
    def initialize_player_features(self, player_ids=None, force_recalculate=False):
        """
        Initialize features for specified players or all players in the dataset.
        
        Args:
            player_ids (list, optional): List of player IDs to initialize
            force_recalculate (bool): Whether to force recalculation for existing features
            
        Returns:
            dict: Dictionary of initialized player features
        """
        if self.feature_extractor is None:
            logger.error("Feature extractor not set. Call set_data_sources() first.")
            return self.player_features
        
        # If no player IDs specified, use all unique players in matches_df
        if player_ids is None and self.matches_df is not None:
            winner_ids = set(self.matches_df['winner_id'].unique())
            loser_ids = set(self.matches_df['loser_id'].unique())
            player_ids = list(winner_ids.union(loser_ids))
        
        if not player_ids:
            logger.error("No player IDs specified and no match data available")
            return self.player_features
        
        # Initialize feature extraction for each player
        for player_id in player_ids:
            # Skip if already initialized and not forcing recalculation
            if str(player_id) in self.player_features and not force_recalculate:
                continue
                
            # Extract initial features
            latest_features = self._extract_latest_player_features(player_id)
            
            if latest_features:
                self.player_features[str(player_id)] = latest_features
                self.last_update[str(player_id)] = datetime.now()
                logger.info(f"Initialized features for player {player_id}")
            else:
                logger.warning(f"Failed to initialize features for player {player_id}")
        
        return self.player_features
    
    def _extract_latest_player_features(self, player_id):
        """
        Extract the latest features for a player using all available data.
        
        Args:
            player_id (int): ID of the player
            
        Returns:
            dict: Dictionary of the player's latest features
        """
        if self.matches_df is None or self.feature_extractor is None:
            logger.error("Match data or feature extractor not available")
            return {}
        
        # Create a placeholder "next match" to extract features
        latest_date = self.matches_df['tournament_date'].max() + timedelta(days=1)
        dummy_opponent_id = -1  # Placeholder opponent
        
        # Use feature extractor to get current features
        try:
            features = self.feature_extractor.extract_features(
                player1_id=player_id,
                player2_id=dummy_opponent_id,
                match_date=latest_date
            )
            
            # Filter features relevant only to this player (player1)
            player_features = {
                k: v for k, v in features.items()
                if k.startswith('p1_') or not (k.startswith('p2_') or k.startswith('h2h_'))
            }
            
            return player_features
            
        except Exception as e:
            logger.error(f"Error extracting features for player {player_id}: {e}")
            return {}
    
    def update_features_from_match(self, match_info):
        """
        Update player features based on a new match result.
        
        Args:
            match_info (dict): Information about the new match including:
                - player1_id, player2_id: IDs of the players
                - winner_id: ID of the match winner
                - match_date: Date of the match
                - surface: Surface of the match
                - tournament_id: ID of the tournament
                - tournament_level: Level of the tournament
                - match_stats: Dictionary with match statistics
                
        Returns:
            dict: Dictionary with updated features for both players
        """
        if not match_info or 'player1_id' not in match_info or 'player2_id' not in match_info:
            logger.error("Invalid match info provided")
            return {}
        
        player1_id = str(match_info['player1_id'])
        player2_id = str(match_info['player2_id'])
        
        # Ensure players are initialized
        if player1_id not in self.player_features:
            logger.info(f"Player {player1_id} not initialized. Initializing...")
            self.initialize_player_features([int(player1_id)])
        
        if player2_id not in self.player_features:
            logger.info(f"Player {player2_id} not initialized. Initializing...")
            self.initialize_player_features([int(player2_id)])
        
        # Use feature extractor to calculate updates if available
        updated_features = {}
        if self.feature_extractor:
            try:
                feature_changes = self.feature_extractor.update_features(match_info)
                
                # Apply changes to player1
                p1_changes = feature_changes.get('player1', {})
                if player1_id in self.player_features:
                    self._apply_feature_changes(player1_id, p1_changes, match_info)
                    updated_features[player1_id] = self.player_features[player1_id]
                
                # Apply changes to player2
                p2_changes = feature_changes.get('player2', {})
                if player2_id in self.player_features:
                    self._apply_feature_changes(player2_id, p2_changes, match_info)
                    updated_features[player2_id] = self.player_features[player2_id]
                
            except Exception as e:
                logger.error(f"Error using feature extractor for updates: {e}")
                logger.info("Falling back to manual feature updates")
                updated_features = self._manual_feature_update(match_info)
        else:
            # If no feature extractor, use manual update method
            updated_features = self._manual_feature_update(match_info)
        
        # Add to match buffer
        self.match_buffer.append(match_info)
        
        # Trim buffer if too large
        if len(self.match_buffer) > 1000:
            self.match_buffer = self.match_buffer[-1000:]
        
        # Update last update timestamp
        current_time = datetime.now()
        self.last_update[player1_id] = current_time
        self.last_update[player2_id] = current_time
        
        # Record update in history
        match_date = pd.to_datetime(match_info.get('match_date', pd.Timestamp.now()))
        update_key = f"{match_date.strftime('%Y%m%d')}_{player1_id}_{player2_id}"
        
        self.update_history[update_key] = {
            'match_info': {k: v for k, v in match_info.items() if k != 'match_stats'},
            'update_time': current_time.isoformat(),
            'player1_id': player1_id,
            'player2_id': player2_id
        }
        
        logger.info(f"Features updated for players {player1_id} and {player2_id}")
        return updated_features
    
    def _apply_feature_changes(self, player_id, changes, match_info):
        """
        Apply feature changes to a player's feature set.
        
        Args:
            player_id (str): ID of the player
            changes (dict): Dictionary of feature changes
            match_info (dict): Information about the match
        """
        if not changes:
            return
        
        # Initialize tracking of changes
        changes_applied = []
        
        # Apply each change
        for feature, value in changes.items():
            if feature in self.player_features[player_id]:
                old_value = self.player_features[player_id][feature]
                
                # Determine how to apply the change based on feature type
                if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                    if feature.endswith('_count') or feature.endswith('_matches'):
                        # For count features, increment
                        self.player_features[player_id][feature] = old_value + value
                    elif '_ratio' in feature or '_rate' in feature or 'win_prob' in feature:
                        # For ratio features, compute weighted average
                        # The weight for new values decreases with more matches
                        match_count = match_info.get('match_count', 20)  # Default weight
                        new_value = ((old_value * match_count) + value) / (match_count + 1)
                        self.player_features[player_id][feature] = new_value
                    elif 'streak' in feature:
                        # For streak features, use the provided value directly
                        # (positive for wins, negative for losses)
                        current_streak = old_value
                        
                        # Update streak based on win/loss
                        if (value > 0 and current_streak > 0) or (value < 0 and current_streak < 0):
                            # Continuing streak
                            self.player_features[player_id][feature] = current_streak + value
                        else:
                            # Streak ended, starting new one
                            self.player_features[player_id][feature] = value
                    else:
                        # For other numerical features, use the provided value
                        self.player_features[player_id][feature] = value
                else:
                    # For non-numerical features, replace with new value
                    self.player_features[player_id][feature] = value
                
                changes_applied.append(feature)
        
        logger.debug(f"Applied {len(changes_applied)} feature changes for player {player_id}")
    
    def _manual_feature_update(self, match_info):
        """
        Manually update features without using the feature extractor.
        
        Args:
            match_info (dict): Information about the new match
            
        Returns:
            dict: Dictionary with updated features for both players
        """
        player1_id = str(match_info['player1_id'])
        player2_id = str(match_info['player2_id'])
        winner_id = str(match_info.get('winner_id', player1_id))
        
        updated_features = {}
        
        # Basic updates for both players
        for player_id in [player1_id, player2_id]:
            if player_id not in self.player_features:
                continue
                
            is_winner = (player_id == winner_id)
            updates = {}
            
            # Update win/loss stats
            updates[f'recent_win_count'] = 1 if is_winner else 0
            updates[f'recent_match_count'] = 1
            
            # Update streak
            current_streak = self.player_features[player_id].get('current_streak', 0)
            if is_winner:
                if current_streak >= 0:
                    updates['current_streak'] = current_streak + 1
                else:
                    updates['current_streak'] = 1
            else:
                if current_streak <= 0:
                    updates['current_streak'] = current_streak - 1
                else:
                    updates['current_streak'] = -1
            
            # Update surface stats if available
            if 'surface' in match_info and match_info['surface']:
                surface = match_info['surface'].lower()
                updates[f'recent_{surface}_match_count'] = 1
                updates[f'recent_{surface}_win_count'] = 1 if is_winner else 0
            
            # Apply updates
            self._apply_feature_changes(player_id, updates, match_info)
            updated_features[player_id] = self.player_features[player_id]
        
        return updated_features
    
    def batch_update_from_matches(self, matches_df):
        """
        Update features from a batch of new matches.
        
        Args:
            matches_df (pd.DataFrame): DataFrame containing new matches
            
        Returns:
            dict: Dictionary with counts of updates by player
        """
        if matches_df.empty:
            logger.warning("Empty matches DataFrame provided")
            return {}
        
        # Sort matches by date to process them chronologically
        sorted_matches = matches_df.sort_values('tournament_date')
        
        update_counts = {}
        
        # Process each match
        for _, match in sorted_matches.iterrows():
            # Convert match row to match_info dictionary
            match_info = {
                'player1_id': match['winner_id'],
                'player2_id': match['loser_id'],
                'winner_id': match['winner_id'],
                'match_date': match['tournament_date'],
                'surface': match.get('surface'),
                'tournament_id': match.get('tourney_id'),
                'tournament_level': match.get('tourney_level')
            }
            
            # Add match statistics if available
            match_stats = {}
            for col in match.index:
                if col.startswith(('w_', 'l_')):
                    match_stats[col] = match[col]
            
            if match_stats:
                match_info['match_stats'] = match_stats
            
            # Update features
            updated = self.update_features_from_match(match_info)
            
            # Track update counts
            for player_id in updated:
                if player_id in update_counts:
                    update_counts[player_id] += 1
                else:
                    update_counts[player_id] = 1
        
        logger.info(f"Batch update completed for {len(update_counts)} players")
        return update_counts
    
    def save_player_features(self, filename=None):
        """
        Save the current player features to a file.
        
        Args:
            filename (str, optional): Path to save the features
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.player_features:
            logger.warning("No player features to save")
            return False
        
        # Auto-generate filename if not provided
        if filename is None:
            if self.feature_store_path:
                os.makedirs(self.feature_store_path, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(self.feature_store_path, f"player_features_{timestamp}.json")
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"player_features_{timestamp}.json"
        
        try:
            # Prepare data for serialization
            serializable_features = {}
            
            for player_id, features in self.player_features.items():
                # Convert numpy and pandas objects to native types
                player_features = {}
                for key, value in features.items():
                    if isinstance(value, (np.integer, np.floating)):
                        player_features[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        player_features[key] = value.tolist()
                    else:
                        player_features[key] = value
                
                serializable_features[player_id] = player_features
            
            # Add metadata
            save_data = {
                'features': serializable_features,
                'last_update': {pid: ts.isoformat() for pid, ts in self.last_update.items()},
                'saved_at': datetime.now().isoformat(),
                'player_count': len(serializable_features)
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"Player features saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving player features: {e}")
            return False
    
    def load_player_features(self, filename):
        """
        Load player features from a file.
        
        Args:
            filename (str): Path to the features file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                load_data = json.load(f)
            
            # Extract features
            if 'features' in load_data:
                self.player_features = load_data['features']
            else:
                # Legacy format - the whole file is features
                self.player_features = load_data
            
            # Extract last update times if available
            if 'last_update' in load_data:
                self.last_update = {
                    pid: datetime.fromisoformat(ts)
                    for pid, ts in load_data['last_update'].items()
                }
            else:
                # Set default time
                now = datetime.now()
                self.last_update = {pid: now for pid in self.player_features}
            
            logger.info(f"Loaded features for {len(self.player_features)} players from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading player features: {e}")
            return False
    
    def get_player_features(self, player_id, ensure_recent=True, max_age_days=30):
        """
        Get features for a specific player, ensuring they are recent if requested.
        
        Args:
            player_id (int or str): ID of the player
            ensure_recent (bool): Whether to check feature freshness
            max_age_days (int): Maximum age of features in days
            
        Returns:
            dict: Dictionary of player features
        """
        player_id = str(player_id)
        
        # Check if player features exist
        if player_id not in self.player_features:
            logger.warning(f"No features found for player {player_id}. Initializing...")
            self.initialize_player_features([int(player_id)])
            
            if player_id not in self.player_features:
                logger.error(f"Failed to initialize features for player {player_id}")
                return {}
        
        # Check feature freshness if requested
        if ensure_recent and player_id in self.last_update:
            last_update_time = self.last_update[player_id]
            age = (datetime.now() - last_update_time).days
            
            if age > max_age_days:
                logger.warning(f"Features for player {player_id} are {age} days old. Refreshing...")
                self.initialize_player_features([int(player_id)], force_recalculate=True)
        
        return self.player_features.get(player_id, {})
    
    def get_player_feature_history(self, player_id):
        """
        Get the history of feature updates for a player.
        
        Args:
            player_id (int or str): ID of the player
            
        Returns:
            dict: Dictionary of update history for the player
        """
        player_id = str(player_id)
        player_history = {}
        
        # Filter update history for this player
        for update_key, update_info in self.update_history.items():
            if update_info['player1_id'] == player_id or update_info['player2_id'] == player_id:
                player_history[update_key] = update_info
        
        return player_history