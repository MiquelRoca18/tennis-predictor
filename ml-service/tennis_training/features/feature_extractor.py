"""
Feature extractor module for tennis match prediction.

This module contains the TennisFeatureExtractor class that extracts advanced
features for tennis match prediction, including head-to-head statistics,
player stats, surface metrics, and temporal features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TennisFeatureExtractor:
    """
    Class for extracting advanced features for tennis match prediction.
    
    This class provides methods to extract various types of features including:
    - Head-to-head statistics
    - Player performance statistics
    - Surface-specific metrics
    - Temporal features (recent form, fatigue, rest)
    
    The extracted features can be used as input for machine learning models
    to predict tennis match outcomes.
    """
    
    def __init__(self, matches_df, players_df, elo_ratings=None, rankings_df=None):
        """
        Initialize the TennisFeatureExtractor with necessary data sources.
        
        Args:
            matches_df (pd.DataFrame): DataFrame containing match data
            players_df (pd.DataFrame): DataFrame containing player information
            elo_ratings (dict, optional): Dictionary containing ELO ratings by player ID
            rankings_df (pd.DataFrame, optional): DataFrame containing official rankings
        """
        self.matches_df = matches_df
        self.players_df = players_df
        self.elo_ratings = elo_ratings or {}
        self.rankings_df = rankings_df
        
        # Ensure tournament_date is datetime
        if 'tournament_date' in self.matches_df.columns:
            self.matches_df['tournament_date'] = pd.to_datetime(self.matches_df['tournament_date'])
        
        # Create a copy of matches with winner/loser as player1/player2 for easier manipulation
        self._prepare_matches_view()
        
        logger.info("TennisFeatureExtractor initialized with %d matches and %d players", 
                   len(matches_df), len(players_df))
    
    def _prepare_matches_view(self):
        """
        Create a standardized view of matches with player1/player2 columns.
        This simplifies extraction of head-to-head and other comparative features.
        """
        # Create a copy to avoid modifying original data
        self.matches_view = self.matches_df.copy()
        
        # Check column names to handle different formats
        if 'winner_id' in self.matches_df.columns and 'loser_id' in self.matches_df.columns:
            # For consistency, create player1 (winner) and player2 (loser) columns
            self.matches_view['player1_id'] = self.matches_df['winner_id']
            self.matches_view['player2_id'] = self.matches_df['loser_id']
            
            # Copy other relevant columns with standardized names
            column_mappings = {
                'winner_rank': 'player1_rank',
                'loser_rank': 'player2_rank',
                'winner_rank_points': 'player1_rank_points',
                'loser_rank_points': 'player2_rank_points',
                'winner_age': 'player1_age',
                'loser_age': 'player2_age',
                'winner_ht': 'player1_height',
                'loser_ht': 'player2_height',
                'winner_hand': 'player1_hand',
                'loser_hand': 'player2_hand'
            }
            
            for src, dst in column_mappings.items():
                if src in self.matches_df.columns:
                    self.matches_view[dst] = self.matches_df[src]
        
        logger.debug("Prepared standardized matches view")
    
    def extract_features(self, player1_id, player2_id, match_date, tournament_id=None, 
                         surface=None, tournament_level=None):
        """
        Extract all features for a match between two players.
        
        Args:
            player1_id (int): ID of the first player
            player2_id (int): ID of the second player
            match_date (datetime or str): Date of the match
            tournament_id (int, optional): ID of the tournament
            surface (str, optional): Surface of the match (e.g., 'Hard', 'Clay', 'Grass')
            tournament_level (str, optional): Level of the tournament
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)
        
        features = {}
        
        # Extract basic player features
        player_features = self.extract_basic_player_features(player1_id, player2_id)
        features.update(player_features)
        
        # Extract head-to-head features
        h2h_features = self.extract_h2h_features(player1_id, player2_id, match_date, surface)
        features.update(h2h_features)
        
        # Extract player style features
        style_features = self.extract_player_style_features(player1_id, player2_id)
        features.update(style_features)
        
        # Extract ELO-based features
        elo_features = self.extract_elo_features(player1_id, player2_id, surface)
        features.update(elo_features)
        
        # Extract temporal features
        temporal_features = self.extract_temporal_features(player1_id, player2_id, match_date)
        features.update(temporal_features)
        
        # Extract tournament-specific features if tournament_id is provided
        if tournament_id is not None:
            tournament_features = self.extract_tournament_features(
                tournament_id, player1_id, player2_id, surface, tournament_level
            )
            features.update(tournament_features)
        
        logger.debug("Extracted %d features for match between players %s and %s", 
                   len(features), player1_id, player2_id)
        
        return features
    
    def extract_basic_player_features(self, player1_id, player2_id):
        """
        Extract basic features about the players.
        
        Args:
            player1_id (int): ID of the first player
            player2_id (int): ID of the second player
            
        Returns:
            dict: Dictionary containing basic player features
        """
        features = {}
        
        # Get player info from players DataFrame
        player1_info = self.players_df[self.players_df['player_id'] == player1_id].iloc[0] if player1_id in self.players_df['player_id'].values else None
        player2_info = self.players_df[self.players_df['player_id'] == player2_id].iloc[0] if player2_id in self.players_df['player_id'].values else None
        
        # Extract available basic features
        if player1_info is not None and player2_info is not None:
            # Handle different column names for different datasets
            for feature in ['hand', 'height', 'country']:
                feature_cols = [col for col in self.players_df.columns if feature in col.lower()]
                if feature_cols:
                    feature_col = feature_cols[0]
                    if not pd.isna(player1_info[feature_col]) and not pd.isna(player2_info[feature_col]):
                        if feature == 'hand':
                            # One-hot encode hand
                            features[f'player1_is_left'] = 1 if player1_info[feature_col].upper() == 'L' else 0
                            features[f'player2_is_left'] = 1 if player2_info[feature_col].upper() == 'L' else 0
                            features[f'same_handedness'] = 1 if player1_info[feature_col] == player2_info[feature_col] else 0
                        elif feature == 'height':
                            # Height difference and ratios
                            p1_height = float(player1_info[feature_col]) if not pd.isna(player1_info[feature_col]) else 0
                            p2_height = float(player2_info[feature_col]) if not pd.isna(player2_info[feature_col]) else 0
                            if p1_height > 0 and p2_height > 0:
                                features[f'height_diff'] = p1_height - p2_height
                                features[f'height_ratio'] = p1_height / p2_height if p2_height > 0 else 1
                        elif feature == 'country':
                            # Same country feature
                            features[f'same_country'] = 1 if player1_info[feature_col] == player2_info[feature_col] else 0
        
        return features
    
    def extract_h2h_features(self, player1_id, player2_id, match_date, surface=None):
        """
        Extract head-to-head features between two players.
        
        Args:
            player1_id (int): ID of the first player
            player2_id (int): ID of the second player
            match_date (datetime): Date of the match
            surface (str, optional): Surface of the match
            
        Returns:
            dict: Dictionary containing head-to-head features
        """
        features = {}
        
        # Filter matches before the current match date
        past_matches = self.matches_df[
            ((self.matches_df['winner_id'] == player1_id) & (self.matches_df['loser_id'] == player2_id) |
             (self.matches_df['winner_id'] == player2_id) & (self.matches_df['loser_id'] == player1_id)) &
            (self.matches_df['tournament_date'] < match_date)
        ]
        
        # Calculate overall head-to-head
        p1_wins = len(past_matches[past_matches['winner_id'] == player1_id])
        p2_wins = len(past_matches[past_matches['winner_id'] == player2_id])
        total_matches = p1_wins + p2_wins
        
        features['h2h_total_matches'] = total_matches
        features['h2h_p1_win_ratio'] = p1_wins / total_matches if total_matches > 0 else 0.5
        
        # Calculate surface-specific head-to-head if surface is provided
        if surface and 'surface' in self.matches_df.columns:
            surface_matches = past_matches[past_matches['surface'] == surface]
            p1_surface_wins = len(surface_matches[surface_matches['winner_id'] == player1_id])
            p2_surface_wins = len(surface_matches[surface_matches['winner_id'] == player2_id])
            total_surface_matches = p1_surface_wins + p2_surface_wins
            
            features['h2h_surface_matches'] = total_surface_matches
            features['h2h_p1_surface_win_ratio'] = p1_surface_wins / total_surface_matches if total_surface_matches > 0 else 0.5
        
        # Calculate recent head-to-head (last 2 years)
        two_years_ago = match_date - timedelta(days=730)
        recent_matches = past_matches[past_matches['tournament_date'] >= two_years_ago]
        p1_recent_wins = len(recent_matches[recent_matches['winner_id'] == player1_id])
        p2_recent_wins = len(recent_matches[recent_matches['winner_id'] == player2_id])
        total_recent_matches = p1_recent_wins + p2_recent_wins
        
        features['h2h_recent_matches'] = total_recent_matches
        features['h2h_p1_recent_win_ratio'] = p1_recent_wins / total_recent_matches if total_recent_matches > 0 else 0.5
        
        # Calculate average match stats if available
        if 'w_ace' in self.matches_df.columns and 'l_ace' in self.matches_df.columns:
            # Process matches where player1 was winner
            p1_winner_matches = past_matches[past_matches['winner_id'] == player1_id]
            p2_winner_matches = past_matches[past_matches['winner_id'] == player2_id]
            
            stat_columns = {
                'ace': ('w_ace', 'l_ace'),
                'df': ('w_df', 'l_df'),
                'svpt': ('w_svpt', 'l_svpt'),
                '1stIn': ('w_1stIn', 'l_1stIn'),
                '1stWon': ('w_1stWon', 'l_1stWon'),
                '2ndWon': ('w_2ndWon', 'l_2ndWon'),
                'bpSaved': ('w_bpSaved', 'l_bpSaved'),
                'bpFaced': ('w_bpFaced', 'l_bpFaced')
            }
            
            # Calculate average stats from matches
            for stat, (w_col, l_col) in stat_columns.items():
                if w_col in self.matches_df.columns and l_col in self.matches_df.columns:
                    # Player 1 stats
                    p1_as_winner_stat = p1_winner_matches[w_col].mean() if len(p1_winner_matches) > 0 else 0
                    p1_as_loser_stat = p2_winner_matches[l_col].mean() if len(p2_winner_matches) > 0 else 0
                    p1_matches = len(p1_winner_matches) + len(p2_winner_matches)
                    p1_avg_stat = ((p1_as_winner_stat * len(p1_winner_matches)) + 
                                  (p1_as_loser_stat * len(p2_winner_matches))) / p1_matches if p1_matches > 0 else 0
                    
                    # Player 2 stats
                    p2_as_winner_stat = p2_winner_matches[w_col].mean() if len(p2_winner_matches) > 0 else 0
                    p2_as_loser_stat = p1_winner_matches[l_col].mean() if len(p1_winner_matches) > 0 else 0
                    p2_matches = len(p2_winner_matches) + len(p1_winner_matches)
                    p2_avg_stat = ((p2_as_winner_stat * len(p2_winner_matches)) + 
                                  (p2_as_loser_stat * len(p1_winner_matches))) / p2_matches if p2_matches > 0 else 0
                    
                    # Add to features
                    features[f'h2h_p1_avg_{stat}'] = p1_avg_stat
                    features[f'h2h_p2_avg_{stat}'] = p2_avg_stat
                    features[f'h2h_diff_{stat}'] = p1_avg_stat - p2_avg_stat
        
        return features
    
    def extract_player_style_features(self, player1_id, player2_id):
        """
        Extract features related to playing styles of both players.
        
        Args:
            player1_id (int): ID of the first player
            player2_id (int): ID of the second player
            
        Returns:
            dict: Dictionary containing player style features
        """
        features = {}
        
        # Filter matches involving player1 and player2
        p1_matches = self.matches_df[(self.matches_df['winner_id'] == player1_id) | 
                                    (self.matches_df['loser_id'] == player1_id)]
        p2_matches = self.matches_df[(self.matches_df['winner_id'] == player2_id) | 
                                    (self.matches_df['loser_id'] == player2_id)]
        
        # Statistical style features
        stat_columns = {
            'ace_rate': ('w_ace', 'l_ace', 'w_svpt', 'l_svpt'),
            'df_rate': ('w_df', 'l_df', 'w_svpt', 'l_svpt'),
            '1st_serve_in': ('w_1stIn', 'l_1stIn', 'w_svpt', 'l_svpt'),
            '1st_serve_win': ('w_1stWon', 'l_1stWon', 'w_1stIn', 'l_1stIn'),
            '2nd_serve_win': ('w_2ndWon', 'l_2ndWon', 'w_svpt', 'l_svpt', 'w_1stIn', 'l_1stIn'),
            'bp_save': ('w_bpSaved', 'l_bpSaved', 'w_bpFaced', 'l_bpFaced'),
            'return_pts_win': ('w_1stWon', 'l_1stWon', 'w_2ndWon', 'l_2ndWon', 'w_svpt', 'l_svpt', 
                              'w_1stIn', 'l_1stIn')
        }
        
        # Check if required columns exist
        for stat, cols in stat_columns.items():
            if all(col in self.matches_df.columns for col in cols):
                # Calculate player 1 stats
                p1_stat = self._calculate_player_style_stat(p1_matches, player1_id, stat, cols)
                
                # Calculate player 2 stats
                p2_stat = self._calculate_player_style_stat(p2_matches, player2_id, stat, cols)
                
                # Add to features
                features[f'p1_style_{stat}'] = p1_stat
                features[f'p2_style_{stat}'] = p2_stat
                features[f'style_diff_{stat}'] = p1_stat - p2_stat
        
        # Calculate set and game dominance
        if 'w_sets' in self.matches_df.columns and 'l_sets' in self.matches_df.columns and \
           'w_games' in self.matches_df.columns and 'l_games' in self.matches_df.columns:
            
            # Player 1 dominance
            p1_set_dominance, p1_game_dominance = self._calculate_dominance(p1_matches, player1_id)
            features['p1_set_dominance'] = p1_set_dominance
            features['p1_game_dominance'] = p1_game_dominance
            
            # Player 2 dominance
            p2_set_dominance, p2_game_dominance = self._calculate_dominance(p2_matches, player2_id)
            features['p2_set_dominance'] = p2_set_dominance
            features['p2_game_dominance'] = p2_game_dominance
            
            # Difference
            features['set_dominance_diff'] = p1_set_dominance - p2_set_dominance
            features['game_dominance_diff'] = p1_game_dominance - p2_game_dominance
        
        return features
    
    def _calculate_player_style_stat(self, matches, player_id, stat, cols):
        """
        Helper method to calculate a specific style statistic for a player.
        
        Args:
            matches (pd.DataFrame): DataFrame containing matches involving the player
            player_id (int): ID of the player
            stat (str): Name of the statistic to calculate
            cols (tuple): Column names needed for calculation
            
        Returns:
            float: Calculated statistic value
        """
        # Filter matches where player was winner or loser
        as_winner = matches[matches['winner_id'] == player_id]
        as_loser = matches[matches['loser_id'] == player_id]
        
        # Calculate based on statistic type
        if stat == 'ace_rate':
            w_aces = as_winner[cols[0]].sum()
            l_aces = as_loser[cols[1]].sum()
            w_svpt = as_winner[cols[2]].sum()
            l_svpt = as_loser[cols[3]].sum()
            total_aces = w_aces + l_aces
            total_svpt = w_svpt + l_svpt
            return total_aces / total_svpt if total_svpt > 0 else 0
            
        elif stat == 'df_rate':
            w_dfs = as_winner[cols[0]].sum()
            l_dfs = as_loser[cols[1]].sum()
            w_svpt = as_winner[cols[2]].sum()
            l_svpt = as_loser[cols[3]].sum()
            total_dfs = w_dfs + l_dfs
            total_svpt = w_svpt + l_svpt
            return total_dfs / total_svpt if total_svpt > 0 else 0
            
        elif stat == '1st_serve_in':
            w_1stIn = as_winner[cols[0]].sum()
            l_1stIn = as_loser[cols[1]].sum()
            w_svpt = as_winner[cols[2]].sum()
            l_svpt = as_loser[cols[3]].sum()
            total_1stIn = w_1stIn + l_1stIn
            total_svpt = w_svpt + l_svpt
            return total_1stIn / total_svpt if total_svpt > 0 else 0
            
        elif stat == '1st_serve_win':
            w_1stWon = as_winner[cols[0]].sum()
            l_1stWon = as_loser[cols[1]].sum()
            w_1stIn = as_winner[cols[2]].sum()
            l_1stIn = as_loser[cols[3]].sum()
            total_1stWon = w_1stWon + l_1stWon
            total_1stIn = w_1stIn + l_1stIn
            return total_1stWon / total_1stIn if total_1stIn > 0 else 0
            
        elif stat == '2nd_serve_win':
            w_2ndWon = as_winner[cols[0]].sum()
            l_2ndWon = as_loser[cols[1]].sum()
            w_svpt = as_winner[cols[2]].sum()
            l_svpt = as_loser[cols[3]].sum()
            w_1stIn = as_winner[cols[4]].sum()
            l_1stIn = as_loser[cols[5]].sum()
            
            w_2ndServes = w_svpt - w_1stIn
            l_2ndServes = l_svpt - l_1stIn
            total_2ndWon = w_2ndWon + l_2ndWon
            total_2ndServes = w_2ndServes + l_2ndServes
            return total_2ndWon / total_2ndServes if total_2ndServes > 0 else 0
            
        elif stat == 'bp_save':
            w_bpSaved = as_winner[cols[0]].sum()
            l_bpSaved = as_loser[cols[1]].sum()
            w_bpFaced = as_winner[cols[2]].sum()
            l_bpFaced = as_loser[cols[3]].sum()
            total_bpSaved = w_bpSaved + l_bpSaved
            total_bpFaced = w_bpFaced + l_bpFaced
            return total_bpSaved / total_bpFaced if total_bpFaced > 0 else 1.0
            
        elif stat == 'return_pts_win':
            # For opponent's service games
            # We need to calculate return points won
            w_1stWonBy = as_winner[cols[0]].sum()  # First serve points won by player as winner
            l_1stWonBy = as_loser[cols[1]].sum()  # First serve points won by player as loser
            w_2ndWonBy = as_winner[cols[2]].sum()  # Second serve points won by player as winner
            l_2ndWonBy = as_loser[cols[3]].sum()  # Second serve points won by player as loser
            
            # Opponent's serve points
            w_opp_svpt = as_loser[cols[4]].sum()  # Total serve points by opponents when player is loser
            l_opp_svpt = as_winner[cols[5]].sum()  # Total serve points by opponents when player is winner
            w_opp_1stIn = as_loser[cols[6]].sum()  # First serves in by opponents when player is loser
            l_opp_1stIn = as_winner[cols[7]].sum()  # First serves in by opponents when player is winner
            
            # Calculate return points won
            opp_1st_serves_won = (w_opp_1stIn - l_1stWonBy) + (l_opp_1stIn - w_1stWonBy)
            opp_2nd_serves_won = ((w_opp_svpt - w_opp_1stIn) - l_2ndWonBy) + ((l_opp_svpt - l_opp_1stIn) - w_2ndWonBy)
            total_return_pts = w_opp_svpt + l_opp_svpt
            
            return_pts_won = total_return_pts - (opp_1st_serves_won + opp_2nd_serves_won)
            return return_pts_won / total_return_pts if total_return_pts > 0 else 0
            
        return 0
    
    def _calculate_dominance(self, matches, player_id):
        """
        Calculate set and game dominance for a player.
        
        Args:
            matches (pd.DataFrame): DataFrame containing matches involving the player
            player_id (int): ID of the player
            
        Returns:
            tuple: (set_dominance, game_dominance)
        """
        # Filter matches where player was winner or loser
        as_winner = matches[matches['winner_id'] == player_id]
        as_loser = matches[matches['loser_id'] == player_id]
        
        # Calculate set dominance
        w_sets_won = as_winner['w_sets'].sum() if 'w_sets' in as_winner.columns else 0
        w_sets_lost = as_winner['l_sets'].sum() if 'l_sets' in as_winner.columns else 0
        l_sets_won = as_loser['l_sets'].sum() if 'l_sets' in as_loser.columns else 0
        l_sets_lost = as_loser['w_sets'].sum() if 'w_sets' in as_loser.columns else 0
        
        total_sets_won = w_sets_won + l_sets_won
        total_sets_played = total_sets_won + w_sets_lost + l_sets_lost
        set_dominance = total_sets_won / total_sets_played if total_sets_played > 0 else 0.5
        
        # Calculate game dominance
        w_games_won = as_winner['w_games'].sum() if 'w_games' in as_winner.columns else 0
        w_games_lost = as_winner['l_games'].sum() if 'l_games' in as_winner.columns else 0
        l_games_won = as_loser['l_games'].sum() if 'l_games' in as_loser.columns else 0
        l_games_lost = as_loser['w_games'].sum() if 'w_games' in as_loser.columns else 0
        
        total_games_won = w_games_won + l_games_won
        total_games_played = total_games_won + w_games_lost + l_games_lost
        game_dominance = total_games_won / total_games_played if total_games_played > 0 else 0.5
        
        return set_dominance, game_dominance
    
    def extract_elo_features(self, player1_id, player2_id, surface=None):
        """
        Extract features based on ELO ratings for both players.
        
        Args:
            player1_id (int): ID of the first player
            player2_id (int): ID of the second player
            surface (str, optional): Surface of the match
            
        Returns:
            dict: Dictionary containing ELO-based features
        """
        features = {}
        
        # Check if we have ELO ratings
        if not self.elo_ratings:
            return features
        
        # Extract general ELO ratings
        if str(player1_id) in self.elo_ratings and 'general' in self.elo_ratings[str(player1_id)]:
            features['p1_elo_general'] = self.elo_ratings[str(player1_id)]['general']
        else:
            features['p1_elo_general'] = 1500  # Default ELO
            
        if str(player2_id) in self.elo_ratings and 'general' in self.elo_ratings[str(player2_id)]:
            features['p2_elo_general'] = self.elo_ratings[str(player2_id)]['general']
        else:
            features['p2_elo_general'] = 1500  # Default ELO
            
        features['elo_diff_general'] = features['p1_elo_general'] - features['p2_elo_general']
        
        # Extract surface-specific ELO ratings if available and surface is provided
        if surface and surface.lower() in ['hard', 'clay', 'grass', 'carpet']:
            surface_key = surface.lower()
            
            # Player 1 surface ELO
            if str(player1_id) in self.elo_ratings and surface_key in self.elo_ratings[str(player1_id)]:
                features[f'p1_elo_{surface_key}'] = self.elo_ratings[str(player1_id)][surface_key]
            else:
                features[f'p1_elo_{surface_key}'] = features['p1_elo_general']
                
            # Player 2 surface ELO
            if str(player2_id) in self.elo_ratings and surface_key in self.elo_ratings[str(player2_id)]:
                features[f'p2_elo_{surface_key}'] = self.elo_ratings[str(player2_id)][surface_key]
            else:
                features[f'p2_elo_{surface_key}'] = features['p2_elo_general']
                
            features[f'elo_diff_{surface_key}'] = features[f'p1_elo_{surface_key}'] - features[f'p2_elo_{surface_key}']
            
            # Calculate win probability based on ELO
            elo_diff = features[f'elo_diff_{surface_key}']
            features[f'p1_win_prob_{surface_key}'] = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        
        # Calculate general win probability based on ELO
        elo_diff_general = features['elo_diff_general']
        features['p1_win_prob_general'] = 1.0 / (1.0 + 10.0 ** (-elo_diff_general / 400.0))
        
        return features
    
    def extract_temporal_features(self, player1_id, player2_id, match_date):
        """
        Extract temporal features (recent form, fatigue, rest) for both players.
        
        Args:
            player1_id (int): ID of the first player
            player2_id (int): ID of the second player
            match_date (datetime): Date of the match
            
        Returns:
            dict: Dictionary containing temporal features
        """
        features = {}
        
        # Convert match_date to datetime if it's a string
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)
        
        # Define time windows for recent form
        time_windows = {
            '1m': 30,    # 1 month
            '3m': 90,    # 3 months
            '6m': 180,   # 6 months
            '1y': 365    # 1 year
        }
        
        # Extract recent form features for each time window
        for window_name, days in time_windows.items():
            window_start = match_date - timedelta(days=days)
            p1_features = self._extract_player_recent_form(player1_id, window_start, match_date)
            p2_features = self._extract_player_recent_form(player2_id, window_start, match_date)
            
            # Add prefix to features
            for feat_name, feat_value in p1_features.items():
                features[f'p1_{window_name}_{feat_name}'] = feat_value
            
            for feat_name, feat_value in p2_features.items():
                features[f'p2_{window_name}_{feat_name}'] = feat_value
            
            # Add difference features
            for feat_name in p1_features.keys():
                if feat_name in p2_features:
                    features[f'diff_{window_name}_{feat_name}'] = p1_features[feat_name] - p2_features[feat_name]
        
        # Extract rest and fatigue features
        p1_rest_fatigue = self._extract_rest_fatigue(player1_id, match_date)
        p2_rest_fatigue = self._extract_rest_fatigue(player2_id, match_date)
        
        features.update(p1_rest_fatigue)
        features.update(p2_rest_fatigue)
        
        # Calculate relative rest advantage
        if 'p1_days_since_last_match' in features and 'p2_days_since_last_match' in features:
            features['rest_advantage'] = features['p1_days_since_last_match'] - features['p2_days_since_last_match']
        
        # Calculate relative fatigue advantage
        if 'p1_fatigue_index' in features and 'p2_fatigue_index' in features:
            features['fatigue_advantage'] = features['p2_fatigue_index'] - features['p1_fatigue_index']  # Higher fatigue is worse
        
        return features
    
    def _extract_player_recent_form(self, player_id, start_date, end_date):
        """
        Extract recent form features for a player within a specific time window.
        
        Args:
            player_id (int): ID of the player
            start_date (datetime): Start date of the time window
            end_date (datetime): End date of the time window
            
        Returns:
            dict: Dictionary containing recent form features
        """
        features = {}
        
        # Filter matches involving the player within the time window
        recent_matches = self.matches_df[
            ((self.matches_df['winner_id'] == player_id) | (self.matches_df['loser_id'] == player_id)) &
            (self.matches_df['tournament_date'] >= start_date) &
            (self.matches_df['tournament_date'] < end_date)
        ]
        
        # Calculate basic form statistics
        total_matches = len(recent_matches)
        features['match_count'] = total_matches
        
        if total_matches > 0:
            wins = len(recent_matches[recent_matches['winner_id'] == player_id])
            features['win_ratio'] = wins / total_matches
            
            # Calculate streak (consecutive wins/losses)
            sorted_matches = recent_matches.sort_values('tournament_date', ascending=False)
            current_streak = 0
            for _, match in sorted_matches.iterrows():
                if match['winner_id'] == player_id:
                    if current_streak >= 0:
                        current_streak += 1
                    else:
                        break  # Streak ended
                else:  # player lost
                    if current_streak <= 0:
                        current_streak -= 1
                    else:
                        break  # Streak ended
            
            features['current_streak'] = current_streak
            
            # Calculate surface-specific win ratios if surface information is available
            if 'surface' in recent_matches.columns:
                for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
                    surface_matches = recent_matches[recent_matches['surface'] == surface]
                    surface_total = len(surface_matches)
                    
                    if surface_total > 0:
                        surface_wins = len(surface_matches[surface_matches['winner_id'] == player_id])
                        features[f'win_ratio_{surface.lower()}'] = surface_wins / surface_total
                    else:
                        features[f'win_ratio_{surface.lower()}'] = 0.5  # Neutral if no matches
            
            # Calculate level-specific win ratios if tournament level information is available
            if 'tourney_level' in recent_matches.columns:
                for level in ['G', 'M', 'A', 'D', 'F', 'C', 'S']:  # Grand Slam, Masters, etc.
                    level_matches = recent_matches[recent_matches['tourney_level'] == level]
                    level_total = len(level_matches)
                    
                    if level_total > 0:
                        level_wins = len(level_matches[level_matches['winner_id'] == player_id])
                        features[f'win_ratio_level_{level.lower()}'] = level_wins / level_total
            
            # Calculate momentum (weighted recent performance)
            if total_matches >= 3:
                sorted_matches = recent_matches.sort_values('tournament_date', ascending=True)
                momentum_score = 0
                weights_sum = 0
                
                for i, (_, match) in enumerate(sorted_matches.iterrows()):
                    # More recent matches have higher weight
                    weight = (i + 1) / sum(range(1, total_matches + 1))
                    weights_sum += weight
                    
                    if match['winner_id'] == player_id:
                        momentum_score += weight
                
                features['momentum'] = momentum_score / weights_sum if weights_sum > 0 else 0.5
            else:
                features['momentum'] = 0.5  # Neutral if not enough matches
        else:
            # Default values if no matches in the period
            features['win_ratio'] = 0.5
            features['current_streak'] = 0
            features['momentum'] = 0.5
            for surface in ['hard', 'clay', 'grass', 'carpet']:
                features[f'win_ratio_{surface}'] = 0.5
        
        return features
    
    def _extract_rest_fatigue(self, player_id, match_date):
        """
        Extract rest and fatigue features for a player.
        
        Args:
            player_id (int): ID of the player
            match_date (datetime): Date of the current match
            
        Returns:
            dict: Dictionary containing rest and fatigue features
        """
        features = {}
        prefix = f'p{1 if player_id == player1_id else 2}'
        
        # Find the player's most recent match before the current one
        player_matches = self.matches_df[
            ((self.matches_df['winner_id'] == player_id) | (self.matches_df['loser_id'] == player_id)) &
            (self.matches_df['tournament_date'] < match_date)
        ].sort_values('tournament_date', ascending=False)
        
        if len(player_matches) > 0:
            last_match = player_matches.iloc[0]
            last_match_date = last_match['tournament_date']
            
            # Calculate days since last match
            days_rest = (match_date - last_match_date).days
            features[f'{prefix}_days_since_last_match'] = days_rest
            
            # Categorize rest periods
            if days_rest < 3:
                features[f'{prefix}_rest_category'] = 0  # Short rest
            elif days_rest < 7:
                features[f'{prefix}_rest_category'] = 1  # Normal rest
            elif days_rest < 14:
                features[f'{prefix}_rest_category'] = 2  # Long rest
            else:
                features[f'{prefix}_rest_category'] = 3  # Extended rest/break
            
            # Calculate fatigue index based on recent match load
            # Look at matches in the last 30 days
            recent_matches = player_matches[player_matches['tournament_date'] >= (match_date - timedelta(days=30))]
            match_count_30d = len(recent_matches)
            
            # More weight to more recent matches and to longer/tougher matches
            fatigue_index = 0
            for _, m in recent_matches.iterrows():
                # Days ago (more recent = higher fatigue impact)
                days_ago = (match_date - m['tournament_date']).days
                recency_factor = max(0, (30 - days_ago) / 30)
                
                # Match difficulty (more sets/games = higher fatigue)
                was_winner = m['winner_id'] == player_id
                sets_played = m['w_sets'] + m['l_sets'] if 'w_sets' in m and 'l_sets' in m else 3  # Default to 3 if unknown
                
                # Calculate match toughness
                toughness = sets_played
                if 'w_games' in m and 'l_games' in m:
                    games_diff = abs(m['w_games'] - m['l_games'])
                    toughness *= (1 + (1 / max(1, games_diff)))  # Closer matches are tougher
                
                # Add to fatigue index
                fatigue_index += recency_factor * toughness
            
            # Normalize by number of matches
            if match_count_30d > 0:
                fatigue_index /= match_count_30d
            
            features[f'{prefix}_match_count_30d'] = match_count_30d
            features[f'{prefix}_fatigue_index'] = fatigue_index
            
            # Calculate performance after different rest periods
            performance_by_rest = self._calculate_performance_by_rest(player_id)
            for rest_period, perf in performance_by_rest.items():
                features[f'{prefix}_perf_after_{rest_period}d_rest'] = perf
            
            # Calculate current match rest similarity
            closest_rest_period = min(performance_by_rest.keys(), key=lambda x: abs(x - days_rest))
            features[f'{prefix}_expected_perf_current_rest'] = performance_by_rest[closest_rest_period]
        else:
            # Default values if no previous matches
            features[f'{prefix}_days_since_last_match'] = 30  # Assume well-rested
            features[f'{prefix}_rest_category'] = 3  # Extended rest
            features[f'{prefix}_match_count_30d'] = 0
            features[f'{prefix}_fatigue_index'] = 0
            features[f'{prefix}_expected_perf_current_rest'] = 0.5
        
        return features
    
    def _calculate_performance_by_rest(self, player_id):
        """
        Calculate player's historical performance after different rest periods.
        
        Args:
            player_id (int): ID of the player
            
        Returns:
            dict: Dictionary mapping rest periods to performance metrics
        """
        performance = {1: 0.5, 2: 0.5, 3: 0.5, 5: 0.5, 7: 0.5, 14: 0.5, 30: 0.5}
        matches_count = {1: 0, 2: 0, 3: 0, 5: 0, 7: 0, 14: 0, 30: 0}
        
        # Sort matches by date
        player_matches = self.matches_df[
            (self.matches_df['winner_id'] == player_id) | (self.matches_df['loser_id'] == player_id)
        ].sort_values('tournament_date')
        
        if len(player_matches) < 2:
            return performance
        
        # Calculate rest periods and outcomes
        prev_match_date = None
        for idx, match in player_matches.iterrows():
            if prev_match_date is not None:
                days_rest = (match['tournament_date'] - prev_match_date).days
                won = match['winner_id'] == player_id
                
                # Find closest rest period bucket
                closest_period = min(performance.keys(), key=lambda x: abs(x - days_rest))
                
                # Update performance for this rest period
                current_count = matches_count[closest_period]
                current_perf = performance[closest_period]
                new_perf = (current_perf * current_count + int(won)) / (current_count + 1)
                
                performance[closest_period] = new_perf
                matches_count[closest_period] += 1
            
            prev_match_date = match['tournament_date']
        
        return performance
    
    def extract_tournament_features(self, tournament_id, player1_id, player2_id, surface=None, tournament_level=None):
        """
        Extract tournament-specific features for both players.
        
        Args:
            tournament_id (int): ID of the tournament
            player1_id (int): ID of the first player
            player2_id (int): ID of the second player
            surface (str, optional): Surface of the match
            tournament_level (str, optional): Level of the tournament
            
        Returns:
            dict: Dictionary containing tournament features
        """
        features = {}
        
        # Filter past matches in this tournament
        past_tournament_matches = self.matches_df[
            (self.matches_df['tourney_id'] == tournament_id) &
            ((self.matches_df['winner_id'].isin([player1_id, player2_id])) | 
             (self.matches_df['loser_id'].isin([player1_id, player2_id])))
        ]
        
        # Calculate historical performance at this tournament
        p1_tournament_matches = past_tournament_matches[
            (past_tournament_matches['winner_id'] == player1_id) | 
            (past_tournament_matches['loser_id'] == player1_id)
        ]
        p2_tournament_matches = past_tournament_matches[
            (past_tournament_matches['winner_id'] == player2_id) | 
            (past_tournament_matches['loser_id'] == player2_id)
        ]
        
        # Player 1 tournament stats
        p1_tourney_matches = len(p1_tournament_matches)
        features['p1_tournament_match_count'] = p1_tourney_matches
        
        if p1_tourney_matches > 0:
            p1_tourney_wins = len(p1_tournament_matches[p1_tournament_matches['winner_id'] == player1_id])
            features['p1_tournament_win_ratio'] = p1_tourney_wins / p1_tourney_matches
        else:
            features['p1_tournament_win_ratio'] = 0.5
        
        # Player 2 tournament stats
        p2_tourney_matches = len(p2_tournament_matches)
        features['p2_tournament_match_count'] = p2_tourney_matches
        
        if p2_tourney_matches > 0:
            p2_tourney_wins = len(p2_tournament_matches[p2_tournament_matches['winner_id'] == player2_id])
            features['p2_tournament_win_ratio'] = p2_tourney_wins / p2_tourney_matches
        else:
            features['p2_tournament_win_ratio'] = 0.5
        
        # Calculate tournament experience difference
        features['tournament_experience_diff'] = p1_tourney_matches - p2_tourney_matches
        
        # If we have tournament level information, calculate performance at similar tournaments
        if tournament_level is not None and 'tourney_level' in self.matches_df.columns:
            p1_level_features = self._calculate_tournament_level_performance(player1_id, tournament_level)
            p2_level_features = self._calculate_tournament_level_performance(player2_id, tournament_level)
            
            for feat_name, feat_value in p1_level_features.items():
                features[f'p1_{feat_name}'] = feat_value
            
            for feat_name, feat_value in p2_level_features.items():
                features[f'p2_{feat_name}'] = feat_value
        
        # If we have surface information, calculate performance on this surface at this tournament
        if surface is not None and 'surface' in self.matches_df.columns:
            # Filter tournament matches on this surface
            surface_matches = past_tournament_matches[past_tournament_matches['surface'] == surface]
            
            # Player 1 surface stats at this tournament
            p1_surface_matches = surface_matches[
                (surface_matches['winner_id'] == player1_id) | 
                (surface_matches['loser_id'] == player1_id)
            ]
            p1_surface_count = len(p1_surface_matches)
            
            if p1_surface_count > 0:
                p1_surface_wins = len(p1_surface_matches[p1_surface_matches['winner_id'] == player1_id])
                features['p1_tournament_surface_win_ratio'] = p1_surface_wins / p1_surface_count
            else:
                features['p1_tournament_surface_win_ratio'] = 0.5
            
            # Player 2 surface stats at this tournament
            p2_surface_matches = surface_matches[
                (surface_matches['winner_id'] == player2_id) | 
                (surface_matches['loser_id'] == player2_id)
            ]
            p2_surface_count = len(p2_surface_matches)
            
            if p2_surface_count > 0:
                p2_surface_wins = len(p2_surface_matches[p2_surface_matches['winner_id'] == player2_id])
                features['p2_tournament_surface_win_ratio'] = p2_surface_wins / p2_surface_count
            else:
                features['p2_tournament_surface_win_ratio'] = 0.5
        
        return features
    
    def _calculate_tournament_level_performance(self, player_id, tournament_level):
        """
        Calculate player's performance at tournaments of a specific level.
        
        Args:
            player_id (int): ID of the player
            tournament_level (str): Level of the tournament (e.g., 'G' for Grand Slam)
            
        Returns:
            dict: Dictionary containing tournament level performance features
        """
        features = {}
        
        # Filter matches at this tournament level
        level_matches = self.matches_df[
            (self.matches_df['tourney_level'] == tournament_level) &
            ((self.matches_df['winner_id'] == player_id) | (self.matches_df['loser_id'] == player_id))
        ]
        
        level_match_count = len(level_matches)
        features['tournament_level_match_count'] = level_match_count
        
        if level_match_count > 0:
            level_wins = len(level_matches[level_matches['winner_id'] == player_id])
            features['tournament_level_win_ratio'] = level_wins / level_match_count
            
            # Calculate round progression stats if available
            if 'round' in level_matches.columns:
                # Convert round to numerical value for comparison
                round_values = {
                    'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'W': 8,
                    'RR': 3,  # Round robin approximately equivalent to R32
                    'Q1': 0, 'Q2': 0, 'Q3': 0  # Qualifying rounds
                }
                
                # Calculate average round reached
                avg_round = 0
                count = 0
                
                for _, match in level_matches.iterrows():
                    round_str = match['round']
                    if round_str in round_values:
                        if match['winner_id'] == player_id:
                            avg_round += round_values[round_str]
                        else:
                            # If player lost, they reached this round but didn't progress
                            avg_round += round_values[round_str] - 1
                        count += 1
                
                if count > 0:
                    features['avg_round_reached'] = avg_round / count
                else:
                    features['avg_round_reached'] = 0
                
                # Calculate best performance
                best_round = 0
                for _, match in level_matches.iterrows():
                    round_str = match['round']
                    if round_str in round_values:
                        round_val = round_values[round_str]
                        if match['winner_id'] == player_id and round_str == 'W':
                            # Won the tournament
                            best_round = max(best_round, round_val)
                        elif match['winner_id'] == player_id and round_str != 'W':
                            # Won this match but not the tournament
                            best_round = max(best_round, round_val)
                        else:
                            # Lost in this round
                            best_round = max(best_round, round_val - 1)
                
                features['best_round_reached'] = best_round
            
            # Calculate surface-specific performance at this level if surface data is available
            if 'surface' in level_matches.columns:
                for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
                    surface_level_matches = level_matches[level_matches['surface'] == surface]
                    surface_level_count = len(surface_level_matches)
                    
                    if surface_level_count > 0:
                        surface_level_wins = len(surface_level_matches[surface_level_matches['winner_id'] == player_id])
                        features[f'tournament_level_win_ratio_{surface.lower()}'] = surface_level_wins / surface_level_count
                    else:
                        features[f'tournament_level_win_ratio_{surface.lower()}'] = 0.5
        else:
            # Default values if no matches at this level
            features['tournament_level_win_ratio'] = 0.5
            features['avg_round_reached'] = 0
            features['best_round_reached'] = 0
            for surface in ['hard', 'clay', 'grass', 'carpet']:
                features[f'tournament_level_win_ratio_{surface}'] = 0.5
        
        return features
    
    def update_features(self, match_info):
        """
        Update features incrementally after a new match without reprocessing all data.
        
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
            dict: Updated features for both players
        """
        if not match_info or 'player1_id' not in match_info or 'player2_id' not in match_info:
            logger.error("Invalid match info provided")
            return None
        
        player1_id = match_info['player1_id']
        player2_id = match_info['player2_id']
        winner_id = match_info.get('winner_id')
        match_date = pd.to_datetime(match_info.get('match_date', pd.Timestamp.now()))
        surface = match_info.get('surface')
        tournament_id = match_info.get('tournament_id')
        tournament_level = match_info.get('tournament_level')
        
        # Create a temporary match record to update data
        temp_match = self._create_temp_match_record(match_info)
        
        # Temporarily append this match to our matches dataframe
        temp_matches_df = self.matches_df.append(temp_match, ignore_index=True)
        old_matches_df = self.matches_df.copy()
        self.matches_df = temp_matches_df
        
        # Update views
        self._prepare_matches_view()
        
        # Extract updated features for both players
        updated_features = {}
        
        # For future matches involving player1
        updated_features['player1'] = self._update_player_features(
            player1_id, match_info, is_winner=(winner_id == player1_id)
        )
        
        # For future matches involving player2
        updated_features['player2'] = self._update_player_features(
            player2_id, match_info, is_winner=(winner_id == player2_id)
        )
        
        # Restore original data
        self.matches_df = old_matches_df
        self._prepare_matches_view()
        
        logger.info(f"Features updated incrementally for players {player1_id} and {player2_id}")
        return updated_features

    def _create_temp_match_record(self, match_info):
        """
        Create a temporary match record from match info for incremental updates.
        
        Args:
            match_info (dict): Information about the new match
            
        Returns:
            pd.Series: Match record in the format of matches_df
        """
        player1_id = match_info['player1_id']
        player2_id = match_info['player2_id']
        winner_id = match_info.get('winner_id')
        
        # If winner is not specified, use player1 as default
        if winner_id is None:
            winner_id = player1_id
            
        loser_id = player2_id if winner_id == player1_id else player1_id
        
        # Create base record with essential fields
        record = {
            'winner_id': winner_id,
            'loser_id': loser_id,
            'tournament_date': pd.to_datetime(match_info.get('match_date', pd.Timestamp.now())),
            'surface': match_info.get('surface'),
            'tourney_id': match_info.get('tournament_id'),
            'tourney_level': match_info.get('tournament_level'),
        }
        
        # Add match statistics if provided
        match_stats = match_info.get('match_stats', {})
        for key, value in match_stats.items():
            record[key] = value
        
        # Ensure record has all necessary columns from matches_df
        for col in self.matches_df.columns:
            if col not in record:
                record[col] = None
        
        return pd.Series(record)

    def _update_player_features(self, player_id, match_info, is_winner=True):
        """
        Update features for a specific player after a new match.
        
        Args:
            player_id (int): ID of the player to update features for
            match_info (dict): Information about the new match
            is_winner (bool): Whether the player won the match
            
        Returns:
            dict: Dictionary of feature changes for the player
        """
        feature_changes = {}
        
        # Update head-to-head features if applicable
        opponent_id = match_info['player2_id'] if player_id == match_info['player1_id'] else match_info['player1_id']
        h2h_features = self._update_h2h_features(player_id, opponent_id, is_winner, match_info)
        feature_changes.update(h2h_features)
        
        # Update temporal features
        temporal_features = self._update_temporal_features(player_id, is_winner, match_info)
        feature_changes.update(temporal_features)
        
        # Update tournament-specific features if applicable
        if 'tournament_id' in match_info and match_info['tournament_id'] is not None:
            tournament_features = self._update_tournament_features(
                player_id, match_info['tournament_id'], is_winner, match_info
            )
            feature_changes.update(tournament_features)
        
        # Update player style metrics
        style_features = self._update_player_style_features(player_id, is_winner, match_info)
        feature_changes.update(style_features)
        
        return feature_changes

    def _update_h2h_features(self, player_id, opponent_id, is_winner, match_info):
        """
        Update head-to-head features for a player after a match.
        
        Args:
            player_id (int): ID of the player
            opponent_id (int): ID of the opponent
            is_winner (bool): Whether the player won the match
            match_info (dict): Information about the match
            
        Returns:
            dict: Updated head-to-head features
        """
        updates = {}
        
        # Basic win/loss update
        h2h_key = f'h2h_{player_id}_{opponent_id}'
        h2h_surface_key = f'h2h_{player_id}_{opponent_id}_{match_info.get("surface", "unknown")}'
        h2h_recent_key = f'h2h_recent_{player_id}_{opponent_id}'
        
        # Update total matches
        updates[f'{h2h_key}_total_matches'] = 1  # Increment
        
        # Update win ratio
        win_value = 1 if is_winner else 0
        updates[f'{h2h_key}_p1_win_ratio'] = win_value  # This is a delta to be applied
        
        # Update surface-specific if available
        if 'surface' in match_info and match_info['surface']:
            updates[f'{h2h_surface_key}_matches'] = 1  # Increment
            updates[f'{h2h_surface_key}_p1_win_ratio'] = win_value  # Delta
        
        # Update recent matches
        updates[f'{h2h_recent_key}_matches'] = 1  # Increment
        updates[f'{h2h_recent_key}_p1_win_ratio'] = win_value  # Delta
        
        # Update match statistics if available
        match_stats = match_info.get('match_stats', {})
        for stat, value in match_stats.items():
            if stat in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'bpSaved', 'bpFaced']:
                updates[f'{h2h_key}_p1_avg_{stat}'] = value  # New value to be averaged in
        
        return updates

    def _update_temporal_features(self, player_id, is_winner, match_info):
        """
        Update temporal features for a player after a match.
        
        Args:
            player_id (int): ID of the player
            is_winner (bool): Whether the player won the match
            match_info (dict): Information about the match
            
        Returns:
            dict: Updated temporal features
        """
        updates = {}
        match_date = pd.to_datetime(match_info.get('match_date', pd.Timestamp.now()))
        
        # Update recent form for different time windows
        for window_name, days in [('1m', 30), ('3m', 90), ('6m', 180), ('1y', 365)]:
            key = f'p_{player_id}_{window_name}'
            
            # Increment match count
            updates[f'{key}_match_count'] = 1  # Increment
            
            # Update win ratio
            win_value = 1 if is_winner else 0
            updates[f'{key}_win_ratio'] = win_value  # Delta to be averaged in
            
            # Update streak
            streak_key = f'{key}_current_streak'
            if is_winner:
                updates[streak_key] = 1  # Increment for win
            else:
                updates[streak_key] = -1  # Decrement for loss
            
            # Update surface-specific win ratios if available
            if 'surface' in match_info and match_info['surface']:
                surface = match_info['surface'].lower()
                updates[f'{key}_win_ratio_{surface}'] = win_value  # Delta
            
            # Update momentum (weighted recent performance)
            updates[f'{key}_momentum'] = win_value  # Delta with higher weight for recency
        
        # Update rest features - would need last match date
        last_match_date = self._get_player_last_match_date(player_id, before_date=match_date)
        if last_match_date is not None:
            days_rest = (match_date - last_match_date).days
            updates[f'p_{player_id}_days_since_last_match'] = days_rest
            
            # Categorize rest
            if days_rest < 3:
                rest_category = 0  # Short rest
            elif days_rest < 7:
                rest_category = 1  # Normal rest
            elif days_rest < 14:
                rest_category = 2  # Long rest
            else:
                rest_category = 3  # Extended rest/break
                
            updates[f'p_{player_id}_rest_category'] = rest_category
            
            # Update performance after this rest period
            updates[f'p_{player_id}_perf_after_{days_rest}d_rest'] = 1 if is_winner else 0
        
        # Update match count in last 30 days
        updates[f'p_{player_id}_match_count_30d'] = 1  # Increment
        
        # Update fatigue index
        match_difficulty = self._calculate_match_difficulty(match_info)
        updates[f'p_{player_id}_fatigue_index'] = match_difficulty  # Delta to be weighted by recency
        
        return updates

    def _get_player_last_match_date(self, player_id, before_date=None):
        """
        Get the date of the player's last match before a specific date.
        
        Args:
            player_id (int): ID of the player
            before_date (datetime, optional): Look for matches before this date
            
        Returns:
            datetime or None: Date of the last match or None if no previous matches
        """
        player_matches = self.matches_df[
            ((self.matches_df['winner_id'] == player_id) | (self.matches_df['loser_id'] == player_id))
        ]
        
        if before_date is not None:
            player_matches = player_matches[player_matches['tournament_date'] < before_date]
        
        if len(player_matches) == 0:
            return None
        
        last_match = player_matches.sort_values('tournament_date', ascending=False).iloc[0]
        return last_match['tournament_date']

    def _calculate_match_difficulty(self, match_info):
        """
        Calculate the difficulty/fatigue impact of a match.
        
        Args:
            match_info (dict): Information about the match
            
        Returns:
            float: Match difficulty score
        """
        difficulty = 1.0  # Base difficulty
        
        # Add difficulty based on match length
        match_stats = match_info.get('match_stats', {})
        
        # If we have set information
        if 'w_sets' in match_stats and 'l_sets' in match_stats:
            sets_played = match_stats['w_sets'] + match_stats['l_sets']
            difficulty *= sets_played / 3.0  # Normalize by expected 3 sets
        
        # If we have game information, close matches are more difficult
        if 'w_games' in match_stats and 'l_games' in match_stats:
            games_diff = abs(match_stats['w_games'] - match_stats['l_games'])
            difficulty *= (1 + (1 / max(1, games_diff)))  # Closer matches are tougher
        
        # Tournament level affects difficulty
        if 'tournament_level' in match_info:
            level_difficulty = {
                'G': 1.5,  # Grand Slam
                'M': 1.3,  # Masters
                'A': 1.2,  # ATP 500/250
                'D': 1.0,  # Davis Cup
                'F': 1.4,  # Tour Finals
                'C': 0.9,  # Challenger
                'S': 0.8   # Satellite/Future
            }
            difficulty *= level_difficulty.get(match_info['tournament_level'], 1.0)
        
        return difficulty

    def _update_tournament_features(self, player_id, tournament_id, is_winner, match_info):
        """
        Update tournament-specific features for a player after a match.
        
        Args:
            player_id (int): ID of the player
            tournament_id (int): ID of the tournament
            is_winner (bool): Whether the player won the match
            match_info (dict): Information about the match
            
        Returns:
            dict: Updated tournament features
        """
        updates = {}
        key = f'p_{player_id}_tournament_{tournament_id}'
        
        # Increment match count
        updates[f'{key}_match_count'] = 1
        
        # Update win ratio
        win_value = 1 if is_winner else 0
        updates[f'{key}_win_ratio'] = win_value  # Delta
        
        # Update surface-specific win ratio if available
        if 'surface' in match_info and match_info['surface']:
            surface = match_info['surface'].lower()
            updates[f'{key}_surface_{surface}_win_ratio'] = win_value  # Delta
        
        # Update round progression if available
        if 'round' in match_info:
            round_str = match_info['round']
            round_values = {
                'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'W': 8,
                'RR': 3,  # Round robin approximately equivalent to R32
                'Q1': 0, 'Q2': 0, 'Q3': 0  # Qualifying rounds
            }
            
            if round_str in round_values:
                round_val = round_values[round_str]
                if is_winner and round_str == 'W':
                    updates[f'{key}_best_round_reached'] = round_val
                elif is_winner:
                    # Won this match but not the tournament
                    updates[f'{key}_best_round_reached'] = round_val
                else:
                    # Lost in this round
                    updates[f'{key}_best_round_reached'] = round_val - 1
        
        # Update tournament level features if available
        if 'tournament_level' in match_info:
            level = match_info['tournament_level']
            level_key = f'p_{player_id}_tournament_level_{level}'
            
            # Increment match count
            updates[f'{level_key}_match_count'] = 1
            
            # Update win ratio
            updates[f'{level_key}_win_ratio'] = win_value  # Delta
            
            # Update surface-specific win ratio if available
            if 'surface' in match_info and match_info['surface']:
                surface = match_info['surface'].lower()
                updates[f'{level_key}_surface_{surface}_win_ratio'] = win_value  # Delta
        
        return updates

    def _update_player_style_features(self, player_id, is_winner, match_info):
        """
        Update player style metrics after a match.
        
        Args:
            player_id (int): ID of the player
            is_winner (bool): Whether the player won the match
            match_info (dict): Information about the match
            
        Returns:
            dict: Updated player style features
        """
        updates = {}
        match_stats = match_info.get('match_stats', {})
        
        # Skip if no detailed stats available
        if not match_stats:
            return updates
        
        # Extract relevant stats based on winner/loser status
        prefix = 'w_' if is_winner else 'l_'
        
        # Extract basic stats
        aces = match_stats.get(f'{prefix}ace', 0)
        dfs = match_stats.get(f'{prefix}df', 0)
        svpt = match_stats.get(f'{prefix}svpt', 0)
        first_in = match_stats.get(f'{prefix}1stIn', 0)
        first_won = match_stats.get(f'{prefix}1stWon', 0)
        second_won = match_stats.get(f'{prefix}2ndWon', 0)
        bp_saved = match_stats.get(f'{prefix}bpSaved', 0)
        bp_faced = match_stats.get(f'{prefix}bpFaced', 0)
        
        # Calculate and update style metrics
        key = f'p_{player_id}_style'
        
        if svpt > 0:
            # Ace rate
            updates[f'{key}_ace_rate'] = aces / svpt
            
            # Double fault rate
            updates[f'{key}_df_rate'] = dfs / svpt
            
            # 1st serve in percentage
            updates[f'{key}_1st_serve_in'] = first_in / svpt if first_in else 0
        
        if first_in > 0:
            # 1st serve win percentage
            updates[f'{key}_1st_serve_win'] = first_won / first_in if first_won else 0
        
        if svpt > first_in:
            # 2nd serve win percentage
            second_serves = svpt - first_in
            updates[f'{key}_2nd_serve_win'] = second_won / second_serves if second_won and second_serves else 0
        
        if bp_faced > 0:
            # Break point save percentage
            updates[f'{key}_bp_save'] = bp_saved / bp_faced
        
        # Set and game dominance updates
        if 'w_sets' in match_stats and 'l_sets' in match_stats and 'w_games' in match_stats and 'l_games' in match_stats:
            if is_winner:
                sets_won = match_stats['w_sets']
                sets_lost = match_stats['l_sets']
                games_won = match_stats['w_games']
                games_lost = match_stats['l_games']
            else:
                sets_won = match_stats['l_sets']
                sets_lost = match_stats['w_sets']
                games_won = match_stats['l_games']
                games_lost = match_stats['w_games']
            
            total_sets = sets_won + sets_lost
            total_games = games_won + games_lost
            
            if total_sets > 0:
                updates[f'{key}_set_dominance'] = sets_won / total_sets
            
            if total_games > 0:
                updates[f'{key}_game_dominance'] = games_won / total_games
        
        return updates