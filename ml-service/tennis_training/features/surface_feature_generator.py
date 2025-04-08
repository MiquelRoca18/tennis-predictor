"""
Surface-specific feature generator for tennis match prediction.

This module provides specialized functionality for extracting detailed
features for different tennis court surfaces, considering their unique characteristics.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SurfaceFeatureGenerator:
    """
    Class for generating detailed surface-specific features for tennis prediction.
    
    This class provides methods to extract specialized features for different
    tennis court surfaces (hard, clay, grass, carpet), considering the unique
    playing characteristics of each surface.
    """
    
    def __init__(self, matches_df=None, players_df=None):
        """
        Initialize the SurfaceFeatureGenerator.
        
        Args:
            matches_df (pd.DataFrame, optional): DataFrame containing match data
            players_df (pd.DataFrame, optional): DataFrame containing player information
        """
        self.matches_df = matches_df
        self.players_df = players_df
        
        # Surface-specific threshold values
        self.surface_thresholds = {
            'hard': {
                'ace_rate_high': 0.08,
                'df_rate_high': 0.05,
                'first_serve_win_high': 0.75,
                'second_serve_win_high': 0.55,
                'return_win_high': 0.35,
                'rally_length_threshold': 5
            },
            'clay': {
                'ace_rate_high': 0.05,
                'df_rate_high': 0.04,
                'first_serve_win_high': 0.70,
                'second_serve_win_high': 0.50,
                'return_win_high': 0.40,
                'rally_length_threshold': 8
            },
            'grass': {
                'ace_rate_high': 0.10,
                'df_rate_high': 0.06,
                'first_serve_win_high': 0.80,
                'second_serve_win_high': 0.60,
                'return_win_high': 0.30,
                'rally_length_threshold': 4
            },
            'carpet': {
                'ace_rate_high': 0.09,
                'df_rate_high': 0.05,
                'first_serve_win_high': 0.78,
                'second_serve_win_high': 0.58,
                'return_win_high': 0.32,
                'rally_length_threshold': 5
            }
        }
        
        logger.info("SurfaceFeatureGenerator initialized")
    
    def set_data(self, matches_df=None, players_df=None):
        """
        Set or update the data sources.
        
        Args:
            matches_df (pd.DataFrame, optional): DataFrame containing match data
            players_df (pd.DataFrame, optional): DataFrame containing player information
        """
        if matches_df is not None:
            self.matches_df = matches_df
        if players_df is not None:
            self.players_df = players_df
            
        logger.info("Data sources updated")
    
    def generate_surface_features(self, player_id, surface=None):
        """
        Generate detailed surface-specific features for a player.
        
        Args:
            player_id (int): ID of the player
            surface (str, optional): Specific surface to generate features for.
                If None, generates features for all surfaces.
                
        Returns:
            dict: Dictionary of surface-specific features
        """
        if self.matches_df is None:
            logger.error("Match data not set. Call set_data() first.")
            return {}
        
        # If surface is specified, generate only those features
        if surface is not None:
            if surface.lower() not in ['hard', 'clay', 'grass', 'carpet']:
                logger.error(f"Invalid surface: {surface}. Must be hard, clay, grass, or carpet.")
                return {}
                
            surfaces = [surface.lower()]
        else:
            # Generate features for all surfaces
            surfaces = ['hard', 'clay', 'grass', 'carpet']
        
        features = {}
        
        # Generate general surface preference feature
        surface_preference = self._calculate_surface_preference(player_id)
        features.update(surface_preference)
        
        # Generate detailed features for each requested surface
        for surface in surfaces:
            surface_features = self._generate_features_for_surface(player_id, surface)
            features.update(surface_features)
        
        logger.info(f"Generated {len(features)} surface-specific features for player {player_id}")
        return features
    
    def _calculate_surface_preference(self, player_id):
        """
        Calculate which surface the player performs best on.
        
        Args:
            player_id (int): ID of the player
            
        Returns:
            dict: Dictionary of surface preference features
        """
        features = {}
        
        # Filter matches for this player
        player_matches = self.matches_df[
            (self.matches_df['winner_id'] == player_id) | 
            (self.matches_df['loser_id'] == player_id)
        ]
        
        if player_matches.empty:
            logger.warning(f"No matches found for player {player_id}")
            return features
        
        # Calculate win percentage by surface
        surface_stats = {}
        
        for surface in ['hard', 'clay', 'grass', 'carpet']:
            surface_matches = player_matches[player_matches['surface'].str.lower() == surface]
            
            if len(surface_matches) >= 5:  # Minimum matches threshold
                wins = len(surface_matches[surface_matches['winner_id'] == player_id])
                total = len(surface_matches)
                win_pct = wins / total if total > 0 else 0
                
                surface_stats[surface] = {
                    'matches': total,
                    'wins': wins,
                    'win_pct': win_pct
                }
                
                features[f'surface_{surface}_match_count'] = total
                features[f'surface_{surface}_win_ratio'] = win_pct
        
        # Determine preferred surface (highest win percentage)
        if surface_stats:
            best_surface = max(surface_stats.items(), key=lambda x: x[1]['win_pct'])
            features['preferred_surface'] = best_surface[0]
            features['preferred_surface_win_ratio'] = best_surface[1]['win_pct']
            
            # Calculate relative performance across surfaces
            avg_win_pct = sum(s['win_pct'] for s in surface_stats.values()) / len(surface_stats)
            
            for surface, stats in surface_stats.items():
                # How much better/worse the player does on this surface vs. their average
                relative_performance = stats['win_pct'] / avg_win_pct if avg_win_pct > 0 else 1.0
                features[f'surface_{surface}_relative_performance'] = relative_performance
        
        return features
    
    def _generate_features_for_surface(self, player_id, surface):
        """
        Generate detailed features for a specific surface.
        
        Args:
            player_id (int): ID of the player
            surface (str): Surface to generate features for
            
        Returns:
            dict: Dictionary of surface-specific features
        """
        features = {}
        surface_key = surface.lower()
        
        # Filter matches for this player on this surface
        player_matches = self.matches_df[
            ((self.matches_df['winner_id'] == player_id) | 
             (self.matches_df['loser_id'] == player_id)) &
            (self.matches_df['surface'].str.lower() == surface_key)
        ]
        
        if player_matches.empty:
            logger.warning(f"No matches found for player {player_id} on {surface}")
            return features
        
        # Surface-specific win/loss record
        features[f'{surface_key}_match_count'] = len(player_matches)
        wins = len(player_matches[player_matches['winner_id'] == player_id])
        features[f'{surface_key}_win_count'] = wins
        features[f'{surface_key}_win_ratio'] = wins / len(player_matches)
        
        # Recent form on this surface (last 10 matches)
        recent_matches = player_matches.sort_values('tournament_date', ascending=False).head(10)
        if not recent_matches.empty:
            recent_wins = len(recent_matches[recent_matches['winner_id'] == player_id])
            features[f'{surface_key}_recent_win_ratio'] = recent_wins / len(recent_matches)
        
        # Extract serve and return stats
        serve_return_features = self._extract_serve_return_stats(player_id, player_matches, surface_key)
        features.update(serve_return_features)
        
        # Extract game style features
        style_features = self._extract_game_style_features(player_id, player_matches, surface_key)
        features.update(style_features)
        
        # Extract matchup features
        matchup_features = self._extract_matchup_features(player_id, surface_key)
        features.update(matchup_features)
        
        # Extract tournament performance
        tournament_features = self._extract_tournament_features(player_id, player_matches, surface_key)
        features.update(tournament_features)
        
        return features
    
    def _extract_serve_return_stats(self, player_id, matches, surface):
        """
        Extract serve and return statistics for a specific surface.
        
        Args:
            player_id (int): ID of the player
            matches (pd.DataFrame): DataFrame with matches on the specific surface
            surface (str): Surface key
            
        Returns:
            dict: Dictionary of serve and return features
        """
        features = {}
        thresholds = self.surface_thresholds.get(surface, {})
        
        # Separate matches where player was winner or loser
        as_winner = matches[matches['winner_id'] == player_id]
        as_loser = matches[matches['loser_id'] == player_id]
        
        # Calculate serve statistics
        ace_rate = 0
        df_rate = 0
        first_serve_in = 0
        first_serve_win = 0
        second_serve_win = 0
        
        if 'w_ace' in matches.columns and 'w_svpt' in matches.columns:
            # Aces when player was winner
            w_aces = as_winner['w_ace'].sum() if not as_winner.empty else 0
            w_svpt = as_winner['w_svpt'].sum() if not as_winner.empty else 0
            
            # Aces when player was loser
            l_aces = as_loser['l_ace'].sum() if not as_loser.empty else 0
            l_svpt = as_loser['l_svpt'].sum() if not as_loser.empty else 0
            
            # Calculate ace rate
            total_aces = w_aces + l_aces
            total_svpt = w_svpt + l_svpt
            
            if total_svpt > 0:
                ace_rate = total_aces / total_svpt
        
        if 'w_df' in matches.columns and 'w_svpt' in matches.columns:
            # Double faults when player was winner
            w_dfs = as_winner['w_df'].sum() if not as_winner.empty else 0
            w_svpt = as_winner['w_svpt'].sum() if not as_winner.empty else 0
            
            # Double faults when player was loser
            l_dfs = as_loser['l_df'].sum() if not as_loser.empty else 0
            l_svpt = as_loser['l_svpt'].sum() if not as_loser.empty else 0
            
            # Calculate double fault rate
            total_dfs = w_dfs + l_dfs
            total_svpt = w_svpt + l_svpt
            
            if total_svpt > 0:
                df_rate = total_dfs / total_svpt
        
        # First serve percentage
        if 'w_1stIn' in matches.columns and 'w_svpt' in matches.columns:
            w_1stIn = as_winner['w_1stIn'].sum() if not as_winner.empty else 0
            w_svpt = as_winner['w_svpt'].sum() if not as_winner.empty else 0
            
            l_1stIn = as_loser['l_1stIn'].sum() if not as_loser.empty else 0
            l_svpt = as_loser['l_svpt'].sum() if not as_loser.empty else 0
            
            total_1stIn = w_1stIn + l_1stIn
            total_svpt = w_svpt + l_svpt
            
            if total_svpt > 0:
                first_serve_in = total_1stIn / total_svpt
        
        # First serve win percentage
        if 'w_1stWon' in matches.columns and 'w_1stIn' in matches.columns:
            w_1stWon = as_winner['w_1stWon'].sum() if not as_winner.empty else 0
            w_1stIn = as_winner['w_1stIn'].sum() if not as_winner.empty else 0
            
            l_1stWon = as_loser['l_1stWon'].sum() if not as_loser.empty else 0
            l_1stIn = as_loser['l_1stIn'].sum() if not as_loser.empty else 0
            
            total_1stWon = w_1stWon + l_1stWon
            total_1stIn = w_1stIn + l_1stIn
            
            if total_1stIn > 0:
                first_serve_win = total_1stWon / total_1stIn
        
        # Second serve win percentage
        if ('w_2ndWon' in matches.columns and 'w_svpt' in matches.columns and 
            'w_1stIn' in matches.columns):
            w_2ndWon = as_winner['w_2ndWon'].sum() if not as_winner.empty else 0
            w_svpt = as_winner['w_svpt'].sum() if not as_winner.empty else 0
            w_1stIn = as_winner['w_1stIn'].sum() if not as_winner.empty else 0
            
            l_2ndWon = as_loser['l_2ndWon'].sum() if not as_loser.empty else 0
            l_svpt = as_loser['l_svpt'].sum() if not as_loser.empty else 0
            l_1stIn = as_loser['l_1stIn'].sum() if not as_loser.empty else 0
            
            w_2ndServes = w_svpt - w_1stIn
            l_2ndServes = l_svpt - l_1stIn
            
            total_2ndWon = w_2ndWon + l_2ndWon
            total_2ndServes = w_2ndServes + l_2ndServes
            
            if total_2ndServes > 0:
                second_serve_win = total_2ndWon / total_2ndServes
        
        # Add serve features
        features[f'{surface}_ace_rate'] = ace_rate
        features[f'{surface}_df_rate'] = df_rate
        features[f'{surface}_first_serve_in'] = first_serve_in
        features[f'{surface}_first_serve_win'] = first_serve_win
        features[f'{surface}_second_serve_win'] = second_serve_win
        
        # Add surface-specific serve classifications
        if thresholds:
            features[f'{surface}_high_ace_server'] = 1 if ace_rate >= thresholds.get('ace_rate_high', 0.08) else 0
            features[f'{surface}_high_df_server'] = 1 if df_rate >= thresholds.get('df_rate_high', 0.05) else 0
            features[f'{surface}_strong_first_serve'] = 1 if first_serve_win >= thresholds.get('first_serve_win_high', 0.75) else 0
            features[f'{surface}_strong_second_serve'] = 1 if second_serve_win >= thresholds.get('second_serve_win_high', 0.55) else 0
        
        # Calculate return statistics if data available
        return_win_pct = 0
        
        if ('w_1stWon' in matches.columns and 'l_1stWon' in matches.columns and
            'w_svpt' in matches.columns and 'l_svpt' in matches.columns and
            'w_1stIn' in matches.columns and 'l_1stIn' in matches.columns and
            'w_2ndWon' in matches.columns and 'l_2ndWon' in matches.columns):
            
            # Get return stats when player was winner (opponent was serving)
            opp_1stIn = as_winner['l_1stIn'].sum() if not as_winner.empty else 0
            opp_1stWon = as_winner['l_1stWon'].sum() if not as_winner.empty else 0
            opp_svpt = as_winner['l_svpt'].sum() if not as_winner.empty else 0
            opp_2ndWon = as_winner['l_2ndWon'].sum() if not as_winner.empty else 0
            
            # Return points won on first serve when player was winner
            return_1st_won_as_winner = opp_1stIn - opp_1stWon if opp_1stIn >= opp_1stWon else 0
            
            # Return points won on second serve when player was winner
            opp_2ndServes = opp_svpt - opp_1stIn
            return_2nd_won_as_winner = opp_2ndServes - opp_2ndWon if opp_2ndServes >= opp_2ndWon else 0
            
            # Get return stats when player was loser (opponent was serving)
            opp_1stIn = as_loser['w_1stIn'].sum() if not as_loser.empty else 0
            opp_1stWon = as_loser['w_1stWon'].sum() if not as_loser.empty else 0
            opp_svpt = as_loser['w_svpt'].sum() if not as_loser.empty else 0
            opp_2ndWon = as_loser['w_2ndWon'].sum() if not as_loser.empty else 0
            
            # Return points won on first serve when player was loser
            return_1st_won_as_loser = opp_1stIn - opp_1stWon if opp_1stIn >= opp_1stWon else 0
            
            # Return points won on second serve when player was loser
            opp_2ndServes = opp_svpt - opp_1stIn
            return_2nd_won_as_loser = opp_2ndServes - opp_2ndWon if opp_2ndServes >= opp_2ndWon else 0
            
            # Total return points won and played
            total_return_points_won = (return_1st_won_as_winner + return_2nd_won_as_winner +
                                     return_1st_won_as_loser + return_2nd_won_as_loser)
            
            # Total opponent serve points
            total_opp_serve_points = (as_winner['l_svpt'].sum() if not as_winner.empty else 0) + \
                                   (as_loser['w_svpt'].sum() if not as_loser.empty else 0)
            
            if total_opp_serve_points > 0:
                return_win_pct = total_return_points_won / total_opp_serve_points
        
        # Add return features
        features[f'{surface}_return_win_pct'] = return_win_pct
        
        # Add surface-specific return classifications
        if thresholds:
            features[f'{surface}_strong_returner'] = 1 if return_win_pct >= thresholds.get('return_win_high', 0.35) else 0
        
        # Break point conversion
        bp_conversion = 0
        
        if 'w_bpFaced' in matches.columns and 'l_bpFaced' in matches.columns:
            # Break points faced by opponents
            opp_bp_faced_as_winner = as_winner['l_bpFaced'].sum() if not as_winner.empty else 0
            opp_bp_faced_as_loser = as_loser['w_bpFaced'].sum() if not as_loser.empty else 0
            
            # Break points saved by opponents
            opp_bp_saved_as_winner = as_winner['l_bpSaved'].sum() if not as_winner.empty else 0
            opp_bp_saved_as_loser = as_loser['w_bpSaved'].sum() if not as_loser.empty else 0
            
            # Break points converted by player
            bp_converted = (opp_bp_faced_as_winner - opp_bp_saved_as_winner) + \
                         (opp_bp_faced_as_loser - opp_bp_saved_as_loser)
            
            total_bp = opp_bp_faced_as_winner + opp_bp_faced_as_loser
            
            if total_bp > 0:
                bp_conversion = bp_converted / total_bp
        
        features[f'{surface}_bp_conversion'] = bp_conversion
        
        return features
    
    def _extract_game_style_features(self, player_id, matches, surface):
        """
        Extract features related to player's game style on a specific surface.
        
        Args:
            player_id (int): ID of the player
            matches (pd.DataFrame): DataFrame with matches on the specific surface
            surface (str): Surface key
            
        Returns:
            dict: Dictionary of game style features
        """
        features = {}
        thresholds = self.surface_thresholds.get(surface, {})
        
        # Calculate average match duration if available
        if 'minutes' in matches.columns:
            avg_duration = matches['minutes'].mean() if not matches['minutes'].isna().all() else 0
            features[f'{surface}_avg_match_duration'] = avg_duration
            
            # Classify match length
            if avg_duration > 0:
                if surface == 'clay' and avg_duration > 160:
                    features[f'{surface}_long_match_player'] = 1
                elif surface == 'grass' and avg_duration < 120:
                    features[f'{surface}_quick_match_player'] = 1
                elif surface == 'hard' and avg_duration > 150:
                    features[f'{surface}_long_match_player'] = 1
                else:
                    features[f'{surface}_long_match_player'] = 0
                    features[f'{surface}_quick_match_player'] = 0
        
        # Calculate game dominance
        total_games_won = 0
        total_games_played = 0
        
        if 'w_games' in matches.columns and 'l_games' in matches.columns:
            # Games won when player was winner
            w_games_won = matches[matches['winner_id'] == player_id]['w_games'].sum()
            w_games_lost = matches[matches['winner_id'] == player_id]['l_games'].sum()
            
            # Games won when player was loser
            l_games_won = matches[matches['loser_id'] == player_id]['l_games'].sum()
            l_games_lost = matches[matches['loser_id'] == player_id]['w_games'].sum()
            
            total_games_won = w_games_won + l_games_won
            total_games_played = total_games_won + w_games_lost + l_games_lost
            
            if total_games_played > 0:
                features[f'{surface}_game_dominance'] = total_games_won / total_games_played
        
        # Extract tiebreak performance if available
        if 'w_tiebreaks' in matches.columns and 'l_tiebreaks' in matches.columns:
            # Tiebreaks won when player was winner
            w_tb_won = matches[matches['winner_id'] == player_id]['w_tiebreaks'].sum()
            
            # Tiebreaks won when player was loser
            l_tb_won = matches[matches['loser_id'] == player_id]['l_tiebreaks'].sum()
            
            # Tiebreaks lost when player was winner
            w_tb_lost = matches[matches['winner_id'] == player_id]['l_tiebreaks'].sum()
            
            # Tiebreaks lost when player was loser
            l_tb_lost = matches[matches['loser_id'] == player_id]['w_tiebreaks'].sum()
            
            total_tb_won = w_tb_won + l_tb_won
            total_tb_played = total_tb_won + w_tb_lost + l_tb_lost
            
            if total_tb_played > 0:
                features[f'{surface}_tiebreak_win_ratio'] = total_tb_won / total_tb_played
        
        # Classify player style based on serve & return stats
        if (f'{surface}_ace_rate' in features and 
            f'{surface}_return_win_pct' in features):
            
            ace_rate = features[f'{surface}_ace_rate']
            return_win_pct = features[f'{surface}_return_win_pct']
            
            # Classification based on surface characteristics
            if surface == 'grass':
                if ace_rate >= 0.10 and return_win_pct < 0.30:
                    style = 'serve_dominant'
                elif ace_rate < 0.05 and return_win_pct >= 0.40:
                    style = 'return_dominant'
                elif ace_rate >= 0.07 and return_win_pct >= 0.35:
                    style = 'all_court'
                else:
                    style = 'balanced'
            elif surface == 'clay':
                if ace_rate >= 0.07 and return_win_pct < 0.35:
                    style = 'serve_dominant'
                elif ace_rate < 0.04 and return_win_pct >= 0.45:
                    style = 'return_dominant'
                elif ace_rate >= 0.05 and return_win_pct >= 0.40:
                    style = 'all_court'
                else:
                    style = 'balanced'
            else:  # hard, carpet
                if ace_rate >= 0.08 and return_win_pct < 0.32:
                    style = 'serve_dominant'
                elif ace_rate < 0.05 and return_win_pct >= 0.38:
                    style = 'return_dominant'
                elif ace_rate >= 0.06 and return_win_pct >= 0.36:
                    style = 'all_court'
                else:
                    style = 'balanced'
            
            features[f'{surface}_play_style'] = style
            
            # One-hot encode the style
            features[f'{surface}_style_serve_dominant'] = 1 if style == 'serve_dominant' else 0
            features[f'{surface}_style_return_dominant'] = 1 if style == 'return_dominant' else 0
            features[f'{surface}_style_all_court'] = 1 if style == 'all_court' else 0
            features[f'{surface}_style_balanced'] = 1 if style == 'balanced' else 0
        
        return features
    
    def _extract_matchup_features(self, player_id, surface):
        """
        Extract features related to player matchups on a specific surface.
        
        Args:
            player_id (int): ID of the player
            surface (str): Surface key
            
        Returns:
            dict: Dictionary of matchup features
        """
        features = {}
        
        if self.matches_df is None:
            return features
        
        # Filter surface-specific matches for this player
        surface_matches = self.matches_df[
            ((self.matches_df['winner_id'] == player_id) | 
             (self.matches_df['loser_id'] == player_id)) &
            (self.matches_df['surface'].str.lower() == surface)
        ]
        
        if surface_matches.empty:
            return features
        
        # Calculate performance against different play styles if we have player data
        if self.players_df is not None and 'hand' in self.players_df.columns:
            # Performance against left-handed players
            left_handed_opponents = self.players_df[self.players_df['hand'] == 'L']['player_id'].tolist()
            
            if left_handed_opponents:
                left_matches = surface_matches[
                    ((surface_matches['winner_id'].isin(left_handed_opponents)) & (surface_matches['loser_id'] == player_id)) |
                    ((surface_matches['loser_id'].isin(left_handed_opponents)) & (surface_matches['winner_id'] == player_id))
                ]
                
                if not left_matches.empty:
                    left_matches_count = len(left_matches)
                    left_matches_won = len(left_matches[left_matches['winner_id'] == player_id])
                    
                    features[f'{surface}_vs_lefty_match_count'] = left_matches_count
                    features[f'{surface}_vs_lefty_win_ratio'] = left_matches_won / left_matches_count if left_matches_count > 0 else 0
        
        # Calculate performance against top players
        if 'loser_rank' in surface_matches.columns and 'winner_rank' in surface_matches.columns:
            # Matches against top 10 players
            top10_matches = surface_matches[
                ((surface_matches['winner_rank'] <= 10) & (surface_matches['loser_id'] == player_id)) |
                ((surface_matches['loser_rank'] <= 10) & (surface_matches['winner_id'] == player_id))
            ]
            
            if not top10_matches.empty:
                top10_matches_count = len(top10_matches)
                top10_matches_won = len(top10_matches[top10_matches['winner_id'] == player_id])
                
                features[f'{surface}_vs_top10_match_count'] = top10_matches_count
                features[f'{surface}_vs_top10_win_ratio'] = top10_matches_won / top10_matches_count if top10_matches_count > 0 else 0
            
            # Matches against top 20 players
            top20_matches = surface_matches[
                ((surface_matches['winner_rank'] <= 20) & (surface_matches['winner_rank'] > 10) & (surface_matches['loser_id'] == player_id)) |
                ((surface_matches['loser_rank'] <= 20) & (surface_matches['loser_rank'] > 10) & (surface_matches['winner_id'] == player_id))
            ]
            
            if not top20_matches.empty:
                top20_matches_count = len(top20_matches)
                top20_matches_won = len(top20_matches[top20_matches['winner_id'] == player_id])
                
                features[f'{surface}_vs_top11_20_match_count'] = top20_matches_count
                features[f'{surface}_vs_top11_20_win_ratio'] = top20_matches_won / top20_matches_count if top20_matches_count > 0 else 0
        
        # Calculate performance in decisive sets
        if 'w_sets' in surface_matches.columns and 'l_sets' in surface_matches.columns:
            # Find matches that went to decisive set
            decisive_matches = surface_matches[
                ((surface_matches['w_sets'] + surface_matches['l_sets']) >= 3) &  # 3+ sets played
                ((surface_matches['w_sets'] == 2) | (surface_matches['l_sets'] == 2))  # Someone won at least 2 sets
            ]
            
            if not decisive_matches.empty:
                decisive_matches_count = len(decisive_matches)
                decisive_matches_won = len(decisive_matches[decisive_matches['winner_id'] == player_id])
                
                features[f'{surface}_decisive_set_match_count'] = decisive_matches_count
                features[f'{surface}_decisive_set_win_ratio'] = decisive_matches_won / decisive_matches_count if decisive_matches_count > 0 else 0
        
        return features
    
    def _extract_tournament_features(self, player_id, matches, surface):
        """
        Extract features related to tournament performance on a specific surface.
        
        Args:
            player_id (int): ID of the player
            matches (pd.DataFrame): DataFrame with matches on the specific surface
            surface (str): Surface key
            
        Returns:
            dict: Dictionary of tournament performance features
        """
        features = {}
        
        if 'tourney_level' not in matches.columns:
            return features
        
        # Performance by tournament level
        for level in ['G', 'M', 'A', 'B', 'C', 'D', 'F']:  # Grand Slam, Masters, etc.
            level_matches = matches[matches['tourney_level'] == level]
            
            if not level_matches.empty:
                level_matches_count = len(level_matches)
                level_matches_won = len(level_matches[level_matches['winner_id'] == player_id])
                
                features[f'{surface}_level_{level}_match_count'] = level_matches_count
                features[f'{surface}_level_{level}_win_ratio'] = level_matches_won / level_matches_count
                
                # Calculate average round reached for Grand Slams and Masters
                if level in ['G', 'M'] and 'round' in level_matches.columns:
                    # Convert round to numerical value
                    round_values = {
                        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'W': 8,
                        'RR': 3  # Round robin approximately equivalent to R32
                    }
                    
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
                        features[f'{surface}_level_{level}_avg_round'] = avg_round / count
        
        # Calculate performance in specific major tournaments on this surface
        if 'tourney_id' in matches.columns:
            # Group matches by tournament
            tourney_groups = matches.groupby('tourney_id')
            
            for tourney_id, tourney_matches in tourney_groups:
                if len(tourney_matches) >= 3:  # Only consider if player has played here multiple times
                    tourney_name = tourney_matches['tourney_name'].iloc[0] if 'tourney_name' in tourney_matches.columns else f"Tournament {tourney_id}"
                    
                    tourney_matches_count = len(tourney_matches)
                    tourney_matches_won = len(tourney_matches[tourney_matches['winner_id'] == player_id])
                    
                    win_ratio = tourney_matches_won / tourney_matches_count
                    
                    # Only add specific tournament features if performance is notable
                    if win_ratio >= 0.7 or tourney_matches_count >= 10:
                        safe_name = tourney_name.lower().replace(' ', '_')[:20]
                        features[f'{surface}_tourney_{safe_name}_match_count'] = tourney_matches_count
                        features[f'{surface}_tourney_{safe_name}_win_ratio'] = win_ratio
        
        return features

    def compare_surface_performance(self, player_id):
        """
        Compare player's performance across different surfaces.
        
        Args:
            player_id (int): ID of the player
            
        Returns:
            dict: Dictionary comparing performance across surfaces
        """
        if self.matches_df is None:
            logger.error("Match data not set. Call set_data() first.")
            return {}
        
        # Generate features for all surfaces
        all_features = self.generate_surface_features(player_id)
        
        # Comparative analysis
        comparison = {}
        
        # Track win ratios across surfaces
        win_ratios = {}
        for surface in ['hard', 'clay', 'grass', 'carpet']:
            win_ratio_key = f'surface_{surface}_win_ratio'
            if win_ratio_key in all_features:
                win_ratios[surface] = all_features[win_ratio_key]
        
        if win_ratios:
            # Calculate strongest and weakest surfaces
            strongest_surface = max(win_ratios.items(), key=lambda x: x[1])
            weakest_surface = min(win_ratios.items(), key=lambda x: x[1])
            
            comparison['strongest_surface'] = strongest_surface[0]
            comparison['strongest_surface_win_ratio'] = strongest_surface[1]
            comparison['weakest_surface'] = weakest_surface[0]
            comparison['weakest_surface_win_ratio'] = weakest_surface[1]
            
            # Calculate relative surface adaptability
            if len(win_ratios) >= 2:
                win_ratio_values = list(win_ratios.values())
                adaptability = 1 - (max(win_ratio_values) - min(win_ratio_values))
                comparison['surface_adaptability'] = adaptability
                
                # Classify adaptability
                if adaptability >= 0.8:
                    comparison['adaptability_level'] = 'excellent'
                elif adaptability >= 0.6:
                    comparison['adaptability_level'] = 'good'
                elif adaptability >= 0.4:
                    comparison['adaptability_level'] = 'moderate'
                else:
                    comparison['adaptability_level'] = 'poor'
        
        # Compare serve effectiveness across surfaces
        serve_effectiveness = {}
        for surface in ['hard', 'clay', 'grass', 'carpet']:
            ace_key = f'{surface}_ace_rate'
            first_serve_key = f'{surface}_first_serve_win'
            
            if ace_key in all_features and first_serve_key in all_features:
                # Weighted combination of ace rate and first serve win percentage
                serve_effectiveness[surface] = (all_features[ace_key] * 0.4) + (all_features[first_serve_key] * 0.6)
        
        if serve_effectiveness:
            best_serve_surface = max(serve_effectiveness.items(), key=lambda x: x[1])
            comparison['best_serve_surface'] = best_serve_surface[0]
            comparison['best_serve_effectiveness'] = best_serve_surface[1]
        
        # Compare return effectiveness across surfaces
        return_effectiveness = {}
        for surface in ['hard', 'clay', 'grass', 'carpet']:
            return_key = f'{surface}_return_win_pct'
            bp_key = f'{surface}_bp_conversion'
            
            if return_key in all_features and bp_key in all_features:
                # Weighted combination of return win percentage and break point conversion
                return_effectiveness[surface] = (all_features[return_key] * 0.5) + (all_features[bp_key] * 0.5)
        
        if return_effectiveness:
            best_return_surface = max(return_effectiveness.items(), key=lambda x: x[1])
            comparison['best_return_surface'] = best_return_surface[0]
            comparison['best_return_effectiveness'] = best_return_surface[1]
        
        return comparison