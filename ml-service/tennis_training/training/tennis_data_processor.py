"""
Tennis Data Processor

This module provides specialized functionality for processing tennis match data,
including handling of retirements, walkovers, and other tennis-specific data issues.
It also provides integration with ELO rating systems for tennis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from datetime import datetime, timedelta
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logger
logger = logging.getLogger(__name__)

class TennisDataProcessor:
    """
    Class for processing tennis-specific data and handling domain-specific issues.
    
    This class handles tennis-specific data processing tasks such as:
    - Retirement and walkover handling
    - Match outcome adjustments
    - Integration with ELO rating systems
    - Surface-specific and tournament-level data processing
    
    Attributes:
        elo_data_path (str): Path to ELO ratings data
        elo_ratings (dict): Dictionary of player ELO ratings by surface
        retirement_handling (str): Strategy for handling retirements
        data_dir (str): Directory for storing processed data
    """
    
    def __init__(
        self,
        elo_data_path: Optional[str] = None,
        retirement_handling: str = 'exclude',
        data_dir: str = 'processed_data'
    ):
        """
        Initialize the tennis data processor.
        
        Args:
            elo_data_path: Path to ELO ratings data file
            retirement_handling: Strategy for handling retirements
                Options: 'exclude', 'winner', 'progress' (based on match progress)
            data_dir: Directory for storing processed data
        """
        self.elo_data_path = elo_data_path
        self.retirement_handling = retirement_handling
        self.data_dir = data_dir
        self.elo_ratings = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load ELO ratings if path is provided
        if elo_data_path and os.path.exists(elo_data_path):
            self._load_elo_ratings()
            
        logger.info(f"Initialized TennisDataProcessor with retirement handling: {retirement_handling}")
        
    def _load_elo_ratings(self):
        """Load ELO ratings from file."""
        try:
            with open(self.elo_data_path, 'r') as f:
                self.elo_ratings = json.load(f)
            
            logger.info(f"Loaded ELO ratings for {len(self.elo_ratings.get('general', {}))} players")
        except Exception as e:
            logger.error(f"Error loading ELO ratings: {e}")
            self.elo_ratings = {
                'general': {},
                'hard': {},
                'clay': {},
                'grass': {},
                'carpet': {}
            }
    
    def handle_retirements(
        self,
        matches_df: pd.DataFrame,
        outcome_column: str = 'winner_id',
        retirement_column: str = 'retirement',
        score_column: str = 'score',
        player1_id_column: str = 'player1_id',
        player2_id_column: str = 'player2_id',
        progress_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle retirement cases in tennis matches.
        
        Args:
            matches_df: DataFrame of tennis matches
            outcome_column: Column indicating match winner
            retirement_column: Column indicating retirement (1 for retirement)
            score_column: Column with match score
            player1_id_column: Column with player 1 ID
            player2_id_column: Column with player 2 ID
            progress_threshold: Threshold for considering match progress (0-1)
            
        Returns:
            Processed DataFrame with adjusted outcomes
        """
        # Create a copy to avoid modifying the original
        processed_df = matches_df.copy()
        
        # Check if retirement column exists
        if retirement_column not in processed_df.columns:
            logger.warning(f"Retirement column '{retirement_column}' not found. Adding with default value 0.")
            processed_df[retirement_column] = 0
        
        # Get retirement matches
        retirement_matches = processed_df[processed_df[retirement_column] == 1]
        logger.info(f"Found {len(retirement_matches)} retirement matches out of {len(processed_df)} total matches")
        
        if self.retirement_handling == 'exclude':
            # Exclude retirement matches
            processed_df = processed_df[processed_df[retirement_column] != 1]
            logger.info(f"Excluded {len(retirement_matches)} retirement matches")
        
        elif self.retirement_handling == 'winner':
            # Keep as is (winner is recorded correctly)
            logger.info(f"Keeping {len(retirement_matches)} retirement matches with recorded winners")
        
        elif self.retirement_handling == 'progress':
            # Determine match progress and adjust outcome if necessary
            if score_column in processed_df.columns:
                for idx, match in retirement_matches.iterrows():
                    # Calculate match progress
                    progress = self._calculate_match_progress(match[score_column])
                    
                    if progress < progress_threshold:
                        # Match didn't progress enough, exclude it
                        processed_df.drop(idx, inplace=True)
                    # else keep the match with recorded winner
                
                logger.info(f"Processed {len(retirement_matches)} retirement matches based on match progress")
            else:
                logger.warning(f"Score column '{score_column}' not found. Keeping retirement matches as is.")
        
        return processed_df
    
    def _calculate_match_progress(self, score_str: str) -> float:
        """
        Calculate the progress of a match based on the score.
        
        Args:
            score_str: String representation of the match score
            
        Returns:
            Float value between 0 and 1 indicating match progress
        """
        if not score_str or pd.isna(score_str):
            return 0.0
        
        try:
            # Parse score (handling various formats)
            sets = score_str.split()
            
            # Remove retirement notation if present
            sets = [s for s in sets if 'RET' not in s and 'ret' not in s and 'DEF' not in s]
            
            # Count completed sets
            completed_sets = len(sets)
            
            # Determine match format (best of 3 or best of 5)
            if any(s in score_str for s in ['Australian Open', 'Roland Garros', 'Wimbledon', 'US Open', 'Grand Slam']):
                # Grand Slam men's singles (best of 5)
                total_possible_sets = 5
            else:
                # Most other matches (best of 3)
                total_possible_sets = 3
            
            # Calculate progress based on completed sets
            progress = min(completed_sets / total_possible_sets, 1.0)
            
            return progress
        
        except Exception as e:
            logger.error(f"Error calculating match progress from score '{score_str}': {e}")
            return 0.0
    
    def handle_walkovers(
        self,
        matches_df: pd.DataFrame,
        walkover_column: str = 'walkover',
        strategy: str = 'exclude'
    ) -> pd.DataFrame:
        """
        Handle walkover cases in tennis matches.
        
        Args:
            matches_df: DataFrame of tennis matches
            walkover_column: Column indicating walkover (1 for walkover)
            strategy: Strategy for handling walkovers ('exclude' or 'keep')
            
        Returns:
            Processed DataFrame with walkovers handled
        """
        # Create a copy to avoid modifying the original
        processed_df = matches_df.copy()
        
        # Check if walkover column exists
        if walkover_column not in processed_df.columns:
            logger.warning(f"Walkover column '{walkover_column}' not found. Adding with default value 0.")
            processed_df[walkover_column] = 0
        
        # Handle walkovers based on strategy
        if strategy == 'exclude':
            walkover_count = processed_df[processed_df[walkover_column] == 1].shape[0]
            processed_df = processed_df[processed_df[walkover_column] != 1]
            logger.info(f"Excluded {walkover_count} walkover matches")
        elif strategy == 'keep':
            walkover_count = processed_df[processed_df[walkover_column] == 1].shape[0]
            logger.info(f"Kept {walkover_count} walkover matches")
        else:
            logger.warning(f"Unknown walkover handling strategy: {strategy}. Using 'exclude'.")
            walkover_count = processed_df[processed_df[walkover_column] == 1].shape[0]
            processed_df = processed_df[processed_df[walkover_column] != 1]
            logger.info(f"Excluded {walkover_count} walkover matches")
        
        return processed_df
    
    def add_elo_features(
        self,
        matches_df: pd.DataFrame,
        player1_id_column: str = 'player1_id',
        player2_id_column: str = 'player2_id',
        surface_column: str = 'surface',
        date_column: str = 'match_date'
    ) -> pd.DataFrame:
        """
        Add ELO rating features to match data.
        
        Args:
            matches_df: DataFrame of tennis matches
            player1_id_column: Column with player 1 ID
            player2_id_column: Column with player 2 ID
            surface_column: Column with match surface
            date_column: Column with match date
            
        Returns:
            DataFrame with ELO features added
        """
        if not self.elo_ratings:
            logger.warning("No ELO ratings loaded. Cannot add ELO features.")
            return matches_df
        
        # Create a copy to avoid modifying the original
        enhanced_df = matches_df.copy()
        
        # Ensure date column is datetime
        if date_column in enhanced_df.columns:
            enhanced_df[date_column] = pd.to_datetime(enhanced_df[date_column])
        
        # Add ELO rating columns
        enhanced_df['player1_elo'] = np.nan
        enhanced_df['player2_elo'] = np.nan
        enhanced_df['player1_surface_elo'] = np.nan
        enhanced_df['player2_surface_elo'] = np.nan
        enhanced_df['elo_difference'] = np.nan
        enhanced_df['surface_elo_difference'] = np.nan
        enhanced_df['elo_win_probability'] = np.nan
        
        # Sort by date if available
        if date_column in enhanced_df.columns:
            enhanced_df.sort_values(date_column, inplace=True)
        
        # Process each match
        for idx, match in enhanced_df.iterrows():
            player1_id = str(match[player1_id_column])
            player2_id = str(match[player2_id_column])
            
            # Get general ELO ratings
            player1_elo = self._get_player_elo(player1_id, 'general')
            player2_elo = self._get_player_elo(player2_id, 'general')
            
            # Get surface-specific ELO ratings if available
            if surface_column in enhanced_df.columns and not pd.isna(match[surface_column]):
                surface = match[surface_column].lower()
                if surface in ['hard', 'clay', 'grass', 'carpet']:
                    player1_surface_elo = self._get_player_elo(player1_id, surface)
                    player2_surface_elo = self._get_player_elo(player2_id, surface)
                else:
                    player1_surface_elo = player1_elo
                    player2_surface_elo = player2_elo
            else:
                player1_surface_elo = player1_elo
                player2_surface_elo = player2_elo
            
            # Store ELO values
            enhanced_df.at[idx, 'player1_elo'] = player1_elo
            enhanced_df.at[idx, 'player2_elo'] = player2_elo
            enhanced_df.at[idx, 'player1_surface_elo'] = player1_surface_elo
            enhanced_df.at[idx, 'player2_surface_elo'] = player2_surface_elo
            
            # Calculate ELO differences
            enhanced_df.at[idx, 'elo_difference'] = player1_elo - player2_elo
            enhanced_df.at[idx, 'surface_elo_difference'] = player1_surface_elo - player2_surface_elo
            
            # Calculate win probability based on ELO
            enhanced_df.at[idx, 'elo_win_probability'] = self._calculate_elo_win_probability(
                player1_surface_elo, player2_surface_elo
            )
        
        logger.info(f"Added ELO features to {len(enhanced_df)} matches")
        return enhanced_df
    
    def _get_player_elo(self, player_id: str, surface: str = 'general') -> float:
        """
        Get ELO rating for a player on a specific surface.
        
        Args:
            player_id: Player ID
            surface: Playing surface
            
        Returns:
            ELO rating (default 1500 if not found)
        """
        surface = surface.lower()
        
        if surface not in self.elo_ratings:
            surface = 'general'
        
        return self.elo_ratings.get(surface, {}).get(player_id, 1500.0)
    
    def _calculate_elo_win_probability(self, player1_elo: float, player2_elo: float) -> float:
        """
        Calculate win probability based on ELO ratings.
        
        Args:
            player1_elo: ELO rating for player 1
            player2_elo: ELO rating for player 2
            
        Returns:
            Probability of player 1 winning
        """
        return 1.0 / (1.0 + 10.0 ** ((player2_elo - player1_elo) / 400.0))
    
    def update_elo_ratings(
        self,
        matches_df: pd.DataFrame,
        k_factor: Union[float, Dict[str, float]] = 32.0,
        player1_id_column: str = 'player1_id',
        player2_id_column: str = 'player2_id',
        winner_id_column: str = 'winner_id',
        surface_column: str = 'surface',
        tournament_level_column: str = 'tournament_level',
        date_column: str = 'match_date',
        save_results: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Update ELO ratings based on match results.
        
        Args:
            matches_df: DataFrame of tennis matches
            k_factor: K-factor for ELO calculations (can be dict by surface)
            player1_id_column: Column with player 1 ID
            player2_id_column: Column with player 2 ID
            winner_id_column: Column with winner ID
            surface_column: Column with match surface
            tournament_level_column: Column with tournament level
            date_column: Column with match date
            save_results: Whether to save updated ratings
            
        Returns:
            Updated ELO ratings dictionary
        """
        # Create a copy of current ratings
        updated_ratings = {
            'general': self.elo_ratings.get('general', {}).copy(),
            'hard': self.elo_ratings.get('hard', {}).copy(),
            'clay': self.elo_ratings.get('clay', {}).copy(),
            'grass': self.elo_ratings.get('grass', {}).copy(),
            'carpet': self.elo_ratings.get('carpet', {}).copy()
        }
        
        # Ensure date column is datetime if present
        if date_column in matches_df.columns:
            matches_df = matches_df.copy()
            matches_df[date_column] = pd.to_datetime(matches_df[date_column])
            matches_df.sort_values(date_column, inplace=True)
        
        # Process each match
        for idx, match in matches_df.iterrows():
            player1_id = str(match[player1_id_column])
            player2_id = str(match[player2_id_column])
            
            # Determine winner
            if winner_id_column in match:
                winner_id = str(match[winner_id_column])
                player1_won = winner_id == player1_id
            else:
                logger.warning(f"Winner column '{winner_id_column}' not found in match {idx}")
                continue
            
            # Determine surface
            if surface_column in match and not pd.isna(match[surface_column]):
                surface = match[surface_column].lower()
                if surface not in ['hard', 'clay', 'grass', 'carpet']:
                    surface = 'general'
            else:
                surface = 'general'
            
            # Determine K-factor
            if isinstance(k_factor, dict):
                k = k_factor.get(surface, k_factor.get('general', 32.0))
            else:
                k = k_factor
            
            # Adjust K-factor based on tournament level if available
            if tournament_level_column in match and not pd.isna(match[tournament_level_column]):
                level = match[tournament_level_column]
                k = self._adjust_k_factor_by_tournament(k, level)
            
            # Update general ELO
            self._update_player_elo(
                updated_ratings['general'],
                player1_id,
                player2_id,
                player1_won,
                k
            )
            
            # Update surface-specific ELO
            if surface != 'general':
                self._update_player_elo(
                    updated_ratings[surface],
                    player1_id,
                    player2_id,
                    player1_won,
                    k
                )
        
        # Save updated ratings if requested
        if save_results:
            self.elo_ratings = updated_ratings
            self._save_elo_ratings()
        
        logger.info(f"Updated ELO ratings based on {len(matches_df)} matches")
        return updated_ratings
    
    def _adjust_k_factor_by_tournament(self, base_k: float, tournament_level: str) -> float:
        """
        Adjust K-factor based on tournament level.
        
        Args:
            base_k: Base K-factor
            tournament_level: Tournament level
            
        Returns:
            Adjusted K-factor
        """
        level_adjustments = {
            'grand_slam': 1.5,
            'masters': 1.2,
            'atp500': 1.0,
            'atp250': 0.8,
            'challenger': 0.6,
            'futures': 0.4,
            'itf': 0.4
        }
        
        adjustment = level_adjustments.get(tournament_level.lower(), 1.0)
        return base_k * adjustment
    
    def _update_player_elo(
        self,
        ratings: Dict[str, float],
        player1_id: str,
        player2_id: str,
        player1_won: bool,
        k_factor: float
    ) -> None:
        """
        Update ELO ratings for two players based on match result.
        
        Args:
            ratings: Dictionary of ELO ratings
            player1_id: Player 1 ID
            player2_id: Player 2 ID
            player1_won: Whether player 1 won
            k_factor: K-factor for update
        """
        # Get current ratings
        player1_elo = ratings.get(player1_id, 1500.0)
        player2_elo = ratings.get(player2_id, 1500.0)
        
        # Calculate expected outcome
        expected_player1 = 1.0 / (1.0 + 10.0 ** ((player2_elo - player1_elo) / 400.0))
        expected_player2 = 1.0 - expected_player1
        
        # Actual outcome
        actual_player1 = 1.0 if player1_won else 0.0
        actual_player2 = 1.0 - actual_player1
        
        # Update ratings
        player1_new_elo = player1_elo + k_factor * (actual_player1 - expected_player1)
        player2_new_elo = player2_elo + k_factor * (actual_player2 - expected_player2)
        
        # Store updated ratings
        ratings[player1_id] = player1_new_elo
        ratings[player2_id] = player2_new_elo
    
    def _save_elo_ratings(self) -> None:
        """Save ELO ratings to file."""
        if not self.elo_data_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.elo_data_path = os.path.join(self.data_dir, f"elo_ratings_{timestamp}.json")
        
        try:
            with open(self.elo_data_path, 'w') as f:
                json.dump(self.elo_ratings, f, indent=2)
            
            logger.info(f"Saved ELO ratings to {self.elo_data_path}")
        except Exception as e:
            logger.error(f"Error saving ELO ratings: {e}")
    
    def visualize_elo_distribution(
        self,
        surface: str = 'general',
        figsize: Tuple[int, int] = (10, 6),
        top_n: int = 20
    ) -> plt.Figure:
        """
        Visualize the distribution of ELO ratings.
        
        Args:
            surface: Surface to visualize
            figsize: Figure size
            top_n: Number of top players to highlight
            
        Returns:
            Matplotlib figure
        """
        surface = surface.lower()
        if surface not in self.elo_ratings:
            logger.warning(f"Surface '{surface}' not found in ELO ratings. Using 'general'.")
            surface = 'general'
        
        ratings = self.elo_ratings.get(surface, {})
        
        if not ratings:
            logger.warning(f"No ratings found for surface '{surface}'")
            return None
        
        # Convert ratings to list
        rating_values = list(ratings.values())
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot distribution
        sns.histplot(rating_values, kde=True, ax=ax)
        
        # Add mean and median lines
        mean_elo = np.mean(rating_values)
        median_elo = np.median(rating_values)
        
        ax.axvline(mean_elo, color='r', linestyle='--', label=f'Mean: {mean_elo:.1f}')
        ax.axvline(median_elo, color='g', linestyle='--', label=f'Median: {median_elo:.1f}')
        
        # Set title and labels
        ax.set_title(f'Distribution of {surface.capitalize()} ELO Ratings')
        ax.set_xlabel('ELO Rating')
        ax.set_ylabel('Count')
        ax.legend()
        
        # Create a separate plot for top players
        if top_n > 0:
            # Get top players
            top_players = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Create figure
            top_fig, top_ax = plt.subplots(figsize=figsize)
            
            # Plot top players
            player_ids = [p[0] for p in top_players]
            player_ratings = [p[1] for p in top_players]
            
            # Create bar plot
            top_ax.barh(range(len(player_ids)), player_ratings, align='center')
            top_ax.set_yticks(range(len(player_ids)))
            top_ax.set_yticklabels(player_ids)
            
            # Set title and labels
            top_ax.set_title(f'Top {top_n} Players by {surface.capitalize()} ELO Rating')
            top_ax.set_xlabel('ELO Rating')
            top_ax.invert_yaxis()  # Highest rating at the top
            
            # Save plot
            top_plot_path = os.path.join(self.data_dir, f"top_players_{surface}_elo.png")
            top_fig.tight_layout()
            top_fig.savefig(top_plot_path)
            plt.close(top_fig)
            
            logger.info(f"Saved top players plot to {top_plot_path}")
        
        # Save distribution plot
        dist_plot_path = os.path.join(self.data_dir, f"elo_distribution_{surface}.png")
        fig.tight_layout()
        fig.savefig(dist_plot_path)
        
        logger.info(f"Saved ELO distribution plot to {dist_plot_path}")
        
        return fig
    
    def calculate_h2h_adjustments(
        self,
        matches_df: pd.DataFrame,
        player1_id_column: str = 'player1_id',
        player2_id_column: str = 'player2_id',
        winner_id_column: str = 'winner_id',
        date_column: str = 'match_date',
        min_matches: int = 2,
        weight_factor: float = 0.1
    ) -> pd.DataFrame:
        """
        Calculate head-to-head adjustments for ELO predictions.
        
        Args:
            matches_df: DataFrame of tennis matches
            player1_id_column: Column with player 1 ID
            player2_id_column: Column with player 2 ID
            winner_id_column: Column with winner ID
            date_column: Column with match date
            min_matches: Minimum number of H2H matches required
            weight_factor: Weight factor for H2H adjustment
            
        Returns:
            DataFrame with H2H adjustment features
        """
        # Create a copy to avoid modifying the original
        enhanced_df = matches_df.copy()
        
        # Add H2H columns
        enhanced_df['h2h_matches'] = 0
        enhanced_df['h2h_player1_wins'] = 0
        enhanced_df['h2h_player2_wins'] = 0
        enhanced_df['h2h_win_pct_player1'] = 0.5
        enhanced_df['h2h_adjustment'] = 0.0
        
        # Ensure date column is datetime if present
        if date_column in enhanced_df.columns:
            enhanced_df[date_column] = pd.to_datetime(enhanced_df[date_column])
            enhanced_df.sort_values(date_column, inplace=True)
        
        # Build H2H records
        h2h_records = {}
        
        # First pass to build complete H2H records
        for idx, match in enhanced_df.iterrows():
            player1_id = str(match[player1_id_column])
            player2_id = str(match[player2_id_column])
            
            # Create unique pair key (always sort player IDs for consistency)
            pair_key = tuple(sorted([player1_id, player2_id]))
            
            # Initialize record if needed
            if pair_key not in h2h_records:
                h2h_records[pair_key] = {
                    'matches': 0,
                    'wins': {player1_id: 0, player2_id: 0}
                }
            
            # Skip this match in the H2H calculation since we're calculating what was
            # known BEFORE this match
            continue
            
            # In a complete implementation, we would update the H2H record here
            # but we're skipping it to avoid calculating a record that includes the current match
        
        # Second pass to apply H2H info to each match
        for idx, match in enhanced_df.iterrows():
            player1_id = str(match[player1_id_column])
            player2_id = str(match[player2_id_column])
            
            # Create unique pair key (always sort for consistency)
            pair_key = tuple(sorted([player1_id, player2_id]))
            
            # Get H2H record prior to this match
            record = h2h_records.get(pair_key, {'matches': 0, 'wins': {player1_id: 0, player2_id: 0}})
            
            # Store H2H stats
            enhanced_df.at[idx, 'h2h_matches'] = record['matches']
            enhanced_df.at[idx, 'h2h_player1_wins'] = record['wins'].get(player1_id, 0)
            enhanced_df.at[idx, 'h2h_player2_wins'] = record['wins'].get(player2_id, 0)
            
            # Calculate win percentage and adjustment
            if record['matches'] >= min_matches:
                h2h_win_pct_player1 = record['wins'].get(player1_id, 0) / record['matches']
                enhanced_df.at[idx, 'h2h_win_pct_player1'] = h2h_win_pct_player1
                
                # Calculate adjustment (deviation from 0.5 win rate)
                adjustment = (h2h_win_pct_player1 - 0.5) * weight_factor
                enhanced_df.at[idx, 'h2h_adjustment'] = adjustment
            
            # Update H2H record for next matches
            if winner_id_column in match:
                winner_id = str(match[winner_id_column])
                
                # Update record
                record['matches'] += 1
                record['wins'][winner_id] = record['wins'].get(winner_id, 0) + 1
                
                # Update in the dictionary
                h2h_records[pair_key] = record
        
        logger.info(f"Added H2H adjustments to {len(enhanced_df)} matches")
        return enhanced_df
    
    def combine_elo_with_h2h(
        self,
        matches_df: pd.DataFrame,
        h2h_weight: float = 0.2
    ) -> pd.DataFrame:
        """
        Combine ELO predictions with H2H adjustments.
        
        Args:
            matches_df: DataFrame with ELO and H2H features
            h2h_weight: Weight for H2H component
            
        Returns:
            DataFrame with combined predictions
        """
        # Create a copy to avoid modifying the original
        combined_df = matches_df.copy()
        
        # Check if required columns exist
        required_columns = ['elo_win_probability', 'h2h_adjustment']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for combining ELO with H2H: {missing_columns}")
            return combined_df
        
        # Calculate combined probability
        combined_df['combined_win_probability'] = combined_df['elo_win_probability'] + (
            combined_df['h2h_adjustment'] * h2h_weight
        )
        
        # Ensure probability is between 0 and 1
        combined_df['combined_win_probability'] = combined_df['combined_win_probability'].clip(0, 1)
        
        logger.info(f"Combined ELO with H2H adjustments for {len(combined_df)} matches")
        return combined_df
    
    def preprocess_for_modeling(
        self,
        matches_df: pd.DataFrame,
        features_to_include: List[str] = None,
        target_column: str = 'winner_id',
        player1_id_column: str = 'player1_id',
        test_size: float = 0.2,
        temporal_split: bool = True,
        date_column: str = 'match_date',
        handle_retirements: bool = True,
        handle_walkovers: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Preprocess tennis matches data for modeling.
        
        Args:
            matches_df: DataFrame of tennis matches
            features_to_include: List of feature columns to include
            target_column: Column with target variable
            player1_id_column: Column with player 1 ID (target is 1 if player1 wins)
            test_size: Proportion of data for testing
            temporal_split: Whether to split data temporally
            date_column: Column with match date
            handle_retirements: Whether to handle retirement matches
            handle_walkovers: Whether to handle walkover matches
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, dates_train, dates_test)
        """
        # Create a copy to avoid modifying the original
        processed_df = matches_df.copy()
        
        # Handle retirements if requested
        if handle_retirements:
            processed_df = self.handle_retirements(processed_df)
        
        # Handle walkovers if requested
        if handle_walkovers:
            processed_df = self.handle_walkovers(processed_df)
        
        # Prepare target variable (1 if player1 wins, 0 otherwise)
        if target_column in processed_df.columns and player1_id_column in processed_df.columns:
            processed_df['target'] = (processed_df[target_column] == processed_df[player1_id_column]).astype(int)
        else:
            logger.error(f"Missing columns for target creation: {target_column} or {player1_id_column}")
            raise ValueError(f"Missing columns for target creation: {target_column} or {player1_id_column}")
        
        # Select features
        if features_to_include is None:
            # Default features (exclude non-feature columns)
            exclude_columns = [
                target_column, 'target', 'match_id', 'tournament_id', 'score',
                'player1_name', 'player2_name', 'winner_name'
            ]
            feature_columns = [
                col for col in processed_df.columns 
                if col not in exclude_columns and not col.endswith('_id')
            ]
        else:
            feature_columns = features_to_include
        
        # Check if all feature columns exist
        missing_features = [col for col in feature_columns if col not in processed_df.columns]
        if missing_features:
            logger.warning(f"Missing feature columns: {missing_features}")
            feature_columns = [col for col in feature_columns if col in processed_df.columns]
        
        # Extract features and target
        X = processed_df[feature_columns]
        y = processed_df['target'].values
        
        # Get dates for temporal split
        if date_column in processed_df.columns:
            dates = processed_df[date_column]
        else:
            logger.warning(f"Date column '{date_column}' not found. Using row indices for temporal ordering.")
            dates = pd.Series(range(len(processed_df)))
        
        # Split data
        if temporal_split:
            # Sort by date
            sorted_indices = dates.sort_values().index
            X_sorted = X.loc[sorted_indices]
            y_sorted = y[sorted_indices]
            dates_sorted = dates.loc[sorted_indices]
            
            # Calculate split point
            split_idx = int((1 - test_size) * len(X_sorted))
            
            # Split data
            X_train = X_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_train = y_sorted[:split_idx]
            y_test = y_sorted[split_idx:]
            dates_train = dates_sorted.iloc[:split_idx]
            dates_test = dates_sorted.iloc[split_idx:]
            
            logger.info(f"Temporal split: train={len(X_train)}, test={len(X_test)}")
            if date_column in processed_df.columns:
                logger.info(f"Training period: {dates_train.min()} to {dates_train.max()}")
                logger.info(f"Testing period: {dates_test.min()} to {dates_test.max()}")
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X, y, np.arange(len(X)), test_size=test_size, random_state=42
            )
            
            dates_train = dates.iloc[train_idx]
            dates_test = dates.iloc[test_idx]
            
            logger.info(f"Random split: train={len(X_train)}, test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test, dates_train, dates_test
    
    def analyze_surface_impact(
        self,
        matches_df: pd.DataFrame,
        surface_column: str = 'surface',
        elo_win_prob_column: str = 'elo_win_probability',
        target_column: str = 'target',
        player1_id_column: str = 'player1_id',
        player2_id_column: str = 'player2_id'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the impact of playing surface on match outcomes and predictions.
        
        Args:
            matches_df: DataFrame of tennis matches
            surface_column: Column with surface information
            elo_win_prob_column: Column with ELO win probability
            target_column: Column with target outcome
            player1_id_column: Column with player 1 ID
            player2_id_column: Column with player 2 ID
            
        Returns:
            Dictionary with surface analysis results
        """
        if surface_column not in matches_df.columns:
            logger.warning(f"Surface column '{surface_column}' not found.")
            return {}
        
        # Create a copy to avoid modifying the original
        df = matches_df.copy()
        
        # Ensure we have target variable
        if target_column not in df.columns:
            if 'winner_id' in df.columns and player1_id_column in df.columns:
                df['target'] = (df['winner_id'] == df[player1_id_column]).astype(int)
                target_column = 'target'
            else:
                logger.warning("Cannot determine target outcome. Returning empty results.")
                return {}
        
        # Check for prediction column
        if elo_win_prob_column not in df.columns:
            logger.warning(f"ELO win probability column '{elo_win_prob_column}' not found.")
            has_predictions = False
        else:
            has_predictions = True
        
        # Get unique surfaces
        surfaces = df[surface_column].dropna().unique()
        
        # Analyze each surface
        results = {}
        overall_accuracy = None
        
        if has_predictions:
            # Calculate overall prediction accuracy
            df['predicted_winner'] = (df[elo_win_prob_column] > 0.5).astype(int)
            overall_accuracy = np.mean(df['predicted_winner'] == df[target_column])
        
        for surface in surfaces:
            surface_df = df[df[surface_column] == surface]
            
            if len(surface_df) < 10:
                logger.info(f"Too few matches on {surface} surface: {len(surface_df)}")
                continue
            
            # Basic stats
            total_matches = len(surface_df)
            if 'winner_id' in surface_df.columns:
                player1_wins = surface_df[surface_df['winner_id'] == surface_df[player1_id_column]].shape[0]
                player1_win_rate = player1_wins / total_matches
            else:
                player1_wins = surface_df[surface_df[target_column] == 1].shape[0]
                player1_win_rate = player1_wins / total_matches
            
            # Prediction accuracy if available
            if has_predictions:
                surface_df['predicted_winner'] = (surface_df[elo_win_prob_column] > 0.5).astype(int)
                accuracy = np.mean(surface_df['predicted_winner'] == surface_df[target_column])
                accuracy_vs_overall = accuracy / overall_accuracy if overall_accuracy else None
            else:
                accuracy = None
                accuracy_vs_overall = None
            
            # Store results
            results[surface] = {
                'total_matches': total_matches,
                'player1_wins': player1_wins,
                'player1_win_rate': player1_win_rate,
                'prediction_accuracy': accuracy,
                'accuracy_vs_overall': accuracy_vs_overall
            }
        
        # Create visualization
        if has_predictions and len(results) > 1:
            self._plot_surface_analysis(results)
        
        return results
    
    def _plot_surface_analysis(self, surface_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Create visualization of surface analysis results.
        
        Args:
            surface_results: Dictionary with surface analysis results
        """
        surfaces = list(surface_results.keys())
        accuracies = [results['prediction_accuracy'] for results in surface_results.values()]
        matches = [results['total_matches'] for results in surface_results.values()]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        ax1 = plt.gca()
        bars = ax1.bar(surfaces, accuracies, alpha=0.7)
        
        # Add accuracy values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f"{accuracies[i]:.3f}",
                ha='center',
                va='bottom',
                rotation=0
            )
        
        # Add match counts as text
        for i, surface in enumerate(surfaces):
            plt.text(
                i,
                0.05,
                f"n={matches[i]}",
                ha='center',
                va='bottom',
                color='red'
            )
        
        # Set title and labels
        plt.title('Prediction Accuracy by Surface')
        plt.xlabel('Surface')
        plt.ylabel('Accuracy')
        plt.ylim([0, max(accuracies) * 1.2])
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, "surface_accuracy_analysis.png"))
        plt.close()
        
        logger.info(f"Saved surface analysis plot to {os.path.join(self.data_dir, 'surface_accuracy_analysis.png')}")