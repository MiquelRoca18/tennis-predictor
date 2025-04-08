"""
Feature transformer module for tennis match prediction.

This module contains the FeatureTransformer class that handles preprocessing,
normalization, scaling, and encoding of features for tennis match prediction models.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    OneHotEncoder, OrdinalEncoder, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Class for preprocessing and transforming features for tennis match prediction.
    
    This class provides methods for:
    - Handling missing values in different ways
    - Scaling/normalizing numerical features
    - Encoding categorical features
    - Feature transformation for better distribution
    
    It is compatible with scikit-learn pipelines and provides a unified interface
    for all preprocessing needed for tennis prediction models.
    """
    
    def __init__(
        self,
        numerical_features=None,
        categorical_features=None,
        binary_features=None,
        ordinal_features=None,
        scaling_method='standard',
        missing_method='median',
        power_transform=False,
        ordinal_mappings=None
    ):
        """
        Initialize the FeatureTransformer with feature type specifications.
        
        Args:
            numerical_features (list, optional): List of numerical feature names
            categorical_features (list, optional): List of categorical feature names
            binary_features (list, optional): List of binary feature names
            ordinal_features (list, optional): List of ordinal feature names
            scaling_method (str): Method for scaling numerical features
                ('standard', 'minmax', 'robust', or None)
            missing_method (str): Method for handling missing values
                ('mean', 'median', 'constant', 'knn', or None)
            power_transform (bool): Whether to apply power transform to numerical features
            ordinal_mappings (dict, optional): Mappings for ordinal features
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.binary_features = binary_features or []
        self.ordinal_features = ordinal_features or []
        self.scaling_method = scaling_method
        self.missing_method = missing_method
        self.power_transform = power_transform
        self.ordinal_mappings = ordinal_mappings or {}
        
        self.column_transformer = None
        self.fitted = False
        
        logger.info("FeatureTransformer initialized")
    
    def fit(self, X, y=None):
        """
        Fit the transformer on the training data.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable (not used in transformation)
            
        Returns:
            self: Returns self
        """
        # If feature lists are empty, try to infer feature types
        if not any([self.numerical_features, self.categorical_features, 
                   self.binary_features, self.ordinal_features]):
            self._infer_feature_types(X)
        
        # Create transformers for different feature types
        transformers = []
        
        # Numerical features transformer
        if self.numerical_features:
            num_transformer = self._build_numerical_transformer()
            transformers.append(('num', num_transformer, self.numerical_features))
            logger.info(f"Added numerical transformer for {len(self.numerical_features)} features")
        
        # Categorical features transformer
        if self.categorical_features:
            cat_transformer = self._build_categorical_transformer()
            transformers.append(('cat', cat_transformer, self.categorical_features))
            logger.info(f"Added categorical transformer for {len(self.categorical_features)} features")
        
        # Binary features transformer
        if self.binary_features:
            bin_transformer = self._build_binary_transformer()
            transformers.append(('bin', bin_transformer, self.binary_features))
            logger.info(f"Added binary transformer for {len(self.binary_features)} features")
        
        # Ordinal features transformer
        if self.ordinal_features:
            ord_transformer = self._build_ordinal_transformer()
            transformers.append(('ord', ord_transformer, self.ordinal_features))
            logger.info(f"Added ordinal transformer for {len(self.ordinal_features)} features")
        
        # Create column transformer
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        # Fit the transformer
        self.column_transformer.fit(X)
        self.fitted = True
        
        # Store original columns and transformed columns
        self.original_columns = X.columns.tolist()
        self._set_output_columns()
        
        logger.info("FeatureTransformer fitted successfully")
        return self
    
    def transform(self, X):
        """
        Transform the input features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Transformed features
        """
        if not self.fitted:
            logger.error("Transformer not fitted. Call fit() first.")
            return X
        
        # Transform features
        X_transformed = self.column_transformer.transform(X)
        
        # Convert to DataFrame with proper column names
        return pd.DataFrame(X_transformed, columns=self.output_columns, index=X.index)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the input features.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            
        Returns:
            pd.DataFrame: Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def _infer_feature_types(self, X):
        """
        Infer feature types from the data.
        
        Args:
            X (pd.DataFrame): Input features
        """
        # Reset all feature lists
        self.numerical_features = []
        self.categorical_features = []
        self.binary_features = []
        self.ordinal_features = []
        
        for col in X.columns:
            # Skip columns with all NaN values
            if X[col].isna().all():
                logger.warning(f"Column {col} has all NaN values. Skipping.")
                continue
            
            # Check for binary features
            if X[col].nunique() <= 2:
                self.binary_features.append(col)
            
            # Check for categorical features
            elif X[col].dtype == 'object' or X[col].dtype.name == 'category':
                if X[col].nunique() <= 20:  # Arbitrary cutoff for categorical vs high-cardinality
                    self.categorical_features.append(col)
                else:
                    logger.warning(f"Column {col} has high cardinality ({X[col].nunique()} values). Treating as numeric.")
                    self.numerical_features.append(col)
            
            # Default to numerical
            else:
                self.numerical_features.append(col)
        
        logger.info(f"Inferred feature types: {len(self.numerical_features)} numerical, "
                   f"{len(self.categorical_features)} categorical, "
                   f"{len(self.binary_features)} binary, "
                   f"{len(self.ordinal_features)} ordinal")
    
    def _build_numerical_transformer(self):
        """
        Build a transformer pipeline for numerical features.
        
        Returns:
            Pipeline: Scikit-learn pipeline for numerical feature transformation
        """
        steps = []
        
        # Missing value imputation
        if self.missing_method:
            if self.missing_method == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif self.missing_method == 'median':
                imputer = SimpleImputer(strategy='median')
            elif self.missing_method == 'constant':
                imputer = SimpleImputer(strategy='constant', fill_value=0)
            elif self.missing_method == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                logger.warning(f"Unknown missing method: {self.missing_method}. Using median.")
                imputer = SimpleImputer(strategy='median')
                
            steps.append(('imputer', imputer))
        
        # Scaling
        if self.scaling_method:
            if self.scaling_method == 'standard':
                scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.scaling_method}. Using standard.")
                scaler = StandardScaler()
                
            steps.append(('scaler', scaler))
        
        # Power transform for better distribution
        if self.power_transform:
            steps.append(('power', PowerTransformer(method='yeo-johnson')))
        
        return Pipeline(steps)
    
    def _build_categorical_transformer(self):
        """
        Build a transformer pipeline for categorical features.
        
        Returns:
            Pipeline: Scikit-learn pipeline for categorical feature transformation
        """
        steps = []
        
        # Missing value imputation
        imputer = SimpleImputer(strategy='constant', fill_value='missing')
        steps.append(('imputer', imputer))
        
        # One-hot encoding
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        steps.append(('encoder', encoder))
        
        return Pipeline(steps)
    
    def _build_binary_transformer(self):
        """
        Build a transformer pipeline for binary features.
        
        Returns:
            Pipeline: Scikit-learn pipeline for binary feature transformation
        """
        steps = []
        
        # Missing value imputation
        imputer = SimpleImputer(strategy='most_frequent')
        steps.append(('imputer', imputer))
        
        # No additional encoding needed for binary features
        
        return Pipeline(steps)
    
    def _build_ordinal_transformer(self):
        """
        Build a transformer pipeline for ordinal features.
        
        Returns:
            Pipeline: Scikit-learn pipeline for ordinal feature transformation
        """
        steps = []
        
        # Missing value imputation
        imputer = SimpleImputer(strategy='most_frequent')
        steps.append(('imputer', imputer))
        
        # Ordinal encoding
        if self.ordinal_mappings:
            # Create ordered categories for each ordinal feature
            categories = []
            for feature in self.ordinal_features:
                if feature in self.ordinal_mappings:
                    categories.append(self.ordinal_mappings[feature])
                else:
                    logger.warning(f"No mapping provided for ordinal feature {feature}. Using default ordering.")
                    categories.append('auto')
            
            encoder = OrdinalEncoder(categories=categories)
        else:
            # Use auto-inferred ordering if no mappings provided
            encoder = OrdinalEncoder()
            
        steps.append(('encoder', encoder))
        
        return Pipeline(steps)
    
    def _set_output_columns(self):
        """
        Set column names for the transformed output.
        """
        # Initialize output columns
        self.output_columns = []
        
        # Get names for each transformer
        for name, transformer, columns in self.column_transformer.transformers_:
            if name == 'num' or name == 'bin':
                # Numerical and binary columns keep their names
                self.output_columns.extend(columns)
            elif name == 'cat':
                # Get categories for one-hot encoded features
                one_hot_encoder = transformer.named_steps['encoder']
                categories = one_hot_encoder.categories_
                
                for i, col in enumerate(columns):
                    for category in categories[i]:
                        self.output_columns.append(f"{col}_{category}")
            elif name == 'ord':
                # Ordinal columns keep their names
                self.output_columns.extend(columns)
        
        # Add passthrough columns
        passthrough_mask = np.ones(len(self.original_columns), dtype=bool)
        for _, _, columns in self.column_transformer.transformers_:
            for col in columns:
                idx = self.original_columns.index(col)
                passthrough_mask[idx] = False
                
        passthrough_columns = [col for i, col in enumerate(self.original_columns) if passthrough_mask[i]]
        self.output_columns.extend(passthrough_columns)
    
    def get_feature_names_out(self):
        """
        Get the column names of the transformed output.
        
        Returns:
            list: Column names of transformed output
        """
        if not self.fitted:
            logger.error("Transformer not fitted. Call fit() first.")
            return None
            
        return self.output_columns
    
    def set_feature_types(self, feature_types):
        """
        Set feature types explicitly.
        
        Args:
            feature_types (dict): Dictionary mapping feature types to feature names
                Keys should be 'numerical', 'categorical', 'binary', 'ordinal'
        """
        self.numerical_features = feature_types.get('numerical', [])
        self.categorical_features = feature_types.get('categorical', [])
        self.binary_features = feature_types.get('binary', [])
        self.ordinal_features = feature_types.get('ordinal', [])
        
        # Reset fit state
        self.fitted = False
        self.column_transformer = None
        
        logger.info(f"Feature types set: {len(self.numerical_features)} numerical, "
                   f"{len(self.categorical_features)} categorical, "
                   f"{len(self.binary_features)} binary, "
                   f"{len(self.ordinal_features)} ordinal")
    
    def get_scaling_params(self):
        """
        Get scaling parameters for numerical features after fitting.
        
        Returns:
            dict: Dictionary with scaling parameters
        """
        if not self.fitted or 'num' not in dict(self.column_transformer.named_transformers_):
            logger.error("Transformer not fitted or no numerical transformer found.")
            return None
        
        num_transformer = self.column_transformer.named_transformers_['num']
        
        if 'scaler' not in num_transformer.named_steps:
            logger.warning("No scaler found in numerical transformer.")
            return None
        
        scaler = num_transformer.named_steps['scaler']
        params = {}
        
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            params['mean'] = scaler.mean_
            params['scale'] = scaler.scale_
            
            # Map to feature names
            params['feature_mean'] = dict(zip(self.numerical_features, scaler.mean_))
            params['feature_scale'] = dict(zip(self.numerical_features, scaler.scale_))
        
        if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
            params['data_min'] = scaler.data_min_
            params['data_max'] = scaler.data_max_
            params['feature_min'] = dict(zip(self.numerical_features, scaler.data_min_))
            params['feature_max'] = dict(zip(self.numerical_features, scaler.data_max_))
        
        return params
    
    def get_categorical_encodings(self):
        """
        Get encoding information for categorical features after fitting.
        
        Returns:
            dict: Dictionary with categorical encodings
        """
        if not self.fitted or 'cat' not in dict(self.column_transformer.named_transformers_):
            logger.error("Transformer not fitted or no categorical transformer found.")
            return None
        
        cat_transformer = self.column_transformer.named_transformers_['cat']
        
        if 'encoder' not in cat_transformer.named_steps:
            logger.warning("No encoder found in categorical transformer.")
            return None
        
        encoder = cat_transformer.named_steps['encoder']
        encodings = {}
        
        if hasattr(encoder, 'categories_'):
            for i, feature in enumerate(self.categorical_features):
                encodings[feature] = encoder.categories_[i].tolist()
        
        return encodings
    
    def save_transformer_state(self, filename):
        """
        Save the transformer state to a JSON file.
        
        Args:
            filename (str): Filename for the saved state
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.fitted:
            logger.error("Transformer not fitted. Nothing to save.")
            return False
        
        try:
            state = {
                'numerical_features': self.numerical_features,
                'categorical_features': self.categorical_features,
                'binary_features': self.binary_features,
                'ordinal_features': self.ordinal_features,
                'scaling_method': self.scaling_method,
                'missing_method': self.missing_method,
                'power_transform': self.power_transform,
                'ordinal_mappings': self.ordinal_mappings,
                'scaling_params': self.get_scaling_params(),
                'categorical_encodings': self.get_categorical_encodings(),
                'output_columns': self.output_columns
            }
            
            # Save to file
            import json
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Transformer state saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving transformer state: {e}")
            return False
    
    @classmethod
    def from_saved_state(cls, filename):
        """
        Create a FeatureTransformer from a saved state.
        
        Args:
            filename (str): Filename of the saved state
            
        Returns:
            FeatureTransformer: New instance with loaded state
        """
        try:
            import json
            with open(filename, 'r') as f:
                state = json.load(f)
                
            # Create new instance with loaded parameters
            transformer = cls(
                numerical_features=state['numerical_features'],
                categorical_features=state['categorical_features'],
                binary_features=state['binary_features'],
                ordinal_features=state['ordinal_features'],
                scaling_method=state['scaling_method'],
                missing_method=state['missing_method'],
                power_transform=state['power_transform'],
                ordinal_mappings=state['ordinal_mappings']
            )
            
            # Set output columns if available
            if 'output_columns' in state:
                transformer.output_columns = state['output_columns']
                
            logger.info(f"Loaded transformer state from {filename}")
            return transformer
            
        except Exception as e:
            logger.error(f"Error loading transformer state: {e}")
            return None


class TennisMissingValueHandler:
    """
    Class for handling missing values in tennis match prediction features.
    
    This class provides specialized methods for filling missing values
    in tennis-specific features based on domain knowledge.
    """
    
    def __init__(self, matches_df=None, players_df=None):
        """
        Initialize the TennisMissingValueHandler.
        
        Args:
            matches_df (pd.DataFrame, optional): DataFrame containing match data
            players_df (pd.DataFrame, optional): DataFrame containing player information
        """
        self.matches_df = matches_df
        self.players_df = players_df
        
        logger.info("TennisMissingValueHandler initialized")
    
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
    
    def fill_missing_player_data(self, features_df):
        """
        Fill missing player-specific features with reasonable values.
        
        Args:
            features_df (pd.DataFrame): DataFrame with features to process
            
        Returns:
            pd.DataFrame: DataFrame with filled values
        """
        # Create a copy to avoid modifying the original
        df = features_df.copy()
        
        # Handle missing physical attributes
        height_cols = [col for col in df.columns if 'height' in col.lower()]
        for col in height_cols:
            if df[col].isna().any():
                # For height, use median height by gender if available
                if self.players_df is not None and 'height' in self.players_df.columns:
                    median_height = self.players_df['height'].median()
                    df[col].fillna(median_height, inplace=True)
                else:
                    # Tennis average height is around 185cm for men, 170cm for women
                    df[col].fillna(df[col].mean() if not df[col].isna().all() else 180, inplace=True)
        
        # Handle missing handedness (dominant hand)
        hand_cols = [col for col in df.columns if 'hand' in col.lower() or 'left' in col.lower()]
        for col in hand_cols:
            if df[col].isna().any():
                # About 10% of players are left-handed
                if 'is_left' in col:
                    df[col].fillna(0, inplace=True)  # Most players are right-handed
                else:
                    df[col].fillna('R', inplace=True)  # Right-handed is most common
        
        # Handle missing age
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        for col in age_cols:
            if df[col].isna().any():
                # Average professional tennis player is in mid-20s
                df[col].fillna(df[col].mean() if not df[col].isna().all() else 25, inplace=True)
        
        logger.info("Filled missing player data")
        return df
    
    def fill_missing_match_stats(self, features_df):
        """
        Fill missing match statistics with reasonable values.
        
        Args:
            features_df (pd.DataFrame): DataFrame with features to process
            
        Returns:
            pd.DataFrame: DataFrame with filled values
        """
        # Create a copy to avoid modifying the original
        df = features_df.copy()
        
        # Handle serve statistics
        serve_cols = [col for col in df.columns if any(s in col.lower() for s in 
                                                    ['ace', 'df', 'svpt', '1st', '2nd', 'serve'])]
        for col in serve_cols:
            if df[col].isna().any():
                # For rates and percentages, use median value
                if 'rate' in col or 'ratio' in col or 'pct' in col or 'perc' in col:
                    if 'ace' in col.lower():
                        # Ace rate is typically 5-8%
                        df[col].fillna(df[col].median() if not df[col].isna().all() else 0.06, inplace=True)
                    elif 'df' in col.lower() or 'double' in col.lower():
                        # Double fault rate is typically 2-5%
                        df[col].fillna(df[col].median() if not df[col].isna().all() else 0.03, inplace=True)
                    elif '1st' in col and ('in' in col.lower() or 'pct' in col.lower()):
                        # 1st serve percentage is typically 55-65%
                        df[col].fillna(df[col].median() if not df[col].isna().all() else 0.60, inplace=True)
                    elif '1st' in col and 'won' in col.lower():
                        # 1st serve win percentage is typically 70-80%
                        df[col].fillna(df[col].median() if not df[col].isna().all() else 0.75, inplace=True)
                    elif '2nd' in col and 'won' in col.lower():
                        # 2nd serve win percentage is typically 45-55%
                        df[col].fillna(df[col].median() if not df[col].isna().all() else 0.50, inplace=True)
                    else:
                        df[col].fillna(df[col].median() if not df[col].isna().all() else 0.5, inplace=True)
                else:
                    # For raw counts, use median
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)
        
        # Handle break point statistics
        bp_cols = [col for col in df.columns if 'bp' in col.lower() or 'break' in col.lower()]
        for col in bp_cols:
            if df[col].isna().any():
                if 'ratio' in col or 'pct' in col or 'perc' in col or 'save' in col.lower():
                    # Break point save percentage is typically 55-65%
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0.60, inplace=True)
                elif 'conv' in col.lower() or 'conv' in col.lower():
                    # Break point conversion is typically 35-45%
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0.40, inplace=True)
                else:
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)
        
        logger.info("Filled missing match statistics")
        return df
    
    def fill_missing_temporal_features(self, features_df):
        """
        Fill missing temporal features with reasonable values.
        
        Args:
            features_df (pd.DataFrame): DataFrame with features to process
            
        Returns:
            pd.DataFrame: DataFrame with filled values
        """
        # Create a copy to avoid modifying the original
        df = features_df.copy()
        
        # Handle win ratio features
        win_ratio_cols = [col for col in df.columns if 'win_ratio' in col.lower()]
        for col in win_ratio_cols:
            if df[col].isna().any():
                # Default to 0.5 (neutral) for missing win ratios
                df[col].fillna(0.5, inplace=True)
        
        # Handle rest and fatigue features
        rest_cols = [col for col in df.columns if 'rest' in col.lower() or 'days_since' in col.lower()]
        for col in rest_cols:
            if df[col].isna().any():
                # Default to 7 days rest (typical between tournaments)
                df[col].fillna(df[col].median() if not df[col].isna().all() else 7, inplace=True)
        
        fatigue_cols = [col for col in df.columns if 'fatigue' in col.lower()]
        for col in fatigue_cols:
            if df[col].isna().any():
                # Default to median or low fatigue
                df[col].fillna(df[col].median() if not df[col].isna().all() else 0.3, inplace=True)
        
        # Handle streak features
        streak_cols = [col for col in df.columns if 'streak' in col.lower()]
        for col in streak_cols:
            if df[col].isna().any():
                # Default to 0 (no streak) for missing streak values
                df[col].fillna(0, inplace=True)
        
        # Handle momentum features
        momentum_cols = [col for col in df.columns if 'momentum' in col.lower()]
        for col in momentum_cols:
            if df[col].isna().any():
                # Default to 0.5 (neutral) for missing momentum
                df[col].fillna(0.5, inplace=True)
        
        logger.info("Filled missing temporal features")
        return df
    
    def fill_missing_elo_features(self, features_df):
        """
        Fill missing ELO rating features with reasonable values.
        
        Args:
            features_df (pd.DataFrame): DataFrame with features to process
            
        Returns:
            pd.DataFrame: DataFrame with filled values
        """
        # Create a copy to avoid modifying the original
        df = features_df.copy()
        
        # Handle ELO rating features
        elo_cols = [col for col in df.columns if 'elo' in col.lower()]
        for col in elo_cols:
            if df[col].isna().any():
                if 'win_prob' in col.lower():
                    # Default to 0.5 (50-50 chance) for missing win probability
                    df[col].fillna(0.5, inplace=True)
                elif 'diff' in col.lower():
                    # Default to 0 (even match) for missing ELO difference
                    df[col].fillna(0, inplace=True)
                else:
                    # Default to 1500 (starting ELO) for missing ELO ratings
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 1500, inplace=True)
        
        logger.info("Filled missing ELO features")
        return df
    
    def fill_missing_h2h_features(self, features_df):
        """
        Fill missing head-to-head features with reasonable values.
        
        Args:
            features_df (pd.DataFrame): DataFrame with features to process
            
        Returns:
            pd.DataFrame: DataFrame with filled values
        """
        # Create a copy to avoid modifying the original
        df = features_df.copy()
        
        # Handle head-to-head count features
        h2h_count_cols = [col for col in df.columns if 'h2h' in col.lower() and 
                         ('matches' in col.lower() or 'count' in col.lower())]
        for col in h2h_count_cols:
            if df[col].isna().any():
                # Default to 0 (no previous meetings) for missing H2H count
                df[col].fillna(0, inplace=True)
        
        # Handle head-to-head win ratio features
        h2h_ratio_cols = [col for col in df.columns if 'h2h' in col.lower() and 
                         ('ratio' in col.lower() or 'win' in col.lower())]
        for col in h2h_ratio_cols:
            if df[col].isna().any():
                # Default to 0.5 (even) for missing H2H win ratio
                df[col].fillna(0.5, inplace=True)
        
        # Handle head-to-head statistic features
        h2h_stat_cols = [col for col in df.columns if 'h2h' in col.lower() and 
                        any(s in col.lower() for s in ['avg', 'ace', 'df', 'svpt', '1st', '2nd', 'bp'])]
        for col in h2h_stat_cols:
            if df[col].isna().any():
                if 'ace' in col.lower():
                    # Default to average ace count (around 5-8 per match)
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 6, inplace=True)
                elif 'df' in col.lower():
                    # Default to average double fault count (around 3-5 per match)
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 4, inplace=True)
                elif '1st' in col.lower() and 'in' in col.lower():
                    # Default to average first serves in percentage (around 60%)
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0.6, inplace=True)
                elif '1st' in col.lower() and 'won' in col.lower():
                    # Default to average first serve points won (around 70%)
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0.7, inplace=True)
                elif '2nd' in col.lower() and 'won' in col.lower():
                    # Default to average second serve points won (around 50%)
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0.5, inplace=True)
                elif 'bp' in col.lower() and 'saved' in col.lower():
                    # Default to average break points saved (around 60%)
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0.6, inplace=True)
                elif 'bp' in col.lower() and 'faced' in col.lower():
                    # Default to average break points faced per match (around 8)
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 8, inplace=True)
                else:
                    # Default to median for other statistics
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)
        
        logger.info("Filled missing head-to-head features")
        return df
    
    def fill_missing_tournament_features(self, features_df):
        """
        Fill missing tournament-specific features with reasonable values.
        
        Args:
            features_df (pd.DataFrame): DataFrame with features to process
            
        Returns:
            pd.DataFrame: DataFrame with filled values
        """
        # Create a copy to avoid modifying the original
        df = features_df.copy()
        
        # Handle tournament match count features
        count_cols = [col for col in df.columns if 'tournament' in col.lower() and 
                     ('count' in col.lower() or 'match' in col.lower())]
        for col in count_cols:
            if df[col].isna().any():
                # Default to 0 (no previous matches) for missing tournament match count
                df[col].fillna(0, inplace=True)
        
        # Handle tournament win ratio features
        ratio_cols = [col for col in df.columns if 'tournament' in col.lower() and 
                     ('ratio' in col.lower() or 'win' in col.lower())]
        for col in ratio_cols:
            if df[col].isna().any():
                # Default to 0.5 (neutral) for missing tournament win ratio
                df[col].fillna(0.5, inplace=True)
        
        # Handle tournament experience difference
        exp_cols = [col for col in df.columns if 'tournament' in col.lower() and 
                   'experience' in col.lower()]
        for col in exp_cols:
            if df[col].isna().any():
                # Default to 0 (equal experience) for missing tournament experience difference
                df[col].fillna(0, inplace=True)
        
        # Handle round-related features
        round_cols = [col for col in df.columns if 'round' in col.lower()]
        for col in round_cols:
            if df[col].isna().any():
                if 'reached' in col.lower():
                    # Default to early rounds (1-2) for missing best round reached
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 2, inplace=True)
                else:
                    # Default to median for other round features
                    df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)
        
        logger.info("Filled missing tournament features")
        return df
    
    def fill_all_missing_values(self, features_df):
        """
        Apply all missing value handling methods in sequence.
        
        Args:
            features_df (pd.DataFrame): DataFrame with features to process
            
        Returns:
            pd.DataFrame: DataFrame with all missing values filled
        """
        df = features_df.copy()
        
        # Apply all filling methods
        df = self.fill_missing_player_data(df)
        df = self.fill_missing_match_stats(df)
        df = self.fill_missing_temporal_features(df)
        df = self.fill_missing_elo_features(df)
        df = self.fill_missing_h2h_features(df)
        df = self.fill_missing_tournament_features(df)
        
        # Check for any remaining missing values
        remaining_na = df.isna().sum().sum()
        if remaining_na > 0:
            logger.warning(f"There are still {remaining_na} missing values after applying all filling methods")
            
            # Fill any remaining missing values with column median or 0
            for col in df.columns:
                if df[col].isna().any():
                    if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                        df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)
                    else:
                        df[col].fillna('missing', inplace=True)
        
        logger.info("Applied all missing value handling methods")
        return df
    
    def add_polynomial_features(self, degree=2, interaction_only=False, include_bias=False):
        """
        Add polynomial features and interactions between numerical features.
        
        Args:
            degree (int): Maximum degree of polynomial features
            interaction_only (bool): If True, only interaction features are produced
            include_bias (bool): If True, include a bias column (all 1s)
            
        Returns:
            self: Returns self
        """
        if not self.numerical_features:
            logger.warning("No numerical features defined for polynomial transformation")
            return self
        
        # Ensure we're not already fitted
        if self.fitted:
            logger.warning("Transformer already fitted. Cannot add polynomial features.")
            return self
        
        from sklearn.preprocessing import PolynomialFeatures
        
        # Store original numerical features
        original_num_features = self.numerical_features.copy()
        
        # Add polynomial transformer to numerical pipeline
        if hasattr(self, '_num_poly_transformer'):
            logger.warning("Polynomial transformer already added. Updating parameters.")
            # Update existing transformer
            self._num_poly_transformer = PolynomialFeatures(
                degree=degree, 
                interaction_only=interaction_only,
                include_bias=include_bias
            )
        else:
            # Create new transformer
            self._num_poly_transformer = PolynomialFeatures(
                degree=degree, 
                interaction_only=interaction_only,
                include_bias=include_bias
            )
        
        # Flag to indicate that polynomial features should be added
        self._add_polynomial = True
        
        logger.info(f"Polynomial features (degree={degree}) will be added during fit")
        return self

    def _build_numerical_transformer(self):
        """
        Build a transformer pipeline for numerical features.
        
        Returns:
            Pipeline: Scikit-learn pipeline for numerical feature transformation
        """
        steps = []
        
        # Missing value imputation
        if self.missing_method:
            if self.missing_method == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif self.missing_method == 'median':
                imputer = SimpleImputer(strategy='median')
            elif self.missing_method == 'constant':
                imputer = SimpleImputer(strategy='constant', fill_value=0)
            elif self.missing_method == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                logger.warning(f"Unknown missing method: {self.missing_method}. Using median.")
                imputer = SimpleImputer(strategy='median')
                
            steps.append(('imputer', imputer))
        
        # Add polynomial features if requested
        if hasattr(self, '_add_polynomial') and self._add_polynomial:
            if hasattr(self, '_num_poly_transformer'):
                steps.append(('poly', self._num_poly_transformer))
        
        # Scaling
        if self.scaling_method:
            if self.scaling_method == 'standard':
                scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.scaling_method}. Using standard.")
                scaler = StandardScaler()
                
            steps.append(('scaler', scaler))
        
        # Power transform for better distribution
        if self.power_transform:
            steps.append(('power', PowerTransformer(method='yeo-johnson')))
        
        return Pipeline(steps)

    def _set_output_columns(self):
        """
        Set column names for the transformed output.
        """
        # Initialize output columns
        self.output_columns = []
        
        # Get names for each transformer
        for name, transformer, columns in self.column_transformer.transformers_:
            if name == 'bin':
                # Binary columns keep their names
                self.output_columns.extend(columns)
            elif name == 'num':
                # For numerical features, check if polynomial transformer was applied
                if hasattr(self, '_add_polynomial') and self._add_polynomial:
                    if 'poly' in transformer.named_steps:
                        poly = transformer.named_steps['poly']
                        # Get polynomial feature names
                        poly_features = poly.get_feature_names_out(columns)
                        self.output_columns.extend(poly_features)
                    else:
                        self.output_columns.extend(columns)
                else:
                    self.output_columns.extend(columns)
            elif name == 'cat':
                # Get categories for one-hot encoded features
                one_hot_encoder = transformer.named_steps['encoder']
                categories = one_hot_encoder.categories_
                
                for i, col in enumerate(columns):
                    for category in categories[i]:
                        self.output_columns.append(f"{col}_{category}")
            elif name == 'ord':
                # Ordinal columns keep their names
                self.output_columns.extend(columns)
            
        # Add passthrough columns
        passthrough_mask = np.ones(len(self.original_columns), dtype=bool)
        for _, _, columns in self.column_transformer.transformers_:
            for col in columns:
                idx = self.original_columns.index(col)
                passthrough_mask[idx] = False
                
        passthrough_columns = [col for i, col in enumerate(self.original_columns) if passthrough_mask[i]]
        self.output_columns.extend(passthrough_columns)