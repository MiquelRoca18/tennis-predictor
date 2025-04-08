"""
Feature validator module for tennis match prediction.

This module provides functionality to verify the quality and consistency
of extracted features, identify outliers, and validate feature distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class TennisFeatureValidator:
    """
    Class for validating the quality and consistency of tennis features.
    
    This class provides methods to identify outliers, check for inconsistent values,
    validate feature distributions, and ensure features are suitable for modeling.
    """
    
    def __init__(self, features_df=None, reference_features=None, player_features=None):
        """
        Initialize the TennisFeatureValidator.
        
        Args:
            features_df (pd.DataFrame, optional): DataFrame containing features
            reference_features (dict, optional): Reference feature stats for validation
            player_features (dict, optional): Dictionary of player features
        """
        self.features_df = features_df
        self.reference_features = reference_features or {}
        self.player_features = player_features or {}
        
        # Store validation results
        self.validation_results = {}
        self.feature_issues = defaultdict(list)
        
        # Feature type tracking
        self.feature_types = {}
        
        # Expected valid ranges for common features
        self.expected_ranges = {
            'elo': (1000, 3000),
            'win_ratio': (0, 1),
            'ace_rate': (0, 0.3),
            'df_rate': (0, 0.2),
            'first_serve_in': (0, 1),
            'first_serve_win': (0, 1),
            'second_serve_win': (0, 1),
            'bp_conversion': (0, 1),
            'bp_save': (0, 1),
            'height': (150, 220),
            'age': (15, 45)
        }
        
        logger.info("TennisFeatureValidator initialized")
    
    def set_data(self, features_df=None, reference_features=None, player_features=None):
        """
        Set or update data sources.
        
        Args:
            features_df (pd.DataFrame, optional): DataFrame containing features
            reference_features (dict, optional): Reference feature stats for validation
            player_features (dict, optional): Dictionary of player features
        """
        if features_df is not None:
            self.features_df = features_df
        if reference_features is not None:
            self.reference_features = reference_features
        if player_features is not None:
            self.player_features = player_features
            
        # Reset validation results when data changes
        self.validation_results = {}
        self.feature_issues = defaultdict(list)
        
        logger.info("Data sources updated")
    
    def infer_feature_types(self):
        """
        Infer the type of each feature based on its name and values.
        
        Returns:
            dict: Dictionary of feature types
        """
        # Reset feature types
        self.feature_types = {}
        
        # Infer from features_df if available
        if self.features_df is not None:
            for col in self.features_df.columns:
                self.feature_types[col] = self._infer_feature_type(col, self.features_df[col])
        
        # Infer from player_features if available
        elif self.player_features:
            # Get a sample player
            sample_player_id = next(iter(self.player_features))
            sample_features = self.player_features[sample_player_id]
            
            for feature, value in sample_features.items():
                feature_type = self._infer_feature_type(feature, pd.Series([value]))
                self.feature_types[feature] = feature_type
        
        logger.info(f"Inferred types for {len(self.feature_types)} features")
        return self.feature_types
    
    def _infer_feature_type(self, feature_name, values):
        """
        Infer the type of a feature based on its name and values.
        
        Args:
            feature_name (str): Name of the feature
            values (pd.Series): Values of the feature
            
        Returns:
            str: Inferred feature type
        """
        # Check by name first
        name_lower = feature_name.lower()
        
        if any(term in name_lower for term in ['ratio', 'rate', 'pct', 'percent', 'win']):
            return 'ratio'
        elif any(term in name_lower for term in ['elo', 'rating', 'score']):
            return 'rating'
        elif 'count' in name_lower or 'matches' in name_lower:
            return 'count'
        elif any(term in name_lower for term in ['height', 'age', 'weight']):
            return 'physical'
        elif any(term in name_lower for term in ['rank', 'ranking']):
            return 'ranking'
        elif any(term in name_lower for term in ['dominance', 'advantage']):
            return 'dominance'
        elif any(term in name_lower for term in ['style', 'preferred', 'type']):
            return 'category'
        elif any(term in name_lower for term in ['ace', 'df', '1st', '2nd', 'serve', 'bp']):
            return 'statistic'
        elif 'days' in name_lower or 'time' in name_lower or 'date' in name_lower:
            return 'temporal'
        
        # Check by values if name doesn't give clear indication
        if values.dtype in ['int64', 'float64']:
            # If all values are between 0 and 1, likely a ratio
            if values.min() >= 0 and values.max() <= 1:
                return 'ratio'
            # If all values are positive integers, likely a count
            elif values.min() >= 0 and values.equals(values.astype(int)):
                return 'count'
            # If values have a wide range, might be a rating
            elif values.min() >= 1000 and values.max() <= 3000:
                return 'rating'
            # Otherwise general numeric
            else:
                return 'numeric'
        elif values.dtype == 'object' or values.dtype.name == 'category':
            return 'category'
        elif pd.api.types.is_bool_dtype(values):
            return 'binary'
        else:
            return 'unknown'
    
    def validate_features(self, method='all'):
        """
        Run a comprehensive validation on the features.
        
        Args:
            method (str): Validation method ('all', 'outliers', 'consistency', 
                          'completeness', 'distribution')
            
        Returns:
            dict: Validation results
        """
        # Check if we have data to validate
        if self.features_df is None and not self.player_features:
            logger.error("No feature data available for validation")
            return {}
        
        # Reset validation results
        self.validation_results = {}
        self.feature_issues = defaultdict(list)
        
        # Infer feature types if not already done
        if not self.feature_types:
            self.infer_feature_types()
        
        # Run selected validation methods
        if method in ['all', 'outliers']:
            self.validate_outliers()
        
        if method in ['all', 'consistency']:
            self.validate_consistency()
        
        if method in ['all', 'completeness']:
            self.validate_completeness()
        
        if method in ['all', 'distribution']:
            self.validate_distributions()
        
        # Generate summary
        self.validation_results['summary'] = self.generate_validation_summary()
        
        return self.validation_results
    
    def validate_outliers(self):
        """
        Identify outliers in the feature values.
        
        Returns:
            dict: Outlier validation results
        """
        outlier_results = {
            'outlier_features': [],
            'outlier_counts': {},
            'outlier_values': {},
            'outlier_indices': {}
        }
        
        # Prepare data for validation
        if self.features_df is not None:
            feature_data = self.features_df
        else:
            # Convert player_features to DataFrame
            records = []
            for player_id, features in self.player_features.items():
                record = {'player_id': player_id}
                record.update(features)
                records.append(record)
            
            feature_data = pd.DataFrame(records)
        
        # Check numeric features for outliers
        numeric_features = feature_data.select_dtypes(include=['float64', 'int64']).columns
        
        for feature in numeric_features:
            feature_type = self.feature_types.get(feature, 'unknown')
            values = feature_data[feature].dropna()
            
            if len(values) < 5:  # Need enough data for outlier detection
                continue
            
            # Check for expected range violations
            range_violations = 0
            for range_key, (min_val, max_val) in self.expected_ranges.items():
                if range_key in feature.lower():
                    range_violations = ((values < min_val) | (values > max_val)).sum()
                    
                    if range_violations > 0:
                        self.feature_issues[feature].append(
                            f"Range violation: {range_violations} values outside expected range [{min_val}, {max_val}]"
                        )
            
            # Use Z-score method for outlier detection
            z_scores = np.abs(stats.zscore(values))
            outliers_z = np.where(z_scores > 3)[0]
            
            # Use IQR method for outlier detection
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            outliers_iqr = values[((values < (q1 - 1.5 * iqr)) | (values > (q3 + 1.5 * iqr)))].index
            
            # Combine outliers from both methods
            all_outliers = set(outliers_z).union(set(outliers_iqr))
            
            if len(all_outliers) > 0:
                outlier_values = values.iloc[list(outliers_z)] if self.features_df is not None else values.loc[outliers_iqr]
                
                outlier_results['outlier_features'].append(feature)
                outlier_results['outlier_counts'][feature] = len(all_outliers)
                outlier_results['outlier_values'][feature] = outlier_values.tolist()
                outlier_results['outlier_indices'][feature] = list(all_outliers)
                
                # Add to feature issues
                self.feature_issues[feature].append(
                    f"Contains {len(all_outliers)} outliers ({len(all_outliers)/len(values)*100:.1f}% of values)"
                )
        
        # Store results
        self.validation_results['outliers'] = outlier_results
        
        logger.info(f"Found outliers in {len(outlier_results['outlier_features'])} features")
        return outlier_results
    
    def validate_consistency(self):
        """
        Check for logical consistency in feature values.
        
        Returns:
            dict: Consistency validation results
        """
        consistency_results = {
            'inconsistent_features': [],
            'logical_violations': {}
        }
        
        # Prepare data for validation
        if self.features_df is not None:
            feature_data = self.features_df
        else:
            # Convert player_features to DataFrame
            records = []
            for player_id, features in self.player_features.items():
                record = {'player_id': player_id}
                record.update(features)
                records.append(record)
            
            feature_data = pd.DataFrame(records)
        
        # Define consistency rules
        consistency_rules = [
            {
                'name': 'win_ratio_bounds',
                'features': [f for f in feature_data.columns if 'win_ratio' in f.lower()],
                'condition': lambda x: (x >= 0) & (x <= 1),
                'message': 'Win ratio outside valid range [0, 1]'
            },
            {
                'name': 'ace_df_ratio',
                'features': [f for f in feature_data.columns if 'ace_rate' in f.lower() or 'df_rate' in f.lower()],
                'condition': lambda x: (x >= 0) & (x <= 0.3),
                'message': 'Ace or double fault rate outside reasonable range [0, 0.3]'
            },
            {
                'name': 'first_serve_in',
                'features': [f for f in feature_data.columns if 'first_serve_in' in f.lower()],
                'condition': lambda x: (x >= 0.4) & (x <= 0.85),
                'message': 'First serve in percentage outside reasonable range [0.4, 0.85]'
            },
            {
                'name': 'serve_win_rates',
                'features': [f for f in feature_data.columns if 'serve_win' in f.lower()],
                'condition': lambda x: (x >= 0.3) & (x <= 0.9),
                'message': 'Serve win rate outside reasonable range [0.3, 0.9]'
            }
        ]
        
        # Apply consistency rules
        for rule in consistency_rules:
            for feature in rule['features']:
                if feature in feature_data.columns:
                    values = feature_data[feature].dropna()
                    
                    # Check if values satisfy the condition
                    violations = (~rule['condition'](values)).sum()
                    
                    if violations > 0:
                        if feature not in consistency_results['inconsistent_features']:
                            consistency_results['inconsistent_features'].append(feature)
                        
                        if feature not in consistency_results['logical_violations']:
                            consistency_results['logical_violations'][feature] = []
                        
                        consistency_results['logical_violations'][feature].append({
                            'rule': rule['name'],
                            'violations': violations,
                            'message': rule['message'],
                            'violation_rate': violations / len(values)
                        })
                        
                        # Add to feature issues
                        self.feature_issues[feature].append(
                            f"{rule['message']}: {violations} violations ({violations/len(values)*100:.1f}% of values)"
                        )
        
        # Check pair-wise consistency
        feature_pairs = [
            {
                'name': 'first_vs_second_serve',
                'feature1': [f for f in feature_data.columns if 'first_serve_win' in f.lower()],
                'feature2': [f for f in feature_data.columns if 'second_serve_win' in f.lower()],
                'condition': lambda x, y: x >= y,
                'message': 'First serve win rate should be higher than second serve win rate'
            },
            {
                'name': 'win_vs_loss_ratio',
                'feature1': [f for f in feature_data.columns if 'win_ratio' in f.lower()],
                'feature2': [f for f in feature_data.columns if 'loss_ratio' in f.lower()],
                'condition': lambda x, y: np.isclose(x + y, 1, atol=0.01),
                'message': 'Win ratio and loss ratio should sum to approximately 1'
            }
        ]
        
        for pair in feature_pairs:
            for f1 in pair['feature1']:
                if f1 not in feature_data.columns:
                    continue
                    
                for f2 in pair['feature2']:
                    if f2 not in feature_data.columns:
                        continue
                    
                    # Match features (e.g., same surface or same player)
                    if not self._are_related_features(f1, f2):
                        continue
                    
                    # Get values and drop rows with NaN in either feature
                    mask = feature_data[[f1, f2]].notna().all(axis=1)
                    values1 = feature_data.loc[mask, f1]
                    values2 = feature_data.loc[mask, f2]
                    
                    if len(values1) < 1:
                        continue
                    
                    # Check consistency
                    violations = (~pair['condition'](values1, values2)).sum()
                    
                    if violations > 0:
                        for feature in [f1, f2]:
                            if feature not in consistency_results['inconsistent_features']:
                                consistency_results['inconsistent_features'].append(feature)
                            
                            if feature not in consistency_results['logical_violations']:
                                consistency_results['logical_violations'][feature] = []
                            
                            consistency_results['logical_violations'][feature].append({
                                'rule': pair['name'],
                                'related_feature': f2 if feature == f1 else f1,
                                'violations': violations,
                                'message': pair['message'],
                                'violation_rate': violations / len(values1)
                            })
                            
                            # Add to feature issues
                            self.feature_issues[feature].append(
                                f"{pair['message']} with {f2 if feature == f1 else f1}: "
                                f"{violations} violations ({violations/len(values1)*100:.1f}% of values)"
                            )
        
        # Store results
        self.validation_results['consistency'] = consistency_results
        
        logger.info(f"Found consistency issues in {len(consistency_results['inconsistent_features'])} features")
        return consistency_results
    
    def _are_related_features(self, feature1, feature2):
        """
        Check if two features are related and should be compared.
        
        Args:
            feature1 (str): First feature name
            feature2 (str): Second feature name
            
        Returns:
            bool: True if features are related
        """
        # Extract prefixes and suffixes
        parts1 = feature1.split('_')
        parts2 = feature2.split('_')
        
        # Check if they have the same player prefix
        if feature1.startswith('p1_') and feature2.startswith('p1_'):
            return True
        if feature1.startswith('p2_') and feature2.startswith('p2_'):
            return True
        
        # Check if they have the same surface
        surfaces = ['hard', 'clay', 'grass', 'carpet']
        for surface in surfaces:
            if surface in feature1.lower() and surface in feature2.lower():
                return True
        
        # Check if they share common prefixes/suffixes
        common_prefix = len(set(parts1[:-1]).intersection(set(parts2[:-1]))) > 0
        if common_prefix:
            return True
        
        return False
    
    def validate_completeness(self):
        """
        Check for missing values and completeness of features.
        
        Returns:
            dict: Completeness validation results
        """
        completeness_results = {
            'missing_counts': {},
            'missing_rates': {},
            'incomplete_features': []
        }
        
        # Prepare data for validation
        if self.features_df is not None:
            feature_data = self.features_df
        else:
            # Convert player_features to DataFrame
            records = []
            for player_id, features in self.player_features.items():
                record = {'player_id': player_id}
                record.update(features)
                records.append(record)
            
            feature_data = pd.DataFrame(records)
        
        # Calculate missing values for each feature
        missing_counts = feature_data.isna().sum()
        missing_rates = missing_counts / len(feature_data)
        
        # Identify features with high missing rates
        high_missing_features = missing_rates[missing_rates > 0.1].index.tolist()
        
        # Store results
        completeness_results['missing_counts'] = missing_counts.to_dict()
        completeness_results['missing_rates'] = missing_rates.to_dict()
        completeness_results['incomplete_features'] = high_missing_features
        
        # Add to feature issues
        for feature in high_missing_features:
            missing_rate = missing_rates[feature]
            self.feature_issues[feature].append(
                f"High missing value rate: {missing_counts[feature]} missing values ({missing_rate*100:.1f}%)"
            )
        
        # Store results
        self.validation_results['completeness'] = completeness_results
        
        logger.info(f"Found {len(high_missing_features)} features with high missing rates")
        return completeness_results
    
    def validate_distributions(self):
        """
        Check distribution properties of features.
        
        Returns:
            dict: Distribution validation results
        """
        distribution_results = {
            'skewed_features': [],
            'low_variance_features': [],
            'multimodal_features': [],
            'distribution_stats': {}
        }
        
        # Prepare data for validation
        if self.features_df is not None:
            feature_data = self.features_df
        else:
            # Convert player_features to DataFrame
            records = []
            for player_id, features in self.player_features.items():
                record = {'player_id': player_id}
                record.update(features)
                records.append(record)
            
            feature_data = pd.DataFrame(records)
        
        # Check numeric features
        numeric_features = feature_data.select_dtypes(include=['float64', 'int64']).columns
        
        for feature in numeric_features:
            values = feature_data[feature].dropna()
            
            if len(values) < 5:  # Need enough data for distribution analysis
                continue
            
            # Calculate basic statistics
            stats_dict = {
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'range': values.max() - values.min(),
                'iqr': values.quantile(0.75) - values.quantile(0.25),
                'skewness': values.skew(),
                'kurtosis': values.kurtosis()
            }
            
            # Check for skewness
            skewness = stats_dict['skewness']
            if abs(skewness) > 1.0:
                distribution_results['skewed_features'].append(feature)
                
                # Add to feature issues
                skew_direction = 'positive' if skewness > 0 else 'negative'
                self.feature_issues[feature].append(
                    f"Highly {skew_direction} skewed distribution (skewness = {skewness:.2f})"
                )
            
            # Check for low variance
            variance = stats_dict['std'] ** 2
            cv = stats_dict['std'] / stats_dict['mean'] if stats_dict['mean'] != 0 else float('inf')
            
            if cv < 0.1 and feature_data[feature].nunique() > 1:
                distribution_results['low_variance_features'].append(feature)
                
                # Add to feature issues
                self.feature_issues[feature].append(
                    f"Low variance: coefficient of variation = {cv:.3f}"
                )
            
            # Check for multimodality (using kernel density estimation)
            try:
                kde = stats.gaussian_kde(values)
                x = np.linspace(values.min(), values.max(), 1000)
                y = kde(x)
                
                # Find peaks in density
                peaks = []
                for i in range(1, len(y) - 1):
                    if y[i] > y[i-1] and y[i] > y[i+1]:
                        peaks.append((x[i], y[i]))
                
                # If more than one peak, might be multimodal
                if len(peaks) > 1:
                    # Check if the peaks are significant
                    significant_peaks = []
                    max_peak = max(peaks, key=lambda p: p[1])
                    
                    for peak in peaks:
                        if peak[1] > 0.3 * max_peak[1]:  # Significant if at least 30% of max peak
                            significant_peaks.append(peak)
                    
                    if len(significant_peaks) > 1:
                        distribution_results['multimodal_features'].append(feature)
                        
                        # Add to feature issues
                        self.feature_issues[feature].append(
                            f"Potentially multimodal distribution with {len(significant_peaks)} peaks"
                        )
            except Exception as e:
                logger.warning(f"Error checking multimodality for feature {feature}: {e}")
            
            # Store statistics
            distribution_results['distribution_stats'][feature] = stats_dict
        
        # Store results
        self.validation_results['distributions'] = distribution_results
        
        logger.info(f"Found distribution issues: {len(distribution_results['skewed_features'])} skewed, "
                   f"{len(distribution_results['low_variance_features'])} low variance, "
                   f"{len(distribution_results['multimodal_features'])} multimodal")
        return distribution_results
    
    def generate_validation_summary(self):
        """
        Generate a summary of all validation results.
        
        Returns:
            dict: Validation summary
        """
        summary = {
            'total_features': 0,
            'features_with_issues': 0,
            'total_issues': 0,
            'issue_counts_by_type': {
                'outliers': 0,
                'consistency': 0,
                'completeness': 0,
                'distribution': 0
            },
            'critical_features': [],
            'features_by_severity': {
                'high': [],
                'medium': [],
                'low': []
            }
        }
        
        # Count features
        if self.features_df is not None:
            summary['total_features'] = len(self.features_df.columns)
        elif self.player_features:
            # Get a sample player
            sample_player_id = next(iter(self.player_features))
            summary['total_features'] = len(self.player_features[sample_player_id])
        
        # Count issues
        summary['features_with_issues'] = len(self.feature_issues)
        summary['total_issues'] = sum(len(issues) for issues in self.feature_issues.values())
        
        # Count issues by type
        if 'outliers' in self.validation_results:
            summary['issue_counts_by_type']['outliers'] = len(self.validation_results['outliers']['outlier_features'])
        
        if 'consistency' in self.validation_results:
            summary['issue_counts_by_type']['consistency'] = len(self.validation_results['consistency']['inconsistent_features'])
        
        if 'completeness' in self.validation_results:
            summary['issue_counts_by_type']['completeness'] = len(self.validation_results['completeness']['incomplete_features'])
        
        if 'distributions' in self.validation_results:
            distribution_issues = len(self.validation_results['distributions']['skewed_features'])
            distribution_issues += len(self.validation_results['distributions']['low_variance_features'])
            distribution_issues += len(self.validation_results['distributions']['multimodal_features'])
            summary['issue_counts_by_type']['distribution'] = distribution_issues
        
        # Categorize features by severity
        for feature, issues in self.feature_issues.items():
            # Assign severity based on issue count and types
            severity = self._assess_feature_severity(feature, issues)
            
            summary['features_by_severity'][severity].append(feature)
            
            # Check if it's a critical feature
            feature_type = self.feature_types.get(feature, 'unknown')
            is_critical = False
            
            # Features like ELO, win ratios, and serve stats are typically critical
            if feature_type in ['rating', 'ratio'] and len(issues) > 0:
                is_critical = True
            elif any(term in feature.lower() for term in ['elo', 'win_ratio', 'first_serve', 'ace']):
                is_critical = True
            
            if is_critical and severity in ['high', 'medium']:
                summary['critical_features'].append({
                    'feature': feature,
                    'severity': severity,
                    'issues': issues
                })
        
        return summary
    
    def _assess_feature_severity(self, feature, issues):
        """
        Assess the severity of issues for a feature.
        
        Args:
            feature (str): Feature name
            issues (list): List of issues for the feature
            
        Returns:
            str: Severity level ('high', 'medium', 'low')
        """
        # Count issues
        issue_count = len(issues)
        
        # Check issue types
        has_outliers = any('outlier' in issue.lower() for issue in issues)
        has_consistency = any('should be' in issue.lower() or 'violation' in issue.lower() for issue in issues)
        has_completeness = any('missing value' in issue.lower() for issue in issues)
        has_distribution = any('skewed' in issue.lower() or 'variance' in issue.lower() for issue in issues)
        
        # Check issue severity
        has_high_rate = any('% of values' in issue and float(issue.split('(')[1].split('%')[0]) > 30 for issue in issues)
        
        # Assess severity
        if issue_count >= 3 or (issue_count >= 2 and has_high_rate):
            return 'high'
        elif issue_count >= 2 or has_consistency or has_completeness:
            return 'medium'
        else:
            return 'low'
    
    def plot_feature_distributions(self, features=None, top_n=10, figsize=(15, 10), save_path=None):
        """
        Plot distributions of selected features.
        
        Args:
            features (list, optional): List of features to plot
            top_n (int): Number of features to plot if not specified
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Prepare data for validation
        if self.features_df is not None:
            feature_data = self.features_df
        else:
            # Convert player_features to DataFrame
            records = []
            for player_id, features in self.player_features.items():
                record = {'player_id': player_id}
                record.update(features)
                records.append(record)
            
            feature_data = pd.DataFrame(records)
        
        # If no features specified, use features with issues or top numeric features
        if features is None:
            if self.feature_issues:
                # Get top features with issues
                features = sorted(self.feature_issues.keys(), 
                                 key=lambda f: len(self.feature_issues[f]), 
                                 reverse=True)[:top_n]
            else:
                # Use top numeric features
                numeric_features = feature_data.select_dtypes(include=['float64', 'int64']).columns
                features = numeric_features[:min(top_n, len(numeric_features))]
        
        # Filter to numeric features
        numeric_features = [f for f in features if feature_data[f].dtype in ['float64', 'int64']]
        
        if not numeric_features:
            logger.warning("No numeric features available for plotting")
            return None
        
        # Determine grid size
        n_features = len(numeric_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each feature
        for i, feature in enumerate(numeric_features):
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = feature_data[feature].dropna()
            
            # Plot distribution
            sns.histplot(values, kde=True, ax=ax)
            
            # Get feature information
            feature_type = self.feature_types.get(feature, 'unknown')
            issues = self.feature_issues.get(feature, [])
            
            # Set title
            title = f"{feature} ({feature_type})"
            if issues:
                title += f"\n({len(issues)} issues)"
            ax.set_title(title)
            
            # Add distribution statistics
            if 'distributions' in self.validation_results and feature in self.validation_results['distributions']['distribution_stats']:
                stats = self.validation_results['distributions']['distribution_stats'][feature]
                stats_text = f"Mean: {stats['mean']:.3f}\nMedian: {stats['median']:.3f}\nStd: {stats['std']:.3f}\nSkew: {stats['skewness']:.3f}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Highlight issues if any
            if feature in self.validation_results.get('outliers', {}).get('outlier_features', []):
                ax.set_facecolor('#fff3f3')  # Light red background
            elif feature in self.validation_results.get('consistency', {}).get('inconsistent_features', []):
                ax.set_facecolor('#f3fff3')  # Light green background
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        plt.suptitle("Feature Distributions", fontsize=16, y=1.02)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_feature_correlations(self, features=None, top_n=20, figsize=(12, 10), save_path=None):
        """
        Plot correlation matrix of selected features.
        
        Args:
            features (list, optional): List of features to plot
            top_n (int): Number of features to plot if not specified
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Prepare data for validation
        if self.features_df is not None:
            feature_data = self.features_df
        else:
            # Convert player_features to DataFrame
            records = []
            for player_id, features in self.player_features.items():
                record = {'player_id': player_id}
                record.update(features)
                records.append(record)
            
            feature_data = pd.DataFrame(records)
        
        # Filter to numeric features
        numeric_data = feature_data.select_dtypes(include=['float64', 'int64'])
        
        # If no features specified, use top correlated features
        if features is None:
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Get top correlated features
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr = corr_matrix.iloc[i, j]
                    corr_pairs.append((col1, col2, abs(corr)))
            
            # Sort by correlation strength
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Get unique features from top correlations
            top_features = set()
            for col1, col2, _ in corr_pairs[:top_n]:
                top_features.add(col1)
                top_features.add(col2)
                if len(top_features) >= top_n:
                    break
            
            features = list(top_features)
        
        # Filter to selected features
        features = [f for f in features if f in numeric_data.columns]
        
        if len(features) < 2:
            logger.warning("Not enough numeric features for correlation plot")
            return None
        
        # Subset data
        subset_data = numeric_data[features]
        
        # Calculate correlation matrix
        corr_matrix = subset_data.corr()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        
        # Add title
        plt.title("Feature Correlation Matrix", fontsize=16)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return plt.gcf()
    
    def get_feature_recommendations(self):
        """
        Get recommendations for fixing feature issues.
        
        Returns:
            dict: Dictionary of recommendations by feature
        """
        if not self.feature_issues:
            logger.warning("No feature issues found. Run validate_features() first.")
            return {}
        
        recommendations = {}
        
        for feature, issues in self.feature_issues.items():
            feature_type = self.feature_types.get(feature, 'unknown')
            recommendations[feature] = []
            
            for issue in issues:
                if 'outlier' in issue.lower():
                    # Recommend handling outliers
                    if feature_type in ['ratio', 'statistic']:
                        recommendations[feature].append(
                            "Cap extreme values to reasonable bounds (e.g., winsorize at 1% and 99% percentiles)"
                        )
                    else:
                        recommendations[feature].append(
                            "Investigate outliers and consider removing or transforming them"
                        )
                
                elif 'missing value' in issue.lower():
                    # Recommend handling missing values
                    if feature_type in ['ratio', 'statistic']:
                        recommendations[feature].append(
                            "Impute missing values with median or mean of similar players"
                        )
                    elif feature_type in ['count']:
                        recommendations[feature].append(
                            "Impute missing counts with 0 or median based on context"
                        )
                    else:
                        recommendations[feature].append(
                            "Impute missing values or consider dropping feature if too many values are missing"
                        )
                
                elif 'range violation' in issue.lower() or 'should be' in issue.lower():
                    # Recommend fixing consistency issues
                    recommendations[feature].append(
                        "Fix inconsistent values to ensure they satisfy logical constraints"
                    )
                
                elif 'skewed' in issue.lower():
                    # Recommend handling skewed distributions
                    if 'positive' in issue.lower():
                        recommendations[feature].append(
                            "Apply log or square root transformation to reduce positive skew"
                        )
                    else:
                        recommendations[feature].append(
                            "Apply power transformation to normalize distribution"
                        )
                
                elif 'low variance' in issue.lower():
                    # Recommend handling low variance
                    recommendations[feature].append(
                        "Consider dropping feature due to low information content"
                    )
                
                elif 'multimodal' in issue.lower():
                    # Recommend handling multimodality
                    recommendations[feature].append(
                        "Investigate if feature represents distinct groups and consider creating separate features"
                    )
            
            # Remove duplicates
            recommendations[feature] = list(set(recommendations[feature]))
        
        return recommendations