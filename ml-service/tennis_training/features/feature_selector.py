"""
Feature selector module for tennis match prediction.

This module contains the FeatureSelector class that provides various methods
for selecting the most important features for tennis match prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Class for selecting the most relevant features for tennis match prediction.
    
    This class provides methods for feature selection using various techniques:
    - Correlation-based selection
    - Tree-based feature importance
    - Recursive feature elimination
    - Variance threshold
    - Mutual information
    
    It also allows comparing different feature sets to identify the optimal subset.
    """
    
    def __init__(self, X=None, y=None):
        """
        Initialize the FeatureSelector with optional data.
        
        Args:
            X (pd.DataFrame, optional): Feature matrix
            y (pd.Series, optional): Target variable
        """
        self.X = X
        self.y = y
        self.selection_results = {}
        
        logger.info("FeatureSelector initialized")
    
    def set_data(self, X, y):
        """
        Set or update the data for feature selection.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        self.X = X
        self.y = y
        logger.info(f"Data set with {X.shape[1]} features and {X.shape[0]} samples")
    
    def select_by_correlation(self, threshold=0.7, method='pearson', target_correlation_threshold=0.05):
        """
        Select features based on correlation with target and remove highly correlated features.
        
        Args:
            threshold (float): Correlation threshold for removing correlated features
            method (str): Correlation method ('pearson', 'spearman', or 'kendall')
            target_correlation_threshold (float): Minimum absolute correlation with target
            
        Returns:
            list: Names of selected features
        """
        if self.X is None or self.y is None:
            logger.error("Data not set. Call set_data() first.")
            return None
        
        # Combine features and target
        data = pd.concat([self.X, self.y.rename('target')], axis=1)
        
        # Calculate correlation matrix
        correlation_matrix = data.corr(method=method)
        
        # Find features with sufficient correlation to target
        target_correlations = correlation_matrix['target'].drop('target')
        relevant_features = target_correlations[abs(target_correlations) >= target_correlation_threshold].index.tolist()
        
        logger.info(f"Found {len(relevant_features)} features with correlation >= {target_correlation_threshold}")
        
        # Remove highly correlated features among themselves
        selected_features = []
        
        # Start with feature most correlated with target
        sorted_features = target_correlations[relevant_features].abs().sort_values(ascending=False).index.tolist()
        
        for feature in sorted_features:
            if not selected_features:
                selected_features.append(feature)
            else:
                # Check correlation with already selected features
                correlated = False
                for selected in selected_features:
                    if abs(correlation_matrix.loc[feature, selected]) >= threshold:
                        correlated = True
                        logger.debug(f"Feature {feature} dropped due to correlation {correlation_matrix.loc[feature, selected]:.4f} with {selected}")
                        break
                
                if not correlated:
                    selected_features.append(feature)
        
        logger.info(f"Selected {len(selected_features)} features after removing highly correlated ones")
        
        # Store results
        self.selection_results['correlation'] = {
            'selected_features': selected_features,
            'method': method,
            'threshold': threshold,
            'target_correlation_threshold': target_correlation_threshold,
            'target_correlations': target_correlations.to_dict()
        }
        
        return selected_features
    
    def select_by_tree_importance(self, model_type='rf', n_features=None, threshold=None):
        """
        Select features based on importance derived from tree-based models.
        
        Args:
            model_type (str): Type of tree model ('rf' for Random Forest, 'gb' for Gradient Boosting)
            n_features (int, optional): Number of top features to select
            threshold (float, optional): Importance threshold for feature selection
            
        Returns:
            list: Names of selected features
        """
        if self.X is None or self.y is None:
            logger.error("Data not set. Call set_data() first.")
            return None
        
        # Choose model based on type
        if model_type.lower() == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type.lower() == 'gb':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            logger.error(f"Unknown model type: {model_type}. Use 'rf' or 'gb'.")
            return None
        
        # Fit model
        model.fit(self.X, self.y)
        
        # Get feature importances
        importances = model.feature_importances_
        feature_importances = pd.Series(importances, index=self.X.columns)
        
        # Select features based on parameters
        if n_features is not None:
            selected_features = feature_importances.nlargest(n_features).index.tolist()
            logger.info(f"Selected top {n_features} features based on {model_type} importance")
        elif threshold is not None:
            selected_features = feature_importances[feature_importances >= threshold].index.tolist()
            logger.info(f"Selected {len(selected_features)} features with importance >= {threshold}")
        else:
            # Default to selecting features with importance > mean importance
            mean_importance = np.mean(importances)
            selected_features = feature_importances[feature_importances >= mean_importance].index.tolist()
            logger.info(f"Selected {len(selected_features)} features with importance >= mean importance ({mean_importance:.4f})")
        
        # Store results
        self.selection_results[f'{model_type}_importance'] = {
            'selected_features': selected_features,
            'model_type': model_type,
            'feature_importances': feature_importances.to_dict(),
            'n_features': n_features,
            'threshold': threshold
        }
        
        return selected_features
    
    def select_by_rfe(self, n_features=None, cv=5, step=1, scoring='accuracy'):
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            n_features (int, optional): Number of features to select, if None will use cross-validation
            cv (int): Number of cross-validation folds if using RFECV
            step (int): Number of features to remove at each iteration
            scoring (str): Scoring metric for cross-validation
            
        Returns:
            list: Names of selected features
        """
        if self.X is None or self.y is None:
            logger.error("Data not set. Call set_data() first.")
            return None
        
        # Base estimator for RFE
        estimator = LogisticRegression(random_state=42, max_iter=1000)
        
        # Choose between RFE and RFECV
        if n_features is not None:
            selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
            logger.info(f"Using RFE to select {n_features} features")
        else:
            selector = RFECV(estimator=estimator, step=step, cv=cv, scoring=scoring, n_jobs=-1)
            logger.info(f"Using RFECV to select optimal number of features with CV={cv}")
        
        # Fit selector
        selector.fit(self.X, self.y)
        
        # Get selected features
        selected_features = self.X.columns[selector.support_].tolist()
        
        # Additional info for RFECV
        n_selected = len(selected_features)
        if n_features is None:
            logger.info(f"RFECV selected {n_selected} features as optimal")
            optimal_n = selector.n_features_
            cv_results = selector.cv_results_
        else:
            logger.info(f"RFE selected {n_selected} features")
        
        # Store results
        self.selection_results['rfe'] = {
            'selected_features': selected_features,
            'n_features': n_features if n_features is not None else selector.n_features_,
            'ranking': {feature: rank for feature, rank in zip(self.X.columns, selector.ranking_)},
            'cv_used': n_features is None,
        }
        
        if n_features is None:
            self.selection_results['rfe']['cv_results'] = {
                'mean_score': list(cv_results['mean_test_score']),
                'std_score': list(cv_results['std_test_score']),
                'n_features': list(range(1, len(cv_results['mean_test_score']) + 1))
            }
        
        return selected_features
    
    def select_by_mutual_info(self, n_features=None, threshold=None, discrete_features='auto'):
        """
        Select features based on mutual information with the target.
        
        Args:
            n_features (int, optional): Number of top features to select
            threshold (float, optional): Mutual information threshold
            discrete_features (str or array): Specification of discrete features
            
        Returns:
            list: Names of selected features
        """
        if self.X is None or self.y is None:
            logger.error("Data not set. Call set_data() first.")
            return None
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(self.X, self.y, discrete_features=discrete_features, random_state=42)
        feature_mi = pd.Series(mi_scores, index=self.X.columns)
        
        # Select features based on parameters
        if n_features is not None:
            selected_features = feature_mi.nlargest(n_features).index.tolist()
            logger.info(f"Selected top {n_features} features based on mutual information")
        elif threshold is not None:
            selected_features = feature_mi[feature_mi >= threshold].index.tolist()
            logger.info(f"Selected {len(selected_features)} features with mutual information >= {threshold}")
        else:
            # Default to selecting features with MI > mean MI
            mean_mi = np.mean(mi_scores)
            selected_features = feature_mi[feature_mi >= mean_mi].index.tolist()
            logger.info(f"Selected {len(selected_features)} features with mutual information >= mean ({mean_mi:.4f})")
        
        # Store results
        self.selection_results['mutual_info'] = {
            'selected_features': selected_features,
            'feature_mi': feature_mi.to_dict(),
            'n_features': n_features,
            'threshold': threshold
        }
        
        return selected_features
    
    def select_by_variance(self, threshold=0.01):
        """
        Remove low-variance features.
        
        Args:
            threshold (float): Variance threshold for feature selection
            
        Returns:
            list: Names of selected features
        """
        if self.X is None:
            logger.error("Data not set. Call set_data() first.")
            return None
        
        # Scale features to make variance threshold more meaningful
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_scaled)
        
        # Get selected features
        selected_features = self.X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} features with variance >= {threshold}")
        
        # Calculate variances for reporting
        variances = selector.variances_
        feature_variances = pd.Series(variances, index=self.X.columns)
        
        # Store results
        self.selection_results['variance'] = {
            'selected_features': selected_features,
            'threshold': threshold,
            'feature_variances': feature_variances.to_dict()
        }
        
        return selected_features
    
    def compare_feature_sets(self, feature_sets, model=None, cv=5, scoring='accuracy'):
        """
        Compare different feature sets using cross-validation.
        
        Args:
            feature_sets (dict): Dictionary of feature set name to list of features
            model (object, optional): Sklearn-compatible model, default is RandomForestClassifier
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for cross-validation
            
        Returns:
            dict: Dictionary with cross-validation results for each feature set
        """
        if self.X is None or self.y is None:
            logger.error("Data not set. Call set_data() first.")
            return None
        
        # Default model
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        results = {}
        
        for name, features in feature_sets.items():
            # Select only available features that are present in X
            valid_features = [f for f in features if f in self.X.columns]
            
            if len(valid_features) < len(features):
                logger.warning(f"Some features in set '{name}' are not present in the data. {len(features) - len(valid_features)} features removed.")
            
            if not valid_features:
                logger.error(f"No valid features in set '{name}'. Skipping.")
                continue
            
            # Select subset of features
            X_subset = self.X[valid_features]
            
            # Create pipeline with standardization
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Perform cross-validation
            cv_scores = cross_val_score(pipeline, X_subset, self.y, cv=cv, scoring=scoring, n_jobs=-1)
            
            results[name] = {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'cv_scores': list(cv_scores),
                'n_features': len(valid_features),
                'features': valid_features
            }
            
            logger.info(f"Feature set '{name}' ({len(valid_features)} features): mean {scoring} = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Find best feature set
        best_set = max(results.items(), key=lambda x: x[1]['mean_score'])
        logger.info(f"Best feature set: '{best_set[0]}' with mean {scoring} = {best_set[1]['mean_score']:.4f}")
        
        # Store results
        self.selection_results['comparison'] = results
        
        return results
    
    def find_optimal_feature_count(self, method='tree', max_features=None, cv=5, scoring='accuracy'):
        """
        Find the optimal number of features using cross-validation.
        
        Args:
            method (str): Feature selection method ('tree', 'mutual_info', or 'rfe')
            max_features (int, optional): Maximum number of features to consider
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for cross-validation
            
        Returns:
            dict: Results including optimal number of features and scores
        """
        if self.X is None or self.y is None:
            logger.error("Data not set. Call set_data() first.")
            return None
        
        # Determine max features if not specified
        if max_features is None:
            max_features = min(50, self.X.shape[1])
        
        # Model for evaluation
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Get feature importances/scores based on method
        if method == 'tree':
            model.fit(self.X, self.y)
            feature_scores = pd.Series(model.feature_importances_, index=self.X.columns)
        elif method == 'mutual_info':
            mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
            feature_scores = pd.Series(mi_scores, index=self.X.columns)
        elif method == 'rfe':
            # For RFE, we use RFECV directly
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            selector = RFECV(estimator=estimator, step=1, cv=cv, scoring=scoring, n_jobs=-1)
            selector.fit(self.X, self.y)
            
            # Get RFECV results
            results = {
                'optimal_n_features': selector.n_features_,
                'selected_features': list(self.X.columns[selector.support_]),
                'mean_scores': list(selector.cv_results_['mean_test_score']),
                'std_scores': list(selector.cv_results_['std_test_score']),
                'n_features_list': list(range(1, len(selector.cv_results_['mean_test_score']) + 1))
            }
            
            logger.info(f"Optimal number of features: {results['optimal_n_features']}")
            return results
        else:
            logger.error(f"Unknown method: {method}. Use 'tree', 'mutual_info', or 'rfe'.")
            return None
        
        # Sort features by importance
        sorted_features = feature_scores.sort_values(ascending=False).index
        
        # Test different feature counts
        results = {
            'n_features_list': [],
            'mean_scores': [],
            'std_scores': [],
            'feature_sets': {}
        }
        
        # Define feature counts to test
        if max_features <= 10:
            feature_counts = list(range(1, max_features + 1))
        else:
            # Use logarithmic spacing for larger feature sets
            feature_counts = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
            feature_counts = [n for n in feature_counts if n <= max_features]
            if max_features not in feature_counts:
                feature_counts.append(max_features)
        
        # Evaluate each feature count
        for n_features in feature_counts:
            features = sorted_features[:n_features].tolist()
            X_subset = self.X[features]
            
            # Create pipeline with standardization
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Perform cross-validation
            cv_scores = cross_val_score(pipeline, X_subset, self.y, cv=cv, scoring=scoring, n_jobs=-1)
            
            results['n_features_list'].append(n_features)
            results['mean_scores'].append(np.mean(cv_scores))
            results['std_scores'].append(np.std(cv_scores))
            results['feature_sets'][n_features] = features
            
            logger.info(f"{n_features} features: mean {scoring} = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Find optimal number of features
        optimal_idx = np.argmax(results['mean_scores'])
        optimal_n_features = results['n_features_list'][optimal_idx]
        optimal_score = results['mean_scores'][optimal_idx]
        
        results['optimal_n_features'] = optimal_n_features
        results['optimal_score'] = optimal_score
        results['selected_features'] = results['feature_sets'][optimal_n_features]
        
        logger.info(f"Optimal number of features: {optimal_n_features} with {scoring} = {optimal_score:.4f}")
        
        # Store results
        self.selection_results['optimal_count'] = results
        
        return results
    
    def plot_feature_importances(self, method='tree', n_top=20, figsize=(12, 8)):
        """
        Plot feature importances or scores from a specified method.
        
        Args:
            method (str): Feature selection method key in selection_results
            n_top (int): Number of top features to display
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if method not in self.selection_results:
            logger.error(f"Results for method '{method}' not found. Run the corresponding selection method first.")
            return None
        
        # Get feature importances/scores
        if method == 'tree_importance':
            feature_scores = pd.Series(self.selection_results[method]['feature_importances'])
            title = 'Feature Importances from Tree-based Model'
            xlabel = 'Importance'
        elif method == 'mutual_info':
            feature_scores = pd.Series(self.selection_results[method]['feature_mi'])
            title = 'Mutual Information with Target'
            xlabel = 'Mutual Information'
        elif method == 'correlation':
            feature_scores = pd.Series(self.selection_results[method]['target_correlations'])
            title = 'Correlation with Target'
            xlabel = 'Correlation Coefficient'
        elif method == 'rfe':
            feature_scores = pd.Series({
                feature: 1.0 / rank for feature, rank in self.selection_results[method]['ranking'].items()
            })
            title = 'Feature Ranking from RFE'
            xlabel = 'Inverse Ranking (higher is better)'
        elif method == 'variance':
            feature_scores = pd.Series(self.selection_results[method]['feature_variances'])
            title = 'Feature Variances'
            xlabel = 'Variance'
        else:
            logger.error(f"Plotting not supported for method '{method}'")
            return None
        
        # Sort and select top features
        top_features = feature_scores.abs().sort_values(ascending=False).head(n_top)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        top_features.sort_values().plot(kind='barh', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Features')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_optimal_feature_count(self, figsize=(10, 6)):
        """
        Plot cross-validation scores for different feature counts.
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if 'optimal_count' not in self.selection_results:
            logger.error("Optimal feature count results not found. Run find_optimal_feature_count() first.")
            return None
        
        results = self.selection_results['optimal_count']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mean scores with error bars
        ax.errorbar(
            results['n_features_list'], 
            results['mean_scores'], 
            yerr=results['std_scores'], 
            marker='o', 
            linestyle='-', 
            elinewidth=1, 
            capsize=3
        )
        
        # Mark optimal point
        optimal_idx = results['n_features_list'].index(results['optimal_n_features'])
        ax.plot(
            results['optimal_n_features'], 
            results['mean_scores'][optimal_idx], 
            'ro', 
            ms=10, 
            label=f"Optimal: {results['optimal_n_features']} features"
        )
        
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Cross-Validation Score')
        ax.set_title('Performance vs Number of Features')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Use log scale for x-axis if range is large
        if max(results['n_features_list']) / min(results['n_features_list']) > 10:
            ax.set_xscale('log')
            ax.set_xticks(results['n_features_list'])
            ax.set_xticklabels(results['n_features_list'])
        
        return fig
    
    def get_feature_stability(self, methods=None, threshold=0.5):
        """
        Analyze feature stability across different selection methods.
        
        Args:
            methods (list, optional): List of method keys to include
            threshold (float): Minimum fraction of methods for a feature to be considered stable
            
        Returns:
            dict: Feature stability metrics and stable feature list
        """
        if not self.selection_results:
            logger.error("No selection results found. Run selection methods first.")
            return None
        
        # Determine which methods to include
        if methods is None:
            methods = [m for m in self.selection_results.keys() 
                      if m != 'comparison' and m != 'optimal_count']
        
        # Collect all unique features and their occurrence count
        feature_counts = {}
        method_counts = 0
        
        for method in methods:
            if method not in self.selection_results:
                logger.warning(f"Method '{method}' not found in selection results. Skipping.")
                continue
                
            if 'selected_features' in self.selection_results[method]:
                features = self.selection_results[method]['selected_features']
                method_counts += 1
                
                for feature in features:
                    if feature in feature_counts:
                        feature_counts[feature] += 1
                    else:
                        feature_counts[feature] = 1
        
        if method_counts == 0:
            logger.error("No valid methods with selected features found.")
            return None
        
        # Calculate stability scores
        stability_scores = {feature: count / method_counts for feature, count in feature_counts.items()}
        
        # Get stable features based on threshold
        stable_features = [feature for feature, score in stability_scores.items() 
                          if score >= threshold]
        
        # Sort by stability score
        sorted_stability = pd.Series(stability_scores).sort_values(ascending=False)
        
        stability_results = {
            'stability_scores': stability_scores,
            'stable_features': stable_features,
            'sorted_stability': sorted_stability.to_dict(),
            'method_count': method_counts,
            'threshold': threshold
        }
        
        logger.info(f"Found {len(stable_features)} stable features appearing in at least {threshold*100:.0f}% of methods")
        
        return stability_results
    
    def get_final_feature_set(self, strategy='stability', **kwargs):
        """
        Get the final recommended feature set based on a specified strategy.
        
        Args:
            strategy (str): Strategy for final feature selection
                - 'stability': Select stable features across methods
                - 'optimal_count': Use results from find_optimal_feature_count
                - 'best_comparison': Use best set from compare_feature_sets
                - 'method': Use results from a specific method
            **kwargs: Additional arguments for the chosen strategy
            
        Returns:
            list: Final selected features
        """
        if strategy == 'stability':
            # Default parameters
            methods = kwargs.get('methods', None)
            threshold = kwargs.get('threshold', 0.5)
            
            # Get stable features
            stability_results = self.get_feature_stability(methods, threshold)
            if stability_results is None:
                return None
                
            return stability_results['stable_features']
            
        elif strategy == 'optimal_count':
            if 'optimal_count' not in self.selection_results:
                logger.error("Optimal feature count results not found. Run find_optimal_feature_count() first.")
                return None
                
            return self.selection_results['optimal_count']['selected_features']
            
        elif strategy == 'best_comparison':
            if 'comparison' not in self.selection_results:
                logger.error("Comparison results not found. Run compare_feature_sets() first.")
                return None
                
            # Find best set
            results = self.selection_results['comparison']
            best_set = max(results.items(), key=lambda x: x[1]['mean_score'])[0]
            
            return results[best_set]['features']
            
        elif strategy == 'method':
            method = kwargs.get('method')
            if method is None or method not in self.selection_results:
                logger.error(f"Method not specified or not found in results.")
                return None
                
            return self.selection_results[method]['selected_features']
            
        else:
            logger.error(f"Unknown strategy: {strategy}")
            return None
        
    def export_selection_results(self, filename):
        """
        Export feature selection results to a JSON file.
        
        Args:
            filename (str): Path to save the selection results
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.selection_results:
            logger.error("No selection results to export")
            return False
        
        try:
            # Create a serializable copy of the results
            serializable_results = {}
            
            for method_name, results in self.selection_results.items():
                # Convert numpy arrays and pandas objects to lists/dicts
                method_results = {}
                
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        method_results[key] = value.tolist()
                    elif isinstance(value, pd.Series):
                        method_results[key] = value.to_dict()
                    elif isinstance(value, dict):
                        # Recursively convert dict values
                        serializable_dict = {}
                        for k, v in value.items():
                            if isinstance(v, np.ndarray):
                                serializable_dict[k] = v.tolist()
                            elif isinstance(v, pd.Series):
                                serializable_dict[k] = v.to_dict()
                            else:
                                serializable_dict[k] = v
                        method_results[key] = serializable_dict
                    else:
                        method_results[key] = value
                
                serializable_results[method_name] = method_results
            
            # Save to file
            import json
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Selection results exported to {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting selection results: {e}")
            return False

    def import_selection_results(self, filename):
        """
        Import feature selection results from a JSON file.
        
        Args:
            filename (str): Path to the selection results file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import json
            with open(filename, 'r') as f:
                imported_results = json.load(f)
            
            # Convert imported results to appropriate types
            for method_name, results in imported_results.items():
                # If 'selected_features' is present, convert to list
                if 'selected_features' in results:
                    results['selected_features'] = list(results['selected_features'])
                
                # Convert feature importances/scores to pandas Series if needed
                for key in ['feature_importances', 'feature_mi', 'target_correlations', 'feature_variances']:
                    if key in results and isinstance(results[key], dict):
                        results[key] = pd.Series(results[key])
            
            # Update selection results
            self.selection_results = imported_results
            
            logger.info(f"Selection results imported from {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Error importing selection results: {e}")
            return False

    def merge_selection_results(self, other_results):
        """
        Merge selection results from another instance or dictionary.
        
        Args:
            other_results (dict or FeatureSelector): Results to merge
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get results dictionary from input
            if isinstance(other_results, FeatureSelector):
                merge_from = other_results.selection_results
            elif isinstance(other_results, dict):
                merge_from = other_results
            else:
                logger.error("Input must be a FeatureSelector instance or dictionary")
                return False
            
            # Merge results
            for method_name, results in merge_from.items():
                if method_name in self.selection_results:
                    logger.warning(f"Method '{method_name}' already exists, overwriting")
                
                self.selection_results[method_name] = results
            
            logger.info(f"Merged {len(merge_from)} selection results")
            return True
        
        except Exception as e:
            logger.error(f"Error merging selection results: {e}")
            return False