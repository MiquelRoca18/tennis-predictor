#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tennis Match Prediction Model Training Script

This script serves as the main entry point for training tennis prediction models.
It orchestrates the complete workflow from data loading to model evaluation and serialization.

Usage:
    python train_models.py --config path/to/config.json [--model {ensemble,xgboost,neural_net,all}]
                           [--features {basic,advanced,all}] [--evaluation {basic,detailed}]
                           [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
                           [--surfaces {all,hard,clay,grass,carpet}] [--tours {atp,wta,all}]
                           [--experiment-name NAME] [--debug] [--force]

Example:
    python train_models.py --config config/training_config.json --model all 
                           --features advanced --evaluation detailed
                           --start-date 2015-01-01 --end-date 2023-12-31
                           --surfaces all --tours atp --experiment-name atp_full_2015_2023

Author: Your Name
Date: April 2025
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to sys.path to allow imports from sibling packages
sys.path.insert(0, str(Path(__file__).parent.parent))

# Internal imports from the ml_training package
from utils.logging_config import setup_logging
from utils.data_loader import DataLoader
from utils.serialization import ModelSerializer
from utils.memory_manager import MemoryManager

from features.feature_extractor import TennisFeatureExtractor
from features.feature_selector import FeatureSelector
from features.feature_transformer import FeatureTransformer
from features.feature_validator import TennisFeatureValidator as FeatureValidator

from models.ensemble_model import TennisEnsembleModel
from models.xgboost_model import TennisXGBoostModel
from models.neural_net_model import TennisNeuralNetModel

from training.model_trainer import ModelTrainer
from training.cross_validation import TemporalCrossValidator
from training.hyperparameter_tuning import HyperparameterTuner
from training.tennis_data_processor import TennisDataProcessor

from evaluation.metrics import TennisMetrics
from evaluation.results_analyzer import ResultsAnalyzer
from evaluation.temporal_analysis import TemporalAnalyzer
from evaluation.bias_detector import BiasDetector
from evaluation.model_comparator import ModelComparator


def parse_arguments():
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description='Train tennis prediction models')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the training configuration JSON file')
    
    # Optional arguments with defaults
    parser.add_argument('--model', type=str, default='all',
                        choices=['ensemble', 'xgboost', 'neural_net', 'all'],
                        help='Type of model to train (default: all)')
    
    parser.add_argument('--features', type=str, default='advanced',
                        choices=['basic', 'advanced', 'all'],
                        help='Level of feature engineering to use (default: advanced)')
    
    parser.add_argument('--evaluation', type=str, default='detailed',
                        choices=['basic', 'detailed'],
                        help='Level of evaluation detail (default: detailed)')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for training data (YYYY-MM-DD format)')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for training data (YYYY-MM-DD format)')
    
    parser.add_argument('--surfaces', type=str, default='all',
                        choices=['all', 'hard', 'clay', 'grass', 'carpet'],
                        help='Court surfaces to include (default: all)')
    
    parser.add_argument('--tours', type=str, default='all',
                        choices=['atp', 'wta', 'all'],
                        help='Tennis tours to include (default: all)')
    
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for this training experiment (for logging and model naming)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose logging')
    
    parser.add_argument('--force', action='store_true',
                        help='Force retraining even if model files exist')
    
    return parser.parse_args()


def load_config(config_path):
    """Load and validate the training configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required configuration sections
        required_sections = ['data_sources', 'output_paths', 'features', 'training_params']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to load configuration from {config_path}: {str(e)}")
        sys.exit(1)


def override_config_with_args(config, args):
    """Override configuration values with command line arguments."""
    if args.start_date:
        config['training_params']['start_date'] = args.start_date
    
    if args.end_date:
        config['training_params']['end_date'] = args.end_date
    
    if args.surfaces != 'all':
        config['training_params']['surfaces'] = [args.surfaces]
    
    if args.tours != 'all':
        config['training_params']['tours'] = [args.tours]
    
    # Set feature level based on argument
    if args.features == 'basic':
        config['features']['use_advanced_features'] = False
    elif args.features == 'advanced':
        config['features']['use_advanced_features'] = True
    
    # Handle model selection
    if args.model != 'all':
        for model_type in config['training_params']['models']:
            config['training_params']['models'][model_type]['enabled'] = (model_type == args.model)
    
    return config


def setup_experiment(config, args):
    """Set up experiment directories and logging."""
    # Create experiment name if not provided
    if not args.experiment_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tours = '_'.join(config['training_params'].get('tours', ['all']))
        surfaces = '_'.join(config['training_params'].get('surfaces', ['all']))
        args.experiment_name = f"{tours}_{surfaces}_{timestamp}"
    
    # Set up experiment directories
    experiment_dir = os.path.join(config['output_paths']['models_dir'], args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up subdirectories
    logs_dir = os.path.join(experiment_dir, 'logs')
    models_dir = os.path.join(experiment_dir, 'models')
    results_dir = os.path.join(experiment_dir, 'results')
    visualizations_dir = os.path.join(experiment_dir, 'visualizations')
    
    for directory in [logs_dir, models_dir, results_dir, visualizations_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = os.path.join(logs_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_file, log_level)
    
    # Save experiment configuration
    experiment_config = {
        'name': args.experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'command_line_args': vars(args)
    }
    
    with open(os.path.join(experiment_dir, 'experiment_config.json'), 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    # Add directories to config for later use
    config['experiment'] = {
        'name': args.experiment_name,
        'dir': experiment_dir,
        'logs_dir': logs_dir,
        'models_dir': models_dir,
        'results_dir': results_dir,
        'visualizations_dir': visualizations_dir
    }
    
    return config


def load_model_params(config, model_type):
    """Load model-specific parameters from configuration files."""
    model_params_file = None
    
    if model_type == 'ensemble':
        model_params_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'config', 'ensemble_params.json'
        )
    elif model_type == 'xgboost':
        model_params_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'config', 'xgboost_params.json'
        )
    elif model_type == 'neural_net':
        model_params_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'config', 'neural_net_params.json'
        )
    
    if model_params_file and os.path.exists(model_params_file):
        try:
            with open(model_params_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load model parameters from {model_params_file}: {str(e)}")
            logging.warning("Using default parameters instead.")
    
    # Return default parameters if file not found or error occurred
    return config['training_params']['models'].get(model_type, {}).get('params', {})


def train_model(model_type, data, features, config, experiment_dir):
    """Train a specific model type with the given data and features."""
    logging.info(f"Starting training for model type: {model_type}")
    start_time = time.time()
    
    # Load model-specific parameters
    model_params = load_model_params(config, model_type)
    
    # Set up cross-validation
    cv = TemporalCrossValidator(
        n_splits=config['training_params'].get('cv_splits', 5),
        test_period=config['training_params'].get('test_period', '3M')
    )
    
    # Create appropriate model instance
    model = None
    if model_type == 'ensemble':
        model = TennisEnsembleModel(**model_params)
    elif model_type == 'xgboost':
        model = TennisXGBoostModel(**model_params)
    elif model_type == 'neural_net':
        model = TennisNeuralNetModel(**model_params)
    else:
        logging.error(f"Unknown model type: {model_type}")
        return None
    
    # Set up model trainer
    trainer = ModelTrainer(
        model=model,
        cv=cv,
        model_type=model_type,
        experiment_dir=experiment_dir,
        config=config
    )
    
    # Train the model
    trained_model = trainer.train(
        X=features, 
        y=data['target'],
        optimize_hyperparams=config['training_params'].get('optimize_hyperparams', True)
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    logging.info(f"Model {model_type} training completed in {training_time:.2f} seconds")
    
    # Save the model
    model_file = os.path.join(
        config['experiment']['models_dir'], 
        f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.model"
    )
    serializer = ModelSerializer()
    serializer.save(
        model=trained_model,
        filepath=model_file,
        metadata={
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'training_time': training_time,
            'data_info': {
                'start_date': config['training_params'].get('start_date'),
                'end_date': config['training_params'].get('end_date'),
                'surfaces': config['training_params'].get('surfaces'),
                'tours': config['training_params'].get('tours'),
                'sample_count': len(features)
            }
        }
    )
    
    logging.info(f"Model saved to {model_file}")
    
    return trained_model


def evaluate_models(models, data, features, config):
    """Evaluate trained models and generate performance reports."""
    logging.info("Starting model evaluation")
    
    # Split data for evaluation
    data_processor = TennisDataProcessor()
    X_train, X_test, y_train, y_test = data_processor.temporal_train_test_split(
        features, 
        data['target'],
        test_size=config['training_params'].get('test_size', 0.2),
        test_start_date=config['training_params'].get('test_start_date')
    )
    
    # Create metrics calculator
    metrics = TennisMetrics()
    
    # Create results analyzer
    analyzer = ResultsAnalyzer(
        metrics=metrics,
        output_dir=config['experiment']['results_dir'],
        visualizations_dir=config['experiment']['visualizations_dir']
    )
    
    # Evaluate each model
    evaluation_results = {}
    for model_type, model in models.items():
        logging.info(f"Evaluating model: {model_type}")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        results = metrics.calculate_all_metrics(y_test, y_pred_proba)
        evaluation_results[model_type] = results
        
        # Generate detailed analysis if requested
        if config['training_params'].get('detailed_evaluation', True):
            # Analyze results by surface, tournament level, etc.
            analyzer.analyze_by_surface(
                model, X_test, y_test, 
                surface_column='surface',
                model_name=model_type
            )
            
            analyzer.analyze_by_tournament_level(
                model, X_test, y_test, 
                tourney_level_column='tourney_level',
                model_name=model_type
            )
            
            # Analyze temporal performance
            temporal_analyzer = TemporalAnalyzer(
                output_dir=config['experiment']['results_dir'],
                visualizations_dir=config['experiment']['visualizations_dir']
            )
            temporal_analyzer.analyze_performance_over_time(
                model, X_test, y_test,
                date_column='tourney_date',
                model_name=model_type
            )
            
            # Check for bias
            bias_detector = BiasDetector(
                output_dir=config['experiment']['results_dir']
            )
            bias_detector.detect_bias(
                model, X_test, y_test, 
                protected_attributes=['player1_hand', 'player2_hand'],
                model_name=model_type
            )
    
    # Compare models if multiple models were trained
    if len(models) > 1:
        comparator = ModelComparator(
            metrics=metrics,
            output_dir=config['experiment']['results_dir'],
            visualizations_dir=config['experiment']['visualizations_dir']
        )
        comparator.compare_models(
            models=models,
            X_test=X_test,
            y_test=y_test,
            metrics_to_compare=['accuracy', 'roc_auc', 'log_loss', 'brier_score']
        )
    
    # Save evaluation results
    results_file = os.path.join(
        config['experiment']['results_dir'], 
        f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logging.info(f"Evaluation results saved to {results_file}")
    
    return evaluation_results


def main():
    """Main function to orchestrate the tennis prediction model training process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    config = override_config_with_args(config, args)
    
    # Set up experiment directories and logging
    config = setup_experiment(config, args)
    
    logging.info(f"Starting tennis prediction model training experiment: {args.experiment_name}")
    logging.info(f"Configuration loaded from: {args.config}")
    
    # Initialize memory manager to monitor and optimize memory usage
    memory_manager = MemoryManager(threshold_mb=config.get('memory_threshold_mb', 1000))
    
    try:
        # Load data
        logging.info("Loading data...")
        data_loader = DataLoader(
            data_dir=config['data_sources']['data_dir'],
            atp_dir=config['data_sources'].get('atp_dir'),
            wta_dir=config['data_sources'].get('wta_dir')
        )
        
        # Filter data based on configuration
        matches_data = data_loader.load_matches(
            tours=config['training_params'].get('tours', ['atp', 'wta']),
            start_date=config['training_params'].get('start_date'),
            end_date=config['training_params'].get('end_date'),
            surfaces=config['training_params'].get('surfaces')
        )
        
        players_data = data_loader.load_players(
            tours=config['training_params'].get('tours', ['atp', 'wta'])
        )
        
        rankings_data = data_loader.load_rankings(
            tours=config['training_params'].get('tours', ['atp', 'wta']),
            start_date=config['training_params'].get('start_date'),
            end_date=config['training_params'].get('end_date')
        )
        
        # Process data for training
        data_processor = TennisDataProcessor()
        processed_data = data_processor.prepare_data_for_training(
            matches=matches_data,
            players=players_data,
            rankings=rankings_data,
            target_column=config['training_params'].get('target_column', 'winner'),
            filter_retirements=config['training_params'].get('filter_retirements', True),
            filter_walkovers=config['training_params'].get('filter_walkovers', True)
        )
        
        # Feature engineering
        logging.info("Extracting features...")
        feature_extractor = TennisFeatureExtractor(
            use_advanced_features=config['features'].get('use_advanced_features', True),
            include_elo=config['features'].get('include_elo', True),
            include_h2h=config['features'].get('include_h2h', True),
            include_surface_specific=config['features'].get('include_surface_specific', True),
            include_temporal=config['features'].get('include_temporal', True)
        )
        
        all_features = feature_extractor.extract_all_features(
            matches=processed_data['matches'],
            players=processed_data['players'],
            rankings=processed_data['rankings']
        )
        
        # Feature selection
        if config['features'].get('perform_selection', True):
            logging.info("Selecting features...")
            feature_selector = FeatureSelector(
                selection_method=config['features'].get('selection_method', 'recursive'),
                n_features=config['features'].get('n_features', 50)
            )
            selected_features = feature_selector.select_features(
                X=all_features,
                y=processed_data['target']
            )
        else:
            selected_features = all_features
        
        # Feature transformation
        logging.info("Transforming features...")
        feature_transformer = FeatureTransformer(
            normalization=config['features'].get('normalization', 'standard'),
            handle_missing=config['features'].get('handle_missing', 'mean')
        )
        
        transformed_features = feature_transformer.fit_transform(selected_features)
        
        # Validate features
        logging.info("Validating features...")
        feature_validator = FeatureValidator()
        validation_result = feature_validator.validate(
            features=transformed_features,
            target=processed_data['target']
        )
        
        if not validation_result['valid']:
            logging.warning(f"Feature validation found issues: {validation_result['issues']}")
            if not config.get('ignore_validation_errors', False):
                raise ValueError("Feature validation failed. Use --force to ignore.")
        
        # Train models based on configuration
        logging.info("Training models...")
        models = {}
        model_types = []
        
        if args.model == 'all':
            model_types = ['ensemble', 'xgboost', 'neural_net']
        else:
            model_types = [args.model]
        
        for model_type in model_types:
            if config['training_params']['models'].get(model_type, {}).get('enabled', True):
                logging.info(f"Training {model_type} model...")
                model = train_model(
                    model_type=model_type,
                    data=processed_data,
                    features=transformed_features,
                    config=config,
                    experiment_dir=config['experiment']['dir']
                )
                
                if model:
                    models[model_type] = model
        
        # Evaluate models
        if models and config['training_params'].get('perform_evaluation', True):
            logging.info("Evaluating models...")
            evaluation_results = evaluate_models(
                models=models,
                data=processed_data,
                features=transformed_features,
                config=config
            )
            
            # Log summary of results
            logging.info("Evaluation results summary:")
            for model_type, results in evaluation_results.items():
                logging.info(f"  Model: {model_type}")
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        logging.info(f"    {metric}: {value:.4f}")
        
        logging.info(f"Training experiment {args.experiment_name} completed successfully")
        return 0
    
    except Exception as e:
        logging.exception(f"Error during model training: {str(e)}")
        return 1
    finally:
        # Release memory
        memory_manager.cleanup()
        logging.info("Cleaned up resources")


if __name__ == "__main__":
    sys.exit(main())