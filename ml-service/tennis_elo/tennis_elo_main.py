#!/usr/bin/env python3
"""
Tennis ELO Rating System - Main Execution Module
================================================

This module serves as the entry point for the Tennis ELO rating system.
It provides a command-line interface for accessing various functionalities
of the system including processing matches, calculating ratings, generating
visualizations, and producing reports.

Usage:
    python tennis_elo_main.py --action <action> [additional options]

Actions:
    calculate-ratings     - Process matches and calculate ELO ratings
    predict-matches       - Predict outcomes of upcoming matches
    evaluate-performance  - Evaluate predictive performance of the system
    generate-visuals      - Generate visualizations of player ratings
    export-reports        - Export reports and data to various formats
    run-full-pipeline     - Execute the complete analysis pipeline
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Internal imports from the package
from tennis_elo.core.processor import EnhancedTennisEloProcessor
from tennis_elo.core.ratings import RatingCalculator
from tennis_elo.utils.data_loader import DataLoader
from tennis_elo.utils.error_tracking import setup_error_tracking, log_error
from tennis_elo.analytics.evaluation import EloEvaluator
from tennis_elo.visualization import distribution_plots, evolution_plots, player_comparison
from tennis_elo.io.persistence import EloDataManager
from tennis_elo.io.reporting import ReportGenerator


def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    return logging.getLogger('tennis_elo')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Tennis ELO Rating System')
    
    # Main action argument
    parser.add_argument('--action', type=str, required=True,
                        choices=['calculate-ratings', 'predict-matches', 
                                 'evaluate-performance', 'generate-visuals',
                                 'export-reports', 'run-full-pipeline'],
                        help='Action to perform')
    
    # Common arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing input data files')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory for output files')
    parser.add_argument('--config', type=str, default='./config.json',
                        help='Configuration file path')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    # Action-specific arguments
    parser.add_argument('--start-date', type=str,
                        help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                        help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--player-ids', type=str, nargs='+',
                        help='Player IDs for specific player analysis')
    parser.add_argument('--surface', type=str,
                        choices=['hard', 'clay', 'grass', 'carpet', 'all'],
                        default='all',
                        help='Court surface to analyze')
    parser.add_argument('--tournament-level', type=str,
                        choices=['grand_slam', 'atp1000', 'atp500', 'atp250', 'challenger', 'all'],
                        default='all',
                        help='Tournament level to analyze')
    
    return parser.parse_args()


def validate_paths(args):
    """Validate and create necessary directories."""
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    return data_path, output_path


def calculate_ratings(args, logger, data_path, output_path):
    """Process matches and calculate ELO ratings."""
    logger.info("Starting ELO rating calculation...")
    
    try:
        # Load data
        data_loader = DataLoader(data_dir=str(data_path))
        matches = data_loader.load_matches(
            start_date=args.start_date,
            end_date=args.end_date,
            surface=args.surface,
            tournament_level=args.tournament_level
        )
        
        logger.info(f"Loaded {len(matches)} matches for processing")
        
        # Initialize processor
        processor = EnhancedTennisEloProcessor()
        
        # Process matches and calculate ratings
        ratings = processor.process_matches(matches)
        
        # Save results
        data_manager = EloDataManager(output_dir=str(output_path))
        data_manager.save_ratings(ratings)
        
        logger.info(f"Successfully calculated ratings for {len(ratings)} players")
        return ratings
    
    except Exception as e:
        log_error(e, "Error during rating calculation")
        raise


def predict_matches(args, logger, data_path, output_path):
    """Predict outcomes of upcoming matches."""
    logger.info("Starting match prediction...")
    
    try:
        # Load data
        data_loader = DataLoader(data_dir=str(data_path))
        upcoming_matches = data_loader.load_upcoming_matches()
        
        # Load existing ratings
        data_manager = EloDataManager(output_dir=str(output_path))
        ratings = data_manager.load_latest_ratings()
        
        # Initialize rating calculator
        calculator = RatingCalculator()
        
        # Generate predictions
        predictions = []
        for match in upcoming_matches:
            prob = calculator.calculate_win_probability(
                ratings.get(match['player1_id'], 1500),
                ratings.get(match['player2_id'], 1500),
                surface=match.get('surface')
            )
            predictions.append({
                'match_id': match.get('match_id'),
                'player1_id': match['player1_id'],
                'player2_id': match['player2_id'],
                'player1_win_probability': prob,
                'predicted_winner': match['player1_id'] if prob > 0.5 else match['player2_id']
            })
        
        # Save predictions
        data_manager.save_predictions(predictions)
        
        logger.info(f"Successfully generated predictions for {len(predictions)} matches")
        return predictions
    
    except Exception as e:
        log_error(e, "Error during match prediction")
        raise


def evaluate_performance(args, logger, data_path, output_path):
    """Evaluate predictive performance of the system."""
    logger.info("Starting performance evaluation...")
    
    try:
        # Load data
        data_loader = DataLoader(data_dir=str(data_path))
        test_matches = data_loader.load_matches(
            start_date=args.start_date,
            end_date=args.end_date,
            purpose='testing'
        )
        
        # Load predictions if available, or generate them
        data_manager = EloDataManager(output_dir=str(output_path))
        try:
            predictions = data_manager.load_predictions()
        except FileNotFoundError:
            logger.info("No existing predictions found, generating new ones...")
            predictions = predict_matches(args, logger, data_path, output_path)
        
        # Initialize evaluator
        evaluator = EloEvaluator()
        
        # Run evaluation
        evaluation_results = evaluator.evaluate(test_matches, predictions)
        
        # Save evaluation results
        data_manager.save_evaluation(evaluation_results)
        
        # Generate evaluation report
        report_generator = ReportGenerator(output_dir=str(output_path))
        report_generator.generate_evaluation_report(evaluation_results)
        
        logger.info("Performance evaluation completed successfully")
        return evaluation_results
    
    except Exception as e:
        log_error(e, "Error during performance evaluation")
        raise


def generate_visuals(args, logger, data_path, output_path):
    """Generate visualizations of player ratings."""
    logger.info("Starting visualization generation...")
    
    try:
        # Load ratings
        data_manager = EloDataManager(output_dir=str(output_path))
        ratings = data_manager.load_latest_ratings()
        rating_history = data_manager.load_rating_history()
        
        # Generate distribution plots
        distribution_plots.plot_rating_distribution(
            ratings, 
            output_file=str(output_path / "rating_distribution.png")
        )
        
        # Generate evolution plots for specific players if requested
        if args.player_ids:
            for player_id in args.player_ids:
                evolution_plots.plot_rating_evolution(
                    rating_history, 
                    player_id=player_id,
                    output_file=str(output_path / f"rating_evolution_{player_id}.png")
                )
        
        # Generate player comparison if multiple players specified
        if args.player_ids and len(args.player_ids) > 1:
            player_comparison.plot_player_comparison(
                rating_history,
                player_ids=args.player_ids,
                output_file=str(output_path / "player_comparison.png")
            )
        
        logger.info("Visualization generation completed successfully")
    
    except Exception as e:
        log_error(e, "Error during visualization generation")
        raise


def export_reports(args, logger, data_path, output_path):
    """Export reports and data to various formats."""
    logger.info("Starting report generation...")
    
    try:
        # Load data
        data_manager = EloDataManager(output_dir=str(output_path))
        ratings = data_manager.load_latest_ratings()
        
        # Initialize report generator
        report_generator = ReportGenerator(output_dir=str(output_path))
        
        # Generate player ratings report
        report_generator.generate_player_ratings_report(ratings)
        
        # Generate tournament prediction report if upcoming data available
        try:
            data_loader = DataLoader(data_dir=str(data_path))
            upcoming_tournaments = data_loader.load_upcoming_tournaments()
            report_generator.generate_tournament_prediction_report(
                upcoming_tournaments, 
                ratings
            )
        except FileNotFoundError:
            logger.warning("No upcoming tournament data found, skipping tournament prediction report")
        
        # Generate system performance report if evaluation data available
        try:
            evaluation_results = data_manager.load_evaluation()
            report_generator.generate_system_performance_report(evaluation_results)
        except FileNotFoundError:
            logger.warning("No evaluation data found, skipping system performance report")
        
        logger.info("Report generation completed successfully")
    
    except Exception as e:
        log_error(e, "Error during report generation")
        raise


def run_full_pipeline(args, logger, data_path, output_path):
    """Execute the complete analysis pipeline."""
    logger.info("Starting full analysis pipeline...")
    
    try:
        # Step 1: Calculate ratings
        ratings = calculate_ratings(args, logger, data_path, output_path)
        
        # Step 2: Generate predictions
        predictions = predict_matches(args, logger, data_path, output_path)
        
        # Step 3: Evaluate performance
        evaluation_results = evaluate_performance(args, logger, data_path, output_path)
        
        # Step 4: Generate visualizations
        generate_visuals(args, logger, data_path, output_path)
        
        # Step 5: Export reports
        export_reports(args, logger, data_path, output_path)
        
        logger.info("Full analysis pipeline completed successfully")
        
    except Exception as e:
        log_error(e, "Error during full pipeline execution")
        raise


def main():
    """Main entry point of the application."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup logging
        logger = setup_logging(args.verbose)
        logger.info(f"Starting Tennis ELO System - Action: {args.action}")
        
        # Setup error tracking
        setup_error_tracking()
        
        # Validate and create paths
        data_path, output_path = validate_paths(args)
        
        # Execute requested action
        if args.action == 'calculate-ratings':
            calculate_ratings(args, logger, data_path, output_path)
        elif args.action == 'predict-matches':
            predict_matches(args, logger, data_path, output_path)
        elif args.action == 'evaluate-performance':
            evaluate_performance(args, logger, data_path, output_path)
        elif args.action == 'generate-visuals':
            generate_visuals(args, logger, data_path, output_path)
        elif args.action == 'export-reports':
            export_reports(args, logger, data_path, output_path)
        elif args.action == 'run-full-pipeline':
            run_full_pipeline(args, logger, data_path, output_path)
        
        logger.info(f"Tennis ELO System completed successfully")
        return 0
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 130
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("\nDetailed error information:")
        print(f"  File: {sys.exc_info()[2].tb_frame.f_code.co_filename}")
        print(f"  Line: {sys.exc_info()[2].tb_lineno}")
        print(f"  Function: {sys.exc_info()[2].tb_frame.f_code.co_name}")
        print(f"  Context: {traceback.format_exc()}")
        print("\nSuggested solution:")
        
        if isinstance(e, FileNotFoundError):
            print("  Please check that all required data files exist and paths are correct")
        elif isinstance(e, PermissionError):
            print("  Please check file permissions for input and output directories")
        elif isinstance(e, ValueError) and "date" in str(e).lower():
            print("  Please check date format (should be YYYY-MM-DD)")
        else:
            print("  Check the error message and traceback for more information")
            print("  If the issue persists, please report it with the error details above")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())