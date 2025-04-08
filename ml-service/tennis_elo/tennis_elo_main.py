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
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from tennis_elo.utils.error_tracking import save_error_report

# Internal imports from the package
from tennis_elo.core.processor import EnhancedTennisEloProcessor
from tennis_elo.core.ratings import RatingCalculator
from tennis_elo.utils.data_loader import TennisDataLoader as DataLoader
from tennis_elo.utils.error_tracking import clear_error_history as setup_error_tracking, save_error_report as log_error
from tennis_elo.analytics.evaluation import evaluate_predictive_power as EloEvaluator
from tennis_elo.visualization import distribution_plots, evolution_plots, player_comparison
from tennis_elo.io.persistence import EloDataManager
from tennis_elo.io.reporting import EloReportGenerator as ReportGenerator

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
                        choices=['grand_slam', 'atp1000', 'atp500', 'atp250', 'challenger', 'futures', 'all'],
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
        # Determinar los archivos a cargar según el nivel de torneo
        tournament_level = args.tournament_level or 'all'
        logger.info(f"Procesando datos para nivel de torneo: {tournament_level}")
        
        # Determinar qué tipos de archivos cargar basado en el nivel de torneo
        if tournament_level == 'all':
            match_types = ['main', 'challenger', 'futures']
        elif tournament_level == 'challenger':
            match_types = ['challenger']
        elif tournament_level == 'futures':
            match_types = ['futures']
        else:  # grand_slam, atp1000, atp500, atp250
            match_types = ['main']  # Estos se filtrarán por nivel más adelante
        
        # Cargar archivos según los tipos de partidos determinados
        matches_dfs = []
        
        for match_type in match_types:
            atp_file = os.path.join(data_path, "atp", f"atp_matches_{match_type}_2000_2024.csv")
            wta_file = os.path.join(data_path, "wta", f"wta_matches_{match_type}_2000_2024.csv")
            
            logger.info(f"Buscando archivo ATP {match_type}: {atp_file}")
            logger.info(f"Buscando archivo WTA {match_type}: {wta_file}")
            
            if os.path.exists(atp_file):
                atp_df = pd.read_csv(atp_file, low_memory=False)
                
                # Si es 'main', filtrar por nivel de torneo específico si es necesario
                if match_type == 'main' and tournament_level not in ['all', 'main']:
                    if 'tourney_level' in atp_df.columns:
                        atp_df = atp_df[atp_df['tourney_level'] == tournament_level]
                        logger.info(f"Filtrado por nivel de torneo {tournament_level}: {len(atp_df)} partidos")
                
                logger.info(f"Cargado archivo ATP {match_type}: {len(atp_df)} partidos")
                matches_dfs.append(atp_df)
            
            if os.path.exists(wta_file):
                wta_df = pd.read_csv(wta_file, low_memory=False)
                
                # Si es 'main', filtrar por nivel de torneo específico si es necesario
                if match_type == 'main' and tournament_level not in ['all', 'main']:
                    if 'tourney_level' in wta_df.columns:
                        wta_df = wta_df[wta_df['tourney_level'] == tournament_level]
                        logger.info(f"Filtrado por nivel de torneo {tournament_level}: {len(wta_df)} partidos")
                
                logger.info(f"Cargado archivo WTA {match_type}: {len(wta_df)} partidos")
                matches_dfs.append(wta_df)
        
        # Combinar dataframes
        if matches_dfs:
            matches = pd.concat(matches_dfs, ignore_index=True)
            logger.info(f"Total de partidos combinados: {len(matches)}")
        else:
            matches = pd.DataFrame()
            logger.warning(f"No se encontraron archivos para el nivel de torneo: {tournament_level}")
        
        # Initialize processor
        processor = EnhancedTennisEloProcessor()
        
        # Process matches and calculate ratings
        if not matches.empty:
            processed_df = processor.process_matches_dataframe(matches)
            logger.info(f"Procesados {len(processed_df)} partidos")
        else:
            logger.warning("No hay partidos para procesar")
        
        # Save results - Pasar el nivel de torneo para incluirlo en los nombres de archivo
        data_manager = EloDataManager(data_dir=str(output_path))
        data_manager.save_ratings(
            processor.player_ratings,
            processor.player_ratings_by_surface,
            processor.player_match_count,
            processor.player_match_count_by_surface,
            processor.player_rating_uncertainty,
            tournament_level=tournament_level  # Añadir nivel de torneo aquí
        )
        
        logger.info(f"Successfully calculated ratings for {len(processor.player_ratings)} players")
        return processor.player_ratings
    
    except Exception as e:
        logger.error(f"Error during rating calculation: {str(e)}")
        raise

def predict_matches(args, logger, data_path, output_path):
    """Predict outcomes of upcoming matches."""
    logger.info("Starting match prediction...")
    
    try:
        # Load data
        data_loader = DataLoader(data_dir=str(data_path))
        upcoming_matches = data_loader.load_upcoming_matches()
        
        # Load existing ratings
        data_manager = EloDataManager(data_dir=str(output_path))
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
        # Verificar rutas exactas para depuración
        logger.info(f"Buscando archivos en: {data_path}")
        atp_file = os.path.join(data_path, "atp", "atp_matches_main_2000_2024.csv")
        wta_file = os.path.join(data_path, "wta", "wta_matches_main_2000_2024.csv")
        
        logger.info(f"¿Existe archivo ATP? {os.path.exists(atp_file)}")
        logger.info(f"¿Existe archivo WTA? {os.path.exists(wta_file)}")
        
        # Cargar archivos directamente
        test_matches = None
        
        if os.path.exists(atp_file):
            atp_df = pd.read_csv(atp_file)
            logger.info(f"Cargado archivo ATP: {len(atp_df)} partidos")
            test_matches = atp_df
        
        if os.path.exists(wta_file):
            wta_df = pd.read_csv(wta_file)
            logger.info(f"Cargado archivo WTA: {len(wta_df)} partidos")
            if test_matches is not None:
                # Intentar encontrar columnas comunes
                common_cols = list(set(test_matches.columns) & set(wta_df.columns))
                test_matches = pd.concat([test_matches[common_cols], wta_df[common_cols]])
            else:
                test_matches = wta_df
        
        # Verificar si pudimos cargar datos
        if test_matches is None or test_matches.empty:
            logger.warning("No hay datos de test disponibles para evaluación")
            return
            
        # Asegurar que la fecha es datetime
        if 'match_date' in test_matches.columns:
            test_matches['match_date'] = pd.to_datetime(test_matches['match_date'], errors='coerce')
        
        # Filtrar por fechas si es necesario
        if args.start_date:
            start_date = pd.to_datetime(args.start_date)
            test_matches = test_matches[test_matches['match_date'] >= start_date]
        if args.end_date:
            end_date = pd.to_datetime(args.end_date)
            test_matches = test_matches[test_matches['match_date'] <= end_date]
        
        if test_matches.empty:
            logger.warning("No hay datos para el rango de fechas especificado")
            return
            
        # Verificar que tenemos las columnas necesarias para la evaluación
        if not all(col in test_matches.columns for col in ['winner_id', 'loser_id']):
            logger.error("Los datos no contienen las columnas requeridas (winner_id, loser_id)")
            return
        
        # Cargar el modelo ELO
        data_manager = EloDataManager(data_dir=str(output_path))
        ratings_data = data_manager.load_ratings()
        
        if not ratings_data.get('player_ratings'):
            logger.error("No se pudieron cargar los ratings ELO")
            return
        
        # Inicializar el procesador con los ratings cargados
        processor = EnhancedTennisEloProcessor()
        processor.player_ratings = ratings_data.get('player_ratings', {})
        processor.player_ratings_by_surface = ratings_data.get('player_ratings_by_surface', {})
        processor.player_match_count = ratings_data.get('player_match_count', {})
        processor.player_match_count_by_surface = ratings_data.get('player_match_count_by_surface', {})
        
        # Generar predicciones para los partidos de test
        predictions = []
        for _, match in test_matches.iterrows():
            try:
                winner_id = str(match['winner_id'])
                loser_id = str(match['loser_id'])
                surface = match['surface'] if 'surface' in match else 'hard'
                
                prediction = processor.predict_match(winner_id, loser_id, surface)
                prob = prediction['prediction']['p1_win_probability']
                
                predictions.append({
                    'match_id': match.get('match_id', ''),
                    'winner_id': winner_id,
                    'loser_id': loser_id,
                    'predicted_prob': prob,
                    'predicted_winner': winner_id if prob > 0.5 else loser_id,
                    'actual_winner': winner_id
                })
            except Exception as e:
                logger.debug(f"Error prediciendo partido: {str(e)}")
                continue
        
        if not predictions:
            logger.warning("No se pudieron generar predicciones")
            return
            
        # Evaluar predicciones
        logger.info(f"Evaluando {len(predictions)} predicciones...")
        correct_predictions = sum(1 for p in predictions if p['predicted_winner'] == p['actual_winner'])
        accuracy = correct_predictions / len(predictions)
        
        # Crear resultados de evaluación
        evaluation_results = {
            'total_matches': len(predictions),
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Guardar resultados de evaluación
        result_path = os.path.join(output_path, 'evaluation_results.json')
        with open(result_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluación completada con precisión del {accuracy*100:.2f}%")
        logger.info(f"Resultados guardados en {result_path}")
        
        return evaluation_results
    
    except Exception as e:
        save_error_report(e, "Error during performance evaluation")
        raise


def load_player_names(data_path):
    """
    Carga los nombres de jugadores desde archivos CSV en el directorio de datos.
    
    Args:
        data_path: Ruta base de datos
        
    Returns:
        Dict: Diccionario {player_id: player_name}
    """
    import os
    import glob
    import pandas as pd
    
    player_names = {}
    
    # Buscar en archivos CSV de jugadores estándar
    for tour in ['atp', 'wta']:
        # Buscar en carpetas típicas
        csv_patterns = [
            os.path.join(data_path, tour, f"{tour}_players.csv"),
            os.path.join(data_path, f"{tour}_players.csv"),
            os.path.join(data_path, "players", f"{tour}_players.csv"),
            os.path.join(data_path, "..", "raw", tour, f"{tour}_players.csv")
        ]
        
        for pattern in csv_patterns:
            player_files = glob.glob(pattern)
            for file_path in player_files:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Detectar columnas apropiadas
                    id_col = None
                    name_col = None
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'id' in col_lower and not id_col:
                            id_col = col
                        elif ('name' in col_lower or 'player' in col_lower) and not name_col:
                            name_col = col
                    
                    if id_col and name_col:
                        # Construir diccionario de nombres
                        for _, row in df.iterrows():
                            player_id = str(row[id_col])
                            player_name = row[name_col]
                            player_names[player_id] = player_name
                            
                except Exception as e:
                    print(f"Error cargando archivo {file_path}: {str(e)}")
    
    # Si no se encontraron nombres, usar diccionario para jugadores conocidos
    if not player_names:
        # Diccionario básico de jugadores conocidos
        player_names = {
            # ATP
            "104925": "Novak Djokovic",
            "106421": "Rafael Nadal",
            "103819": "Roger Federer",
            "207989": "Carlos Alcaraz",
            "106233": "Alexander Zverev",
            "111815": "Jannik Sinner",
            "126774": "Stefanos Tsitsipas",
            "106421": "Rafael Nadal",
            "105223": "Andy Murray",
            "105453": "Stan Wawrinka",
            "134770": "Daniil Medvedev",
            "144895": "Andrey Rublev",
            "126094": "Karen Khachanov",
            "200282": "Felix Auger-Aliassime",
            "206173": "Holger Rune",
            "207989": "Carlos Alcaraz",
            "106401": "Kei Nishikori",
            "106432": "Milos Raonic",
            # WTA
            "230234": "Iga Swiatek",
            "105583": "Serena Williams",
            "106043": "Victoria Azarenka",
            "106034": "Simona Halep",
            "105314": "Maria Sharapova",
            "106400": "Naomi Osaka",
            "105723": "Angelique Kerber",
            "106223": "Petra Kvitova",
            "201520": "Bianca Andreescu",
            "125824": "Sloane Stephens",
            "117644": "Madison Keys",
            "227772": "Emma Raducanu",
            "201520": "Bianca Andreescu",
            "106450": "Karolina Pliskova",
            "105544": "Venus Williams",
        }
    
    return player_names

def generate_visuals(args, logger, data_path, output_path):
    """Generate visualizations of player ratings."""
    logger.info("Starting visualization generation...")
    
    try:
        # Crear el directorio de visualizaciones si no existe
        viz_output_dir = os.path.join(output_path, 'visualizations')
        os.makedirs(viz_output_dir, exist_ok=True)
        
        # La ruta correcta es directamente output_path/ratings
        ratings_file_path = os.path.join(output_path, 'ratings')
        
        logger.info(f"Buscando archivos de ratings en: {ratings_file_path}")
        
        # Buscar los archivos de ratings en formato JSON
        import glob
        import json
        
        # Buscar archivos de ratings generales
        general_rating_files = glob.glob(os.path.join(ratings_file_path, "elo_ratings_general*.json"))
        surface_rating_files = glob.glob(os.path.join(ratings_file_path, "elo_ratings_by_surface*.json"))
        
        if not general_rating_files and not surface_rating_files:
            logger.error(f"No se encontraron archivos de ratings en {ratings_file_path}")
            return
        
        # Cargar nombres de jugadores
        player_names = load_player_names(data_path)
        logger.info(f"Cargados nombres para {len(player_names)} jugadores")
        
        # Cargar el archivo de ratings más reciente
        if general_rating_files:
            # Ordenar archivos por fecha de modificación (más reciente primero)
            general_rating_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_general_file = general_rating_files[0]
            
            logger.info(f"Cargando ratings generales desde: {latest_general_file}")
            with open(latest_general_file, 'r') as f:
                general_ratings = json.load(f)
        else:
            general_ratings = {}
        
        # Cargar ratings por superficie si están disponibles
        surface_ratings = {}
        if surface_rating_files:
            surface_rating_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_surface_file = surface_rating_files[0]
            
            logger.info(f"Cargando ratings por superficie desde: {latest_surface_file}")
            with open(latest_surface_file, 'r') as f:
                surface_ratings = json.load(f)
        
        # Generar distribución de ratings
        distribution_plot_file = os.path.join(viz_output_dir, "rating_distribution.png")
        logger.info(f"Generando distribución de ratings en {distribution_plot_file}")
        
        # Usar la función existente con los ratings generales
        from tennis_elo.visualization.distribution_plots import plot_rating_distribution
        fig = plot_rating_distribution(general_ratings, save_path=distribution_plot_file)
        
        # Generar gráfico de mejores jugadores
        if general_ratings:
            # Convertir a DataFrame para facilitar el ordenamiento
            import pandas as pd
            ratings_df = pd.DataFrame([
                {"player_id": player_id, "elo_rating": rating} 
                for player_id, rating in general_ratings.items()
            ])
            
            # Añadir nombres de jugadores
            ratings_df['player_name'] = ratings_df['player_id'].map(lambda x: player_names.get(x, f"Player {x}"))
            
            # Obtener top 20 jugadores
            top_n = min(20, len(ratings_df))
            top_players = ratings_df.sort_values(by='elo_rating', ascending=False).head(top_n)
            
            # Generar gráfico de mejores jugadores
            top_players_file = os.path.join(viz_output_dir, f"top_{top_n}_players.png")
            logger.info(f"Generando gráfico de top {top_n} jugadores en {top_players_file}")
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            # Usar player_name en lugar de player_id
            bars = plt.barh(top_players['player_name'], top_players['elo_rating'])
            
            # Añadir valores numéricos a las barras
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                         f"{top_players['elo_rating'].iloc[i]:.0f}", 
                         va='center')
                
            plt.title(f'Top {top_n} Players by ELO Rating')
            plt.xlabel('ELO Rating')
            plt.ylabel('Player Name')
            plt.tight_layout()
            plt.savefig(top_players_file, dpi=300)
            plt.close()
        
        # Generar gráficos por superficie si hay datos
        if surface_ratings:
            for surface, ratings in surface_ratings.items():
                if not ratings:
                    continue
                    
                surface_plot_file = os.path.join(viz_output_dir, f"rating_distribution_{surface}.png")
                logger.info(f"Generando distribución de ratings para superficie {surface} en {surface_plot_file}")
                
                # Usar la función existente con los ratings de esta superficie
                fig = plot_rating_distribution(ratings, surface=surface, save_path=surface_plot_file)
                
                # Convertir a DataFrame para el top de jugadores por superficie
                ratings_df = pd.DataFrame([
                    {"player_id": player_id, "elo_rating": rating} 
                    for player_id, rating in ratings.items()
                ])
                
                # Añadir nombres de jugadores
                ratings_df['player_name'] = ratings_df['player_id'].map(lambda x: player_names.get(x, f"Player {x}"))
                
                # Obtener top 20 jugadores
                top_n = min(20, len(ratings_df))
                top_players = ratings_df.sort_values(by='elo_rating', ascending=False).head(top_n)
                
                # Generar gráfico de mejores jugadores por superficie
                top_players_file = os.path.join(viz_output_dir, f"top_{top_n}_players_{surface}.png")
                logger.info(f"Generando gráfico de top {top_n} jugadores en superficie {surface}")
                
                plt.figure(figsize=(12, 8))
                # Usar player_name en lugar de player_id
                bars = plt.barh(top_players['player_name'], top_players['elo_rating'])
                
                # Añadir valores numéricos a las barras
                for i, bar in enumerate(bars):
                    plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                             f"{top_players['elo_rating'].iloc[i]:.0f}", 
                             va='center')
                    
                plt.title(f'Top {top_n} Players by ELO Rating - {surface.upper()} Surface')
                plt.xlabel('ELO Rating')
                plt.ylabel('Player Name')
                plt.tight_layout()
                plt.savefig(top_players_file, dpi=300)
                plt.close()
        
        # Generar visualizaciones específicas por jugador si se solicitaron
        if args.player_ids:
            logger.info(f"Generando visualizaciones para jugadores específicos: {args.player_ids}")
            # Aquí puedes añadir código para generar visualizaciones por jugador
            
        logger.info("Visualization generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during visualization generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Usar la función correctamente
        log_error(e)  # Pasar solo el error, no el mensaje adicional

def export_reports(args, logger, data_path, output_path):
    """Export reports and data to various formats."""
    logger.info("Starting report generation...")
    
    try:
        # Load data
        data_manager = EloDataManager(data_dir=str(output_path))
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
        logger.error(f"Error during report generation: {str(e)}")
        log_error(e)  # Corregido para pasar solo un argumento
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
        logger.error(f"Error during full pipeline execution: {str(e)}")
        log_error(e)  # Corregido para pasar solo un argumento
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