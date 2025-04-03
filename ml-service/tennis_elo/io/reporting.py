"""
io/reporting.py

Módulo para la generación de informes y reportes del sistema ELO de tenis.
Proporciona funciones para crear informes detallados sobre el estado del sistema,
jugadores, predicciones y evaluaciones.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

# Configurar logging
logger = logging.getLogger(__name__)

class EloReportGenerator:
    """
    Clase para generar informes del sistema ELO de tenis.
    Proporciona métodos para crear diversos tipos de reportes basados
    en los datos del sistema ELO.
    """
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Inicializa el generador de informes.
        
        Args:
            output_dir: Directorio donde se guardarán los informes
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_system_summary_report(self, elo_stats: Dict[str, Any], 
                                     output_file: Optional[str] = None) -> str:
        """
        Genera un informe de resumen del sistema ELO.
        
        Args:
            elo_stats: Diccionario con estadísticas generales del sistema ELO
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"elo_system_summary_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Asegurar que valores no serializables se conviertan correctamente
        serializable_stats = self._make_json_serializable(elo_stats)
        
        # Guardar el informe
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2)
            
            logger.info(f"Informe de resumen del sistema guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando informe de resumen: {str(e)}")
            raise
    
    def generate_top_players_report(self, top_players_data: Dict[str, Any], 
                                  include_details: bool = True,
                                  output_file: Optional[str] = None) -> str:
        """
        Genera un informe con los mejores jugadores del sistema.
        
        Args:
            top_players_data: Datos de los mejores jugadores
            include_details: Si debe incluir detalles adicionales de cada jugador
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"top_players_report_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Preparar datos para el informe
        report_data = {
            "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "top_players": {}
        }
        
        # Procesar datos por categoría (general y por superficie)
        for category, players_data in top_players_data.items():
            if isinstance(players_data, list):
                # Convertir a formato más legible y serializable
                processed_players = []
                for player in players_data:
                    processed_player = dict(player)
                    
                    # Redondear valores numéricos para mejor legibilidad
                    for key, value in processed_player.items():
                        if isinstance(value, (float, np.float64, np.float32)):
                            processed_player[key] = round(value, 2)
                    
                    processed_players.append(processed_player)
                
                report_data["top_players"][category] = processed_players
        
        # Guardar el informe
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Informe de mejores jugadores guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando informe de mejores jugadores: {str(e)}")
            raise
    
    def generate_player_profile_report(self, player_data: Dict[str, Any],
                                     output_file: Optional[str] = None) -> str:
        """
        Genera un informe detallado del perfil de un jugador.
        
        Args:
            player_data: Datos del perfil del jugador
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        player_id = player_data.get('id', 'unknown')
        player_name = player_data.get('name', player_id)
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sanitized_name = player_name.replace(' ', '_').lower()
            output_file = f"player_profile_{sanitized_name}_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Hacer los datos serializables
        serializable_data = self._make_json_serializable(player_data)
        
        # Añadir metadatos al informe
        report_data = {
            "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "player_profile": serializable_data
        }
        
        # Guardar el informe
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Perfil del jugador guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando perfil del jugador: {str(e)}")
            raise
    
    def generate_prediction_report(self, prediction_data: Dict[str, Any],
                                 match_context: Optional[Dict[str, Any]] = None,
                                 output_file: Optional[str] = None) -> str:
        """
        Genera un informe de predicción para un partido.
        
        Args:
            prediction_data: Datos de la predicción
            match_context: Información contextual del partido (opcional)
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        # Determinar nombres de jugadores para el nombre del archivo
        player1_name = prediction_data.get('player1', {}).get('name', 'player1')
        player2_name = prediction_data.get('player2', {}).get('name', 'player2')
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sanitized_p1 = player1_name.replace(' ', '_').lower()
            sanitized_p2 = player2_name.replace(' ', '_').lower()
            output_file = f"prediction_{sanitized_p1}_vs_{sanitized_p2}_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Preparar datos para el informe
        report_data = {
            "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "prediction": self._make_json_serializable(prediction_data)
        }
        
        # Añadir contexto si está disponible
        if match_context:
            report_data["match_context"] = self._make_json_serializable(match_context)
        
        # Guardar el informe
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Informe de predicción guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando informe de predicción: {str(e)}")
            raise
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any],
                                 test_set_info: Optional[Dict[str, Any]] = None,
                                 output_file: Optional[str] = None) -> str:
        """
        Genera un informe de evaluación del sistema.
        
        Args:
            evaluation_results: Resultados de la evaluación
            test_set_info: Información sobre el conjunto de prueba usado (opcional)
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"elo_evaluation_report_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Preparar datos para el informe
        report_data = {
            "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "evaluation_results": self._make_json_serializable(evaluation_results)
        }
        
        # Añadir información del conjunto de prueba si está disponible
        if test_set_info:
            report_data["test_set_info"] = self._make_json_serializable(test_set_info)
        
        # Guardar el informe
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Informe de evaluación guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando informe de evaluación: {str(e)}")
            raise
    
    def generate_surface_analysis_report(self, surface_data: Dict[str, Any],
                                      output_file: Optional[str] = None) -> str:
        """
        Genera un informe de análisis por superficie.
        
        Args:
            surface_data: Datos de análisis por superficie
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"surface_analysis_report_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Preparar datos para el informe
        report_data = {
            "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "surface_analysis": self._make_json_serializable(surface_data)
        }
        
        # Guardar el informe
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Informe de análisis por superficie guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando informe de análisis por superficie: {str(e)}")
            raise
    
    def generate_pipeline_report(self, pipeline_results: Dict[str, Any],
                              output_file: Optional[str] = None) -> str:
        """
        Genera un informe de ejecución del pipeline completo.
        
        Args:
            pipeline_results: Resultados de la ejecución del pipeline
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"pipeline_execution_report_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Preparar datos para el informe
        report_data = {
            "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "pipeline_results": self._make_json_serializable(pipeline_results)
        }
        
        # Añadir un resumen ejecutivo de los resultados
        try:
            summary = {
                "status": pipeline_results.get("status", "unknown"),
                "processing_time_seconds": pipeline_results.get("processing_time_seconds", 0),
                "total_matches_processed": pipeline_results.get("total_matches_processed", 0),
                "total_players": pipeline_results.get("total_players", 0)
            }
            
            # Añadir datos de evaluación si están disponibles
            if "evaluation" in pipeline_results and pipeline_results["evaluation"]:
                eval_data = pipeline_results["evaluation"]
                summary["evaluation"] = {
                    "accuracy": eval_data.get("accuracy", 0),
                    "log_loss": eval_data.get("log_loss", 0),
                    "brier_score": eval_data.get("brier_score", 0)
                }
            
            report_data["summary"] = summary
        except Exception as e:
            logger.warning(f"Error generando resumen ejecutivo: {str(e)}")
        
        # Guardar el informe
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Informe de ejecución del pipeline guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando informe de ejecución del pipeline: {str(e)}")
            raise
    
    def generate_csv_report(self, data: Union[List[Dict], pd.DataFrame], 
                         output_file: str,
                         index: bool = False) -> str:
        """
        Genera un informe en formato CSV.
        
        Args:
            data: Datos para el informe (lista de diccionarios o DataFrame)
            output_file: Nombre del archivo de salida
            index: Si se debe incluir el índice en el CSV
            
        Returns:
            Ruta del archivo generado
        """
        if not output_file.endswith('.csv'):
            output_file += '.csv'
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Convertir a DataFrame si es necesario
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Guardar como CSV
        try:
            df.to_csv(output_path, index=index)
            logger.info(f"Informe CSV guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando informe CSV: {str(e)}")
            raise
    
    def _make_json_serializable(self, data: Any) -> Any:
        """
        Convierte datos a un formato serializable para JSON.
        
        Args:
            data: Datos a convertir
            
        Returns:
            Datos en formato serializable
        """
        if isinstance(data, (datetime, np.datetime64, pd.Timestamp)):
            return data.isoformat()
        elif isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32, np.float16)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._make_json_serializable(item) for item in data)
        elif isinstance(data, set):
            return list(self._make_json_serializable(item) for item in data)
        else:
            # Intentar serializar, o convertir a string como último recurso
            try:
                json.dumps(data)
                return data
            except (TypeError, OverflowError):
                return str(data)


class TextReportGenerator:
    """
    Clase para generar informes en formato de texto plano o Markdown.
    Útil para crear informes más legibles para humanos.
    """
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Inicializa el generador de informes de texto.
        
        Args:
            output_dir: Directorio donde se guardarán los informes
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_player_profile_text(self, player_data: Dict[str, Any],
                                   markdown: bool = True,
                                   output_file: Optional[str] = None) -> str:
        """
        Genera un informe de perfil de jugador en formato texto o Markdown.
        
        Args:
            player_data: Datos del perfil del jugador
            markdown: Si debe usar formato Markdown (True) o texto plano (False)
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        player_id = player_data.get('id', 'unknown')
        player_name = player_data.get('name', player_id)
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sanitized_name = player_name.replace(' ', '_').lower()
            extension = '.md' if markdown else '.txt'
            output_file = f"player_profile_{sanitized_name}_{timestamp}{extension}"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Generar el contenido del informe
        lines = []
        
        if markdown:
            lines.append(f"# Perfil del Jugador: {player_name}")
            lines.append(f"*ID: {player_id}*")
            lines.append("")
            lines.append(f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            lines.append("")
            lines.append("## Información de ELO")
            
            # Ratings ELO
            elo_data = player_data.get('elo', {})
            lines.append(f"- **ELO General**: {elo_data.get('general', 'N/A')}")
            lines.append(f"- **Incertidumbre**: {elo_data.get('uncertainty', 'N/A')}")
            lines.append("")
            
            # ELO por superficie
            if 'by_surface' in elo_data:
                lines.append("### ELO por Superficie")
                for surface, rating in elo_data['by_surface'].items():
                    lines.append(f"- **{surface.capitalize()}**: {rating}")
                lines.append("")
            
            # Estadísticas generales
            if 'stats' in player_data:
                stats = player_data['stats']
                lines.append("## Estadísticas")
                lines.append(f"- **Partidos totales**: {stats.get('total_matches', 'N/A')}")
                lines.append(f"- **Victorias**: {stats.get('wins', 'N/A')}")
                lines.append(f"- **Derrotas**: {stats.get('losses', 'N/A')}")
                
                if 'win_rate' in stats:
                    win_rate = stats['win_rate'] * 100 if isinstance(stats['win_rate'], float) else stats['win_rate']
                    lines.append(f"- **Win rate**: {win_rate:.1f}%")
                
                lines.append("")
                
                # Estadísticas por superficie
                if 'by_surface' in stats:
                    lines.append("### Rendimiento por Superficie")
                    lines.append("| Superficie | Partidos | Victorias | Derrotas | Win Rate |")
                    lines.append("|------------|----------|-----------|----------|----------|")
                    
                    for surface, surface_stats in stats['by_surface'].items():
                        matches = surface_stats.get('matches', 0)
                        wins = surface_stats.get('wins', 0)
                        losses = surface_stats.get('losses', 0)
                        win_rate = surface_stats.get('win_rate', 0) * 100 if isinstance(surface_stats.get('win_rate', 0), float) else 0
                        
                        lines.append(f"| {surface.capitalize()} | {matches} | {wins} | {losses} | {win_rate:.1f}% |")
                    
                    lines.append("")
                
                # Estadísticas por nivel de torneo
                if 'by_tourney_level' in stats:
                    lines.append("### Rendimiento por Nivel de Torneo")
                    lines.append("| Nivel | Partidos | Victorias | Derrotas | Win Rate |")
                    lines.append("|-------|----------|-----------|----------|----------|")
                    
                    levels_map = {
                        'G': 'Grand Slam',
                        'M': 'Masters',
                        'A': 'ATP 500',
                        'D': 'ATP 250',
                        'F': 'Tour Finals',
                        'C': 'Challenger',
                        'S': 'Satellite/ITF',
                        'O': 'Other'
                    }
                    
                    for level, level_stats in stats['by_tourney_level'].items():
                        level_name = levels_map.get(level, level)
                        matches = level_stats.get('matches', 0)
                        wins = level_stats.get('wins', 0)
                        losses = level_stats.get('losses', 0)
                        win_rate = level_stats.get('win_rate', 0) * 100 if isinstance(level_stats.get('win_rate', 0), float) else 0
                        
                        lines.append(f"| {level_name} | {matches} | {wins} | {losses} | {win_rate:.1f}% |")
                    
                    lines.append("")
            
            # Forma reciente
            if 'form' in player_data:
                form = player_data['form']
                form_value = form * 100 if isinstance(form, float) and form <= 1 else form
                lines.append(f"## Forma Reciente: {form_value:.1f}%")
                lines.append("")
            
            # Rivales principales
            if 'rivals' in player_data and player_data['rivals']:
                lines.append("## Principales Rivales")
                lines.append("| Rival | Enfrentamientos | Victorias | Derrotas | Win Rate |")
                lines.append("|-------|----------------|-----------|----------|----------|")
                
                rivals = player_data['rivals']
                # Ordenar por número de enfrentamientos
                sorted_rivals = sorted(rivals.items(), 
                                     key=lambda x: x[1].get('total', 0) if isinstance(x[1], dict) else 0, 
                                     reverse=True)
                
                for rival_id, rival_stats in sorted_rivals[:10]:  # Top 10 rivales
                    rival_name = rival_stats.get('name', rival_id)
                    total = rival_stats.get('total', 0)
                    wins = rival_stats.get('wins', 0)
                    losses = rival_stats.get('losses', 0)
                    win_rate = rival_stats.get('win_rate', 0) * 100 if isinstance(rival_stats.get('win_rate', 0), float) else 0
                    
                    lines.append(f"| {rival_name} | {total} | {wins} | {losses} | {win_rate:.1f}% |")
                
                lines.append("")
            
            # Partidos recientes
            if 'recent_matches' in player_data and player_data['recent_matches']:
                lines.append("## Partidos Recientes")
                lines.append("| Fecha | Oponente | Superficie | Resultado | Score |")
                lines.append("|-------|----------|------------|-----------|-------|")
                
                for match in player_data['recent_matches']:
                    date = match.get('date', '')
                    if isinstance(date, (datetime, pd.Timestamp)):
                        date = date.strftime('%Y-%m-%d')
                    
                    opponent_id = match.get('opponent_id', '')
                    surface = match.get('surface', '').capitalize()
                    result = match.get('result', '').upper()
                    score = match.get('score', '')
                    
                    lines.append(f"| {date} | {opponent_id} | {surface} | {result} | {score} |")
                
                lines.append("")
        
        else:
            # Versión de texto plano
            lines.append(f"PERFIL DEL JUGADOR: {player_name}")
            lines.append(f"ID: {player_id}")
            lines.append("")
            lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            lines.append("INFORMACIÓN DE ELO")
            lines.append("=" * 20)
            
            # Ratings ELO
            elo_data = player_data.get('elo', {})
            lines.append(f"ELO General: {elo_data.get('general', 'N/A')}")
            lines.append(f"Incertidumbre: {elo_data.get('uncertainty', 'N/A')}")
            lines.append("")
            
            # Muchas más secciones similares a la versión Markdown, pero con formato texto
            # Se omite por brevedad
        
        # Guardar el archivo
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"Perfil de texto guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando perfil de texto: {str(e)}")
            raise
    
    def generate_prediction_text(self, prediction_data: Dict[str, Any],
                              markdown: bool = True,
                              output_file: Optional[str] = None) -> str:
        """
        Genera un informe de predicción en formato texto o Markdown.
        
        Args:
            prediction_data: Datos de la predicción
            markdown: Si debe usar formato Markdown (True) o texto plano (False)
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        player1_name = prediction_data.get('player1', {}).get('name', 'Player 1')
        player2_name = prediction_data.get('player2', {}).get('name', 'Player 2')
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sanitized_p1 = player1_name.replace(' ', '_').lower()
            sanitized_p2 = player2_name.replace(' ', '_').lower()
            extension = '.md' if markdown else '.txt'
            output_file = f"prediction_{sanitized_p1}_vs_{sanitized_p2}_{timestamp}{extension}"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Generar el contenido del informe
        lines = []
        
        if markdown:
            # Versión Markdown
            lines.append(f"# Predicción: {player1_name} vs {player2_name}")
            lines.append("")
            lines.append(f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            lines.append("")
            
            # Contexto del partido
            if 'context' in prediction_data:
                context = prediction_data['context']
                surface = context.get('surface', 'Desconocida').capitalize()
                tourney_level = context.get('tourney_level', 'Desconocido')
                round_name = context.get('round', 'Desconocida')
                
                lines.append(f"**Superficie**: {surface}")
                lines.append(f"**Torneo**: {tourney_level}")
                lines.append(f"**Ronda**: {round_name}")
                lines.append("")
            
            # Resultado de la predicción
            lines.append("## Probabilidades de Victoria")
            lines.append("")
            
            prediction = prediction_data.get('prediction', {})
            p1_prob = prediction.get('p1_win_probability', 0) * 100
            p2_prob = prediction.get('p2_win_probability', 0) * 100
            
            lines.append(f"- **{player1_name}**: {p1_prob:.1f}%")
            lines.append(f"- **{player2_name}**: {p2_prob:.1f}%")
            lines.append("")
            
            # Favorito
            favorite_id = prediction.get('favorite', '')
            favorite_name = prediction.get('favorite_name', '')
            if favorite_name:
                lines.append(f"**Favorito**: {favorite_name}")
                lines.append("")
            
            # Intervalo de confianza
            if 'confidence_interval' in prediction:
                interval = prediction['confidence_interval']
                lines.append(f"**Intervalo de confianza**: {interval[0]*100:.1f}% - {interval[1]*100:.1f}%")
                lines.append("")
            
            # Certeza de la predicción
            certainty = prediction.get('prediction_certainty', 0) * 100
            lines.append(f"**Certeza de la predicción**: {certainty:.1f}%")
            lines.append("")
            
            # Datos de los jugadores
            lines.append("## Comparativa de Jugadores")
            lines.append("")
            lines.append("| Métrica | " + player1_name + " | " + player2_name + " |")
            lines.append("|---------|" + "-" * len(player1_name) + "|" + "-" * len(player2_name) + "|")
            
            # ELO general
            p1_elo = prediction_data.get('player1', {}).get('elo_general', 'N/A')
            p2_elo = prediction_data.get('player2', {}).get('elo_general', 'N/A')
            lines.append(f"| ELO General | {p1_elo} | {p2_elo} |")
            
            # ELO por superficie
            p1_elo_surface = prediction_data.get('player1', {}).get('elo_surface', 'N/A')
            p2_elo_surface = prediction_data.get('player2', {}).get('elo_surface', 'N/A')
            lines.append(f"| ELO Superficie | {p1_elo_surface} | {p2_elo_surface} |")
            
            # Forma
            p1_form = prediction_data.get('player1', {}).get('form', 'N/A')
            p2_form = prediction_data.get('player2', {}).get('form', 'N/A')
            
            if isinstance(p1_form, float):
                p1_form = f"{p1_form*100:.1f}%" if p1_form <= 1 else f"{p1_form:.2f}"
            if isinstance(p2_form, float):
                p2_form = f"{p2_form*100:.1f}%" if p2_form <= 1 else f"{p2_form:.2f}"
                
            lines.append(f"| Forma | {p1_form} | {p2_form} |")
            
            # Partidos
            p1_matches = prediction_data.get('player1', {}).get('matches', 'N/A')
            p2_matches = prediction_data.get('player2', {}).get('matches', 'N/A')
            lines.append(f"| Partidos | {p1_matches} | {p2_matches} |")
            
            # Partidos en superficie
            p1_surf_matches = prediction_data.get('player1', {}).get('surface_matches', 'N/A')
            p2_surf_matches = prediction_data.get('player2', {}).get('surface_matches', 'N/A')
            lines.append(f"| Partidos en superficie | {p1_surf_matches} | {p2_surf_matches} |")
            
            # Incertidumbre
            p1_uncertainty = prediction_data.get('player1', {}).get('uncertainty', 'N/A')
            p2_uncertainty = prediction_data.get('player2', {}).get('uncertainty', 'N/A')
            lines.append(f"| Incertidumbre | {p1_uncertainty} | {p2_uncertainty} |")
            lines.append("")
            
            # Factores que influyen en la predicción
            if 'factors' in prediction_data:
                lines.append("## Factores de Influencia")
                lines.append("")
                
                factors = prediction_data['factors']
                
                h2h_factor = factors.get('h2h_factor', 1.0)
                form_ratio = factors.get('form_ratio', 1.0)
                style_factor = factors.get('style_factor', 1.0)
                exp_ratio = factors.get('experience_ratio', 1.0)
                surface_exp_ratio = factors.get('surface_experience_ratio', 1.0)
                
                lines.append(f"- **Factor Head-to-Head**: {h2h_factor:.3f}")
                lines.append(f"- **Ratio de Forma**: {form_ratio:.3f}")
                lines.append(f"- **Compatibilidad de Estilos**: {style_factor:.3f}")
                lines.append(f"- **Ratio de Experiencia**: {exp_ratio:.3f}")
                lines.append(f"- **Ratio de Experiencia en Superficie**: {surface_exp_ratio:.3f}")
                lines.append("")
        else:
            # Versión texto plano
            lines.append(f"PREDICCIÓN: {player1_name} vs {player2_name}")
            lines.append("=" * 40)
            lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            # Similar a versión markdown pero con formato texto plano
            # Omitido por brevedad
        
        # Guardar el archivo
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"Informe de predicción guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando informe de predicción: {str(e)}")
            raise
    
    def generate_evaluation_text(self, evaluation_results: Dict[str, Any],
                              markdown: bool = True,
                              output_file: Optional[str] = None) -> str:
        """
        Genera un informe de evaluación en formato texto o Markdown.
        
        Args:
            evaluation_results: Resultados de la evaluación
            markdown: Si debe usar formato Markdown (True) o texto plano (False)
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extension = '.md' if markdown else '.txt'
            output_file = f"evaluation_report_{timestamp}{extension}"
        
        output_path = os.path.join(self.output_dir, output_file)
        
        # Generar el contenido del informe
        lines = []
        
        if markdown:
            # Versión Markdown
            lines.append("# Informe de Evaluación del Sistema ELO de Tenis")
            lines.append("")
            lines.append(f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            lines.append("")
            
            # Resumen de rendimiento
            accuracy = evaluation_results.get('accuracy', 0) * 100 if isinstance(evaluation_results.get('accuracy', 0), float) else 0
            total_matches = evaluation_results.get('total_matches', 0)
            correct_predictions = evaluation_results.get('correct_predictions', 0)
            
            lines.append("## Resumen de Rendimiento")
            lines.append("")
            lines.append(f"- **Precisión global**: {accuracy:.2f}%")
            lines.append(f"- **Partidos evaluados**: {total_matches}")
            lines.append(f"- **Predicciones correctas**: {correct_predictions}")
            lines.append("")
            
            # Métricas adicionales
            log_loss = evaluation_results.get('log_loss', 'N/A')
            brier_score = evaluation_results.get('brier_score', 'N/A')
            
            if log_loss != 'N/A' or brier_score != 'N/A':
                lines.append("## Métricas Adicionales")
                lines.append("")
                
                if log_loss != 'N/A':
                    lines.append(f"- **Log Loss**: {log_loss:.4f}")
                
                if brier_score != 'N/A':
                    lines.append(f"- **Brier Score**: {brier_score:.4f}")
                
                lines.append("")
            
            # Rendimiento por superficie
            if 'accuracy_by_surface' in evaluation_results:
                lines.append("## Rendimiento por Superficie")
                lines.append("")
                lines.append("| Superficie | Precisión |")
                lines.append("|------------|-----------|")
                
                for surface, acc in evaluation_results['accuracy_by_surface'].items():
                    surface_name = surface.capitalize()
                    acc_pct = acc * 100 if isinstance(acc, float) else acc
                    lines.append(f"| {surface_name} | {acc_pct:.2f}% |")
                
                lines.append("")
            
            # Rendimiento por nivel de torneo
            if 'accuracy_by_tourney_level' in evaluation_results:
                lines.append("## Rendimiento por Nivel de Torneo")
                lines.append("")
                lines.append("| Nivel de Torneo | Precisión |")
                lines.append("|----------------|-----------|")
                
                levels_map = {
                    'G': 'Grand Slam',
                    'M': 'Masters 1000',
                    'A': 'ATP 500',
                    'D': 'ATP 250',
                    'F': 'Tour Finals',
                    'C': 'Challenger',
                    'S': 'Satellite/ITF',
                    'O': 'Other'
                }
                
                for level, acc in evaluation_results['accuracy_by_tourney_level'].items():
                    level_name = levels_map.get(level, level)
                    acc_pct = acc * 100 if isinstance(acc, float) else acc
                    lines.append(f"| {level_name} | {acc_pct:.2f}% |")
                
                lines.append("")
            
            # Rendimiento por umbral
            if 'accuracy_by_threshold' in evaluation_results:
                lines.append("## Rendimiento por Umbral de Confianza")
                lines.append("")
                lines.append("| Umbral | Precisión | Cobertura |")
                lines.append("|--------|-----------|-----------|")
                
                for threshold, data in sorted(evaluation_results['accuracy_by_threshold'].items(), 
                                          key=lambda x: float(x[0])):
                    if isinstance(data, dict):
                        accuracy = data.get('accuracy', 0) * 100 if isinstance(data.get('accuracy', 0), float) else 0
                        coverage = data.get('coverage', 0) * 100 if isinstance(data.get('coverage', 0), float) else 0
                        lines.append(f"| {threshold} | {accuracy:.2f}% | {coverage:.2f}% |")
                
                lines.append("")
            
            # Calibración
            if 'calibration' in evaluation_results:
                lines.append("## Calibración del Modelo")
                lines.append("")
                lines.append("| Probabilidad Predicha | Probabilidad Empírica | Error | Partidos |")
                lines.append("|----------------------|------------------------|-------|----------|")
                
                for bin_range, data in sorted(evaluation_results['calibration'].items(), 
                                          key=lambda x: float(x[0].split('-')[0])):
                    if isinstance(data, dict):
                        pred_prob = data.get('predicted_probability', 0) * 100
                        emp_prob = data.get('empirical_probability', 0) * 100
                        error = data.get('error', 0) * 100
                        count = data.get('count', 0)
                        
                        lines.append(f"| {pred_prob:.1f}% | {emp_prob:.1f}% | {error:+.1f}% | {count} |")
                
                lines.append("")
                lines.append("*Nota: Error positivo indica sobreestimación, negativo indica subestimación.*")
                lines.append("")
            
            # Confianza
            if 'confidence' in evaluation_results:
                lines.append("## Precisión por Nivel de Confianza")
                lines.append("")
                lines.append("| Nivel de Confianza | Precisión | Partidos | Porcentaje |")
                lines.append("|-------------------|-----------|----------|------------|")
                
                for bin_key, data in sorted(evaluation_results['confidence'].items()):
                    if isinstance(data, dict):
                        accuracy = data.get('accuracy', 0) * 100
                        count = data.get('count', 0)
                        percentage = data.get('percentage', 0)
                        
                        lines.append(f"| {bin_key} | {accuracy:.2f}% | {count} | {percentage:.1f}% |")
                
                lines.append("")
        else:
            # Versión texto plano
            lines.append("INFORME DE EVALUACIÓN DEL SISTEMA ELO DE TENIS")
            lines.append("=" * 50)
            lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            # Similar a versión markdown pero con formato texto plano
            # Omitido por brevedad
        
        # Guardar el archivo
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"Informe de evaluación guardado en: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error guardando informe de evaluación: {str(e)}")
            raise


def generate_tournament_performance_report(players_data: List[Dict[str, Any]],
                                     tournament_info: Dict[str, Any],
                                     output_file: Optional[str] = None) -> str:
    """
    Genera un informe sobre el rendimiento de jugadores en un torneo específico.
    
    Args:
        players_data: Lista de datos de jugadores que participaron en el torneo
        tournament_info: Información sobre el torneo
        output_file: Nombre del archivo de salida (opcional)
        
    Returns:
        Ruta del archivo generado
    """
    # Inicializar el generador de reportes de texto
    report_generator = TextReportGenerator()
    
    tournament_name = tournament_info.get('name', 'Torneo')
    tournament_year = tournament_info.get('year', datetime.now().year)
    tournament_surface = tournament_info.get('surface', 'Desconocida')
    
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sanitized_name = tournament_name.replace(' ', '_').lower()
        output_file = f"tournament_report_{sanitized_name}_{tournament_year}_{timestamp}.md"
    
    output_path = os.path.join(report_generator.output_dir, output_file)
    
    # Generar el contenido del informe
    lines = []
    
    # Título y encabezado
    lines.append(f"# Análisis de Rendimiento: {tournament_name} {tournament_year}")
    lines.append("")
    lines.append(f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    
    # Información del torneo
    lines.append("## Información del Torneo")
    lines.append("")
    lines.append(f"- **Nombre**: {tournament_name}")
    lines.append(f"- **Año**: {tournament_year}")
    lines.append(f"- **Superficie**: {tournament_surface.capitalize()}")
    
    if 'level' in tournament_info:
        levels_map = {
            'G': 'Grand Slam',
            'M': 'Masters 1000',
            'A': 'ATP 500',
            'D': 'ATP 250',
            'F': 'Tour Finals',
            'C': 'Challenger',
            'S': 'Satellite/ITF',
            'O': 'Other'
        }
        level_code = tournament_info['level']
        level_name = levels_map.get(level_code, level_code)
        lines.append(f"- **Nivel**: {level_name}")
    
    if 'location' in tournament_info:
        lines.append(f"- **Ubicación**: {tournament_info['location']}")
    
    if 'dates' in tournament_info:
        lines.append(f"- **Fechas**: {tournament_info['dates']}")
    
    lines.append("")
    
    # Análisis de rendimiento de jugadores
    if players_data:
        lines.append("## Rendimiento de Jugadores")
        lines.append("")
        
        # Ordenar jugadores por rating ELO
        sorted_players = sorted(players_data, 
                               key=lambda x: x.get('elo', {}).get('general', 0) 
                               if isinstance(x.get('elo', {}), dict) else 0, 
                               reverse=True)
        
        # Tabla de jugadores
        lines.append("| Jugador | ELO General | ELO en Superficie | Probabilidad Victoria |")
        lines.append("|---------|-------------|-------------------|------------------------|")
        
        for player in sorted_players[:16]:  # Mostrar top 16 jugadores
            name = player.get('name', 'Desconocido')
            
            elo_general = player.get('elo', {}).get('general', 'N/A')
            if isinstance(elo_general, (int, float)):
                elo_general = f"{elo_general:.0f}"
            
            elo_surface = player.get('elo', {}).get('by_surface', {}).get(tournament_surface, 'N/A')
            if isinstance(elo_surface, (int, float)):
                elo_surface = f"{elo_surface:.0f}"
            
            win_prob = player.get('win_probability', 'N/A')
            if isinstance(win_prob, (int, float)):
                win_prob = f"{win_prob*100:.1f}%" if win_prob <= 1 else f"{win_prob:.1f}%"
            
            lines.append(f"| {name} | {elo_general} | {elo_surface} | {win_prob} |")
        
        lines.append("")
        
        # Análisis de favoritos
        lines.append("## Análisis de Favoritos")
        lines.append("")
        
        # Ordenar por probabilidad de victoria
        win_prob_sorted = sorted(players_data, 
                              key=lambda x: x.get('win_probability', 0) 
                              if isinstance(x.get('win_probability', 0), (int, float)) else 0, 
                              reverse=True)
        
        # Top 3 favoritos
        for i, player in enumerate(win_prob_sorted[:3], 1):
            name = player.get('name', 'Desconocido')
            
            win_prob = player.get('win_probability', 0)
            if isinstance(win_prob, (int, float)):
                win_prob_str = f"{win_prob*100:.1f}%" if win_prob <= 1 else f"{win_prob:.1f}%"
            else:
                win_prob_str = "N/A"
            
            lines.append(f"### {i}. {name} ({win_prob_str})")
            lines.append("")
            
            # Factores que favorecen al jugador
            strengths = []
            
            # ELO elevado
            elo_surface = player.get('elo', {}).get('by_surface', {}).get(tournament_surface, 0)
            if isinstance(elo_surface, (int, float)) and elo_surface > 2000:
                strengths.append(f"Alto rating ELO en {tournament_surface.capitalize()}: {elo_surface:.0f}")
            
            # Buena forma reciente
            form = player.get('form', 0)
            if isinstance(form, (int, float)) and ((form <= 1 and form > 0.7) or form > 70):
                form_str = f"{form*100:.0f}%" if form <= 1 else f"{form:.0f}%"
                strengths.append(f"Excelente forma reciente: {form_str}")
            
            # Experiencia en la superficie
            surface_matches = player.get('matches_by_surface', {}).get(tournament_surface, 0)
            if isinstance(surface_matches, (int, float)) and surface_matches > 30:
                strengths.append(f"Gran experiencia en la superficie: {surface_matches} partidos")
            
            # Si encontramos fortalezas, mostrarlas
            if strengths:
                lines.append("**Factores favorables:**")
                lines.append("")
                for strength in strengths:
                    lines.append(f"- {strength}")
                lines.append("")
        
        # Análisis sobre el vencedor más probable
        lines.append("## Conclusión")
        lines.append("")
        
        if win_prob_sorted and isinstance(win_prob_sorted[0].get('win_probability', 0), (int, float)):
            top_player = win_prob_sorted[0]
            top_player_name = top_player.get('name', 'Desconocido')
            top_player_prob = top_player.get('win_probability', 0)
            
            if top_player_prob <= 1:
                top_player_prob_str = f"{top_player_prob*100:.1f}%"
            else:
                top_player_prob_str = f"{top_player_prob:.1f}%"
            
            if top_player_prob > 0.3:  # 30% o más
                lines.append(f"Según el análisis ELO, **{top_player_name}** es el claro favorito para ganar este torneo con una probabilidad del {top_player_prob_str}.")
            else:
                lines.append(f"Este torneo presenta un campo muy competitivo sin un claro favorito. **{top_player_name}** tiene la mayor probabilidad de victoria ({top_player_prob_str}), pero varios jugadores tienen opciones reales de ganar.")
            
            # Comparar con segundo favorito si la diferencia es significativa
            if len(win_prob_sorted) > 1:
                second_player = win_prob_sorted[1]
                second_player_name = second_player.get('name', 'Desconocido')
                second_player_prob = second_player.get('win_probability', 0)
                
                if isinstance(second_player_prob, (int, float)) and isinstance(top_player_prob, (int, float)):
                    prob_ratio = top_player_prob / second_player_prob if second_player_prob > 0 else 2
                    
                    if prob_ratio > 1.5:
                        lines.append("")
                        lines.append(f"Hay una diferencia significativa entre {top_player_name} y el segundo favorito, {second_player_name}, lo que refuerza la posición dominante del primero en este torneo.")
        else:
            lines.append("No hay suficientes datos para determinar un favorito claro en este torneo.")
    
    # Guardar archivo
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Informe de rendimiento en torneo guardado en: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generando informe de torneo: {str(e)}")
        raise


def generate_surface_comparison_report(player_id: str, player_data: Dict[str, Any], 
                                    output_file: Optional[str] = None) -> str:
    """
    Genera un informe detallado sobre el rendimiento de un jugador en diferentes superficies.
    
    Args:
        player_id: ID del jugador
        player_data: Datos completos del jugador
        output_file: Nombre del archivo de salida (opcional)
        
    Returns:
        Ruta del archivo generado
    """
    # Inicializar el generador de reportes de texto
    report_generator = TextReportGenerator()
    
    player_name = player_data.get('name', player_id)
    
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sanitized_name = player_name.replace(' ', '_').lower()
        output_file = f"surface_analysis_{sanitized_name}_{timestamp}.md"
    
    output_path = os.path.join(report_generator.output_dir, output_file)
    
    # Generar el contenido del informe
    lines = []
    
    # Título y encabezado
    lines.append(f"# Análisis por Superficie: {player_name}")
    lines.append("")
    lines.append(f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    
    # Información general del jugador
    lines.append("## Información General")
    lines.append("")
    lines.append(f"- **ID**: {player_id}")
    lines.append(f"- **Nombre**: {player_name}")
    
    total_matches = player_data.get('stats', {}).get('total_matches', 0)
    lines.append(f"- **Partidos totales**: {total_matches}")
    
    elo_general = player_data.get('elo', {}).get('general', 0)
    if isinstance(elo_general, (int, float)):
        lines.append(f"- **ELO general**: {elo_general:.0f}")
    
    lines.append("")
    
    # Análisis por superficie
    lines.append("## Rendimiento por Superficie")
    lines.append("")
    
    # Tabla comparativa
    lines.append("| Superficie | ELO | Partidos | Victorias | Derrotas | Win Rate |")
    lines.append("|------------|-----|----------|-----------|----------|----------|")
    
    # Obtener datos por superficie
    surfaces = ['hard', 'clay', 'grass', 'carpet']
    surface_data = []
    
    for surface in surfaces:
        elo = player_data.get('elo', {}).get('by_surface', {}).get(surface, 0)
        stats = player_data.get('stats', {}).get('by_surface', {}).get(surface, {})
        
        matches = stats.get('matches', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        win_rate = stats.get('win_rate', 0)
        
        if isinstance(win_rate, (int, float)):
            win_rate_str = f"{win_rate*100:.1f}%"
        else:
            win_rate_str = "N/A"
        
        if isinstance(elo, (int, float)):
            elo_str = f"{elo:.0f}"
        else:
            elo_str = "N/A"
        
        surface_data.append({
            'surface': surface,
            'elo': elo if isinstance(elo, (int, float)) else 0,
            'matches': matches,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate if isinstance(win_rate, (int, float)) else 0,
            'win_rate_str': win_rate_str,
            'elo_str': elo_str
        })
        
        lines.append(f"| {surface.capitalize()} | {elo_str} | {matches} | {wins} | {losses} | {win_rate_str} |")
    
    lines.append("")
    
    # Ordenar superficies por rendimiento
    sorted_surfaces = sorted(surface_data, key=lambda x: x['win_rate'], reverse=True)
    
    # Análisis de la mejor y peor superficie
    lines.append("## Análisis de Rendimiento")
    lines.append("")
    
    if sorted_surfaces and sorted_surfaces[0]['matches'] > 5:
        best_surface = sorted_surfaces[0]
        best_surface_name = best_surface['surface'].capitalize()
        
        lines.append(f"### Mejor Superficie: {best_surface_name}")
        lines.append("")
        lines.append(f"{player_name} muestra su mejor rendimiento en {best_surface_name} con un win rate de {best_surface['win_rate_str']} en {best_surface['matches']} partidos.")
        
        if best_surface['elo'] > 0:
            lines.append(f"Su rating ELO en esta superficie es {best_surface['elo_str']}, lo que indica un nivel de juego muy competitivo.")
        
        lines.append("")
    
    if len(sorted_surfaces) > 1 and sorted_surfaces[-1]['matches'] > 5:
        worst_surface = sorted_surfaces[-1]
        worst_surface_name = worst_surface['surface'].capitalize()
        
        lines.append(f"### Superficie con Mayor Dificultad: {worst_surface_name}")
        lines.append("")
        lines.append(f"La superficie donde {player_name} muestra más dificultades es {worst_surface_name} con un win rate de {worst_surface['win_rate_str']} en {worst_surface['matches']} partidos.")
        
        if worst_surface['elo'] > 0:
            lines.append(f"Su rating ELO en esta superficie es {worst_surface['elo_str']}.")
        
        lines.append("")
    
    # Comparar superficies e identificar diferencias significativas
    if len(sorted_surfaces) >= 2:
        best = sorted_surfaces[0]
        worst = sorted_surfaces[-1]
        
        if best['matches'] > 5 and worst['matches'] > 5 and (best['win_rate'] - worst['win_rate']) > 0.15:
            lines.append("### Diferencias Significativas")
            lines.append("")
            lines.append(f"Existe una diferencia importante de rendimiento entre {best['surface'].capitalize()} ({best['win_rate_str']}) y {worst['surface'].capitalize()} ({worst['win_rate_str']}), lo que sugiere una clara adaptación del estilo de juego a las características de la primera superficie.")
            lines.append("")
    
    # Guardar archivo
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Informe de análisis por superficie guardado en: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generando informe de superficie: {str(e)}")
        raise