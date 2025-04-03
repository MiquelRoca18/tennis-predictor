"""
match_analysis.py

Este módulo contiene funciones para el análisis de partidos de tenis,
incluyendo el parsing de scores, cálculo de factores de importancia,
análisis de estadísticas y cálculo de márgenes de victoria.
"""

from tennis_elo.utils.normalizers import normalize_tournament_level, normalize_surface

import re
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any
from datetime import datetime

# Configurar logging
logger = logging.getLogger(__name__)

class MatchAnalyzer:
    """
    Clase para analizar partidos de tenis, calcular factores de importancia,
    y evaluar estadísticas detalladas.
    """
    
    def __init__(self, 
                 tourney_level_mapping: Optional[Dict[str, str]] = None,
                 round_multipliers: Optional[Dict[str, float]] = None,
                 surface_specificity: Optional[Dict[str, float]] = None,
                 tournament_k_factors: Optional[Dict[str, float]] = None,
                 initial_rating: float = 1500):
        """
        Inicializa el analizador de partidos con los parámetros necesarios.
        
        Args:
            tourney_level_mapping: Mapeo de niveles de torneo a códigos
            round_multipliers: Multiplicadores por ronda
            surface_specificity: Especificidad por superficie
            tournament_k_factors: Factores K por tipo de torneo
        """
        # Mapeo para nombres de columnas de nivel de torneo
        self.tourney_level_mapping = tourney_level_mapping or {
            'G': 'G',          # Grand Slam
            'M': 'M',          # Masters 1000
            'A': 'A',          # ATP 500
            'D': 'D',          # ATP 250
            'F': 'F',          # Tour Finals
            'C': 'C',          # Challenger
            'S': 'S',          # Satellite/ITF
            'O': 'O',          # Other
            'Grand Slam': 'G',
            'Masters 1000': 'M',
            'ATP500': 'A',
            'ATP250': 'D',
            'Tour Finals': 'F',
            'Challenger': 'C',
            'Satellite': 'S',
            'Futures': 'S',
            'ITF': 'S'
        }
        
        # Multiplicadores por ronda (aumentan en rondas finales)
        self.round_multipliers = round_multipliers or {
            'F': 1.5,     # Final
            'SF': 1.3,    # Semifinal
            'QF': 1.2,    # Cuartos de final
            'R16': 1.1,   # Octavos
            'R32': 1.0,   # 1/16
            'R64': 0.9,   # 1/32
            'R128': 0.8,  # 1/64
            'RR': 1.0     # Round Robin
        }
        
        # Multiplicadores para superficies (mayor especificidad)
        self.surface_specificity = surface_specificity or {
            'hard': 1.0,
            'clay': 1.1,    # Mayor especificidad en tierra
            'grass': 1.2,   # Mayor especificidad en hierba
            'carpet': 1.0
        }
        
        # Parámetros dinámicos del sistema ELO
        self.tournament_k_factors = tournament_k_factors or {
            'G': 2.0,      # Grand Slam
            'M': 1.5,      # Masters 1000
            'A': 1.2,      # ATP 500
            'D': 1.0,      # ATP 250
            'F': 1.8,      # Tour Finals
            'C': 0.8,      # Challenger
            'S': 0.5,      # Satellite/ITF
            'O': 0.7       # Other
        }
        
        # Mapeo de estadísticas de partidos a importancia para ELO
        self.stat_importance_weights = {
            'ace': 0.05,
            'df': -0.03,
            'svpt': 0.02,
            '1stIn': 0.04,
            '1stWon': 0.08,
            '2ndWon': 0.07,
            'bpSaved': 0.06,
            'bpFaced': -0.04,
            'SvGms': 0.03
        }
        
        # Valor de rating inicial para jugadores nuevos
        self.initial_rating = initial_rating

    def parse_score(self, score: str) -> Tuple[int, int, int, bool, Dict]:
        """
        Analiza el score de un partido para determinar sets, games y dominancia.
        
        Args:
            score: String con el resultado (e.g. '6-4 7-5')
            
        Returns:
            Tupla de (sets_ganador, sets_perdedor, diferencia_games, fue_completo, stats)
            donde stats es un diccionario con estadísticas adicionales
        """
        # Validar tipo de score
        if not isinstance(score, str):
            score = str(score) if not pd.isna(score) else ''
        
        if pd.isna(score) or not score:
            return 2, 0, 4, False, {'avg_game_diff': 2.0, 'tiebreaks': 0, 'total_games': 4}
        
        try:
            # Limpiar y dividir el score
            score = str(score).strip()
            
            # Variables para análisis
            sets_won_winner = 0
            sets_won_loser = 0
            games_diff = 0
            total_games = 0
            tiebreaks = 0
            sets_analyzed = 0
            game_diff_values = []
            complete_match = True
            
            # Manejar casos especiales: abandono, walkover, etc.
            special_terms = ['ret', 'reti', 'def', 'w/o', 'walkover', 'default']
            if any(term in score.lower() for term in special_terms):
                complete_match = False
            
            # Dividir el score por sets - considerar que puede haber espacios o no
            # Manejar casos como "64 75" o "6-4 7-5" o "6:4,7:5"
            normalized_score = score.replace(':', '-').replace(',', ' ').replace(';', ' ')
            sets = normalized_score.split()
            
            # Filtrar solo los elementos que parecen sets válidos
            valid_sets = []
            for set_part in sets:
                # Si tiene un guión o se puede dividir en números, considerarlo un set potencial
                if '-' in set_part or (len(set_part) >= 2 and not any(term in set_part.lower() for term in special_terms)):
                    valid_sets.append(set_part)
            
            # Si no hay sets válidos después del filtrado, usar valores predeterminados
            if not valid_sets:
                return 2, 0, 4, False, {'avg_game_diff': 2.0, 'tiebreaks': 0, 'total_games': 4}
            
            for set_score in valid_sets:
                # Detectar tiebreaks
                has_tiebreak = '(' in set_score
                tb_score = None
                
                if has_tiebreak:
                    # Extraer puntaje de tiebreak
                    tb_part = re.search(r'\((.*?)\)', set_score)
                    if tb_part:
                        tb_score = tb_part.group(1)
                    
                    # Limpiar para procesar el set
                    set_score = set_score.split('(')[0]
                    tiebreaks += 1
                
                # Manejar diferentes formatos de sets
                if '-' in set_score:
                    # Formato común "6-4"
                    games = set_score.split('-')
                elif len(set_score) == 2:
                    # Formato compacto "64"
                    games = [set_score[0], set_score[1]]
                elif len(set_score) >= 3 and set_score[1].isdigit():
                    # Posible formato raro "61"+"75" pegados
                    continue  # Ignorar este caso por ahora
                else:
                    # Intentar extraer dos dígitos consecutivos
                    games_match = re.findall(r'\d+', set_score)
                    if len(games_match) >= 2:
                        games = games_match[:2]  # Tomar los dos primeros números
                    else:
                        continue  # Ignorar este set si no podemos parsearlo
                
                # Validar que tenemos dos puntuaciones
                if len(games) == 2:
                    try:
                        # Convertir a enteros y validar
                        winner_games = int(games[0])
                        loser_games = int(games[1])
                        
                        # Validar que son puntuaciones razonables de tenis
                        if 0 <= winner_games <= 50 and 0 <= loser_games <= 50:
                            game_diff = winner_games - loser_games
                            games_diff += game_diff
                            total_games += winner_games + loser_games
                            game_diff_values.append(game_diff)
                            sets_analyzed += 1
                            
                            if winner_games > loser_games:
                                sets_won_winner += 1
                            elif loser_games > winner_games:
                                sets_won_loser += 1
                    except (ValueError, TypeError):
                        # Si no se puede convertir a entero, ignorar este set
                        continue
            
            # Si no se pudo analizar ningún set, usar valores predeterminados
            if sets_analyzed == 0:
                return 2, 0, 4, False, {'avg_game_diff': 2.0, 'tiebreaks': 0, 'total_games': 4}
            
            # Calcular estadísticas adicionales
            avg_game_diff = sum(game_diff_values) / max(1, len(game_diff_values))
            max_game_diff = max(game_diff_values) if game_diff_values else 0
            
            stats = {
                'avg_game_diff': avg_game_diff,
                'max_game_diff': max_game_diff,
                'tiebreaks': tiebreaks,
                'total_games': total_games,
                'complete': complete_match
            }
            
            # En caso de no poder analizar correctamente, usar valores predeterminados
            if sets_won_winner == 0 and sets_won_loser == 0:
                return 2, 0, 4, False, stats
                
            return sets_won_winner, sets_won_loser, games_diff, complete_match, stats
            
        except Exception as e:
            logger.debug(f"Error analizando score '{score}': {str(e)}")
            return 2, 0, 4, False, {'avg_game_diff': 2.0, 'tiebreaks': 0, 'total_games': 4}
    
    def get_match_importance_factor(self, tourney_level: str, round_name: str, 
                                    player1_id: str, player2_id: str, 
                                    player1_rating: float, player2_rating: float) -> float:
        """
        Calcula un factor de importancia del partido basado en contexto.
        
        Args:
            tourney_level: Nivel del torneo
            round_name: Ronda del partido
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            player1_rating: Rating ELO actual del jugador 1
            player2_rating: Rating ELO actual del jugador 2
            
        Returns:
            Factor de importancia (0.8-1.5)
        """
        # Normalizar con validación de tipos
        if not isinstance(tourney_level, str):
            tourney_level = str(tourney_level) if not pd.isna(tourney_level) else 'O'
        
        # Normalizar el nivel de torneo
        tourney_level = normalize_tournament_level(tourney_level)
        
        # Base por nivel de torneo
        base_importance = {
            'G': 1.3,  # Grand Slam
            'M': 1.2,  # Masters
            'F': 1.25, # Finals
            'A': 1.1,  # ATP 500
            'D': 1.0,  # ATP 250
            'C': 0.9,  # Challenger
            'S': 0.8,  # Satélite
            'O': 0.8   # Otros
        }.get(tourney_level, 1.0)
        
        # Validar round_name
        if not isinstance(round_name, str):
            round_name = str(round_name) if not pd.isna(round_name) else 'R32'
        
        # Ajuste por ronda
        round_factor = self.round_multipliers.get(round_name, 1.0)
        
        # Validar player IDs
        player1_id = str(player1_id) if not pd.isna(player1_id) else ''
        player2_id = str(player2_id) if not pd.isna(player2_id) else ''
        
        if not player1_id or not player2_id:
            return base_importance * round_factor
        
        try:
            # Ajuste por ranking relativo con validación
            p1_rating = player1_rating
            p2_rating = player2_rating
            
            # Asegurar que son números
            if not isinstance(p1_rating, (int, float)) or not isinstance(p2_rating, (int, float)):
                return base_importance * round_factor
            
            rating_diff = abs(p1_rating - p2_rating)
            
            # Partidos muy igualados son más importantes
            ranking_factor = 1.0
            if rating_diff < 50:
                ranking_factor = 1.1  # Muy igualado
            elif rating_diff < 100:
                ranking_factor = 1.05  # Bastante igualado
            elif rating_diff > 300:
                ranking_factor = 0.9  # Muy desigualado
            
            # Combinar factores
            return base_importance * round_factor * ranking_factor
        except Exception as e:
            # En caso de error, devolver valor base
            logger.debug(f"Error en get_match_importance_factor: {str(e)}")
            return base_importance * round_factor
    
    def get_style_compatibility_factor(self, player1_id: str, player2_id: str, surface: str,
                                   player1_stats: Optional[Dict] = None, 
                                   player2_stats: Optional[Dict] = None) -> float:
        """
        Calcula un factor de compatibilidad de estilos de juego entre dos jugadores.
        Algunos estilos tienen ventaja natural contra otros en ciertas superficies.
        
        Args:
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            surface: Superficie del partido
            player1_stats: Estadísticas del primer jugador (opcional)
            player2_stats: Estadísticas del segundo jugador (opcional)
            
        Returns:
            Factor de compatibilidad (0.95-1.05, >1 favorece a player1)
        """
        # Ventajas por superficie (modelo simplificado)
        surface_advantages = {
            # surface: {estilo_favorecido: {estilo_en_desventaja: factor}}
            'clay': {
                'defensive': {'serve_oriented': 1.05},
                'baseline': {'attacking': 1.03}
            },
            'grass': {
                'serve_oriented': {'defensive': 1.05, 'baseline': 1.03},
                'attacking': {'defensive': 1.04}
            },
            'hard': {
                'serve_oriented': {'defensive': 1.02},
                'attacking': {'baseline': 1.02}
            },
            'carpet': {
                'serve_oriented': {'defensive': 1.05},
                'attacking': {'baseline': 1.04}
            }
        }
        
        # Implementación simplificada
        # En una implementación completa, habría que analizar las estadísticas de los jugadores
        # para determinar sus estilos de juego y luego aplicar las ventajas correspondientes
        
        # Por ahora, retornamos un valor neutral
        return 1.0
    
    def get_dynamic_k_factor(self, player_id: str, tourney_level: str, round_name: str,
                           surface: str, match_importance: float = 1.0,
                           player_match_count: int = 0,
                           player_uncertainty: float = 0.0) -> float:
        """
        Calcula un factor K dinámico basado en múltiples factores contextuales.
        
        Args:
            player_id: ID del jugador
            tourney_level: Nivel del torneo (G, M, A, D, etc.)
            round_name: Ronda del torneo (F, SF, QF, etc.)
            surface: Superficie de juego
            match_importance: Importancia adicional del partido (1.0 por defecto)
            player_match_count: Número de partidos jugados por el jugador
            player_uncertainty: Incertidumbre del rating del jugador
            
        Returns:
            Factor K ajustado
        """
        # Validar tipos
        player_id = str(player_id) if not pd.isna(player_id) else ''
        
        if not isinstance(tourney_level, str):
            tourney_level = str(tourney_level) if not pd.isna(tourney_level) else 'O'
        
        if not isinstance(round_name, str):
            round_name = str(round_name) if not pd.isna(round_name) else 'R32'
        
        if not isinstance(surface, str):
            surface = str(surface) if not pd.isna(surface) else 'hard'
        
        if not isinstance(match_importance, (int, float)) or pd.isna(match_importance) or match_importance <= 0:
            match_importance = 1.0
        
        try:
            # Factor base - usamos un valor predeterminado de 32
            k_factor = 32.0
            
            # Normalizar valores
            tourney_level = normalize_tournament_level(tourney_level)
            surface = normalize_surface(surface)
            
            # Ajuste por tipo de torneo
            tourney_multiplier = self.tournament_k_factors.get(tourney_level, 1.0)
            if not isinstance(tourney_multiplier, (int, float)) or pd.isna(tourney_multiplier) or tourney_multiplier <= 0:
                tourney_multiplier = 1.0
            k_factor *= tourney_multiplier
            
            # Ajuste por ronda
            round_multiplier = self.round_multipliers.get(round_name, 1.0)
            if not isinstance(round_multiplier, (int, float)) or pd.isna(round_multiplier) or round_multiplier <= 0:
                round_multiplier = 1.0
            k_factor *= round_multiplier
            
            # Ajuste por superficie específica (mayor impacto en superficies más especializadas)
            surface_multiplier = self.surface_specificity.get(surface, 1.0)
            if not isinstance(surface_multiplier, (int, float)) or pd.isna(surface_multiplier) or surface_multiplier <= 0:
                surface_multiplier = 1.0
            k_factor *= surface_multiplier
            
            # Ajuste por experiencia (menor K para jugadores con más experiencia)
            if player_match_count > 0:
                # Comienza en 1.2 para novatos y baja hasta 0.8 para veteranos
                experience_multiplier = 1.2 - min(0.4, (player_match_count / 500))
                k_factor *= experience_multiplier
            
            # Ajuste por incertidumbre (mayor K para jugadores con mayor incertidumbre)
            if isinstance(player_uncertainty, (int, float)) and not pd.isna(player_uncertainty) and player_uncertainty > 0:
                uncertainty_multiplier = min(1.5, 1.0 + (player_uncertainty / 300))
                k_factor *= uncertainty_multiplier
            
            # Ajuste por importancia del partido
            k_factor *= match_importance
            
            # Redondear a 2 decimales para estabilidad
            return round(k_factor, 2)
        except Exception as e:
            logger.debug(f"Error calculando factor K dinámico: {str(e)}")
            return 32.0  # Valor por defecto en caso de error
    
    def get_match_stats_factor(self, match_id: Optional[str] = None, 
                            winner_id: Optional[str] = None, 
                            loser_id: Optional[str] = None,
                            w_stats: Optional[Dict] = None,
                            l_stats: Optional[Dict] = None,
                            match_stats_df: Optional[pd.DataFrame] = None) -> float:
        """
        Calcula un factor basado en estadísticas del partido para ajustar el cambio de ELO.
        
        Args:
            match_id: ID del partido (opcional)
            winner_id: ID del ganador
            loser_id: ID del perdedor
            w_stats: Estadísticas del ganador (opcional)
            l_stats: Estadísticas del perdedor (opcional)
            match_stats_df: DataFrame con estadísticas de partidos (opcional)
            
        Returns:
            Factor de ajuste (0.8-1.2) basado en estadísticas
        """
        # Validar tipos de entrada
        if match_id is not None and not isinstance(match_id, str):
            match_id = str(match_id) if not pd.isna(match_id) else None
            
        if winner_id is not None and not isinstance(winner_id, str):
            winner_id = str(winner_id) if not pd.isna(winner_id) else None
            
        if loser_id is not None and not isinstance(loser_id, str):
            loser_id = str(loser_id) if not pd.isna(loser_id) else None
        
        # Validar diccionarios de estadísticas
        if w_stats is not None and not isinstance(w_stats, dict):
            w_stats = None
            
        if l_stats is not None and not isinstance(l_stats, dict):
            l_stats = None
        
        try:
            # Si tenemos estadísticas proporcionadas directamente, usarlas
            if w_stats and l_stats:
                # Variables para análisis
                dominance_score = 0
                stats_count = 0
                
                # Estadísticas de servicio - aces
                if 'ace' in w_stats and 'ace' in l_stats:
                    w_ace = w_stats['ace']
                    l_ace = l_stats['ace']
                    
                    # Validar tipos
                    if not isinstance(w_ace, (int, float)) or pd.isna(w_ace):
                        w_ace = 0
                    if not isinstance(l_ace, (int, float)) or pd.isna(l_ace):
                        l_ace = 0
                    
                    if w_ace > 0 or l_ace > 0:
                        ace_ratio = w_ace / max(1, w_ace + l_ace)
                        # Normalizar a escala -1 a 1 (0.5 es neutral)
                        dominance_score += (ace_ratio - 0.5) * 2
                        stats_count += 1
                
                # Puntos de break
                if all(key in w_stats for key in ['bpSaved', 'bpFaced']) and all(key in l_stats for key in ['bpSaved', 'bpFaced']):
                    # Break points ganados por el ganador
                    w_bp_faced = w_stats['bpFaced']
                    w_bp_saved = w_stats['bpSaved']
                    l_bp_faced = l_stats['bpFaced']
                    l_bp_saved = l_stats['bpSaved']
                    
                    # Validar tipos
                    if not isinstance(w_bp_faced, (int, float)) or pd.isna(w_bp_faced):
                        w_bp_faced = 0
                    if not isinstance(w_bp_saved, (int, float)) or pd.isna(w_bp_saved):
                        w_bp_saved = 0
                    if not isinstance(l_bp_faced, (int, float)) or pd.isna(l_bp_faced):
                        l_bp_faced = 0
                    if not isinstance(l_bp_saved, (int, float)) or pd.isna(l_bp_saved):
                        l_bp_saved = 0
                    
                    if w_bp_faced > 0:
                        w_bp_saved_pct = w_bp_saved / w_bp_faced
                    else:
                        w_bp_saved_pct = 1.0
                    
                    # Break points ganados por el perdedor
                    if l_bp_faced > 0:
                        l_bp_saved_pct = l_bp_saved / l_bp_faced
                    else:
                        l_bp_saved_pct = 1.0
                    
                    # Normalizar a escala -1 a 1
                    dominance_score += (w_bp_saved_pct - l_bp_saved_pct)
                    stats_count += 1
                
                # Eficiencia en primer servicio
                if all(key in w_stats for key in ['1stIn', 'svpt']) and all(key in l_stats for key in ['1stIn', 'svpt']):
                    w_1st_in = w_stats['1stIn']
                    w_svpt = w_stats['svpt']
                    l_1st_in = l_stats['1stIn']
                    l_svpt = l_stats['svpt']
                    
                    # Validar tipos
                    if not isinstance(w_1st_in, (int, float)) or pd.isna(w_1st_in):
                        w_1st_in = 0
                    if not isinstance(w_svpt, (int, float)) or pd.isna(w_svpt):
                        w_svpt = 0
                    if not isinstance(l_1st_in, (int, float)) or pd.isna(l_1st_in):
                        l_1st_in = 0
                    if not isinstance(l_svpt, (int, float)) or pd.isna(l_svpt):
                        l_svpt = 0
                    
                    if w_svpt > 0 and l_svpt > 0:
                        w_1st_pct = w_1st_in / w_svpt
                        l_1st_pct = l_1st_in / l_svpt
                        
                        # Normalizar
                        dominance_score += (w_1st_pct - l_1st_pct) * 2
                        stats_count += 1
                
                # Si tenemos suficientes estadísticas, calcular factor final
                if stats_count >= 2:
                    avg_dominance = dominance_score / stats_count
                    
                    # Mapear a rango 0.8-1.2
                    # Mayor dominancia estadística = mayor ajuste de ELO
                    stats_factor = 1.0 + (avg_dominance * 0.2)
                    
                    # Limitar el rango
                    return max(0.8, min(1.2, stats_factor))
            
            # Si no hemos devuelto nada hasta aquí, intentar buscar en datos almacenados
            elif match_stats_df is not None and not match_stats_df.empty and (match_id is not None or (winner_id is not None and loser_id is not None)):
                # Intentar buscar estadísticas por match_id o combinación de jugadores
                if match_id and 'match_id' in match_stats_df.columns:
                    # Convertir match_id a mismo tipo que en DataFrame
                    if isinstance(match_stats_df['match_id'].iloc[0], str):
                        match_id = str(match_id)
                    
                    match_stats = match_stats_df[match_stats_df['match_id'] == match_id]
                    if not match_stats.empty:
                        # Extraer estadísticas relevantes y recursivamente llamar esta función
                        # Implementar según estructura específica de datos
                        pass  # Implementar si es necesario
                
                # Si no encontramos por match_id, intentar por combinación de jugadores
                if winner_id and loser_id:
                    # Asegurar que IDs son strings para comparar
                    winner_id_str = str(winner_id)
                    loser_id_str = str(loser_id)
                    
                    potential_matches = match_stats_df[
                        (match_stats_df['winner_id'].astype(str) == winner_id_str) & 
                        (match_stats_df['loser_id'].astype(str) == loser_id_str)
                    ]
                    
                    if not potential_matches.empty:
                        # Usar el partido más reciente si hay varios
                        if 'match_date' in potential_matches.columns:
                            match_stats = potential_matches.sort_values('match_date', ascending=False).iloc[0]
                        else:
                            match_stats = potential_matches.iloc[0]
                        
                        # Extraer estadísticas relevantes
                        extracted_w_stats = {}
                        extracted_l_stats = {}
                        
                        # Mapear columnas comunes en los datos de Jeff Sackmann
                        stat_cols = [
                            'ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 
                            'SvGms', 'bpSaved', 'bpFaced'
                        ]
                        
                        for col in stat_cols:
                            w_col = f'w_{col}'
                            l_col = f'l_{col}'
                            
                            if w_col in match_stats:
                                extracted_w_stats[col] = match_stats[w_col]
                            if l_col in match_stats:
                                extracted_l_stats[col] = match_stats[l_col]
                        
                        # Llamar recursivamente con las estadísticas extraídas
                        if extracted_w_stats and extracted_l_stats:
                            return self.get_match_stats_factor(
                                None, winner_id, loser_id, extracted_w_stats, extracted_l_stats
                            )
            
            # Si no hay suficientes datos, valor neutral
            return 1.0
        except Exception as e:
            logger.debug(f"Error calculando factor de estadísticas: {str(e)}")
            return 1.0  # Valor neutral en caso de error
    
    def get_margin_multiplier(self, score: str, retirement: bool = False) -> float:
        """
        Calcula multiplicador basado en el margen de victoria.
        
        Args:
            score: String con el resultado (e.g. '6-4 7-5')
            retirement: Indica si hubo abandono
            
        Returns:
            Multiplicador para el cambio de ELO (0.8-1.2)
        """
        # Validar tipo de retirement
        if not isinstance(retirement, bool):
            retirement = bool(retirement) if not pd.isna(retirement) else False
        
        if retirement:
            return 0.8  # Menor impacto para partidos con abandono
        
        # Validar score
        if not isinstance(score, str):
            score = str(score) if not pd.isna(score) else ''
        
        try:
            # Analizar el score con función mejorada
            sets_won_winner, sets_won_loser, games_diff, complete_match, stats = self.parse_score(score)
            
            # Validar tipos
            if not isinstance(stats, dict):
                return 1.0  # Valor neutral si la estructura es incorrecta
            
            # Para partidos incompletos, reducir impacto
            if not complete_match:
                return 0.85
            
            # Calcular dominancia con validación
            if sets_won_winner + sets_won_loser > 0:
                sets_ratio = sets_won_winner / max(1, sets_won_winner + sets_won_loser)
            else:
                sets_ratio = 0.67  # Valor neutral-positivo
            
            # Verificar que stats tiene las claves esperadas
            total_games = stats.get('total_games', 0)
            if not isinstance(total_games, (int, float)) or total_games <= 0:
                total_games = 1
            
            # Normalizar diferencia de games con validación
            normalized_games_diff = games_diff / total_games * 2
            
            # Considerar tiebreaks (partidos más ajustados)
            tiebreaks = stats.get('tiebreaks', 0)
            if not isinstance(tiebreaks, (int, float)):
                tiebreaks = 0
            tiebreak_factor = max(0.9, 1.0 - (tiebreaks * 0.05))
            
            # Considerar diferencia promedio por set
            avg_diff = stats.get('avg_game_diff', 2.0)
            if not isinstance(avg_diff, (int, float)):
                avg_diff = 2.0
            avg_diff_factor = min(1.2, 1.0 + (avg_diff * 0.05))
            
            # Combinar factores de dominancia
            dominance = (
                0.5 * sets_ratio +
                0.3 * normalized_games_diff +
                0.2 * avg_diff_factor
            ) * tiebreak_factor
            
            # Ajustar a rango 0.8-1.2
            return 0.8 + min(0.4, dominance * 0.4)
                
        except Exception as e:
            logger.debug(f"Error calculando margen: {str(e)}")
            return 1.0  # Valor predeterminado en caso de error
    
    def calculate_victory_impact_factor(self, winner_rating: float, loser_rating: float, 
                                  expected_prob: float, margin_multiplier: float,
                                  match_importance: float) -> float:
        """
        Calcula un factor de impacto para victorias "inesperadas" o importantes.
        Recompensa más las victorias sorprendentes (bajo expected_prob) o
        las victorias contundentes contra rivales fuertes.
        
        Args:
            winner_rating: Rating del ganador
            loser_rating: Rating del perdedor
            expected_prob: Probabilidad esperada de victoria
            margin_multiplier: Multiplicador basado en margen de victoria
            match_importance: Importancia del partido
            
        Returns:
            Factor de impacto de la victoria (1.0-1.5)
        """
        try:
            # Validar tipos de entrada
            if not isinstance(winner_rating, (int, float)) or pd.isna(winner_rating):
                winner_rating = self.initial_rating  # Valor predeterminado 
                
            if not isinstance(loser_rating, (int, float)) or pd.isna(loser_rating):
                loser_rating = self.initial_rating  # Valor predeterminado
                
            if not isinstance(expected_prob, (int, float)) or pd.isna(expected_prob) or expected_prob <= 0 or expected_prob >= 1:
                expected_prob = 0.5  # Valor neutral
                
            if not isinstance(margin_multiplier, (int, float)) or pd.isna(margin_multiplier) or margin_multiplier <= 0:
                margin_multiplier = 1.0  # Valor neutral
                
            if not isinstance(match_importance, (int, float)) or pd.isna(match_importance) or match_importance <= 0:
                match_importance = 1.0  # Valor neutral
            
            # Factores base
            upset_factor = 1.0
            solid_win_factor = 1.0
            rating_gap_factor = 1.0
            
            # 1. Upset factor - victorias inesperadas tienen mayor impacto
            if expected_prob < 0.5:
                # Más impacto cuanto menos esperada sea la victoria
                upset_factor = 1.0 + ((0.5 - expected_prob) * 2 * 0.3)
            
            # 2. Solid win factor - victorias contundentes tienen mayor impacto
            if margin_multiplier > 1.0:
                solid_win_factor = margin_multiplier
            
            # 3. Rating gap factor - vencer a alguien mucho mejor tiene mayor impacto
            rating_diff = loser_rating - winner_rating
            if rating_diff > 0:  # Ganador tenía peor rating
                # Normalizar diferencia de rating
                normalized_diff = min(1.0, rating_diff / 300)
                rating_gap_factor = 1.0 + (normalized_diff * 0.3)
            
            # Combinar todos los factores, pero limitar el impacto total
            combined_factor = 1.0 + (
                0.4 * (upset_factor - 1.0) +  # 40% peso para upsets
                0.3 * (solid_win_factor - 1.0) +  # 30% peso para victorias contundentes
                0.3 * (rating_gap_factor - 1.0)  # 30% peso para diferencia de rating
            ) * match_importance
            
            # Limitar a rango razonable
            return min(1.5, max(1.0, combined_factor))
        except Exception as e:
            logger.debug(f"Error calculando factor de impacto de victoria: {str(e)}")
            return 1.0  # Valor neutral en caso de error