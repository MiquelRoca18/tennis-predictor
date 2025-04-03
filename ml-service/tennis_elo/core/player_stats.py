"""
player_stats.py

Este módulo contiene funciones para gestionar y analizar estadísticas de jugadores,
incluida la forma reciente, historial head-to-head, y obtención de información
detallada de jugadores.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from collections import defaultdict

from tennis_elo.utils.normalizers import normalize_surface

# Configurar logging
logger = logging.getLogger(__name__)

class PlayerStatsManager:
    """
    Gestiona estadísticas de jugadores, incluyendo forma reciente,
    historial head-to-head y recuperación de información detallada.
    """
    
    def __init__(self, initial_rating: float = 1500, form_window_matches: int = 10):
        """
        Inicializa el gestor de estadísticas de jugadores.
        
        Args:
            initial_rating: Rating ELO inicial para nuevos jugadores
            form_window_matches: Ventana de partidos para calcular forma
        """
        self.initial_rating = initial_rating
        self.form_window_matches = form_window_matches
        
        # Diccionarios para seguimiento
        self.player_names = {}
        self.player_recent_form = {}
        self.player_last_match = {}
        self.player_match_history = defaultdict(list)
        self.h2h_records = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0}))
        self.player_fatigue = {}
    
    def get_player_name(self, player_id: str) -> str:
        """
        Obtiene el nombre de un jugador según su ID.
        
        Args:
            player_id: ID del jugador
            
        Returns:
            Nombre del jugador o el ID si no se encuentra
        """
        player_id = str(player_id)
        return self.player_names.get(player_id, player_id)
    
    def get_player_form(self, player_id: str, surface: Optional[str] = None) -> float:
        """
        Obtiene el factor de forma reciente de un jugador (entre 0.8 y 1.2).
        
        Args:
            player_id: ID del jugador
            surface: Superficie específica (opcional)
            
        Returns:
            Factor de forma reciente
        """
        # Validar tipos
        player_id = str(player_id) if not pd.isna(player_id) else ''
        if not player_id:
            return 1.0  # Valor neutral para ID inválido
        
        try:
            # Si no tenemos datos de forma, devolver valor neutral
            if player_id not in self.player_recent_form:
                return 1.0
            
            # Validar la estructura del diccionario
            form_data = self.player_recent_form[player_id]
            
            if not isinstance(form_data, dict):
                return 1.0
            
            if 'form' not in form_data:
                return 1.0
            
            # Obtener forma general
            general_form = form_data.get('form', 1.0)
            if not isinstance(general_form, (int, float)) or pd.isna(general_form) or general_form <= 0:
                general_form = 1.0
            
            # Si no se especifica superficie, devolver forma general
            if not surface:
                return general_form
            
            # Normalizar superficie
            if not isinstance(surface, str):
                surface = str(surface) if not pd.isna(surface) else 'hard'
            surface = normalize_surface(surface)
            
            # Si tenemos forma específica por superficie, combinar
            surface_key = f"{player_id}_{surface}"
            surface_form_dict = self.player_recent_form.get(surface_key, {})
            
            if not surface_form_dict or not isinstance(surface_form_dict, dict):
                return general_form
            
            # Extraer datos específicos por superficie con validación
            surface_form = surface_form_dict.get('form', 1.0)
            if not isinstance(surface_form, (int, float)) or pd.isna(surface_form) or surface_form <= 0:
                surface_form = 1.0
                
            surface_matches = surface_form_dict.get('matches', 0)
            if not isinstance(surface_matches, (int, float)) or pd.isna(surface_matches):
                surface_matches = 0
            
            # Ponderar según número de partidos
            if surface_matches >= 5:
                return surface_form
            elif surface_matches == 0:
                return general_form
            else:
                # Ponderación progresiva
                weight = surface_matches / 5
                return (weight * surface_form) + ((1 - weight) * general_form)
        except Exception as e:
            logger.debug(f"Error calculando forma del jugador: {str(e)}")
            return 1.0  # Valor neutral en caso de error
    
    def update_player_form(self, player_id: str, result: int, surface: Optional[str] = None) -> None:
        """
        Actualiza el factor de forma reciente de un jugador.
        
        Args:
            player_id: ID del jugador
            result: Resultado (1 para victoria, 0 para derrota)
            surface: Superficie del partido (opcional)
        """
        # Validar tipos
        player_id = str(player_id) if not pd.isna(player_id) else ''
        if not player_id:
            return  # No actualizar forma para ID inválido
            
        if not isinstance(result, (int, float)) or pd.isna(result):
            result = 0  # Valor por defecto
        else:
            # Normalizar a 0 o 1
            result = 1 if result > 0 else 0
        
        # Actualizar forma general
        try:
            if player_id not in self.player_recent_form:
                self.player_recent_form[player_id] = {
                    'results': [],
                    'form': 1.0,
                    'matches': 0
                }
            
            # Validar la estructura del diccionario
            form_data = self.player_recent_form[player_id]
            
            if not isinstance(form_data, dict):
                form_data = {'results': [], 'form': 1.0, 'matches': 0}
                self.player_recent_form[player_id] = form_data
            
            # Asegurar que tenemos la estructura de datos correcta
            if 'results' not in form_data:
                form_data['results'] = []
            if 'form' not in form_data:
                form_data['form'] = 1.0
            if 'matches' not in form_data:
                form_data['matches'] = 0
            
            # Añadir nuevo resultado y mantener ventana móvil
            form_data['results'].append(result)
            if len(form_data['results']) > self.form_window_matches:
                form_data['results'].pop(0)
            
            # Recalcular factor de forma con ponderación reciente
            if form_data['results']:
                # Los partidos más recientes tienen más peso
                weighted_sum = 0
                weights_sum = 0
                
                for i, res in enumerate(form_data['results']):
                    # Peso exponencial: partidos más recientes cuentan más
                    weight = 1.5 ** i  # i=0 es el más antiguo
                    weighted_sum += res * weight
                    weights_sum += weight
                
                # Evitar división por cero
                weighted_avg = weighted_sum / weights_sum if weights_sum > 0 else 0.5
                
                # Mapear a un rango de 0.8 a 1.2
                form_factor = 0.8 + (weighted_avg * 0.4)
                form_data['form'] = form_factor
                form_data['matches'] = len(form_data['results'])
        except Exception as e:
            logger.debug(f"Error actualizando forma general: {str(e)}")
        
        # Si se especifica superficie, actualizar forma específica
        if surface:
            try:
                surface = normalize_surface(surface)
                surface_key = f"{player_id}_{surface}"
                
                if surface_key not in self.player_recent_form:
                    self.player_recent_form[surface_key] = {
                        'results': [],
                        'form': 1.0,
                        'matches': 0
                    }
                
                # Validar la estructura del diccionario
                surface_form = self.player_recent_form[surface_key]
                
                if not isinstance(surface_form, dict):
                    surface_form = {'results': [], 'form': 1.0, 'matches': 0}
                    self.player_recent_form[surface_key] = surface_form
                
                # Asegurar que tenemos la estructura de datos correcta
                if 'results' not in surface_form:
                    surface_form['results'] = []
                if 'form' not in surface_form:
                    surface_form['form'] = 1.0
                if 'matches' not in surface_form:
                    surface_form['matches'] = 0
                
                # Actualizar de manera similar a la forma general
                surface_form['results'].append(result)
                if len(surface_form['results']) > self.form_window_matches:
                    surface_form['results'].pop(0)
                
                # Recalcular con los mismos pesos
                if surface_form['results']:
                    weighted_sum = 0
                    weights_sum = 0
                    
                    for i, res in enumerate(surface_form['results']):
                        weight = 1.5 ** i
                        weighted_sum += res * weight
                        weights_sum += weight
                    
                    # Evitar división por cero
                    weighted_avg = weighted_sum / weights_sum if weights_sum > 0 else 0.5
                    surface_form['form'] = 0.8 + (weighted_avg * 0.4)
                    surface_form['matches'] = len(surface_form['results'])
            except Exception as e:
                logger.debug(f"Error actualizando forma por superficie: {str(e)}")
    
    def get_h2h_advantage(self, player1_id: str, player2_id: str) -> float:
        """
        Calcula el factor de ventaja basado en el historial head-to-head.
        
        Args:
            player1_id: ID del primer jugador
            player2_id: ID del segundo jugador
            
        Returns:
            Factor de ventaja (0.9-1.1) basado en historial h2h
        """
        player1_id = str(player1_id)
        player2_id = str(player2_id)
        
        # Verificar que h2h_records es un defaultdict antes de acceder
        if not hasattr(self, 'h2h_records') or not self.h2h_records:
            return 1.0
        
        # Obtener historial h2h con validación de tipos
        try:
            p1_vs_p2 = self.h2h_records[player1_id][player2_id]
            p2_vs_p1 = self.h2h_records[player2_id][player1_id]
            
            # Verificar que los valores son diccionarios y tienen la estructura esperada
            if not isinstance(p1_vs_p2, dict) or 'wins' not in p1_vs_p2:
                return 1.0
            if not isinstance(p2_vs_p1, dict) or 'wins' not in p2_vs_p1:
                return 1.0
            
            p1_wins = p1_vs_p2['wins']
            p2_wins = p2_vs_p1['wins']
            
            # Validar que son números
            if not isinstance(p1_wins, (int, float)) or not isinstance(p2_wins, (int, float)):
                return 1.0
            
            total_matches = p1_wins + p2_wins
            
            # Si no hay historial, no hay ventaja
            if total_matches == 0:
                return 1.0
            
            # Calcular ratio de victorias
            if total_matches >= 2:
                p1_win_ratio = p1_wins / total_matches
                
                # Mapear a un rango de 0.9 a 1.1 (mayores valores favorecen a p1)
                # Con más partidos, la ventaja puede ser mayor
                max_factor = min(0.1, 0.05 + (total_matches * 0.005))
                h2h_factor = 1.0 + ((p1_win_ratio - 0.5) * 2 * max_factor)
                
                return h2h_factor
            elif total_matches == 1:
                # Para un solo partido, efecto menor
                return 1.05 if p1_wins > 0 else 0.95
        except Exception as e:
            # En caso de error, devolver valor neutral
            logger.debug(f"Error en get_h2h_advantage: {str(e)}")
            return 1.0
        
        return 1.0
    
    def update_h2h_record(self, winner_id: str, loser_id: str) -> None:
        """
        Actualiza el registro head-to-head entre dos jugadores.
        
        Args:
            winner_id: ID del jugador ganador
            loser_id: ID del jugador perdedor
        """
        # Validar tipos
        winner_id = str(winner_id) if not pd.isna(winner_id) else ''
        loser_id = str(loser_id) if not pd.isna(loser_id) else ''
        
        # Verificar IDs válidos
        if not winner_id or not loser_id or winner_id == loser_id:
            return  # No actualizar para IDs inválidos
        
        try:
            # Asegurar que tenemos la estructura correcta para el ganador
            if winner_id not in self.h2h_records:
                self.h2h_records[winner_id] = {}
                
            if loser_id not in self.h2h_records[winner_id]:
                self.h2h_records[winner_id][loser_id] = {'wins': 0, 'losses': 0}
            elif not isinstance(self.h2h_records[winner_id][loser_id], dict):
                # Corregir si no es un diccionario
                self.h2h_records[winner_id][loser_id] = {'wins': 0, 'losses': 0}
            elif 'wins' not in self.h2h_records[winner_id][loser_id]:
                self.h2h_records[winner_id][loser_id]['wins'] = 0
            
            # Asegurar que tenemos la estructura correcta para el perdedor
            if loser_id not in self.h2h_records:
                self.h2h_records[loser_id] = {}
                
            if winner_id not in self.h2h_records[loser_id]:
                self.h2h_records[loser_id][winner_id] = {'wins': 0, 'losses': 0}
            elif not isinstance(self.h2h_records[loser_id][winner_id], dict):
                # Corregir si no es un diccionario
                self.h2h_records[loser_id][winner_id] = {'wins': 0, 'losses': 0}
            elif 'losses' not in self.h2h_records[loser_id][winner_id]:
                self.h2h_records[loser_id][winner_id]['losses'] = 0
            
            # Actualizar registro del ganador contra el perdedor
            current_wins = self.h2h_records[winner_id][loser_id]['wins']
            if not isinstance(current_wins, (int, float)):
                current_wins = 0
            self.h2h_records[winner_id][loser_id]['wins'] = current_wins + 1
            
            # Actualizar registro del perdedor contra el ganador
            current_losses = self.h2h_records[loser_id][winner_id]['losses']
            if not isinstance(current_losses, (int, float)):
                current_losses = 0
            self.h2h_records[loser_id][winner_id]['losses'] = current_losses + 1
            
        except Exception as e:
            logger.debug(f"Error actualizando registro head-to-head: {str(e)}")
    
    def update_match_history(self, player_id: str, match_data: Dict) -> None:
        """
        Actualiza el historial de partidos de un jugador.
        
        Args:
            player_id: ID del jugador
            match_data: Datos del partido a añadir
        """
        player_id = str(player_id) if not pd.isna(player_id) else ''
        if not player_id or not isinstance(match_data, dict):
            return
        
        try:
            # Asegurar que match_data tiene los campos mínimos necesarios
            required_fields = ['date', 'opponent_id', 'result']
            if not all(field in match_data for field in required_fields):
                logger.debug(f"Match data incompleto para player_id {player_id}")
                return
            
            # Añadir al historial
            self.player_match_history[player_id].append(match_data)
            
            # Actualizar fecha del último partido
            if 'date' in match_data:
                match_date = match_data['date']
                if isinstance(match_date, (datetime, pd.Timestamp)):
                    current_last_match = self.player_last_match.get(player_id)
                    if current_last_match is None or match_date > current_last_match:
                        self.player_last_match[player_id] = match_date
        except Exception as e:
            logger.debug(f"Error actualizando historial de partidos: {str(e)}")
    
    def get_player_fatigue(self, player_id: str, current_date: datetime) -> float:
        """
        Calcula el nivel de fatiga de un jugador basado en partidos recientes.
        
        Args:
            player_id: ID del jugador
            current_date: Fecha actual
            
        Returns:
            Factor de fatiga (0.8-1.0, valores más bajos indican mayor fatiga)
        """
        player_id = str(player_id) if not pd.isna(player_id) else ''
        if not player_id:
            return 1.0  # Sin fatiga para ID inválido
        
        try:
            # Obtener partidos recientes (últimos 30 días)
            recent_matches = []
            for match in self.player_match_history.get(player_id, []):
                match_date = match.get('date')
                if match_date and isinstance(match_date, (datetime, pd.Timestamp)):
                    # Solo considerar partidos recientes
                    if (current_date - match_date).days <= 30:
                        recent_matches.append(match)
            
            # Si no hay partidos recientes, no hay fatiga
            if not recent_matches:
                return 1.0
            
            # Calcular fatiga basada en número y frecuencia de partidos recientes
            num_matches = len(recent_matches)
            
            # Más de 10 partidos en 30 días genera fatiga significativa
            if num_matches > 10:
                base_fatigue = 0.8 + (0.2 * (1 - min(1.0, (num_matches - 10) / 10)))
            else:
                # Hasta 10 partidos, fatiga moderada a ninguna
                base_fatigue = 1.0 - (0.02 * num_matches)
            
            # Considerar también la intensidad (duración) de los partidos si está disponible
            # y proximidad entre partidos recientes
            # Este sería un modelo más avanzado que requiere datos adicionales
            
            return max(0.8, min(1.0, base_fatigue))
        except Exception as e:
            logger.debug(f"Error calculando fatiga: {str(e)}")
            return 1.0  # Valor por defecto (sin fatiga)

    def get_player_details(self, player_id: str, 
                         player_ratings: Dict[str, float] = None,
                         player_ratings_by_surface: Dict[str, Dict[str, float]] = None,
                         player_match_count: Dict[str, int] = None,
                         player_match_count_by_surface: Dict[str, Dict[str, int]] = None,
                         player_uncertainty: Dict[str, float] = None) -> Dict:
        """
        Obtiene información detallada del jugador incluyendo ratings,
        estadísticas, forma reciente y más.
        
        Args:
            player_id: ID del jugador
            player_ratings: Diccionario de ratings generales (opcional)
            player_ratings_by_surface: Diccionario de ratings por superficie (opcional)
            player_match_count: Diccionario de conteo de partidos (opcional)
            player_match_count_by_surface: Diccionario de conteo por superficie (opcional)
            player_uncertainty: Diccionario de incertidumbres (opcional)
            
        Returns:
            Diccionario con toda la información disponible del jugador
        """
        player_id = str(player_id)
        
        # Información básica y nombre
        player_name = self.get_player_name(player_id)
        
        # Información básica de ELO
        elo_general = player_ratings.get(player_id, self.initial_rating) if player_ratings else self.initial_rating
        
        # ELO por superficie
        elo_by_surface = {}
        if player_ratings_by_surface:
            for surface, ratings in player_ratings_by_surface.items():
                elo_by_surface[surface] = ratings.get(player_id, self.initial_rating)
        
        # Encuentros por superficie
        matches_by_surface = {}
        if player_match_count_by_surface:
            for surface, counts in player_match_count_by_surface.items():
                matches_by_surface[surface] = counts.get(player_id, 0)
        
        # Forma reciente
        recent_form = self.get_player_form(player_id)
        
        # Historial de partidos
        match_history = self.player_match_history.get(player_id, [])
        
        # Extraer estadísticas del historial de partidos
        total_matches = player_match_count.get(player_id, 0) if player_match_count else len(match_history)
        wins = len([m for m in match_history if m['result'] == 'win'])
        losses = total_matches - wins
        
        # Estadísticas por superficie
        surface_stats = {}
        for surface in elo_by_surface.keys() if player_ratings_by_surface else []:
            surface_matches = [m for m in match_history if m.get('surface') == surface]
            surface_wins = len([m for m in surface_matches if m['result'] == 'win'])
            surface_losses = len(surface_matches) - surface_wins
            
            surface_stats[surface] = {
                'matches': len(surface_matches),
                'wins': surface_wins,
                'losses': surface_losses,
                'win_rate': surface_wins / max(1, len(surface_matches))
            }
        
        # Extraer información de rendimiento por tipo de torneo
        tourney_stats = {}
        for match in match_history:
            level = match.get('tourney_level', 'unknown')
            if level not in tourney_stats:
                tourney_stats[level] = {'wins': 0, 'losses': 0}
            
            if match['result'] == 'win':
                tourney_stats[level]['wins'] += 1
            else:
                tourney_stats[level]['losses'] += 1
        
        # Calcular win rate por tipo de torneo
        for level in tourney_stats:
            stats = tourney_stats[level]
            total = stats['wins'] + stats['losses']
            stats['matches'] = total
            stats['win_rate'] = stats['wins'] / max(1, total)
        
        # Información de incertidumbre
        uncertainty = player_uncertainty.get(player_id, 0) if player_uncertainty else 0
        
        # Información de rivales
        rivals = {}
        for opponent_id in self.h2h_records.get(player_id, {}):
            h2h = self.h2h_records[player_id][opponent_id]
            if h2h['wins'] > 0 or h2h['losses'] > 0:
                opponent_name = self.get_player_name(opponent_id)
                rivals[opponent_id] = {
                    'name': opponent_name,
                    'wins': h2h['wins'],
                    'losses': h2h['losses'],
                    'total': h2h['wins'] + h2h['losses'],
                    'win_rate': h2h['wins'] / max(1, h2h['wins'] + h2h['losses'])
                }
        
        # Organizar toda la información
        return {
            'id': player_id,
            'name': player_name,
            'elo': {
                'general': elo_general,
                'by_surface': elo_by_surface,
                'uncertainty': uncertainty
            },
            'stats': {
                'total_matches': total_matches,
                'wins': wins,
                'losses': losses,
                'win_rate': wins / max(1, total_matches),
                'by_surface': surface_stats,
                'by_tourney_level': tourney_stats
            },
            'form': recent_form,
            'match_count': total_matches,
            'matches_by_surface': matches_by_surface,
            'last_match': self.player_last_match.get(player_id),
            'rivals': rivals,
            'recent_matches': sorted(match_history, key=lambda x: x['date'], reverse=True)[:10] if match_history else []
        }