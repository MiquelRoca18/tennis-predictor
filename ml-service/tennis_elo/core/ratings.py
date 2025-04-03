"""
ratings.py

Este módulo contiene la lógica para el cálculo y actualización de ratings ELO
en el sistema de predicción de tenis, incluyendo el enfoque bayesiano,
decaimiento temporal y transferencia entre superficies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

from tennis_elo.utils.normalizers import normalize_surface

# Configurar logging
logger = logging.getLogger(__name__)

class RatingCalculator:
    """
    Clase encargada del cálculo y actualización de ratings ELO para tenis,
    implementando enfoques avanzados como regularización bayesiana y
    transferencia entre superficies.
    """
    
    def __init__(self, 
                initial_rating: float = 1500,
                k_factor_base: float = 32, 
                decay_rate: float = 0.995,
                surface_transfer_matrix: Optional[Dict[str, Dict[str, float]]] = None,
                min_matches_full_rating: int = 10,
                use_bayesian: bool = True):
        """
        Inicializa el calculador de ratings.
        
        Args:
            initial_rating: Rating ELO inicial para nuevos jugadores
            k_factor_base: Factor K base para ajustes de rating
            decay_rate: Tasa de decaimiento mensual para jugadores inactivos
            surface_transfer_matrix: Matriz de transferencia entre superficies
            min_matches_full_rating: Mínimo de partidos para rating completo
            use_bayesian: Si debe usar enfoque bayesiano para ratings
        """
        self.initial_rating = initial_rating
        self.k_factor_base = k_factor_base
        self.decay_rate = decay_rate
        self.min_matches_full_rating = min_matches_full_rating
        self.use_bayesian = use_bayesian
        
        # Matriz de transferencia entre superficies (cuánto influye una superficie en otra)
        if surface_transfer_matrix is None:
            self.surface_transfer_matrix = {
                'hard': {'hard': 1.0, 'clay': 0.7, 'grass': 0.6, 'carpet': 0.8},
                'clay': {'hard': 0.7, 'clay': 1.0, 'grass': 0.5, 'carpet': 0.6},
                'grass': {'hard': 0.6, 'clay': 0.5, 'grass': 1.0, 'carpet': 0.7},
                'carpet': {'hard': 0.8, 'clay': 0.6, 'grass': 0.7, 'carpet': 1.0}
            }
        else:
            self.surface_transfer_matrix = surface_transfer_matrix
    
    def get_player_rating(self, player_id: str, surface: Optional[str] = None,
                        player_ratings: Dict[str, float] = None,
                        player_ratings_by_surface: Dict[str, Dict[str, float]] = None,
                        player_match_count: Dict[str, int] = None,
                        player_match_count_by_surface: Dict[str, Dict[str, int]] = None,
                        player_rating_uncertainty: Dict[str, float] = None) -> float:
        """
        Obtiene el rating ELO actual de un jugador con soporte bayesiano.
        
        Args:
            player_id: ID del jugador
            surface: Superficie específica (opcional)
            player_ratings: Diccionario de ratings generales
            player_ratings_by_surface: Diccionario de ratings por superficie
            player_match_count: Diccionario de conteo de partidos
            player_match_count_by_surface: Diccionario de conteo por superficie
            player_rating_uncertainty: Diccionario de incertidumbres
            
        Returns:
            Rating ELO (general o específico por superficie) con ajuste bayesiano si corresponde
        """
        # Validar que tenemos los diccionarios necesarios
        if player_ratings is None:
            player_ratings = {}
        if player_ratings_by_surface is None:
            player_ratings_by_surface = {'hard': {}, 'clay': {}, 'grass': {}, 'carpet': {}}
        if player_match_count is None:
            player_match_count = {}
        if player_match_count_by_surface is None:
            player_match_count_by_surface = {'hard': {}, 'clay': {}, 'grass': {}, 'carpet': {}}
            
        player_id = str(player_id)
        
        # Caso básico: ELO por superficie específica
        if surface:
            surface = normalize_surface(surface)
            raw_rating = player_ratings_by_surface[surface].get(player_id, self.initial_rating)
            
            # Si no usamos enfoque bayesiano o jugador tiene suficientes partidos, devolver rating directo
            matches_in_surface = player_match_count_by_surface[surface].get(player_id, 0)
            if not self.use_bayesian or matches_in_surface >= self.min_matches_full_rating:
                return raw_rating
            
            # Enfoque bayesiano: combinar con rating general
            general_rating = player_ratings.get(player_id, self.initial_rating)
            total_matches = player_match_count.get(player_id, 0)
            
            # Factor de confianza basado en número de partidos
            confidence = min(1.0, matches_in_surface / self.min_matches_full_rating)
            
            # Calcular rating combinado
            combined_rating = confidence * raw_rating + (1 - confidence) * general_rating
            
            # Transferencia entre superficies si hay muy pocos partidos en esta superficie
            if matches_in_surface < 3 and total_matches > 5:
                # Buscar ratings en otras superficies
                surface_ratings = {}
                for other_surface, ratings in player_ratings_by_surface.items():
                    if other_surface != surface and player_id in ratings:
                        other_matches = player_match_count_by_surface[other_surface].get(player_id, 0)
                        if other_matches > 3:
                            # Ponderar según la matriz de transferencia
                            transfer_weight = self.surface_transfer_matrix[surface][other_surface]
                            surface_ratings[other_surface] = ratings[player_id] * transfer_weight
                
                if surface_ratings:
                    # Calcular rating promedio ponderado de otras superficies
                    transferred_rating = sum(surface_ratings.values()) / len(surface_ratings)
                    
                    # Calcular peso para el rating transferido (inversamente proporcional a matches_in_surface)
                    transfer_weight = max(0, 1 - (matches_in_surface / 3))
                    
                    # Combinar con el rating actual
                    combined_rating = (1 - transfer_weight) * combined_rating + transfer_weight * transferred_rating
            
            return combined_rating
        
        # Caso: ELO general
        raw_rating = player_ratings.get(player_id, self.initial_rating)
        
        # Si no usamos enfoque bayesiano o el jugador tiene suficientes partidos, devolver rating directo
        matches_played = player_match_count.get(player_id, 0)
        if not self.use_bayesian or matches_played >= self.min_matches_full_rating:
            return raw_rating
        
        # Enfoque bayesiano para jugadores con pocos partidos
        # Regularizar hacia la media global
        ratings_values = list(player_ratings.values())
        if ratings_values:
            global_mean = sum(ratings_values) / len(ratings_values)
        else:
            global_mean = self.initial_rating
            
        confidence = min(1.0, matches_played / self.min_matches_full_rating)
        
        # Combinar con la media según nivel de confianza
        return confidence * raw_rating + (1 - confidence) * global_mean
    
    def get_player_uncertainty(self, player_id: str, surface: Optional[str] = None,
                             player_match_count: Dict[str, int] = None,
                             player_match_count_by_surface: Dict[str, Dict[str, int]] = None,
                             player_rating_uncertainty: Dict[str, float] = None) -> float:
        """
        Obtiene la incertidumbre (desviación estándar) del rating de un jugador.
        
        Args:
            player_id: ID del jugador
            surface: Superficie específica (opcional)
            player_match_count: Diccionario de conteo de partidos
            player_match_count_by_surface: Diccionario de conteo por superficie
            player_rating_uncertainty: Diccionario de incertidumbres
            
        Returns:
            Valor de incertidumbre del rating
        """
        player_id = str(player_id)
        
        # Validar diccionarios
        if player_match_count is None:
            player_match_count = {}
        if player_match_count_by_surface is None:
            player_match_count_by_surface = {'hard': {}, 'clay': {}, 'grass': {}, 'carpet': {}}
        if player_rating_uncertainty is None:
            player_rating_uncertainty = {}
        
        # Si hay poca información, alta incertidumbre
        if surface:
            surface = normalize_surface(surface)
            matches = player_match_count_by_surface[surface].get(player_id, 0)
        else:
            matches = player_match_count.get(player_id, 0)
        
        # Incertidumbre base - inversamente proporcional a partidos jugados
        # Comienza alta (~150) y disminuye con cada partido
        base_uncertainty = 350 / (matches + 5)
        
        # Si tenemos un valor específico de incertidumbre, usarlo
        if player_id in player_rating_uncertainty:
            return player_rating_uncertainty.get(player_id, base_uncertainty)
        
        return base_uncertainty
    
    def get_combined_surface_rating(self, player_id: str, surface: str,
                                  player_ratings: Dict[str, float] = None,
                                  player_ratings_by_surface: Dict[str, Dict[str, float]] = None,
                                  player_match_count: Dict[str, int] = None,
                                  player_match_count_by_surface: Dict[str, Dict[str, int]] = None) -> float:
        """
        Obtiene un rating combinado por superficie que integra información de
        todas las superficies ponderadas según la matriz de transferencia.
        
        Args:
            player_id: ID del jugador
            surface: Superficie para la que calcular el rating
            player_ratings: Diccionario de ratings generales
            player_ratings_by_surface: Diccionario de ratings por superficie
            player_match_count: Diccionario de conteo de partidos
            player_match_count_by_surface: Diccionario de conteo por superficie
            
        Returns:
            Rating ELO combinado para la superficie específica
        """
        # Validar tipos y argumentos
        player_id = str(player_id) if not pd.isna(player_id) else ''
        if not player_id:
            return self.initial_rating
        
        if not isinstance(surface, str):
            surface = str(surface) if not pd.isna(surface) else 'hard'
        surface = normalize_surface(surface)
        
        # Validar diccionarios
        if player_ratings is None:
            player_ratings = {}
        if player_ratings_by_surface is None:
            player_ratings_by_surface = {'hard': {}, 'clay': {}, 'grass': {}, 'carpet': {}}
        if player_match_count is None:
            player_match_count = {}
        if player_match_count_by_surface is None:
            player_match_count_by_surface = {'hard': {}, 'clay': {}, 'grass': {}, 'carpet': {}}
        
        try:
            # Rating específico de la superficie
            specific_rating = player_ratings_by_surface[surface].get(player_id, self.initial_rating)
            if not isinstance(specific_rating, (int, float)) or pd.isna(specific_rating):
                specific_rating = self.initial_rating
                
            specific_matches = player_match_count_by_surface[surface].get(player_id, 0)
            if not isinstance(specific_matches, (int, float)) or pd.isna(specific_matches):
                specific_matches = 0
            
            # Si tiene suficientes partidos en esta superficie, usar directamente
            if specific_matches >= self.min_matches_full_rating:
                return specific_rating
            
            # Calcular rating combinado usando la matriz de transferencia
            ratings_sum = specific_rating * specific_matches
            weights_sum = specific_matches
            
            # Combinar con ratings de otras superficies
            for other_surface, transfer_weight in self.surface_transfer_matrix[surface].items():
                if other_surface != surface:
                    # Validar valores
                    if not isinstance(transfer_weight, (int, float)) or pd.isna(transfer_weight):
                        continue
                        
                    other_rating = player_ratings_by_surface[other_surface].get(player_id, self.initial_rating)
                    if not isinstance(other_rating, (int, float)) or pd.isna(other_rating):
                        other_rating = self.initial_rating
                        
                    other_matches = player_match_count_by_surface[other_surface].get(player_id, 0)
                    if not isinstance(other_matches, (int, float)) or pd.isna(other_matches):
                        other_matches = 0
                    
                    # Ponderar según transferencia y número de partidos
                    if other_matches > 0:
                        effective_weight = transfer_weight * other_matches
                        ratings_sum += other_rating * effective_weight
                        weights_sum += effective_weight
            
            # Rating general como fallback
            general_rating = player_ratings.get(player_id, self.initial_rating)
            if not isinstance(general_rating, (int, float)) or pd.isna(general_rating):
                general_rating = self.initial_rating
                
            general_matches = player_match_count.get(player_id, 0)
            if not isinstance(general_matches, (int, float)) or pd.isna(general_matches):
                general_matches = 0
                
            effective_general_matches = general_matches - specific_matches
            if effective_general_matches > 0:
                ratings_sum += general_rating * (effective_general_matches * 0.5)  # Peso reducido
                weights_sum += (effective_general_matches * 0.5)
            
            # Evitar división por cero
            if weights_sum == 0:
                return self.initial_rating
            
            return ratings_sum / weights_sum
        except Exception as e:
            logger.debug(f"Error calculando rating combinado por superficie: {str(e)}")
            return self.initial_rating  # Valor por defecto en caso de error
    
    def calculate_expected_win_probability(self, rating_a: float, rating_b: float, 
                                        uncertainty_a: float = 0, uncertainty_b: float = 0) -> float:
        """
        Calcula la probabilidad esperada de victoria considerando incertidumbre.
        
        Args:
            rating_a: Rating ELO del jugador A
            rating_b: Rating ELO del jugador B
            uncertainty_a: Incertidumbre del rating de A
            uncertainty_b: Incertidumbre del rating de B
            
        Returns:
            Probabilidad de que el jugador A gane (0-1)
        """
        try:
            # Asegurar que los ratings son números
            if not isinstance(rating_a, (int, float)):
                rating_a = float(rating_a) if not pd.isna(rating_a) else self.initial_rating
            if not isinstance(rating_b, (int, float)):
                rating_b = float(rating_b) if not pd.isna(rating_b) else self.initial_rating
                
            # Asegurar que las incertidumbres son números no negativos
            if not isinstance(uncertainty_a, (int, float)) or uncertainty_a < 0:
                uncertainty_a = 0
            if not isinstance(uncertainty_b, (int, float)) or uncertainty_b < 0:
                uncertainty_b = 0
                
            # Considerar incertidumbre para regularizar las probabilidades
            # Mayor incertidumbre lleva la probabilidad hacia 0.5
            uncertainty_factor = (uncertainty_a + uncertainty_b) / 2
            
            # Fórmula ELO tradicional
            base_probability = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
            
            # Suavizar según incertidumbre
            if uncertainty_factor > 0:
                # El factor máximo de incertidumbre (300) suaviza completamente hacia 0.5
                max_uncertainty = 300
                regularization = min(1.0, uncertainty_factor / max_uncertainty)
                
                # Interpolación lineal hacia 0.5
                return base_probability * (1 - regularization) + 0.5 * regularization
            
            return base_probability
        
        except Exception as e:
            # En caso de error, devolver probabilidad neutral
            logger.warning(f"Error en calculate_expected_win_probability: {str(e)}")
            return 0.5
    
    def apply_temporal_decay(self, player_id: str, current_date: datetime,
                          player_ratings: Dict[str, float],
                          player_ratings_by_surface: Dict[str, Dict[str, float]],
                          player_last_match: Dict[str, datetime],
                          player_rating_uncertainty: Dict[str, float]) -> None:
        """
        Aplica decaimiento temporal al rating de un jugador inactivo.
        Implementa una curva de decaimiento exponencial que es
        más lenta al principio y se acelera con el tiempo.
        
        Args:
            player_id: ID del jugador
            current_date: Fecha actual
            player_ratings: Diccionario de ratings generales
            player_ratings_by_surface: Diccionario de ratings por superficie
            player_last_match: Diccionario de fechas del último partido
            player_rating_uncertainty: Diccionario de incertidumbres
        """
        player_id = str(player_id)
        
        # Validar tipos
        if not isinstance(current_date, datetime):
            try:
                current_date = pd.to_datetime(current_date)
            except:
                # Si no se puede convertir, usar fecha actual
                current_date = datetime.now()
        
        # Obtener fecha último partido con validación
        last_match_date = player_last_match.get(player_id)
        
        # Verificar que last_match_date es una fecha válida
        if last_match_date is None:
            return  # No hay fecha anterior, no aplicar decaimiento
        
        if not isinstance(last_match_date, datetime):
            try:
                last_match_date = pd.to_datetime(last_match_date)
            except:
                # Si no es convertible, actualizar con fecha actual y salir
                player_last_match[player_id] = current_date
                return
        
        if current_date > last_match_date:
            try:
                # Calcular días de inactividad
                days_inactive = (current_date - last_match_date).days
                
                # Sólo aplicar decaimiento después de 30 días
                if days_inactive > 30:
                    # Calcular meses de inactividad
                    months_inactive = days_inactive / 30.0
                    
                    # Aplicar decaimiento exponencial con diferentes tasas
                    # según tiempo de inactividad
                    if months_inactive <= 3:
                        # Primera fase: decaimiento lento
                        decay_factor = self.decay_rate ** (months_inactive - 1)
                    elif months_inactive <= 6:
                        # Segunda fase: decaimiento medio
                        decay_factor = self.decay_rate ** (2 + (months_inactive - 3) * 1.5)
                    else:
                        # Tercera fase: decaimiento acelerado para inactividad prolongada
                        decay_factor = self.decay_rate ** (6.5 + (months_inactive - 6) * 2)
                    
                    # Limitar el decaimiento máximo (no bajar más del 40% del rating)
                    decay_factor = max(0.6, decay_factor)
                    
                    # Aplicar a rating general con validación
                    if player_id in player_ratings:
                        current_rating = player_ratings[player_id]
                        
                        # Verificar que el rating es un número
                        if not isinstance(current_rating, (int, float)):
                            current_rating = self.initial_rating
                            player_ratings[player_id] = current_rating
                        
                        # Calcular media global con validación
                        ratings_values = list(player_ratings.values())
                        valid_ratings = [r for r in ratings_values if isinstance(r, (int, float))]
                        
                        if valid_ratings:
                            global_mean = sum(valid_ratings) / len(valid_ratings)
                        else:
                            global_mean = self.initial_rating
                        
                        # Decaer hacia la media, no hacia 0
                        decay_target = global_mean * 0.9  # 90% de la media global
                        player_ratings[player_id] = current_rating * decay_factor + decay_target * (1 - decay_factor)
                    
                    # Aplicar a ratings por superficie con validación
                    for surface in player_ratings_by_surface:
                        if player_id in player_ratings_by_surface[surface]:
                            current_surface_rating = player_ratings_by_surface[surface][player_id]
                            
                            # Verificar que el rating es un número
                            if not isinstance(current_surface_rating, (int, float)):
                                current_surface_rating = self.initial_rating
                                player_ratings_by_surface[surface][player_id] = current_surface_rating
                            
                            # Calcular media de la superficie con validación
                            surface_ratings = list(player_ratings_by_surface[surface].values())
                            valid_surface_ratings = [r for r in surface_ratings if isinstance(r, (int, float))]
                            
                            if valid_surface_ratings:
                                surface_mean = sum(valid_surface_ratings) / len(valid_surface_ratings)
                                decay_target = surface_mean * 0.9
                            else:
                                decay_target = self.initial_rating
                            
                            # Aplicar decaimiento hacia la media de la superficie
                            player_ratings_by_surface[surface][player_id] = (
                                current_surface_rating * decay_factor + 
                                decay_target * (1 - decay_factor)
                            )
                    
                    # Aumentar incertidumbre con el tiempo
                    if player_id in player_rating_uncertainty:
                        current_uncertainty = player_rating_uncertainty[player_id]
                        
                        # Verificar que es un número
                        if not isinstance(current_uncertainty, (int, float)) or current_uncertainty < 0:
                            current_uncertainty = 100.0
                        
                        # Aumentar incertidumbre en función del tiempo
                        uncertainty_increase = min(100, months_inactive * 5)
                        player_rating_uncertainty[player_id] = current_uncertainty + uncertainty_increase
            
            except Exception as e:
                logger.warning(f"Error en temporal decay para {player_id}: {str(e)}")
                # No interrumpir el proceso si hay un error en un jugador específico