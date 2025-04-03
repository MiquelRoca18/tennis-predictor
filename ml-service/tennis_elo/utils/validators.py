"""
utils/validators.py

Funciones para validar datos de tenis:
- Tipos de datos (strings, numéricos, fechas)
- IDs de jugadores
- Resultados de partidos
- Estructuras de datos complejas

Proporciona validación robusta para todas las entradas del sistema.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Union, Set

# Configurar logging
logger = logging.getLogger(__name__)

class TennisDataValidator:
    """
    Clase para validar datos de tenis.
    Proporciona métodos para verificar la validez y consistencia
    de diferentes tipos de datos utilizados en el sistema.
    """
    
    def __init__(self):
        """Inicializa el validador con configuraciones predeterminadas."""
        # Configuración para validaciones
        self.min_player_id_length = 1
        self.max_player_id_length = 50
        self.valid_match_results = {0, 1}  # 0 = derrota, 1 = victoria
        self.min_valid_rating = 0
        self.max_valid_rating = 3000
        self.valid_surfaces = {'hard', 'clay', 'grass', 'carpet'}
        
    def is_valid_string(self, value: Any) -> bool:
        """
        Verifica si un valor es una cadena de texto válida y no vacía.
        
        Args:
            value: Valor a validar
            
        Returns:
            True si es una cadena válida, False en caso contrario
        """
        if pd.isna(value):
            return False
            
        if not isinstance(value, str):
            try:
                value = str(value)
            except:
                return False
                
        return bool(value.strip())
    
    def is_valid_numeric(self, value: Any, min_value: Optional[float] = None, 
                      max_value: Optional[float] = None) -> bool:
        """
        Verifica si un valor es numérico y está dentro del rango especificado.
        
        Args:
            value: Valor a validar
            min_value: Valor mínimo válido (opcional)
            max_value: Valor máximo válido (opcional)
            
        Returns:
            True si es numérico y está en rango, False en caso contrario
        """
        if pd.isna(value):
            return False
            
        try:
            # Convertir a float para validación
            num_value = float(value)
            
            # Verificar rango si se especifica
            if min_value is not None and num_value < min_value:
                return False
                
            if max_value is not None and num_value > max_value:
                return False
                
            return True
        except (ValueError, TypeError):
            return False
    
    def is_valid_date(self, date_value: Any) -> bool:
        """
        Verifica si un valor es una fecha válida.
        
        Args:
            date_value: Valor de fecha a validar
            
        Returns:
            True si es una fecha válida, False en caso contrario
        """
        if pd.isna(date_value):
            return False
            
        # Si ya es datetime, es válido
        if isinstance(date_value, (datetime, pd.Timestamp)):
            return True
            
        # Intentar convertir
        try:
            # Para string de formato estándar ATP (YYYYMMDD)
            date_str = str(date_value).strip()
            
            if len(date_str) == 8 and date_str.isdigit():
                pd.to_datetime(date_str, format='%Y%m%d')
                return True
                
            # Intentar con to_datetime general
            pd.to_datetime(date_str)
            return True
        except:
            return False
    
    def is_valid_player_id(self, player_id: Any) -> bool:
        """
        Verifica si un ID de jugador es válido.
        
        Args:
            player_id: ID de jugador a validar
            
        Returns:
            True si es un ID válido, False en caso contrario
        """
        if pd.isna(player_id):
            return False
            
        # Convertir a string si no lo es
        if not isinstance(player_id, str):
            try:
                player_id = str(player_id)
            except:
                return False
        
        # Validar longitud y contenido
        player_id = player_id.strip()
        if not player_id:
            return False
            
        if len(player_id) < self.min_player_id_length or len(player_id) > self.max_player_id_length:
            return False
            
        return True
    
    def is_valid_match_result(self, result: Any) -> bool:
        """
        Verifica si un resultado de partido es válido (0 o 1).
        
        Args:
            result: Resultado a validar
            
        Returns:
            True si es un resultado válido, False en caso contrario
        """
        if pd.isna(result):
            return False
            
        try:
            # Intentar convertir a entero
            int_result = int(float(result))
            return int_result in self.valid_match_results
        except (ValueError, TypeError):
            return False
    
    def is_valid_elo_rating(self, rating: Any) -> bool:
        """
        Verifica si un rating ELO es válido.
        
        Args:
            rating: Rating a validar
            
        Returns:
            True si es un rating válido, False en caso contrario
        """
        return self.is_valid_numeric(rating, self.min_valid_rating, self.max_valid_rating)
    
    def is_valid_surface(self, surface: Any) -> bool:
        """
        Verifica si una superficie es válida.
        
        Args:
            surface: Superficie a validar
            
        Returns:
            True si es una superficie válida, False en caso contrario
        """
        if not self.is_valid_string(surface):
            return False
            
        surface_lower = str(surface).lower().strip()
        
        # Verificar directamente
        if surface_lower in self.valid_surfaces:
            return True
            
        # Verificar por contenido
        if 'hard' in surface_lower:
            return True
        elif any(term in surface_lower for term in ['clay', 'arcilla', 'terre']):
            return True
        elif any(term in surface_lower for term in ['grass', 'hierba']):
            return True
        elif any(term in surface_lower for term in ['carpet', 'indoor']):
            return True
            
        return False
    
    def is_valid_score(self, score: Any) -> bool:
        """
        Verifica si un string de score de tenis tiene un formato válido.
        
        Args:
            score: Score a validar
            
        Returns:
            True si tiene formato válido, False en caso contrario
        """
        if not self.is_valid_string(score):
            return False
            
        score_str = str(score).strip()
        
        # Detectar abandonos, que son resultados válidos
        special_terms = ['ret', 'reti', 'def', 'w/o', 'walkover', 'default']
        if any(term in score_str.lower() for term in special_terms):
            return True
            
        # Verificar formato básico para scores normales
        # Permitir espacios, guiones, puntos o comas como separadores
        sets = score_str.replace(':', '-').replace(',', ' ').replace(';', ' ').split()
        
        if not sets:
            return False
            
        # Debe tener al menos un set con formato válido
        valid_sets = False
        
        for set_score in sets:
            # Si tiene tiebreak, formato diferente
            if '(' in set_score:
                set_part = set_score.split('(')[0]
            else:
                set_part = set_score
                
            # Verificar formato "X-Y" o "XY" donde X e Y son dígitos
            if '-' in set_part:
                parts = set_part.split('-')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    valid_sets = True
                    break
            elif len(set_part) == 2 and set_part.isdigit():
                valid_sets = True
                break
                
        return valid_sets
    
    def is_valid_match_data(self, match_data: Dict) -> Tuple[bool, Optional[str]]:
        """
        Verifica si los datos de un partido son válidos y completos.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        # Verificar campos requeridos
        required_fields = ['winner_id', 'loser_id', 'match_date', 'surface']
        for field in required_fields:
            if field not in match_data:
                return False, f"Campo requerido '{field}' faltante"
                
        # Validar IDs
        if not self.is_valid_player_id(match_data['winner_id']):
            return False, "ID del ganador inválido"
            
        if not self.is_valid_player_id(match_data['loser_id']):
            return False, "ID del perdedor inválido"
            
        # Verificar que ganador y perdedor sean diferentes
        if str(match_data['winner_id']) == str(match_data['loser_id']):
            return False, "El ganador y el perdedor no pueden ser el mismo jugador"
            
        # Validar fecha
        if not self.is_valid_date(match_data['match_date']):
            return False, "Fecha del partido inválida"
            
        # Validar superficie
        if not self.is_valid_surface(match_data['surface']):
            return False, "Superficie de juego inválida"
            
        # Validar score si está presente
        if 'score' in match_data and match_data['score'] and not self.is_valid_score(match_data['score']):
            return False, "Formato de score inválido"
            
        return True, None
    
    def is_valid_dataframe_structure(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Verifica si un DataFrame tiene la estructura requerida.
        
        Args:
            df: DataFrame a validar
            required_columns: Lista de columnas requeridas
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        if df is None or df.empty:
            return False, "DataFrame vacío o nulo"
            
        # Verificar columnas requeridas
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Columnas faltantes: {missing_columns}"
            
        return True, None
    
    def validate_dataframe_content(self, df: pd.DataFrame, column_validators: Dict[str, callable]) -> pd.DataFrame:
        """
        Valida el contenido de un DataFrame y devuelve un DataFrame con filas válidas.
        
        Args:
            df: DataFrame a validar
            column_validators: Diccionario {columna: función_validadora}
            
        Returns:
            DataFrame filtrado con solo filas válidas
        """
        if df.empty:
            return df
            
        # Empezar asumiendo que todas las filas son válidas
        valid_rows = pd.Series(True, index=df.index)
        
        # Aplicar cada validador
        for column, validator in column_validators.items():
            if column in df.columns:
                # Crear una máscara para filas con valores válidos en esta columna
                column_valid = df[column].apply(validator)
                valid_rows = valid_rows & column_valid
                
                # Reportar estadísticas
                invalid_count = (~column_valid).sum()
                if invalid_count > 0:
                    logger.warning(f"Encontrados {invalid_count} valores inválidos en columna '{column}'")
        
        # Filtrar DataFrame
        filtered_df = df[valid_rows].copy()
        
        # Reportar resultados
        removed_count = len(df) - len(filtered_df)
        if removed_count > 0:
            logger.info(f"Eliminadas {removed_count} filas con datos inválidos ({removed_count/len(df)*100:.1f}%)")
            
        return filtered_df
    
    def validate_numerical_range(self, df: pd.DataFrame, column: str, 
                             min_value: Optional[float] = None,
                             max_value: Optional[float] = None) -> pd.DataFrame:
        """
        Valida que los valores numéricos en una columna estén dentro de un rango.
        
        Args:
            df: DataFrame a validar
            column: Nombre de la columna
            min_value: Valor mínimo aceptable
            max_value: Valor máximo aceptable
            
        Returns:
            DataFrame con valores válidos en la columna especificada
        """
        if df.empty or column not in df.columns:
            return df
            
        # Convertir a numérico, coercionar errores a NaN
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Aplicar filtros
        mask = ~df[column].isna()
        
        if min_value is not None:
            mask = mask & (df[column] >= min_value)
            
        if max_value is not None:
            mask = mask & (df[column] <= max_value)
            
        # Reportar resultados
        invalid_count = (~mask).sum()
        if invalid_count > 0:
            logger.warning(f"Encontrados {invalid_count} valores fuera de rango [{min_value}, {max_value}] en columna '{column}'")
            
        return df[mask].copy()