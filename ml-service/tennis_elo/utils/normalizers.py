"""
utils/normalizers.py

Funciones para normalizar diferentes tipos de datos del tenis:
- Superficies de juego
- Niveles de torneo
- Nombres de rondas
- Formatos de fechas

Proporciona consistencia en los datos independientemente de la fuente.
"""

import logging
import pandas as pd
from typing import Dict, Optional, Union, Any

# Configurar logging
logger = logging.getLogger(__name__)

class TennisDataNormalizer:
    """
    Clase para normalizar datos de tenis.
    Contiene métodos para estandarizar superficies, niveles de torneo,
    nombres de rondas y otros datos para asegurar consistencia.
    """
    
    def __init__(self):
        """Inicializa el normalizador con los mapeos predeterminados."""
        # Mapeo de superficies para normalización
        self.surface_mapping = {
            'hard': 'hard',
            'clay': 'clay', 
            'grass': 'grass',
            'carpet': 'carpet',
            'h': 'hard',
            'c': 'clay',
            'g': 'grass',
            'cr': 'carpet'
        }
        
        # Mapeo para nombres de columnas de nivel de torneo
        self.tourney_level_mapping = {
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
        
        # Mapeo para rondas de torneo
        self.round_mapping = {
            'F': 'F',           # Final
            'SF': 'SF',         # Semifinal
            'QF': 'QF',         # Cuartos de final
            'R16': 'R16',       # Octavos de final
            'R32': 'R32',       # Ronda de 32
            'R64': 'R64',       # Ronda de 64
            'R128': 'R128',     # Ronda de 128
            'RR': 'RR',         # Round Robin
            'final': 'F',
            'semifinals': 'SF',
            'semi-finals': 'SF',
            'quarterfinals': 'QF',
            'quarter-finals': 'QF',
            'fourth round': 'R16',
            'third round': 'R32',
            'second round': 'R64',
            'first round': 'R128',
            'round robin': 'RR'
        }
        
        # Rangos numéricos para rondas
        self.round_rank = {
            'F': 7,       # Final
            'SF': 6,      # Semifinal
            'QF': 5,      # Cuartos de final
            'R16': 4,     # Octavos
            'R32': 3,     # 1/16
            'R64': 2,     # 1/32
            'R128': 1,    # 1/64
            'RR': 4       # Round Robin (similar a octavos)
        }

    def normalize_surface(self, surface: Any) -> str:
        """
        Normaliza el nombre de la superficie a un formato estándar.
        
        Args:
            surface: Nombre de la superficie (puede ser cualquier formato)
                
        Returns:
            Nombre normalizado de la superficie
        """
        # Validar tipo
        if not isinstance(surface, str):
            surface = str(surface) if not pd.isna(surface) else ''
        
        if pd.isna(surface) or not surface or surface == 'unknown':
            return 'hard'  # Usar hard como valor predeterminado
        
        # Normalizar
        surface_lower = str(surface).lower().strip()
        
        # Mapeo directo
        if surface_lower in self.surface_mapping:
            return self.surface_mapping[surface_lower]
        
        # Mapeo por contenido
        try:
            # Checking for common terms
            if 'hard' in surface_lower or 'h' == surface_lower:
                return 'hard'
            elif any(term in surface_lower for term in ['clay', 'arcilla', 'terre', 'c']):
                return 'clay'
            elif any(term in surface_lower for term in ['grass', 'hierba', 'g']):
                return 'grass'
            elif any(term in surface_lower for term in ['carpet', 'indoor', 'cr']):
                return 'carpet'
        except Exception as e:
            logger.debug(f"Error en mapeo por contenido: {str(e)}")
        
        # Si no hay match, retornar hard como valor por defecto
        return 'hard'
    
    def normalize_tournament_level(self, level: Any) -> str:
        """
        Normaliza el nivel del torneo a un código estándar.
        
        Args:
            level: Nivel del torneo (puede ser cualquier formato)
            
        Returns:
            Código normalizado del nivel del torneo
        """
        # Validar tipo
        if not isinstance(level, str):
            level = str(level) if not pd.isna(level) else ''
        
        if pd.isna(level) or not level:
            return 'O'  # Otros como valor predeterminado
        
        # Trim y normalización
        level = level.strip()
        
        # Primero verificar si es un código normalizado directo (G, M, etc.)
        if level in self.tourney_level_mapping:
            return self.tourney_level_mapping[level]
        
        # Si es un nombre más largo, intentar con el mapeo
        try:
            # Hacer case-insensitive para mayor robustez
            level_upper = level.upper()
            
            # Verificar mapeos conocidos
            if 'GRAND' in level_upper or 'SLAM' in level_upper:
                return 'G'
            elif 'MASTER' in level_upper:
                return 'M'
            elif 'ATP500' in level_upper or 'ATP 500' in level_upper:
                return 'A'
            elif 'ATP250' in level_upper or 'ATP 250' in level_upper:
                return 'D'
            elif 'FINAL' in level_upper and ('TOUR' in level_upper or 'ATP' in level_upper):
                return 'F'
            elif 'CHALL' in level_upper:
                return 'C'
            elif any(term in level_upper for term in ['FUTURE', 'ITF']):
                return 'S'
            
            # Si no hay match exacto, buscar en los valores del mapeo
            for key, val in self.tourney_level_mapping.items():
                if key.upper() in level_upper:
                    return val
        except Exception as e:
            logger.debug(f"Error normalizando nivel de torneo: {str(e)}")
        
        # Si no hay match, retornar valor por defecto
        return 'O'  # Otros
    
    def normalize_round(self, round_name: Any) -> str:
        """
        Normaliza el nombre de la ronda a un formato estándar.
        
        Args:
            round_name: Nombre de la ronda (puede ser cualquier formato)
            
        Returns:
            Código normalizado de la ronda
        """
        # Validar tipo
        if not isinstance(round_name, str):
            round_name = str(round_name) if not pd.isna(round_name) else ''
        
        if pd.isna(round_name) or not round_name:
            return 'R32'  # Valor predeterminado
        
        # Normalizar
        round_lower = str(round_name).lower().strip()
        
        # Mapeo directo
        if round_name in self.round_mapping:
            return self.round_mapping[round_name]
        
        if round_lower in [key.lower() for key in self.round_mapping.keys()]:
            for key in self.round_mapping:
                if key.lower() == round_lower:
                    return self.round_mapping[key]
        
        # Buscar por términos
        try:
            if 'final' in round_lower and 'semi' not in round_lower and 'quarter' not in round_lower:
                return 'F'
            elif any(term in round_lower for term in ['semi', 'sf']):
                return 'SF'
            elif any(term in round_lower for term in ['quarter', 'qf']):
                return 'QF'
            elif any(term in round_lower for term in ['r16', '16', 'fourth']):
                return 'R16'
            elif any(term in round_lower for term in ['r32', '32', 'third']):
                return 'R32'
            elif any(term in round_lower for term in ['r64', '64', 'second']):
                return 'R64'
            elif any(term in round_lower for term in ['r128', '128', 'first']):
                return 'R128'
            elif 'robin' in round_lower or 'rr' == round_lower:
                return 'RR'
        except Exception as e:
            logger.debug(f"Error en mapeo de ronda por contenido: {str(e)}")
        
        # Si no hay match, retornar valor por defecto
        return 'R32'
    
    def get_round_rank(self, round_name: str) -> int:
        """
        Obtiene un valor numérico para la ronda (útil para ordenar).
        
        Args:
            round_name: Nombre de la ronda
            
        Returns:
            Valor numérico representando la ronda (mayor para rondas finales)
        """
        # Normalizar primero
        normalized_round = self.normalize_round(round_name)
        
        # Devolver rank
        return self.round_rank.get(normalized_round, 0)
    
    def normalize_date_format(self, date_value: Any, output_format: str = '%Y-%m-%d') -> str:
        """
        Normaliza un valor de fecha a un formato estándar.
        
        Args:
            date_value: Valor de fecha en cualquier formato
            output_format: Formato de salida deseado
            
        Returns:
            Fecha normalizada en el formato especificado
        """
        if pd.isna(date_value) or not date_value:
            return ''
        
        try:
            # Si ya es datetime, convertir directamente
            if isinstance(date_value, pd.Timestamp) or hasattr(date_value, 'strftime'):
                return date_value.strftime(output_format)
            
            # Intentar varios formatos comunes
            date_str = str(date_value).strip()
            
            # Formato estándar de bases de datos ATP (YYYYMMDD)
            if len(date_str) == 8 and date_str.isdigit():
                parsed_date = pd.to_datetime(date_str, format='%Y%m%d')
                return parsed_date.strftime(output_format)
            
            # Otros formatos comunes
            try:
                parsed_date = pd.to_datetime(date_str)
                return parsed_date.strftime(output_format)
            except:
                logger.debug(f"No se pudo parsear la fecha: {date_str}")
                return ''
                
        except Exception as e:
            logger.debug(f"Error normalizando fecha {date_value}: {str(e)}")
            return ''
    
    def normalize_column_names(self, df: pd.DataFrame, column_map: Dict[str, list]) -> pd.DataFrame:
        """
        Normaliza los nombres de columnas en un DataFrame según un mapeo.
        
        Args:
            df: DataFrame a normalizar
            column_map: Diccionario {nombre_estándar: [nombres_alternativos]}
            
        Returns:
            DataFrame con nombres de columnas normalizados
        """
        if df.empty:
            return df
            
        # Hacer copia para no modificar el original
        normalized_df = df.copy()
        
        # Aplicar mapeo
        for target, alternatives in column_map.items():
            if target not in normalized_df.columns:
                for alt in alternatives:
                    if alt in normalized_df.columns:
                        normalized_df[target] = normalized_df[alt]
                        logger.debug(f"Renombrando columna '{alt}' a '{target}'")
                        break
        
        return normalized_df