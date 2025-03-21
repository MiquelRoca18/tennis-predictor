"""
api.py

API robusta para el servicio de predicción de partidos de tenis usando FastAPI.
"""

import os
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from model_ensemble import TennisEnsembleModel
from utils import extract_features
import joblib
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ml-service/logs/api.log'),
        logging.StreamHandler()
    ]
)

# Inicializar FastAPI
app = FastAPI(
    title="Tennis Match Predictor API",
    description="API para predicción de partidos de tenis usando modelos de machine learning",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar autenticación
API_KEY = os.getenv("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="API Key inválida"
        )
    return api_key

# Cargar modelo
try:
    model = joblib.load('ml-service/models/tennis_ensemble_model.joblib')
    logging.info("Modelo cargado exitosamente")
except Exception as e:
    logging.error(f"Error al cargar el modelo: {str(e)}")
    model = None

# Modelos Pydantic
class MatchInput(BaseModel):
    """Modelo de entrada para predicción de partido."""
    tournament_id: str = Field(..., description="ID del torneo")
    player1_id: str = Field(..., description="ID del primer jugador")
    player2_id: str = Field(..., description="ID del segundo jugador")
    match_date: datetime = Field(..., description="Fecha del partido")
    surface: str = Field(..., description="Superficie del partido")
    
    @validator('surface')
    def validate_surface(cls, v):
        valid_surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
        if v not in valid_surfaces:
            raise ValueError(f"Superficie debe ser una de: {', '.join(valid_surfaces)}")
        return v

class PredictionResponse(BaseModel):
    """Modelo de respuesta para predicción."""
    winner_id: str = Field(..., description="ID del jugador predicho como ganador")
    probability: float = Field(..., description="Probabilidad de victoria")
    confidence_level: str = Field(..., description="Nivel de confianza de la predicción")
    prediction_time: datetime = Field(default_factory=datetime.now)

class AnalysisResponse(BaseModel):
    """Modelo de respuesta para análisis detallado."""
    prediction: PredictionResponse
    h2h_stats: Dict = Field(..., description="Estadísticas head-to-head")
    surface_stats: Dict = Field(..., description="Estadísticas por superficie")
    key_factors: List[str] = Field(..., description="Factores clave que influyen en la predicción")

# Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_match(
    match: MatchInput,
    api_key: str = Depends(get_api_key)
):
    """
    Endpoint para predicción básica del ganador de un partido.
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="El modelo no está disponible"
            )
        
        # Extraer características
        features = extract_features(
            match.tournament_id,
            match.player1_id,
            match.player2_id,
            match.match_date,
            match.surface
        )
        
        if features is None:
            raise HTTPException(
                status_code=400,
                detail="No se pudieron extraer características suficientes"
            )
        
        # Realizar predicción
        probability = model.predict_proba([features])[0]
        winner_id = match.player1_id if probability[1] > 0.5 else match.player2_id
        
        # Determinar nivel de confianza
        confidence = abs(probability[1] - 0.5)
        if confidence > 0.3:
            confidence_level = "Alta"
        elif confidence > 0.15:
            confidence_level = "Media"
        else:
            confidence_level = "Baja"
        
        return PredictionResponse(
            winner_id=winner_id,
            probability=float(max(probability)),
            confidence_level=confidence_level
        )
        
    except Exception as e:
        logging.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {str(e)}"
        )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_match(
    match: MatchInput,
    api_key: str = Depends(get_api_key)
):
    """
    Endpoint para análisis detallado de un partido.
    """
    try:
        # Obtener predicción básica
        prediction = await predict_match(match, api_key)
        
        # Obtener estadísticas head-to-head
        h2h_stats = get_h2h_stats(match.player1_id, match.player2_id)
        
        # Obtener estadísticas por superficie
        surface_stats = get_surface_stats(
            match.player1_id,
            match.player2_id,
            match.surface
        )
        
        # Identificar factores clave
        key_factors = identify_key_factors(
            h2h_stats,
            surface_stats,
            prediction.probability
        )
        
        return AnalysisResponse(
            prediction=prediction,
            h2h_stats=h2h_stats,
            surface_stats=surface_stats,
            key_factors=key_factors
        )
        
    except Exception as e:
        logging.error(f"Error en análisis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en el análisis: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info(api_key: str = Depends(get_api_key)):
    """
    Endpoint para obtener información sobre el modelo actual.
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="El modelo no está disponible"
            )
        
        return {
            "model_type": "Ensemble Model",
            "last_training": "2024-03-20",  # TODO: Implementar tracking
            "features": model.feature_names,
            "performance_metrics": {
                "accuracy": 0.75,  # TODO: Implementar tracking
                "precision": 0.78,
                "recall": 0.72,
                "f1_score": 0.75
            }
        }
        
    except Exception as e:
        logging.error(f"Error al obtener información del modelo: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener información del modelo: {str(e)}"
        )

@app.post("/train")
async def train_model(api_key: str = Depends(get_api_key)):
    """
    Endpoint para iniciar el entrenamiento de un nuevo modelo.
    """
    try:
        # TODO: Implementar entrenamiento en segundo plano
        return {
            "status": "success",
            "message": "Entrenamiento iniciado",
            "job_id": "12345"  # TODO: Implementar tracking de jobs
        }
        
    except Exception as e:
        logging.error(f"Error al iniciar entrenamiento: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al iniciar entrenamiento: {str(e)}"
        )

# Funciones auxiliares
def get_h2h_stats(player1_id: str, player2_id: str) -> Dict:
    """Obtiene estadísticas head-to-head entre dos jugadores."""
    # TODO: Implementar consulta a base de datos
    return {
        "total_matches": 10,
        "player1_wins": 6,
        "player2_wins": 4,
        "last_meeting": "2024-02-15",
        "last_winner": player1_id
    }

def get_surface_stats(player1_id: str, player2_id: str, surface: str) -> Dict:
    """Obtiene estadísticas por superficie para ambos jugadores."""
    # TODO: Implementar consulta a base de datos
    return {
        "player1": {
            "matches": 50,
            "wins": 35,
            "win_rate": 0.7
        },
        "player2": {
            "matches": 45,
            "wins": 30,
            "win_rate": 0.67
        }
    }

def identify_key_factors(
    h2h_stats: Dict,
    surface_stats: Dict,
    probability: float
) -> List[str]:
    """Identifica factores clave que influyen en la predicción."""
    factors = []
    
    # Analizar head-to-head
    if h2h_stats["total_matches"] > 5:
        if h2h_stats["player1_wins"] > h2h_stats["player2_wins"]:
            factors.append("Ventaja histórica en enfrentamientos directos")
    
    # Analizar estadísticas por superficie
    if surface_stats["player1"]["win_rate"] > surface_stats["player2"]["win_rate"]:
        factors.append("Mejor rendimiento en esta superficie")
    
    # Analizar probabilidad
    if probability > 0.8:
        factors.append("Alta confianza en la predicción")
    elif probability < 0.6:
        factors.append("Predicción con incertidumbre significativa")
    
    return factors

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 