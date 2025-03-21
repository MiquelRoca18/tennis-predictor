"""
dashboard.py

Dashboard para visualizar métricas del sistema de predicción de tenis.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ml-service/logs/dashboard.log'),
        logging.StreamHandler()
    ]
)

# Inicializar FastAPI
app = FastAPI(
    title="Tennis Predictor Dashboard",
    description="Dashboard para visualizar métricas del sistema de predicción",
    version="1.0.0"
)

# Configurar templates y archivos estáticos
templates = Jinja2Templates(directory="ml-service/templates")
app.mount("/static", StaticFiles(directory="ml-service/static"), name="static")

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

# Rutas del dashboard
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, api_key: str = Depends(get_api_key)):
    """Página principal del dashboard."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Tennis Predictor Dashboard"
        }
    )

@app.get("/api/metrics/performance")
async def get_performance_metrics(api_key: str = Depends(get_api_key)):
    """Obtener métricas de rendimiento del modelo."""
    try:
        # TODO: Implementar carga de métricas desde base de datos
        return {
            "accuracy": 0.75,
            "precision": 0.78,
            "recall": 0.72,
            "f1_score": 0.75,
            "last_update": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error al obtener métricas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener métricas: {str(e)}"
        )

@app.get("/api/metrics/predictions")
async def get_prediction_metrics(api_key: str = Depends(get_api_key)):
    """Obtener métricas de predicciones."""
    try:
        # TODO: Implementar carga de métricas desde base de datos
        return {
            "total_predictions": 1000,
            "correct_predictions": 750,
            "accuracy_by_surface": {
                "Hard": 0.78,
                "Clay": 0.72,
                "Grass": 0.75
            }
        }
    except Exception as e:
        logging.error(f"Error al obtener métricas de predicciones: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener métricas de predicciones: {str(e)}"
        )

@app.get("/api/metrics/model-updates")
async def get_model_update_metrics(api_key: str = Depends(get_api_key)):
    """Obtener métricas de actualizaciones del modelo."""
    try:
        # TODO: Implementar carga de métricas desde base de datos
        return {
            "last_update": "2024-03-21T00:00:00",
            "update_frequency": "daily",
            "model_version": "1.0.0",
            "training_samples": 10000
        }
    except Exception as e:
        logging.error(f"Error al obtener métricas de actualizaciones: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener métricas de actualizaciones: {str(e)}"
        )

@app.get("/api/charts/performance-trend")
async def get_performance_trend(api_key: str = Depends(get_api_key)):
    """Obtener datos para el gráfico de tendencia de rendimiento."""
    try:
        # TODO: Implementar carga de datos desde base de datos
        dates = pd.date_range(start='2024-01-01', end='2024-03-21', freq='D')
        data = {
            'dates': dates.tolist(),
            'accuracy': [0.7 + i * 0.01 for i in range(len(dates))],
            'precision': [0.68 + i * 0.01 for i in range(len(dates))],
            'recall': [0.65 + i * 0.01 for i in range(len(dates))]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['dates'], y=data['accuracy'], name='Accuracy'))
        fig.add_trace(go.Scatter(x=data['dates'], y=data['precision'], name='Precision'))
        fig.add_trace(go.Scatter(x=data['dates'], y=data['recall'], name='Recall'))
        
        fig.update_layout(
            title='Tendencia de Rendimiento del Modelo',
            xaxis_title='Fecha',
            yaxis_title='Métrica',
            template='plotly_dark'
        )
        
        return json.loads(fig.to_json())
        
    except Exception as e:
        logging.error(f"Error al generar gráfico de tendencia: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar gráfico de tendencia: {str(e)}"
        )

@app.get("/api/charts/surface-performance")
async def get_surface_performance(api_key: str = Depends(get_api_key)):
    """Obtener datos para el gráfico de rendimiento por superficie."""
    try:
        # TODO: Implementar carga de datos desde base de datos
        data = {
            'surface': ['Hard', 'Clay', 'Grass'],
            'accuracy': [0.78, 0.72, 0.75],
            'predictions': [500, 300, 200]
        }
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy por Superficie', 'Predicciones por Superficie'))
        
        fig.add_trace(
            go.Bar(x=data['surface'], y=data['accuracy'], name='Accuracy'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=data['surface'], y=data['predictions'], name='Predicciones'),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            template='plotly_dark'
        )
        
        return json.loads(fig.to_json())
        
    except Exception as e:
        logging.error(f"Error al generar gráfico de superficies: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar gráfico de superficies: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 