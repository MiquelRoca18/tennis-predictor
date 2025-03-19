from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os
import logging
import json
from datetime import datetime
import joblib

# Importar nuestros módulos actualizados
from utils import TennisFeatureEngineering, load_model, preprocess_match_data
import uvicorn

# Configurar logging
logging.basicConfig(
    filename='ml-service/logs/api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

# Crear directorios necesarios
os.makedirs('ml-service/logs', exist_ok=True)
os.makedirs('ml-service/model', exist_ok=True)

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Tennis Match Predictor API",
    description="API avanzada para predicción de partidos de tenis",
    version="2.0.0"
)

# Definir modelos de datos
class MatchData(BaseModel):
    player_1: str
    player_2: str
    ranking_1: Optional[int] = None
    ranking_2: Optional[int] = None
    winrate_1: Optional[float] = None
    winrate_2: Optional[float] = None
    surface: str = "hard"  # Valor por defecto

class PredictionResponse(BaseModel):
    player_1: str
    player_2: str
    predicted_winner: str
    probability: float
    surface: str
    prediction_timestamp: str
    features_used: Optional[List[str]] = None

class AnalysisResponse(BaseModel):
    prediction: Dict[str, Any]
    key_factors: List[str]
    head_to_head: Optional[Dict[str, Any]] = None
    surface_analysis: Optional[Dict[str, Any]] = None

class ModelInfoResponse(BaseModel):
    model_type: str
    features_count: int
    samples_count: int
    accuracy: float
    training_date: str
    model_version: str = "2.0.0"

# Variables globales
model = None
feature_engineering = TennisFeatureEngineering()

@app.on_event("startup")
async def startup_event():
    """Evento que se ejecuta al iniciar la aplicación"""
    global model, feature_engineering
    try:
        model = load_model()
        logging.info("Modelo cargado correctamente")
    except Exception as e:
        logging.warning(f"Error cargando modelo: {e}")
        # Continuar sin modelo, permitiendo entrenamiento posterior
    
    # Preparar feature engineering
    try:
        # Usar rutas relativas al directorio actual
        if os.path.exists("tennis_matches.csv"):
            feature_engineering = TennisFeatureEngineering(data_path="tennis_matches.csv")
            # Calcular estadísticas para predicciones futuras
            feature_engineering.build_player_statistics()
            feature_engineering.build_head_to_head_statistics()
            logging.info("Estadísticas de jugadores calculadas correctamente")
        elif os.path.exists("../tennis_matches.csv"):
            # Intentar buscar en el directorio padre
            feature_engineering = TennisFeatureEngineering(data_path="../tennis_matches.csv")
            feature_engineering.build_player_statistics()
            feature_engineering.build_head_to_head_statistics()
            logging.info("Estadísticas de jugadores calculadas desde directorio padre")
        elif os.path.exists("ml-service/tennis_matches.csv"):
            # Intentar con la ruta absoluta
            feature_engineering = TennisFeatureEngineering(data_path="ml-service/tennis_matches.csv")
            feature_engineering.build_player_statistics()
            feature_engineering.build_head_to_head_statistics()
            logging.info("Estadísticas de jugadores calculadas con ruta absoluta")
        else:
            logging.error("No se encontró el archivo de datos de tenis en ninguna ubicación")
            # Intentar verificar dónde está realmente el archivo
            import subprocess
            try:
                result = subprocess.run(["find", ".", "-name", "tennis_matches.csv"], capture_output=True, text=True)
                found_files = result.stdout.strip()
                if found_files:
                    logging.info(f"Archivos encontrados: {found_files}")
                    # Intentar cargar el primer archivo encontrado
                    first_file = found_files.split('\n')[0]
                    feature_engineering = TennisFeatureEngineering(data_path=first_file)
                    feature_engineering.build_player_statistics()
                    feature_engineering.build_head_to_head_statistics()
                    logging.info(f"Estadísticas de jugadores calculadas usando archivo encontrado: {first_file}")
            except Exception as find_error:
                logging.error(f"Error buscando archivos: {find_error}")
    except Exception as e:
        logging.warning(f"Error inicializando estadísticas: {e}")
        logging.exception("Detalles del error:")

@app.get("/")
def read_root():
    """Endpoint principal"""
    model_exists = os.path.exists('ml-service/model/model.pkl')
    return {
        "message": "Tennis Match Predictor API - Versión 2.0",
        "status": "active",
        "model_loaded": model is not None,
        "model_exists": model_exists,
        "docs_url": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_winner(match_data: MatchData):
    """
    Predice el ganador de un partido de tenis utilizando características avanzadas.
    """
    global model
    
    if model is None:
        try:
            model = load_model()
        except:
            raise HTTPException(status_code=503, detail="Model not trained yet. Please train the model first.")
    
    # Preprocesar datos utilizando el sistema avanzado
    try:
        data_dict = match_data.dict()
        
        # Usar feature engineering avanzado
        processed_data = feature_engineering.transform_match_data(data_dict)
        
        # Hacer predicción
        prediction = model.predict(processed_data)[0]
        probability = max(model.predict_proba(processed_data)[0])
        
        # Determinar ganador
        winner = match_data.player_1 if prediction == 0 else match_data.player_2
        
        # Obtener lista de características usadas
        features_used = list(processed_data.columns)[:5]  # Mostrar las 5 más importantes
        
        response = {
            "player_1": match_data.player_1,
            "player_2": match_data.player_2,
            "predicted_winner": winner,
            "probability": float(probability),
            "surface": match_data.surface,
            "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_used": features_used
        }
        
        # Guardar predicción para análisis futuro
        _save_prediction(match_data.dict(), winner, float(probability))
        
        return response
        
    except Exception as e:
        logging.error(f"Error en predicción: {e}")
        
        # Fallback al método antiguo
        logging.info("Usando método básico para predicción")
        
        data_dict = match_data.dict()
        features = preprocess_match_data(data_dict)
        
        # Convertir a DataFrame para la predicción
        df = pd.DataFrame([features])
        
        # Hacer predicción
        prediction = model.predict(df)[0]
        probability = max(model.predict_proba(df)[0])
        
        # Determinar ganador
        winner = match_data.player_1 if prediction == 0 else match_data.player_2
        
        return {
            "player_1": match_data.player_1,
            "player_2": match_data.player_2,
            "predicted_winner": winner,
            "probability": float(probability),
            "surface": match_data.surface,
            "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_used": list(df.columns)
        }

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_match(match_data: MatchData):
    """
    Analiza un partido en profundidad, mostrando qué factores influyen en la predicción.
    """
    global model, feature_engineering
    
    if model is None:
        try:
            model = load_model()
        except:
            raise HTTPException(status_code=503, detail="Model not trained yet. Please train the model first.")
    
    # Preparar datos para análisis
    data_dict = match_data.dict()
    
    # Asegurar que tenemos estadísticas
    if not feature_engineering.players_stats:
        try:
            feature_engineering.build_player_statistics()
            feature_engineering.build_head_to_head_statistics()
        except Exception as e:
            logging.warning(f"No se pudieron calcular estadísticas: {e}")
    
    # Realizar predicción
    processed_data = feature_engineering.transform_match_data(data_dict)
    prediction = model.predict(processed_data)[0]
    probability = max(model.predict_proba(processed_data)[0])
    
    # Determinar ganador
    winner = match_data.player_1 if prediction == 0 else match_data.player_2
    
    # Analizar factores importantes
    key_factors = []
    
    if hasattr(model, 'feature_importances_'):
        # Obtener importancia de características
        feature_importances = list(zip(processed_data.columns, model.feature_importances_))
        # Ordenar por importancia
        feature_importances.sort(key=lambda x: x[1], reverse=True)
        # Extraer los 5 factores más importantes
        key_factors = [factor[0] for factor in feature_importances[:5]]
    
    # Análisis head-to-head
    player1 = match_data.player_1
    player2 = match_data.player_2
    head_to_head = {}
    
    player_pair = tuple(sorted([player1, player2]))
    if player_pair in feature_engineering.head_to_head_stats:
        h2h = feature_engineering.head_to_head_stats[player_pair]
        head_to_head = {
            "total_matches": h2h.get('total_matches', 0),
            "player1_wins": h2h.get('player1_wins', 0),
            "player2_wins": h2h.get('player2_wins', 0),
            "player1_win_pct": h2h.get('player1_win_pct', 50.0),
            "player2_win_pct": h2h.get('player2_win_pct', 50.0)
        }
    
    # Análisis de superficie
    surface = match_data.surface
    surface_analysis = {
        "surface": surface,
        "player1_surface_stats": feature_engineering.players_stats.get(player1, {}).get('surface_stats', {}).get(surface, {}),
        "player2_surface_stats": feature_engineering.players_stats.get(player2, {}).get('surface_stats', {}).get(surface, {})
    }
    
    return {
        "prediction": {
            "predicted_winner": winner,
            "probability": float(probability)
        },
        "key_factors": key_factors,
        "head_to_head": head_to_head,
        "surface_analysis": surface_analysis
    }

@app.post("/train")
def train_endpoint(background_tasks: BackgroundTasks,
                  data_path: Optional[str] = None,
                  model_type: str = Query("rf", description="Tipo de modelo: 'rf' o 'gb'"),
                  optimize: bool = Query(True, description="Optimizar hiperparámetros")):
    """
    Entrena un nuevo modelo con los datos actualizados.
    El entrenamiento se ejecuta en segundo plano.
    """
    from train import train_model
    
    # Función para ejecutar en segundo plano
    def train_in_background(path, type, opt):
        global model
        try:
            model = train_model(data_path=path, model_type=type, optimize_hyperparams=opt)
            logging.info("Modelo entrenado correctamente en segundo plano")
        except Exception as e:
            logging.error(f"Error entrenando modelo: {e}")
    
    # Iniciar entrenamiento en segundo plano
    background_tasks.add_task(train_in_background, data_path, model_type, optimize)
    
    return {
        "message": "Entrenamiento iniciado en segundo plano",
        "model_type": model_type,
        "optimization": optimize
    }

@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """
    Devuelve información sobre el modelo entrenado actual.
    """
    if not os.path.exists('ml-service/model/model.pkl'):
        raise HTTPException(status_code=404, detail="No model found")
    
    # Intentar cargar metadatos
    try:
        with open('ml-service/model/training_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return {
            "model_type": metadata.get('model_type', 'Unknown'),
            "features_count": metadata.get('features_count', 0),
            "samples_count": metadata.get('samples_count', 0),
            "accuracy": metadata.get('accuracy', 0.0),
            "training_date": metadata.get('timestamp', 'Unknown'),
            "model_version": "2.0.0"
        }
    except:
        # Si no hay metadatos, devolver información básica
        model_time = datetime.fromtimestamp(os.path.getmtime('ml-service/model/model.pkl'))
        
        return {
            "model_type": "Unknown",
            "features_count": 0,
            "samples_count": 0,
            "accuracy": 0.0,
            "training_date": model_time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": "2.0.0"
        }

def _save_prediction(match_data: dict, predicted_winner: str, probability: float):
    """
    Guarda la predicción para análisis futuro.
    """
    try:
        prediction_log_path = 'ml-service/logs/predictions.jsonl'
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "match_data": match_data,
            "predicted_winner": predicted_winner,
            "probability": probability
        }
        
        # Guardar en formato JSONL (una línea por registro)
        with open(prediction_log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        logging.warning(f"No se pudo guardar predicción: {e}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)