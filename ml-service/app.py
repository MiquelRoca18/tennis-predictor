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
import traceback
import asyncio

# Importar nuestros módulos actualizados
from utils import TennisFeatureEngineering, load_model, preprocess_match_data, PlayerStatsManager
import uvicorn

# Configurar logging
os.makedirs('ml-service/logs', exist_ok=True)
logging.basicConfig(
    filename='ml-service/logs/api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

# Crear directorios necesarios
os.makedirs('ml-service/logs', exist_ok=True)
os.makedirs('ml-service/model', exist_ok=True)
os.makedirs('data', exist_ok=True)

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
stats_manager = None

# Intentar cargar modelo directamente al importar el módulo
try:
    model = load_model()
    print("Modelo cargado correctamente durante la importación")
except Exception as e:
    print(f"No se pudo cargar el modelo durante la importación: {e}")
    # No es crítico, lo intentamos de nuevo bajo demanda

# Usar lifespan para evitar el bloqueo en startup
@app.on_event("startup")
async def startup_event():
    """Evento que se ejecuta al iniciar la aplicación - versión no bloqueante"""
    print("Iniciando startup_event...")
    # No usamos global aquí para evitar bloqueos, lo haremos bajo demanda
    # Simplemente registramos un mensaje para indicar que se inició
    logging.info("Servicio de API iniciado - los recursos se cargarán bajo demanda")
    print("Fin de startup_event - API lista para responder")

@app.get("/")
def read_root():
    """Endpoint principal"""
    model_exists = os.path.exists('model/model.pkl')
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
    Predice el ganador de un partido de tenis utilizando estadísticas históricas.
    """
    global model, stats_manager
    
    # Intentar cargar el modelo solo si es necesario
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            # Fallback a una predicción simple si no podemos cargar el modelo
            logging.warning(f"No se pudo cargar el modelo: {e}")
            # El jugador con mejor ranking (número más bajo) gana
            if match_data.ranking_1 is not None and match_data.ranking_2 is not None:
                if match_data.ranking_1 < match_data.ranking_2:
                    predicted_winner = match_data.player_1
                    probability = 0.75
                else:
                    predicted_winner = match_data.player_2
                    probability = 0.75
            else:
                # Sin rankings, simplemente elige al primer jugador
                predicted_winner = match_data.player_1
                probability = 0.5
            
            return {
                "player_1": match_data.player_1,
                "player_2": match_data.player_2,
                "predicted_winner": predicted_winner,
                "probability": probability,
                "surface": match_data.surface,
                "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "features_used": ["ranking_1", "ranking_2"]
            }
    
    # Intentar inicializar el gestor de estadísticas si no existe
    if stats_manager is None:
        try:
            csv_path = 'data/tennis_matches.csv'
            if os.path.exists(csv_path):
                stats_manager = PlayerStatsManager(data_path=csv_path)
                logging.info("Gestor de estadísticas inicializado correctamente")
            else:
                logging.warning(f"El archivo {csv_path} no existe. Usando estadísticas predeterminadas.")
                # Continuamos sin estadísticas, usaremos valores default
        except Exception as e:
            logging.error(f"Error inicializando gestor de estadísticas: {e}")
            # Continuamos sin estadísticas, el código debe ser robusto a esto
    
    try:
        player1 = match_data.player_1
        player2 = match_data.player_2
        surface = match_data.surface.lower() if match_data.surface else "hard"
        
        # Características básicas si no hay estadísticas
        df = None
        if stats_manager:
            # Obtener características basadas en estadísticas históricas
            df = stats_manager.prepare_prediction_features(player1, player2, surface)
        else:
            # Crear DataFrame básico basado solo en los datos proporcionados
            df = pd.DataFrame([{
                'ranking_1': match_data.ranking_1 or 100,
                'ranking_2': match_data.ranking_2 or 100,
                'winrate_1': match_data.winrate_1 or 50.0,
                'winrate_2': match_data.winrate_2 or 50.0,
                'ranking_diff': (match_data.ranking_1 or 100) - (match_data.ranking_2 or 100),
                'winrate_diff': (match_data.winrate_1 or 50.0) - (match_data.winrate_2 or 50.0),
                'surface_code': {'hard': 0, 'clay': 1, 'grass': 2}.get(surface, 0)
            }])
        
        # Obtener las características exactas que el modelo espera
        expected_features = []
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_.tolist()
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_names_in_'):
            expected_features = model.steps[-1][1].feature_names_in_.tolist()
        
        # Si el modelo espera características específicas, ajustar DataFrame
        if expected_features:
            # Crear DataFrame con solo las características esperadas en el orden correcto
            feature_values = []
            for feature in expected_features:
                if feature in df.columns:
                    feature_values.append(df[feature].values[0])
                else:
                    # Si el modelo espera una característica que no tenemos, usar un valor predeterminado
                    feature_values.append(0.0)
            
            df = pd.DataFrame([feature_values], columns=expected_features)
        
        # Hacer predicción
        prediction = model.predict(df)[0]
        probability = max(model.predict_proba(df)[0])
        
        # Determinar ganador
        winner = player1 if prediction == 0 else player2
        
        # Obtener estadísticas reales para mostrar si están disponibles
        p1_stats = {'winrate': 50.0, 'avg_ranking': 100.0}
        p2_stats = {'winrate': 50.0, 'avg_ranking': 100.0}
        p1_surface_stats = {'winrate': 50.0}
        p2_surface_stats = {'winrate': 50.0}
        
        if stats_manager:
            p1_stats = stats_manager.get_player_stats(player1)
            p2_stats = stats_manager.get_player_stats(player2)
            p1_surface_stats = stats_manager.get_surface_stats(player1, surface)
            p2_surface_stats = stats_manager.get_surface_stats(player2, surface)
        
        # Preparar respuesta detallada
        response = {
            "player_1": player1,
            "player_2": player2,
            "predicted_winner": winner,
            "probability": float(probability),
            "surface": surface,
            "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_used": list(df.columns)[:5],  # Mostrar las 5 primeras características
            "player_stats": {
                "player_1": {
                    "ranking": p1_stats.get("avg_ranking", 100.0),
                    "winrate": p1_stats.get("winrate", 50.0),
                    "surface_winrate": p1_surface_stats.get("winrate", 50.0)
                },
                "player_2": {
                    "ranking": p2_stats.get("avg_ranking", 100.0),
                    "winrate": p2_stats.get("winrate", 50.0),
                    "surface_winrate": p2_surface_stats.get("winrate", 50.0)
                }
            }
        }
        
        return response
        
    except Exception as e:
        logging.error(f"Error en predicción: {e}")
        import traceback
        traceback.print_exc()
        # Retornar una respuesta simplificada en caso de error
        return {
            "player_1": match_data.player_1,
            "player_2": match_data.player_2,
            "predicted_winner": match_data.player_1,  # Usar player_1 por defecto
            "probability": 0.5,
            "surface": match_data.surface,
            "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_used": ["fallback_prediction"],
            "error": str(e)
        }

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_match(match_data: MatchData):
    """
    Analiza un partido en profundidad usando datos históricos reales.
    """
    global model, stats_manager
    
    # Intentar cargar el modelo solo si es necesario
    if model is None:
        try:
            model = load_model()
        except:
            # Si no podemos cargar el modelo, devolvemos un análisis simplificado
            return {
                "prediction": {
                    "predicted_winner": match_data.player_1,
                    "probability": 0.5
                },
                "key_factors": ["ranking", "historial", "superficie"],
                "head_to_head": {
                    "total_matches": 0,
                    "player1_wins": 0,
                    "player2_wins": 0,
                    "player1_win_pct": 50.0,
                    "player2_win_pct": 50.0
                },
                "surface_analysis": {
                    "surface": match_data.surface,
                    "explanation": ["Análisis no disponible - modelo no cargado"]
                }
            }
    
    # Intentar inicializar el gestor de estadísticas si no existe
    if stats_manager is None:
        try:
            csv_path = 'data/tennis_matches.csv'
            if os.path.exists(csv_path):
                stats_manager = PlayerStatsManager(data_path=csv_path)
                logging.info("Gestor de estadísticas inicializado correctamente")
            else:
                logging.warning(f"El archivo {csv_path} no existe. Usando estadísticas predeterminadas.")
        except Exception as e:
            logging.error(f"Error inicializando gestor de estadísticas: {e}")
    
    try:
        player1 = match_data.player_1
        player2 = match_data.player_2
        surface = match_data.surface.lower() if match_data.surface else "hard"
        
        # Características básicas si no hay estadísticas
        df = None
        if stats_manager:
            # Obtener características basadas en estadísticas históricas
            df = stats_manager.prepare_prediction_features(player1, player2, surface)
        else:
            # Crear DataFrame básico
            df = pd.DataFrame([{
                'ranking_1': match_data.ranking_1 or 100,
                'ranking_2': match_data.ranking_2 or 100,
                'winrate_1': match_data.winrate_1 or 50.0,
                'winrate_2': match_data.winrate_2 or 50.0,
                'ranking_diff': (match_data.ranking_1 or 100) - (match_data.ranking_2 or 100),
                'winrate_diff': (match_data.winrate_1 or 50.0) - (match_data.winrate_2 or 50.0),
                'surface_code': {'hard': 0, 'clay': 1, 'grass': 2}.get(surface, 0)
            }])
        
        # Obtener las características exactas que el modelo espera
        expected_features = []
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_.tolist()
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_names_in_'):
            expected_features = model.steps[-1][1].feature_names_in_.tolist()
        
        # Si el modelo espera características específicas, ajustar DataFrame
        if expected_features:
            # Crear DataFrame con solo las características esperadas
            feature_values = []
            for feature in expected_features:
                if feature in df.columns:
                    feature_values.append(df[feature].values[0])
                else:
                    feature_values.append(0.0)
            
            df = pd.DataFrame([feature_values], columns=expected_features)
        
        # Hacer predicción
        prediction = model.predict(df)[0]
        probability = max(model.predict_proba(df)[0])
        
        # Determinar ganador
        winner = player1 if prediction == 0 else player2
        
        # Obtener estadísticas por defecto si no hay stats_manager
        p1_stats = {'winrate': 50.0, 'avg_ranking': 100.0}
        p2_stats = {'winrate': 50.0, 'avg_ranking': 100.0}
        p1_surface_stats = {'winrate': 50.0, 'matches': 0, 'wins': 0}
        p2_surface_stats = {'winrate': 50.0, 'matches': 0, 'wins': 0}
        h2h_stats = {
            'total_matches': 0,
            'player1_wins': 0,
            'player2_wins': 0,
            'player1_win_pct': 50.0,
            'player2_win_pct': 50.0
        }
        
        # Usar estadísticas reales si están disponibles
        if stats_manager:
            p1_stats = stats_manager.get_player_stats(player1)
            p2_stats = stats_manager.get_player_stats(player2)
            p1_surface_stats = stats_manager.get_surface_stats(player1, surface)
            p2_surface_stats = stats_manager.get_surface_stats(player2, surface)
            h2h_stats = stats_manager.get_head_to_head(player1, player2)
        
        # Analizar factores importantes
        key_factors = []
        
        # Intentar obtener importancia de características si está disponible
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances = [(expected_features[i], importances[i]) for i in range(len(expected_features))]
            feature_importances.sort(key=lambda x: x[1], reverse=True)
            key_factors = [factor[0] for factor in feature_importances[:5]]
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
            importances = model.steps[-1][1].feature_importances_
            feature_importances = [(expected_features[i], importances[i]) for i in range(len(expected_features))]
            feature_importances.sort(key=lambda x: x[1], reverse=True)
            key_factors = [factor[0] for factor in feature_importances[:5]]
        else:
            # Si no hay información de importancia, usar factores predeterminados
            key_factors = ['ranking_diff', 'winrate_diff', 'surface_code']
        
        # Preparar análisis de superficie
        surface_analysis = {
            "surface": surface,
            "player1_surface_stats": {
                "winrate": p1_surface_stats["winrate"],
                "matches": p1_surface_stats.get("matches", 0),
                "wins": p1_surface_stats.get("wins", 0)
            },
            "player2_surface_stats": {
                "winrate": p2_surface_stats["winrate"],
                "matches": p2_surface_stats.get("matches", 0),
                "wins": p2_surface_stats.get("wins", 0)
            }
        }
        
        # Preparar explicaciones basadas en estadísticas
        explanation = []
        for factor in key_factors[:3]:
            if factor == 'ranking_diff' or factor == 'absolute_ranking_diff':
                ranking_1 = p1_stats.get("avg_ranking", match_data.ranking_1 or 100)
                ranking_2 = p2_stats.get("avg_ranking", match_data.ranking_2 or 100)
                if ranking_1 < ranking_2:
                    explanation.append(f"{player1} tiene mejor ranking que {player2} ({ranking_1:.1f} vs {ranking_2:.1f})")
                elif ranking_1 > ranking_2:
                    explanation.append(f"{player2} tiene mejor ranking que {player1} ({ranking_2:.1f} vs {ranking_1:.1f})")
                else:
                    explanation.append(f"{player1} y {player2} tienen rankings similares ({ranking_1:.1f})")
            elif factor == 'winrate_diff':
                winrate_1 = p1_stats.get("winrate", match_data.winrate_1 or 50)
                winrate_2 = p2_stats.get("winrate", match_data.winrate_2 or 50)
                if winrate_1 > winrate_2:
                    explanation.append(f"{player1} tiene mejor porcentaje de victorias que {player2} ({winrate_1:.1f}% vs {winrate_2:.1f}%)")
                elif winrate_1 < winrate_2:
                    explanation.append(f"{player2} tiene mejor porcentaje de victorias que {player1} ({winrate_2:.1f}% vs {winrate_1:.1f}%)")
                else:
                    explanation.append(f"{player1} y {player2} tienen porcentajes de victoria similares ({winrate_1:.1f}%)")
            elif 'surface' in factor:
                if 'surface_winrate_diff' in factor:
                    surface_winrate_1 = p1_surface_stats["winrate"]
                    surface_winrate_2 = p2_surface_stats["winrate"]
                    if surface_winrate_1 > surface_winrate_2:
                        explanation.append(f"{player1} es mejor en superficie {surface} ({surface_winrate_1:.1f}% vs {surface_winrate_2:.1f}%)")
                    elif surface_winrate_1 < surface_winrate_2:
                        explanation.append(f"{player2} es mejor en superficie {surface} ({surface_winrate_2:.1f}% vs {surface_winrate_1:.1f}%)")
                    else:
                        explanation.append(f"Ambos jugadores tienen rendimiento similar en superficie {surface}")
                else:
                    explanation.append(f"El rendimiento en superficie {surface} es un factor importante")
        
        # Añadir explicación a surface_analysis
        surface_analysis["explanation"] = explanation
        
        # Construir respuesta final
        return {
            "prediction": {
                "predicted_winner": winner,
                "probability": float(probability)
            },
            "key_factors": key_factors,
            "head_to_head": h2h_stats,
            "surface_analysis": surface_analysis
        }
        
    except Exception as e:
        logging.error(f"Error en análisis: {e}")
        import traceback
        traceback.print_exc()
        
        # Devolver un análisis simplificado en caso de error
        return {
            "prediction": {
                "predicted_winner": match_data.player_1,
                "probability": 0.5
            },
            "key_factors": ["Error en análisis"],
            "head_to_head": {
                "total_matches": 0,
                "player1_wins": 0,
                "player2_wins": 0
            },
            "surface_analysis": {
                "surface": match_data.surface,
                "explanation": [f"Error durante el análisis: {str(e)}"]
            }
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
    try:
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
    except Exception as e:
        logging.error(f"Error iniciando entrenamiento: {e}")
        return {
            "message": f"Error iniciando entrenamiento: {str(e)}",
            "status": "error"
        }

@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """
    Devuelve información sobre el modelo entrenado actual.
    """
    # Comprobar varias ubicaciones posibles del modelo
    model_paths = [
        'ml-service/model/model.pkl',
        'model/model.pkl',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model.pkl')
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            model_found = True
            model_path = path
            break
    
    if not model_found:
        raise HTTPException(status_code=404, detail="No model found")
    
    # Intentar cargar metadatos
    metadata_paths = [
        os.path.join(os.path.dirname(model_path), 'training_metadata.json'),
        'ml-service/model/training_metadata.json',
        'model/training_metadata.json'
    ]
    
    for metadata_path in metadata_paths:
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
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
            continue
    
    # Si no hay metadatos, devolver información básica
    model_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    
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
    print("Iniciando servicio de predicción de tenis...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)