from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from utils import load_model, preprocess_match_data
import uvicorn

app = FastAPI(title="Tennis Match Predictor API")

# Definir el modelo de datos para la predicción
class MatchData(BaseModel):
    player_1: str
    player_2: str
    ranking_1: int
    ranking_2: int
    winrate_1: float
    winrate_2: float
    surface: str = "hard"  # Valor por defecto

class PredictionResponse(BaseModel):
    player_1: str
    player_2: str
    predicted_winner: str
    probability: float

# Cargar el modelo al iniciar la aplicación
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        # Aún así continuar, para permitir el entrenamiento posterior

@app.get("/")
def read_root():
    return {"message": "Tennis Match Predictor API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_winner(match_data: MatchData):
    global model
    
    if model is None:
        try:
            model = load_model()
        except:
            raise HTTPException(status_code=503, detail="Model not trained yet. Please train the model first.")
    
    # Preprocesar datos
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
        "probability": float(probability)
    }

@app.post("/train")
def train_endpoint():
    try:
        from train import train_model
        global model
        model = train_model()
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)