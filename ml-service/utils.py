import pandas as pd
import joblib

def load_model():
    """Carga el modelo entrenado desde el archivo"""
    return joblib.load('model/model.pkl')

def preprocess_match_data(match_data):
    """Preprocesa los datos del partido para el modelo"""
    # Convertir superficie a código si existe
    if 'surface' in match_data:
        surfaces = {'hard': 0, 'clay': 1, 'grass': 2}
        match_data['surface_code'] = surfaces.get(match_data['surface'], 0)
    
    # Seleccionar solo las features que el modelo necesita
    features = ["ranking_1", "ranking_2", "winrate_1", "winrate_2"]
    if 'surface_code' in match_data:
        features.append('surface_code')
    
    return {key: match_data[key] for key in features if key in match_data}