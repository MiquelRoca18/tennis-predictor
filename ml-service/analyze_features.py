import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import TennisFeatureEngineering
from datetime import datetime

# Crear directorio para análisis
os.makedirs('analysis', exist_ok=True)

# Cargar datos y modelo
data_path = 'tennis_matches_balanced_20250327_165420.csv'
model_path = 'model/ensemble_model_fold_5.pkl'  # Último fold

print(f"Cargando datos desde {data_path}...")
data = pd.read_csv(data_path)
print(f"Cargando modelo desde {model_path}...")
model_dict = joblib.load(model_path)

# Preparar características manualmente para que coincidan con las esperadas por el modelo
print("Preparando características para que coincidan con las esperadas por el modelo...")
X = pd.DataFrame(index=data.index)

# Extraer características básicas
if 'sets_played' in data.columns:
    X['sets_played'] = data['sets_played']
else:
    # Intentar calcular sets_played desde la columna score
    if 'score' in data.columns:
        X['sets_played'] = data['score'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 3)
    else:
        X['sets_played'] = 3  # Valor por defecto

# Otras características básicas
X['minutes'] = data['minutes'] if 'minutes' in data.columns else 120  # Valor por defecto
X['round_numeric'] = data.apply(lambda x: {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7}.get(x['round'], 0) if 'round' in data.columns else 0, axis=1)

# Características temporales
if 'match_date' in data.columns:
    data['match_date'] = pd.to_datetime(data['match_date'])
    X['day_of_week'] = data['match_date'].dt.dayofweek
    X['month'] = data['match_date'].dt.month
    X['year'] = data['match_date'].dt.year
else:
    # Valores por defecto
    X['day_of_week'] = 0
    X['month'] = 1
    X['year'] = 2025

# Características de ELO
X['player1_elo'] = data['elo_winner'] if 'elo_winner' in data.columns else 1500
X['player2_elo'] = data['elo_loser'] if 'elo_loser' in data.columns else 1500
X['player1_surface_elo'] = data['elo_winner_surface'] if 'elo_winner_surface' in data.columns else 1500
X['player2_surface_elo'] = data['elo_loser_surface'] if 'elo_loser_surface' in data.columns else 1500

# Diferencias de ELO
X['elo_difference'] = X['player1_elo'] - X['player2_elo']
X['surface_elo_difference'] = X['player1_surface_elo'] - X['player2_surface_elo']

# Verificar que tenemos todas las características necesarias
expected_features = ['sets_played', 'minutes', 'round_numeric', 'day_of_week', 'month', 'year', 
                     'player1_elo', 'player2_elo', 'player1_surface_elo', 'player2_surface_elo', 
                     'elo_difference', 'surface_elo_difference']

for feature in expected_features:
    if feature not in X.columns:
        print(f"ADVERTENCIA: Falta la característica {feature}. Añadiendo con valores por defecto.")
        X[feature] = 0

# Asegurar que las columnas estén en el orden correcto
X = X[expected_features]

# 1. Extraer importancia de características
feature_importance = None

# Verificar la estructura del modelo
print("\nAnalizando estructura del modelo...")
if isinstance(model_dict, dict):
    print(f"Claves del modelo: {list(model_dict.keys())}")
    
    if 'base_models' in model_dict:
        print(f"Modelos base encontrados: {list(model_dict['base_models'].keys())}")
        
        # Intentar extraer importancia de los modelos base
        importances = []
        for name, base_model in model_dict['base_models'].items():
            print(f"\nAnalizando modelo base: {name}")
            print(f"Tipo de modelo: {type(base_model)}")
            
            if hasattr(base_model, 'feature_importances_'):
                print(f"Extrayendo feature_importances_ de {name}")
                importances.append(base_model.feature_importances_)
            elif hasattr(base_model, 'coef_'):
                print(f"Extrayendo coef_ de {name}")
                if len(base_model.coef_.shape) > 1:
                    importances.append(np.abs(base_model.coef_[0]))
                else:
                    importances.append(np.abs(base_model.coef_))
            else:
                print(f"No se pudo extraer importancia de características de {name}")
        
        if importances:
            # Promediar importancia de características de diferentes modelos
            print("\nCalculando importancia promedio de características...")
            avg_importance = np.mean(importances, axis=0)
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
    
    # Si no se pudo extraer de los modelos base, intentar con el meta-modelo
    if feature_importance is None and 'meta_model' in model_dict:
        print("\nIntentando extraer importancia del meta-modelo...")
        meta_model = model_dict['meta_model']
        print(f"Tipo de meta-modelo: {type(meta_model)}")
        
        if hasattr(meta_model, 'feature_importances_'):
            print("Extrayendo feature_importances_ del meta-modelo")
            meta_importance = meta_model.feature_importances_
            # El meta-modelo trabaja con las salidas de los modelos base
            meta_features = [f"model_{i}" for i in range(len(meta_importance))]
            feature_importance = pd.DataFrame({
                'feature': meta_features,
                'importance': meta_importance
            }).sort_values('importance', ascending=False)
        elif hasattr(meta_model, 'coef_'):
            print("Extrayendo coef_ del meta-modelo")
            if len(meta_model.coef_.shape) > 1:
                meta_importance = np.abs(meta_model.coef_[0])
            else:
                meta_importance = np.abs(meta_model.coef_)
            # El meta-modelo trabaja con las salidas de los modelos base
            meta_features = [f"model_{i}" for i in range(len(meta_importance))]
            feature_importance = pd.DataFrame({
                'feature': meta_features,
                'importance': meta_importance
            }).sort_values('importance', ascending=False)

if feature_importance is not None:
    # Imprimir top 20 características
    print("\nTop 20 características más importantes:")
    print(feature_importance.head(20))
    
    # Visualizar importancia de características
    plt.figure(figsize=(12, 10))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Características Más Importantes')
    plt.tight_layout()
    plt.savefig('analysis/feature_importance.png')
    plt.close()
    
    # Agrupar características por categoría
    feature_categories = {}
    for feature in X.columns:
        # Extraer prefijo como categoría
        parts = feature.split('_')
        if len(parts) > 1:
            category = parts[0]
        else:
            category = 'other'
            
        if category not in feature_categories:
            feature_categories[category] = []
        feature_categories[category].append(feature)
    
    # Calcular importancia por categoría
    category_importance = []
    for category, features in feature_categories.items():
        indices = [list(X.columns).index(f) for f in features if f in X.columns]
        if feature_importance is not None and all(f in feature_importance['feature'].values for f in features):
            category_imp = feature_importance[feature_importance['feature'].isin(features)]['importance'].sum()
            category_importance.append({
                'category': category,
                'importance': category_imp,
                'count': len(features)
            })
    
    if category_importance:
        category_imp_df = pd.DataFrame(category_importance).sort_values('importance', ascending=False)
        
        # Visualizar importancia por categoría
        plt.figure(figsize=(12, 6))
        sns.barplot(data=category_imp_df, x='importance', y='category')
        plt.title('Importancia por Categoría de Características')
        plt.tight_layout()
        plt.savefig('analysis/category_importance.png')
        plt.close()
        
        print("\nImportancia por categoría:")
        print(category_imp_df)
else:
    print("\nNo se pudo extraer la importancia de características para este modelo.")
    
    # Crear gráficos vacíos para mantener la consistencia
    plt.figure(figsize=(12, 10))
    plt.text(0.5, 0.5, "No se pudo extraer la importancia de características", 
             horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.savefig('analysis/feature_importance.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.text(0.5, 0.5, "No se pudo extraer la importancia por categoría", 
             horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.savefig('analysis/category_importance.png')
    plt.close()

print("\nAnálisis guardado en analysis/feature_importance.png y analysis/category_importance.png")