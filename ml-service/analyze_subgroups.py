import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score
from utils import TennisFeatureEngineering

# Crear directorio para análisis
os.makedirs('analysis', exist_ok=True)

# Cargar datos y modelo
data_path = 'tennis_matches_balanced_20250327_165420.csv'
model_path = 'model/ensemble_model_fold_5.pkl'  # Último fold

data = pd.read_csv(data_path)
model = joblib.load(model_path)

# Preparar características
fe = TennisFeatureEngineering()
X = fe.extract_features(data)
y = data['winner']

# Obtener predicciones
y_pred = model.predict(X)

# 1. Análisis por superficie
if 'surface' in data.columns:
    surface_results = []
    
    for surface in data['surface'].unique():
        surface_mask = data['surface'] == surface
        if sum(surface_mask) > 0:  # Asegurar que hay datos para esta superficie
            surface_acc = accuracy_score(y[surface_mask], y_pred[surface_mask])
            surface_results.append({
                'surface': surface,
                'accuracy': surface_acc,
                'count': sum(surface_mask)
            })
    
    surface_df = pd.DataFrame(surface_results).sort_values('accuracy', ascending=False)
    print("Rendimiento por superficie:")
    print(surface_df)
    
    # Visualizar rendimiento por superficie
    plt.figure(figsize=(12, 6))
    sns.barplot(data=surface_df, x='surface', y='accuracy')
    plt.title('Precisión por Superficie')
    plt.ylabel('Precisión')
    plt.ylim(0.5, 0.8)  # Ajustar según tus resultados
    for i, row in enumerate(surface_df.itertuples()):
        plt.text(i, row.accuracy + 0.01, f"{row.count}", ha='center')
    plt.savefig('analysis/surface_accuracy.png')
    plt.close()

# 2. Análisis por diferencia de ranking
if 'ranking_1' in data.columns and 'ranking_2' in data.columns:
    data['ranking_diff'] = data['ranking_1'] - data['ranking_2']
    
    rank_bins = [
        (-float('inf'), -100),
        (-100, -50),
        (-50, -10),
        (-10, 0),
        (0, 10),
        (10, 50),
        (50, 100),
        (100, float('inf'))
    ]
    
    rank_results = []
    
    for lower, upper in rank_bins:
        mask = (data['ranking_diff'] > lower) & (data['ranking_diff'] <= upper)
        if sum(mask) > 0:
            rank_acc = accuracy_score(y[mask], y_pred[mask])
            rank_results.append({
                'range': f"{lower} to {upper}",
                'accuracy': rank_acc,
                'count': sum(mask)
            })
    
    rank_df = pd.DataFrame(rank_results)
    print("\nRendimiento por diferencia de ranking:")
    print(rank_df)
    
    # Visualizar rendimiento por ranking
    plt.figure(figsize=(14, 6))
    sns.barplot(data=rank_df, x='range', y='accuracy')
    plt.title('Precisión por Diferencia de Ranking')
    plt.ylabel('Precisión')
    plt.ylim(0.5, 0.8)  # Ajustar según tus resultados
    plt.xticks(rotation=45)
    for i, row in enumerate(rank_df.itertuples()):
        plt.text(i, row.accuracy + 0.01, f"{row.count}", ha='center')
    plt.tight_layout()
    plt.savefig('analysis/ranking_accuracy.png')
    plt.close()

# 3. Análisis por tipo de torneo
if 'tournament_category' in data.columns:
    tournament_results = []
    
    for tournament in data['tournament_category'].unique():
        tournament_mask = data['tournament_category'] == tournament
        if sum(tournament_mask) > 0:
            tournament_acc = accuracy_score(y[tournament_mask], y_pred[tournament_mask])
            tournament_results.append({
                'tournament': tournament,
                'accuracy': tournament_acc,
                'count': sum(tournament_mask)
            })
    
    tournament_df = pd.DataFrame(tournament_results).sort_values('accuracy', ascending=False)
    print("\nRendimiento por tipo de torneo:")
    print(tournament_df)
    
    # Visualizar rendimiento por torneo
    plt.figure(figsize=(14, 6))
    sns.barplot(data=tournament_df, x='tournament', y='accuracy')
    plt.title('Precisión por Tipo de Torneo')
    plt.ylabel('Precisión')
    plt.ylim(0.5, 0.8)  # Ajustar según tus resultados
    plt.xticks(rotation=45)
    for i, row in enumerate(tournament_df.itertuples()):
        plt.text(i, row.accuracy + 0.01, f"{row.count}", ha='center')
    plt.tight_layout()
    plt.savefig('analysis/tournament_accuracy.png')
    plt.close()

print("\nAnálisis por subgrupos guardado en analysis/")