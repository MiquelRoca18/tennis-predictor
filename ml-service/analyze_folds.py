import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import TennisFeatureEngineering
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Crear directorio para análisis
os.makedirs('analysis', exist_ok=True)

# Cargar datos
data_path = 'tennis_matches_balanced_20250327_165420.csv'
data = pd.read_csv(data_path)

# Definir métricas a evaluar
metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'roc_auc': roc_auc_score
}

# Preparar características manualmente para que coincidan con las esperadas por el modelo
print("Preparando características para que coincidan con las esperadas por el modelo...")
X_current = pd.DataFrame(index=data.index)

# Extraer características básicas
if 'sets_played' in data.columns:
    X_current['sets_played'] = data['sets_played']
else:
    # Intentar calcular sets_played desde la columna score
    if 'score' in data.columns:
        X_current['sets_played'] = data['score'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 3)
    else:
        X_current['sets_played'] = 3  # Valor por defecto

# Otras características básicas
X_current['minutes'] = data['minutes'] if 'minutes' in data.columns else 120  # Valor por defecto
X_current['round_numeric'] = data.apply(lambda x: {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7}.get(x['round'], 0) if 'round' in data.columns else 0, axis=1)

# Características temporales
if 'match_date' in data.columns:
    data['match_date'] = pd.to_datetime(data['match_date'])
    X_current['day_of_week'] = data['match_date'].dt.dayofweek
    X_current['month'] = data['match_date'].dt.month
    X_current['year'] = data['match_date'].dt.year
else:
    # Valores por defecto
    X_current['day_of_week'] = 0
    X_current['month'] = 1
    X_current['year'] = 2025

# Características de ELO
X_current['player1_elo'] = data['elo_winner'] if 'elo_winner' in data.columns else 1500
X_current['player2_elo'] = data['elo_loser'] if 'elo_loser' in data.columns else 1500
X_current['player1_surface_elo'] = data['elo_winner_surface'] if 'elo_winner_surface' in data.columns else 1500
X_current['player2_surface_elo'] = data['elo_loser_surface'] if 'elo_loser_surface' in data.columns else 1500

# Diferencias de ELO
X_current['elo_difference'] = X_current['player1_elo'] - X_current['player2_elo']
X_current['surface_elo_difference'] = X_current['player1_surface_elo'] - X_current['player2_surface_elo']

# Verificar que tenemos todas las características necesarias
expected_features = ['sets_played', 'minutes', 'round_numeric', 'day_of_week', 'month', 'year', 
                     'player1_elo', 'player2_elo', 'player1_surface_elo', 'player2_surface_elo', 
                     'elo_difference', 'surface_elo_difference']

for feature in expected_features:
    if feature not in X_current.columns:
        print(f"ADVERTENCIA: Falta la característica {feature}. Añadiendo con valores por defecto.")
        X_current[feature] = 0

# Asegurar que las columnas estén en el orden correcto
X_current = X_current[expected_features]

# Obtener la variable objetivo
y = data['winner']

# Clase para adaptar características
class FeatureAdapter:
    def __init__(self, expected_features):
        self.expected_features = expected_features
    
    def adapt(self, X):
        """
        Adapta un DataFrame a las características esperadas.
        - Elimina características que no se esperan
        - Añade características que faltan con valores por defecto (0)
        - Reordena las columnas para que coincidan con lo esperado
        """
        # Crear DataFrame con las características esperadas (con ceros)
        adapted_df = pd.DataFrame(0, index=X.index, columns=self.expected_features)
        
        # Copiar valores de las características que existen
        common_features = list(set(X.columns) & set(self.expected_features))
        adapted_df[common_features] = X[common_features]
        
        # Mapear características renombradas (por ejemplo, elo_diff -> elo_difference)
        name_mapping = {
            # Añadir mapeos de nombres según sea necesario
            'elo_diff': 'elo_difference',
            'elo_surface_diff': 'elo_surface_difference'
            # Añadir más mapeos si es necesario
        }
        
        for new_name, old_name in name_mapping.items():
            if new_name in X.columns and old_name in self.expected_features:
                adapted_df[old_name] = X[new_name]
        
        # Devolver el DataFrame con las características esperadas
        return adapted_df[self.expected_features]

# Clase para manejar predicciones con el modelo stacking
class StackingEnsembleWrapper:
    def __init__(self, model_dict):
        self.base_models = model_dict['base_models']
        self.meta_model = model_dict['meta_model']
        self.scaler = model_dict['scaler'] if 'scaler' in model_dict else None
        
        # Extraer nombres de características esperadas
        if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
            self.feature_names = list(self.scaler.feature_names_in_)
            print(f"Características esperadas por el modelo: {len(self.feature_names)}")
            self.adapter = FeatureAdapter(self.feature_names)
        else:
            self.feature_names = None
            self.adapter = None
            print("No se pudieron extraer nombres de características del modelo")
        
    def predict(self, X):
        X_adapted = self._adapt_features(X)
        meta_features = self._get_meta_features(X_adapted)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        X_adapted = self._adapt_features(X)
        meta_features = self._get_meta_features(X_adapted)
        return self.meta_model.predict_proba(meta_features)
    
    def _adapt_features(self, X):
        """Adapta las características de entrada si es necesario"""
        if self.adapter:
            # Solo imprimir el mensaje la primera vez o cuando se solicite explícitamente
            if not hasattr(self, '_adaptation_reported'):
                print(f"Adaptando características: {X.shape[1]} -> {len(self.feature_names)}")
                self._adaptation_reported = True
            return self.adapter.adapt(X)
        return X
    
    def _get_meta_features(self, X):
        # Escalar los datos si hay un scaler
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Obtener predicciones de cada modelo base
        meta_features = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                # Para clasificadores, usamos probabilidades
                proba = model.predict_proba(X_scaled)
                # Nos quedamos con la probabilidad de la clase positiva
                if proba.shape[1] > 1:  # Si hay más de una columna
                    meta_features.append(proba[:, 1].reshape(-1, 1))
                else:
                    meta_features.append(proba)
            else:
                # Para regresores o modelos sin predict_proba
                preds = model.predict(X_scaled).reshape(-1, 1)
                meta_features.append(preds)
        
        # Concatenar todas las predicciones
        return np.hstack(meta_features)

# Evaluar cada fold
results = []
fold_models = []

for fold in range(1, 6):
    model_path = f'model/ensemble_model_fold_{fold}.pkl'
    print(f"\nProcesando fold {fold}...")
    
    try:
        model_dict = joblib.load(model_path)
        
        print(f"Fold {fold}:")
        print(f"Tipo de modelo: {type(model_dict)}")
        
        if isinstance(model_dict, dict) and 'base_models' in model_dict and 'meta_model' in model_dict:
            print("Modelo stacking encontrado. Creando wrapper...")
            model = StackingEnsembleWrapper(model_dict)
        else:
            print("Estructura de modelo no reconocida.")
            continue
        
        fold_models.append(model)
        
        # Hacer predicciones
        print("Realizando predicciones...")
        y_pred = model.predict(X_current)
        y_proba = model.predict_proba(X_current)[:, 1]
        
        # Calcular métricas
        fold_results = {'fold': fold}
        for name, metric_func in metrics.items():
            if name == 'roc_auc':
                score = metric_func(y, y_proba)
            else:
                score = metric_func(y, y_pred)
            fold_results[name] = score
            print(f"{name}: {score:.4f}")
        
        results.append(fold_results)
    except Exception as e:
        print(f"Error al evaluar el fold {fold}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Verificar que hay resultados antes de continuar
if not results:
    print("No se pudieron obtener resultados de ningún fold. Revisa la estructura de los modelos.")
    exit(1)

# Crear DataFrame para análisis
results_df = pd.DataFrame(results)
print("\nComparación de métricas por fold:")
print(results_df)

# Calcular estadísticas básicas
mean_metrics = results_df.drop('fold', axis=1).mean()
std_metrics = results_df.drop('fold', axis=1).std()
print("\nEstadísticas de las métricas:")
for metric in metrics.keys():
    print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

# Visualizar resultados
plt.figure(figsize=(12, 8))
for metric in metrics.keys():
    plt.plot(results_df['fold'], results_df[metric], marker='o', label=metric)

plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Comparación de métricas por fold')
plt.legend()
plt.grid(True)
plt.savefig('analysis/fold_performance_comparison.png')
plt.close()

# Comparar predicciones entre folds
if len(fold_models) >= 2:
    print("\nCalculando matriz de acuerdo entre modelos...")
    agreement_matrix = np.zeros((len(fold_models), len(fold_models)))
    
    # Calcular todas las predicciones una sola vez
    print("Calculando predicciones para cada modelo...")
    all_predictions = []
    for i, model in enumerate(fold_models):
        print(f"Calculando predicciones para modelo {i+1}...")
        all_predictions.append(model.predict(X_current))
    
    # Calcular matriz de acuerdo usando las predicciones pre-calculadas
    for i in range(len(fold_models)):
        preds_i = all_predictions[i]
        for j in range(len(fold_models)):
            preds_j = all_predictions[j]
            agreement = np.mean(preds_i == preds_j) * 100
            agreement_matrix[i, j] = agreement
    
    # Visualizar matriz de acuerdo
    plt.figure(figsize=(10, 8))
    plt.imshow(agreement_matrix, cmap='Blues', interpolation='none')
    plt.colorbar(label='% de acuerdo en predicciones')
    plt.title('Matriz de acuerdo entre modelos de diferentes folds')
    plt.xlabel('Número de fold')
    plt.ylabel('Número de fold')
    plt.xticks(range(len(fold_models)), range(1, len(fold_models)+1))
    plt.yticks(range(len(fold_models)), range(1, len(fold_models)+1))
    
    # Agregar valores a la matriz
    for i in range(len(fold_models)):
        for j in range(len(fold_models)):
            text = plt.text(j, i, f"{agreement_matrix[i, j]:.1f}%",
                          ha="center", va="center", color="black" if agreement_matrix[i, j] > 90 else "white")
    
    plt.savefig('analysis/fold_agreement_matrix.png')
    plt.close()
    
    print("\nAnálisis guardado en analysis/fold_performance_comparison.png y analysis/fold_agreement_matrix.png")
else:
    print("No hay suficientes modelos para analizar acuerdo entre folds")