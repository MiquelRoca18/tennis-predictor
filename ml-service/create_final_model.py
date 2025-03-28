import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from utils import TennisFeatureEngineering
import os

# Cargar datos
data_path = 'tennis_matches_balanced_20250327_165420.csv'
data = pd.read_csv(data_path)

# Preparar características
fe = TennisFeatureEngineering()
X = fe.extract_features(data)
y = data['winner']

# Cargar todos los modelos
fold_models = []
fold_scores = []

print("Evaluando modelos individuales...")
for fold in range(1, 6):
    model_path = f'model/ensemble_model_fold_{fold}.pkl'
    model = joblib.load(model_path)
    fold_models.append(model)
    
    # Evaluar modelo
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    fold_scores.append(accuracy)
    print(f"Fold {fold}: Accuracy = {accuracy:.4f}")

# Método 1: Seleccionar el mejor modelo
best_fold = np.argmax(fold_scores)
best_model = fold_models[best_fold]
print(f"\nMejor modelo: Fold {best_fold+1} con accuracy {fold_scores[best_fold]:.4f}")

# Guardar mejor modelo
os.makedirs('model/final', exist_ok=True)
joblib.dump(best_model, 'model/final/best_model.joblib')
print("Mejor modelo guardado como model/final/best_model.joblib")

# Método 2: Crear un modelo combinado de votación
class VotingEnsemble:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        """Realiza una predicción por votación por mayoría."""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x.astype(int))), 
            axis=0, 
            arr=predictions
        )
    
    def predict_proba(self, X):
        """Promedia las probabilidades de todos los modelos."""
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)

# Crear modelo de votación
voting_model = VotingEnsemble(fold_models)
y_voting_pred = voting_model.predict(X)
voting_accuracy = accuracy_score(y, y_voting_pred)
print(f"\nModelo combinado por votación: Accuracy = {voting_accuracy:.4f}")

# Guardar modelo combinado
joblib.dump(voting_model, 'model/final/voting_model.joblib')
print("Modelo combinado guardado como model/final/voting_model.joblib")

# Método 3: Crear un modelo combinado ponderado
class WeightedEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    
    def predict(self, X):
        """Realiza una predicción ponderada."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X):
        """Promedia las probabilidades de todos los modelos con pesos."""
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.average(probas, axis=0, weights=self.weights)

# Normalizar los puntajes para usar como pesos
weights = np.array(fold_scores) / sum(fold_scores)
weighted_model = WeightedEnsemble(fold_models, weights)
y_weighted_pred = weighted_model.predict(X)
weighted_accuracy = accuracy_score(y, y_weighted_pred)
print(f"\nModelo combinado ponderado: Accuracy = {weighted_accuracy:.4f}")

# Guardar modelo ponderado
joblib.dump(weighted_model, 'model/final/weighted_model.joblib')
print("Modelo ponderado guardado como model/final/weighted_model.joblib")

# Mostrar comparación final
print("\nComparación final de modelos:")
print(f"Mejor modelo individual: {fold_scores[best_fold]:.4f}")
print(f"Modelo de votación: {voting_accuracy:.4f}")
print(f"Modelo ponderado: {weighted_accuracy:.4f}")

# Seleccionar y guardar el mejor modelo final
final_model = None
final_model_path = 'model/tennis_ensemble_model.joblib'

if voting_accuracy >= weighted_accuracy and voting_accuracy >= fold_scores[best_fold]:
    final_model = voting_model
    final_name = "Modelo de votación"
elif weighted_accuracy >= voting_accuracy and weighted_accuracy >= fold_scores[best_fold]:
    final_model = weighted_model
    final_name = "Modelo ponderado"
else:
    final_model = best_model
    final_name = f"Modelo individual (fold {best_fold+1})"

joblib.dump(final_model, final_model_path)
print(f"\nModelo final seleccionado: {final_name}")
print(f"Guardado como: {final_model_path}")