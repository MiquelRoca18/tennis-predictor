import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from utils import TennisFeatureEngineering
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear directorio para resultados
os.makedirs('results', exist_ok=True)

# Cargar datos
data_path = 'tennis_matches_balanced_20250327_165420.csv'
model_path = 'model/tennis_ensemble_model.joblib'

print(f"Cargando modelo desde {model_path}...")
model = joblib.load(model_path)

print(f"Cargando datos desde {data_path}...")
data = pd.read_csv(data_path)

# Preparar características
fe = TennisFeatureEngineering()
X = fe.extract_features(data)
y = data['winner']

# Hacer predicciones
print("Realizando predicciones...")
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# Calcular métricas
metrics = {
    'accuracy': accuracy_score(y, y_pred),
    'precision': precision_score(y, y_pred),
    'recall': recall_score(y, y_pred),
    'f1': f1_score(y, y_pred),
    'roc_auc': roc_auc_score(y, y_proba)
}

print("\n=== Métricas del Modelo Final ===")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")

# Matriz de confusión
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión del Modelo Final')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.savefig('results/final_confusion_matrix.png')
plt.close()

# Curva ROC
plt.figure(figsize=(10, 8))
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y, y_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["roc_auc"]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('results/final_roc_curve.png')
plt.close()

# Reporte de clasificación
report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nReporte de Clasificación:")
print(report_df)

# Guardar resultados en un archivo
with open('results/final_model_performance.txt', 'w') as f:
    f.write("=== Métricas del Modelo Final ===\n")
    for metric_name, value in metrics.items():
        f.write(f"{metric_name}: {value:.4f}\n")
    
    f.write("\nReporte de Clasificación:\n")
    f.write(str(report_df))

print("\nResultados guardados en 'results/final_model_performance.txt'")
print("Visualizaciones guardadas en 'results/final_confusion_matrix.png' y 'results/final_roc_curve.png'")