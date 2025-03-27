import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import logging
from datetime import datetime
import os
import argparse
from typing import Dict, List, Tuple, Any

from validate_and_merge_data import TennisDataValidator
from elo_system import TennisEloSystem
from evaluation import TennisEvaluator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TennisEnsembleModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.meta_model = None
        self.data_validator = TennisDataValidator()
        self.evaluator = TennisEvaluator()
        
        # Configurar modelos base
        self.base_models = {
            'random_forest': RandomForestClassifier(random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
            'xgboost': xgb.XGBClassifier(random_state=random_state),
            'svm': SVC(random_state=random_state, probability=True),
            'neural_network': self._create_neural_network()
        }
        
        # Configurar hiperparámetros para cada modelo
        self.hyperparameters = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        # Configurar métricas
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score
        }
    
    def _create_neural_network(self) -> tf.keras.Model:
        """Crea una red neuronal avanzada con normalización por lotes y dropout."""
        input_dim = 50  # Aumentado para incluir todas las características
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara los datos para el entrenamiento.
        
        Args:
            data: DataFrame con datos de partidos
            
        Returns:
            Tupla con (X, y) para entrenamiento
        """
        logger.info("Preparando datos para entrenamiento...")
        
        # Validar y enriquecer datos
        data = self.data_validator.validate_and_merge_data(data)
        
        logger.info(f"Columnas disponibles después del enriquecimiento: {data.columns.tolist()}")
        
        # Seleccionar características
        feature_columns = [
            # Características ELO
            'player1_elo', 'player2_elo', 'player1_surface_elo', 'player2_surface_elo',
            'elo_difference', 'surface_elo_difference',
            
            # Características head-to-head
            'h2h_matches', 'h2h_win_rate',
            'h2h_surface_matches', 'h2h_surface_win_rate',
            
            # Características temporales
            'days_since_last_match', 'player1_id_days_since_last', 'player2_id_days_since_last',
            
            # Características por superficie
            'surface_winrate',
            
            # Características de ranking y estadísticas
            'winner_rank', 'winner_rank_points', 'winner_ace', 'winner_df', 'winner_svpt',
            'winner_1stIn', 'winner_1stWon', 'winner_2ndWon', 'winner_SvGms',
            'winner_bpSaved', 'winner_bpFaced',
            'loser_rank', 'loser_rank_points', 'loser_ace', 'loser_df', 'loser_svpt',
            'loser_1stIn', 'loser_1stWon', 'loser_2ndWon', 'loser_SvGms',
            'loser_bpSaved', 'loser_bpFaced',
            
            # Características del torneo
            'tourney_level', 'draw_size', 'best_of',
            
            # Características del jugador
            'winner_age', 'loser_age',
            'winner_ht', 'loser_ht'
        ]
        
        # Verificar que todas las columnas existen
        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Columnas faltantes: {missing_columns}")
            raise ValueError(f"Faltan las siguientes columnas en los datos: {missing_columns}")
        
        # Convertir variables categóricas
        data = pd.get_dummies(data, columns=['tourney_level', 'surface'])
        
        # Escalar características numéricas
        X = self.scaler.fit_transform(data[feature_columns])
        y = (data['winner_id'] == data['player1_id']).astype(int)
        
        logger.info(f"Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
        return X, y
    
    def train(self, data: pd.DataFrame, 
              train_end_date: str = '2020-12-31',
              val_end_date: str = '2021-12-31'):
        """
        Entrena el modelo ensemble con validación temporal.
        
        Args:
            data: DataFrame con datos de partidos
            train_end_date: Fecha final para datos de entrenamiento
            val_end_date: Fecha final para datos de validación
        """
        logger.info("Iniciando entrenamiento del modelo ensemble...")
        
        # Dividir datos temporalmente
        train_data, val_data, test_data = self.data_validator.temporal_split(
            data, train_end_date, val_end_date)
        
        # Preparar datos
        X_train, y_train = self.prepare_data(train_data)
        X_val, y_val = self.prepare_data(val_data)
        X_test, y_test = self.prepare_data(test_data)
        
        # Entrenar modelos base
        logger.info("Entrenando modelos base...")
        for name, model in self.base_models.items():
            logger.info(f"Entrenando {name}...")
            if name == 'neural_network':
                self._train_neural_network(X_train, y_train, X_val, y_val)
            else:
                self._train_base_model(name, X_train, y_train)
        
        # Generar características meta
        logger.info("Generando características meta...")
        meta_features_train = self._generate_meta_features(X_train)
        meta_features_val = self._generate_meta_features(X_val)
        
        # Entrenar meta-modelo
        logger.info("Entrenando meta-modelo...")
        self.meta_model = LogisticRegression(random_state=self.random_state)
        self.meta_model.fit(meta_features_train, y_train)
        
        # Evaluar modelo
        logger.info("Evaluando modelo...")
        self._evaluate_model(X_test, y_test, test_data)
        
        logger.info("Entrenamiento completado")
    
    def _train_base_model(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """Entrena un modelo base con búsqueda de hiperparámetros."""
        model = self.base_models[model_name]
        grid_search = GridSearchCV(
            model,
            self.hyperparameters[model_name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        self.models[model_name] = grid_search.best_estimator_
        
        logger.info(f"Mejores parámetros para {model_name}:")
        for param, value in grid_search.best_params_.items():
            logger.info(f"  {param}: {value}")
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray):
        """Entrena la red neuronal con early stopping y reducción de learning rate."""
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        history = self.base_models['neural_network'].fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['neural_network'] = self.base_models['neural_network']
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Genera características meta usando predicciones de modelos base."""
        meta_features = []
        
        for name, model in self.models.items():
            if name == 'neural_network':
                pred = model.predict(X, verbose=0)
            else:
                pred = model.predict_proba(X)[:, 1]
            
            meta_features.append(pred.reshape(-1, 1))
        
        return np.hstack(meta_features)
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, test_data: pd.DataFrame):
        """Evalúa el modelo usando todas las métricas implementadas."""
        # Generar predicciones
        meta_features_test = self._generate_meta_features(X_test)
        y_pred = self.meta_model.predict(meta_features_test)
        y_pred_proba = self.meta_model.predict_proba(meta_features_test)[:, 1]
        
        # Evaluar modelo
        results = self.evaluator.evaluate_model(y_test, y_pred, test_data)
        
        # Evaluar por diferentes categorías
        surface_results = self.evaluator.evaluate_by_surface(y_test, y_pred, test_data)
        ranking_results = self.evaluator.evaluate_by_ranking_difference(y_test, y_pred, test_data)
        tournament_results = self.evaluator.evaluate_by_tournament_type(y_test, y_pred, test_data)
        form_results = self.evaluator.evaluate_by_recent_form(y_test, y_pred, test_data)
        h2h_results = self.evaluator.evaluate_by_h2h_history(y_test, y_pred, test_data)
        
        # Imprimir resultados
        self.evaluator.print_evaluation_results(results)
        
        # Guardar resultados
        self._save_evaluation_results({
            'general': results,
            'surface': surface_results,
            'ranking': ranking_results,
            'tournament': tournament_results,
            'form': form_results,
            'h2h': h2h_results
        })
    
    def _save_evaluation_results(self, results: Dict[str, Dict[str, float]]):
        """Guarda los resultados de la evaluación."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'evaluation_results'
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
        
        # Convertir resultados a formato serializable
        serializable_results = {}
        for category, metrics in results.items():
            serializable_results[category] = {
                k: float(v) if isinstance(v, np.float64) else v
                for k, v in metrics.items()
            }
        
        # Guardar resultados
        import json
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Resultados guardados en {output_file}")
    
    def save_model(self, output_dir: str = 'models'):
        """Guarda el modelo entrenado."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar modelos base
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(os.path.join(output_dir, f'{name}_{timestamp}.h5'))
            else:
                joblib.dump(model, os.path.join(output_dir, f'{name}_{timestamp}.joblib'))
        
        # Guardar meta-modelo
        joblib.dump(self.meta_model, os.path.join(output_dir, f'meta_model_{timestamp}.joblib'))
        
        # Guardar scaler
        joblib.dump(self.scaler, os.path.join(output_dir, f'scaler_{timestamp}.joblib'))
        
        logger.info(f"Modelo guardado en {output_dir}")

def main():
    """Función principal para entrenar el modelo."""
    parser = argparse.ArgumentParser(description='Entrena el modelo de predicción de tenis')
    parser.add_argument('--data', type=str, required=True, help='Ruta al archivo de datos')
    parser.add_argument('--train_end', type=str, default='2020-12-31', help='Fecha final para entrenamiento')
    parser.add_argument('--val_end', type=str, default='2021-12-31', help='Fecha final para validación')
    parser.add_argument('--output', type=str, default='models', help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Cargar datos
    logger.info(f"Cargando datos desde {args.data}")
    data = pd.read_csv(args.data)
    
    # Crear y entrenar modelo
    model = TennisEnsembleModel()
    model.train(data, args.train_end, args.val_end)
    
    # Guardar modelo
    model.save_model(args.output)

if __name__ == '__main__':
    main() 