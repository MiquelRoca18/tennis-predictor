#!/usr/bin/env python3
"""
Implementación de modelos ensemble para predicción de tenis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

class TennisEnsembleModel:
    """
    Clase que implementa un modelo ensemble para predicción de partidos de tenis.
    Combina Random Forest, Gradient Boosting, SVM y XGBoost.
    """
    
    def __init__(self, n_splits=5, random_state=42):
        """
        Inicializa el modelo ensemble.
        
        Args:
            n_splits: Número de splits para validación cruzada
            random_state: Semilla aleatoria para reproducibilidad
        """
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Modelos base
        self.base_models = {
            'rf': RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=random_state
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.01,
                random_state=random_state
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=random_state
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                random_state=random_state
            )
        }
        
        # Meta-modelo
        self.meta_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state
        )
        
        # Escalador
        self.scaler = StandardScaler()
        
        # Atributos para almacenar modelos entrenados
        self.trained_base_models = {}
        self.trained_meta_model = None
    
    def _create_meta_features(self, X, y=None, training=True):
        """
        Crea características meta para el modelo ensemble.
        
        Args:
            X: Características de entrada
            y: Etiquetas (opcional)
            training: Si es True, usa validación cruzada para crear meta-características
        """
        try:
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            if training:
                # Usar validación cruzada para crear meta-características
                kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
                
                # Convertir X e y a numpy arrays si son DataFrames/Series
                X_array = X.values if isinstance(X, pd.DataFrame) else X
                y_array = y.values if isinstance(y, pd.Series) else y
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_array, y_array)):
                    X_train_fold = X_array[train_idx]
                    X_val_fold = X_array[val_idx]
                    y_train_fold = y_array[train_idx]
                    y_val_fold = y_array[val_idx]
                    
                    # Entrenar modelos base en el fold de entrenamiento
                    for name, model in self.base_models.items():
                        if name == 'xgb':
                            # XGBoost con early stopping
                            model.set_params(early_stopping_rounds=50)
                            model.fit(
                                X_train_fold, y_train_fold,
                                eval_set=[(X_val_fold, y_val_fold)],
                                verbose=False
                            )
                        else:
                            model.fit(X_train_fold, y_train_fold)
                        
                        proba = model.predict_proba(X_val_fold)
                        # Si solo hay una columna, usar esa directamente
                        if proba.shape[1] == 1:
                            meta_features[val_idx, list(self.base_models.keys()).index(name)] = proba[:, 0]
                        else:
                            meta_features[val_idx, list(self.base_models.keys()).index(name)] = proba[:, 1]
            else:
                # Usar modelos entrenados para predicción
                for i, (name, model) in enumerate(self.trained_base_models.items()):
                    proba = model.predict_proba(X)
                    # Si solo hay una columna, usar esa directamente
                    if proba.shape[1] == 1:
                        meta_features[:, i] = proba[:, 0]
                    else:
                        meta_features[:, i] = proba[:, 1]
            
            return meta_features
        except Exception as e:
            logging.error(f"Error creando meta-características: {str(e)}")
            raise
    
    def fit(self, X, y):
        """
        Entrena el modelo ensemble.
        
        Args:
            X: Características de entrada
            y: Etiquetas
        """
        try:
            # Escalar características
            X_scaled = self.scaler.fit_transform(X)
            
            # Crear meta-características
            meta_features = self._create_meta_features(X_scaled, y, training=True)
            
            # Entrenar meta-modelo con validación cruzada
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            meta_scores = []
            
            # Convertir a numpy arrays si son DataFrames/Series
            X_array = X_scaled
            y_array = y.values if isinstance(y, pd.Series) else y
            
            for train_idx, val_idx in kf.split(X_array, y_array):
                X_meta_train = meta_features[train_idx]
                X_meta_val = meta_features[val_idx]
                y_train = y_array[train_idx]
                y_val = y_array[val_idx]
                
                self.meta_model.fit(X_meta_train, y_train)
                score = self.meta_model.score(X_meta_val, y_val)
                meta_scores.append(score)
            
            # Entrenar meta-modelo final en todos los datos
            self.meta_model.fit(meta_features, y_array)
            
            # Entrenar modelos finales en todos los datos
            for name, model in self.base_models.items():
                if name == 'xgb':
                    # XGBoost con early stopping
                    model.set_params(early_stopping_rounds=50)
                    model.fit(
                        X_array, y_array,
                        eval_set=[(X_array, y_array)],
                        verbose=False
                    )
                else:
                    model.fit(X_array, y_array)
                self.trained_base_models[name] = model
            
            self.trained_meta_model = self.meta_model
            
            logging.info(f"Modelo ensemble entrenado exitosamente. Score promedio de validación: {np.mean(meta_scores):.4f}")
            
        except Exception as e:
            logging.error(f"Error durante el entrenamiento: {str(e)}")
            raise
    
    def predict_proba(self, X):
        """
        Realiza predicciones de probabilidad.
        
        Args:
            X: Características de entrada
            
        Returns:
            Probabilidades de predicción
        """
        try:
            # Escalar características
            X_scaled = self.scaler.transform(X)
            
            # Crear meta-características
            meta_features = self._create_meta_features(X_scaled, training=False)
            
            # Predecir con meta-modelo
            return self.meta_model.predict_proba(meta_features)
            
        except Exception as e:
            logging.error(f"Error durante la predicción de probabilidades: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Realiza predicciones de clase.
        
        Args:
            X: Características de entrada
            
        Returns:
            Predicciones de clase
        """
        try:
            return self.meta_model.predict(self._create_meta_features(self.scaler.transform(X), training=False))
        except Exception as e:
            logging.error(f"Error durante la predicción: {str(e)}")
            raise
    
    def save(self, path):
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        try:
            model_data = {
                'base_models': self.trained_base_models,
                'meta_model': self.trained_meta_model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, path)
            logging.info(f"Modelo guardado en {path}")
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {str(e)}")
            raise
    
    def load(self, path):
        """
        Carga un modelo entrenado.
        
        Args:
            path: Ruta del modelo a cargar
        """
        try:
            model_data = joblib.load(path)
            self.trained_base_models = model_data['base_models']
            self.trained_meta_model = model_data['meta_model']
            self.scaler = model_data['scaler']
            logging.info(f"Modelo cargado desde {path}")
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise

class TennisXGBoostModel:
    """
    Clase que implementa un modelo XGBoost optimizado para predicción de tenis.
    Incluye búsqueda de hiperparámetros, validación temporal y análisis de características.
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa el modelo XGBoost.
        
        Args:
            random_state: Semilla aleatoria para reproducibilidad
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        
        # Reducir el número de combinaciones para pruebas
        self.param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Configurar el modelo base
        self.base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=random_state,
            n_jobs=-1
        )
        
        # Configurar la búsqueda de hiperparámetros
        self.grid_search = GridSearchCV(
            estimator=self.base_model,
            param_grid=self.param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            error_score='raise'  # Para ver el error específico
        )
    
    def fit(self, X, y, optimize_hyperparams=True):
        """
        Entrena el modelo XGBoost.
        
        Args:
            X: Características de entrada
            y: Etiquetas
            optimize_hyperparams: Si es True, realiza búsqueda de hiperparámetros
        """
        try:
            logging.info(f"Forma de X: {X.shape}, Forma de y: {y.shape}")
            logging.info(f"Tipos de datos - X: {X.dtypes}, y: {y.dtype}")
            logging.info(f"Valores únicos en y: {np.unique(y, return_counts=True)}")
            
            # Escalar características
            X_scaled = self.scaler.fit_transform(X)
            
            if optimize_hyperparams:
                logging.info("Iniciando búsqueda de hiperparámetros...")
                self.grid_search.fit(X_scaled, y)
                self.best_params = self.grid_search.best_params_
                logging.info(f"Mejores hiperparámetros encontrados: {self.best_params}")
                self.model = self.grid_search.best_estimator_
            else:
                logging.info("Entrenando modelo con parámetros por defecto...")
                self.model = self.base_model
                self.model.fit(X_scaled, y)
            
            logging.info("Modelo XGBoost entrenado exitosamente")
            
        except Exception as e:
            logging.error(f"Error durante el entrenamiento: {str(e)}")
            logging.error(f"Detalles del error: {type(e).__name__}")
            raise
    
    def predict_proba(self, X):
        """
        Realiza predicciones de probabilidad.
        
        Args:
            X: Características de entrada
            
        Returns:
            Probabilidades de predicción
        """
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        except Exception as e:
            logging.error(f"Error durante la predicción de probabilidades: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Realiza predicciones de clase.
        
        Args:
            X: Características de entrada
            
        Returns:
            Predicciones de clase
        """
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            logging.error(f"Error durante la predicción: {str(e)}")
            raise
    
    def feature_importance(self, plot=True, top_n=20):
        """
        Calcula y visualiza la importancia de características.
        
        Args:
            plot: Si es True, muestra gráfico de importancia
            top_n: Número de características más importantes a mostrar
            
        Returns:
            DataFrame con importancia de características
        """
        try:
            importance = pd.DataFrame({
                'feature': self.model.feature_names_in_,
                'importance': self.model.feature_importances_
            })
            importance = importance.sort_values('importance', ascending=False)
            
            if plot:
                plt.figure(figsize=(12, 6))
                sns.barplot(x='importance', y='feature', data=importance.head(top_n))
                plt.title(f'Top {top_n} Características más Importantes')
                plt.tight_layout()
                plt.show()
            
            return importance
        except Exception as e:
            logging.error(f"Error al calcular importancia de características: {str(e)}")
            raise
    
    def save(self, path):
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'best_params': self.best_params
            }
            joblib.dump(model_data, path)
            logging.info(f"Modelo guardado en {path}")
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {str(e)}")
            raise
    
    def load(self, path):
        """
        Carga un modelo entrenado.
        
        Args:
            path: Ruta del modelo a cargar
        """
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.best_params = model_data['best_params']
            logging.info(f"Modelo cargado desde {path}")
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise

class TennisNeuralNetwork:
    """
    Implementa una red neuronal para predicción de tenis usando TensorFlow/Keras.
    """
    
    def __init__(self, input_dim):
        """
        Inicializa la red neuronal.
        
        Args:
            input_dim: Dimensión de entrada (número de características)
        """
        self.input_dim = input_dim
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self):
        """
        Construye la arquitectura de la red neuronal.
        """
        model = tf.keras.Sequential([
            # Capa de entrada con normalización por lotes
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            # Primera capa oculta
            tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.3),
            
            # Segunda capa oculta
            tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),
            
            # Tercera capa oculta
            tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.1),
            
            # Capa de salida
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Entrena la red neuronal.
        
        Args:
            X: Características de entrada
            y: Etiquetas
            validation_split: Proporción de datos para validación
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
        """
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Callbacks para optimización
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            # Reducción de learning rate
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Entrenar modelo
        history = self.model.fit(
            X_scaled, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_proba(self, X):
        """
        Realiza predicciones de probabilidad.
        
        Args:
            X: Características de entrada
            
        Returns:
            Probabilidades de predicción
        """
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled)
        return np.column_stack((1 - probs, probs))
    
    def predict(self, X):
        """
        Realiza predicciones de clase.
        
        Args:
            X: Características de entrada
            
        Returns:
            Predicciones de clase
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
    def save(self, path):
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        try:
            # Crear directorio si no existe
            os.makedirs(path, exist_ok=True)
            
            # Guardar modelo de Keras con extensión .keras
            self.model.save(os.path.join(path, 'model.keras'))
            
            # Guardar escalador
            joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
            
            logging.info(f"Modelo guardado en {path}")
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {str(e)}")
            raise
    
    def load(self, path):
        """
        Carga un modelo entrenado.
        
        Args:
            path: Ruta del modelo a cargar
        """
        try:
            # Cargar modelo de Keras
            self.model = tf.keras.models.load_model(os.path.join(path, 'model.keras'))
            
            # Cargar escalador
            self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
            
            logging.info(f"Modelo cargado desde {path}")
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise 