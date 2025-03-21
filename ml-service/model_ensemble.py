#!/usr/bin/env python3
"""
Implementación de modelos ensemble y red neuronal para predicción de tenis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import joblib
import os

class TennisEnsembleModel:
    """
    Clase que implementa un modelo ensemble para predicción de partidos de tenis.
    Combina Random Forest, Gradient Boosting, SVM, XGBoost y una red neuronal.
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
        
        # Red neuronal
        self.neural_net = self._build_neural_net()
        
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
        self.trained_neural_net = None
        self.trained_meta_model = None
        
    def _build_neural_net(self):
        """Construye la arquitectura de la red neuronal."""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(None,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(16, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_meta_features(self, X, y=None, training=True):
        """
        Crea características meta para el modelo ensemble.
        
        Args:
            X: Características de entrada
            y: Etiquetas (opcional)
            training: Si es True, usa validación cruzada para crear meta-características
        """
        meta_features = np.zeros((X.shape[0], len(self.base_models) + 1))  # +1 para red neuronal
        
        if training:
            # Usar validación cruzada para crear meta-características
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y[train_idx]
                
                # Entrenar modelos base en el fold de entrenamiento
                for name, model in self.base_models.items():
                    model.fit(X_train_fold, y_train_fold)
                    meta_features[val_idx, list(self.base_models.keys()).index(name)] = model.predict_proba(X_val_fold)[:, 1]
                
                # Entrenar red neuronal en el fold de entrenamiento
                self.neural_net.fit(
                    X_train_fold, y_train_fold,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                meta_features[val_idx, -1] = self.neural_net.predict(X_val_fold).flatten()
        else:
            # Usar modelos entrenados para predicción
            for i, (name, model) in enumerate(self.trained_base_models.items()):
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            
            meta_features[:, -1] = self.trained_neural_net.predict(X).flatten()
        
        return meta_features
    
    def fit(self, X, y):
        """
        Entrena el modelo ensemble.
        
        Args:
            X: Características de entrada
            y: Etiquetas
        """
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Crear meta-características
        meta_features = self._create_meta_features(X_scaled, y, training=True)
        
        # Entrenar meta-modelo
        self.meta_model.fit(meta_features, y)
        
        # Entrenar modelos finales en todos los datos
        for name, model in self.base_models.items():
            model.fit(X_scaled, y)
            self.trained_base_models[name] = model
        
        # Entrenar red neuronal final
        self.neural_net.fit(
            X_scaled, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        self.trained_neural_net = self.neural_net
    
    def predict_proba(self, X):
        """
        Realiza predicciones de probabilidad.
        
        Args:
            X: Características de entrada
            
        Returns:
            Probabilidades de predicción
        """
        # Escalar características
        X_scaled = self.scaler.transform(X)
        
        # Crear meta-características
        meta_features = self._create_meta_features(X_scaled, training=False)
        
        # Predecir con meta-modelo
        return self.meta_model.predict_proba(meta_features)
    
    def predict(self, X):
        """
        Realiza predicciones de clase.
        
        Args:
            X: Características de entrada
            
        Returns:
            Predicciones de clase
        """
        return self.meta_model.predict(self._create_meta_features(X, training=False))
    
    def save(self, path):
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        model_data = {
            'base_models': self.trained_base_models,
            'neural_net': self.trained_neural_net,
            'meta_model': self.meta_model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
    
    def load(self, path):
        """
        Carga un modelo entrenado.
        
        Args:
            path: Ruta del modelo a cargar
        """
        model_data = joblib.load(path)
        self.trained_base_models = model_data['base_models']
        self.trained_neural_net = model_data['neural_net']
        self.meta_model = model_data['meta_model']
        self.scaler = model_data['scaler']

class TennisXGBoostModel:
    """
    Clase que implementa un modelo XGBoost optimizado para predicción de tenis.
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
        
        # Espacio de búsqueda de hiperparámetros
        self.param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [500, 1000, 1500],
            'scale_pos_weight': [1, 2, 3]
        }
    
    def fit(self, X, y, optimize_hyperparams=True):
        """
        Entrena el modelo XGBoost.
        
        Args:
            X: Características de entrada
            y: Etiquetas
            optimize_hyperparams: Si es True, realiza búsqueda de hiperparámetros
        """
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize_hyperparams:
            # Usar validación temporal
            n_splits = 5
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Búsqueda de hiperparámetros
            grid_search = GridSearchCV(
                xgb.XGBClassifier(random_state=self.random_state),
                self.param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            
            logging.info("Mejores hiperparámetros encontrados:")
            logging.info(grid_search.best_params_)
        else:
            # Entrenar con parámetros por defecto
            self.model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=1000,
                learning_rate=0.05
            )
            self.model.fit(X_scaled, y)
    
    def predict_proba(self, X):
        """
        Realiza predicciones de probabilidad.
        
        Args:
            X: Características de entrada
            
        Returns:
            Probabilidades de predicción
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X):
        """
        Realiza predicciones de clase.
        
        Args:
            X: Características de entrada
            
        Returns:
            Predicciones de clase
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def feature_importance(self, plot=True):
        """
        Calcula y visualiza la importancia de características.
        
        Args:
            plot: Si es True, muestra gráfico de importancia
            
        Returns:
            DataFrame con importancia de características
        """
        importance = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=importance)
            plt.title('Importancia de Características')
            plt.show()
        
        return importance
    
    def save(self, path):
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
    
    def load(self, path):
        """
        Carga un modelo entrenado.
        
        Args:
            path: Ruta del modelo a cargar
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler'] 