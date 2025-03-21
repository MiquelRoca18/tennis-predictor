"""
model_updater.py

Sistema de actualización automática para el modelo de predicción de tenis.
"""

import os
import json
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_ensemble import TennisEnsembleModel
from utils import extract_features
import joblib
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ml-service/logs/model_updater.log'),
        logging.StreamHandler()
    ]
)

class TennisModelUpdater:
    """Clase para gestionar la actualización automática del modelo."""
    
    def __init__(self, config_path: str = 'ml-service/config/model_updater_config.json'):
        """
        Inicializar el actualizador de modelo.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.current_performance = None
        self.last_training = None
        self.is_training = False
        
        # Configurar email para alertas
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'smtp_username': os.getenv('SMTP_USERNAME'),
            'smtp_password': os.getenv('SMTP_PASSWORD'),
            'from_email': os.getenv('ALERT_FROM_EMAIL'),
            'to_email': os.getenv('ALERT_TO_EMAIL')
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Cargar configuración desde archivo JSON."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error al cargar configuración: {str(e)}")
            return {
                'update_frequency': 'daily',
                'performance_threshold': 0.7,
                'min_samples': 1000,
                'retry_attempts': 3,
                'retry_delay': 300  # 5 minutos
            }
    
    def train_model(self) -> bool:
        """
        Entrenar un nuevo modelo con datos recientes.
        
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        if self.is_training:
            logging.warning("Ya hay un entrenamiento en curso")
            return False
        
        try:
            self.is_training = True
            logging.info("Iniciando entrenamiento de nuevo modelo")
            
            # Cargar datos recientes
            recent_data = self._load_recent_data()
            if len(recent_data) < self.config['min_samples']:
                raise ValueError(f"Datos insuficientes: {len(recent_data)} < {self.config['min_samples']}")
            
            # Preparar características
            X, y = self._prepare_features(recent_data)
            
            # Entrenar modelo
            self.model = TennisEnsembleModel()
            self.model.fit(X, y)
            
            # Evaluar rendimiento
            self.current_performance = self._evaluate_model(X, y)
            
            # Guardar modelo
            self._save_model()
            
            self.last_training = datetime.now()
            logging.info("Entrenamiento completado exitosamente")
            return True
            
        except Exception as e:
            logging.error(f"Error en entrenamiento: {str(e)}")
            self._send_alert(f"Error en entrenamiento: {str(e)}")
            return False
            
        finally:
            self.is_training = False
    
    def evaluate_model(self) -> Dict:
        """
        Evaluar el rendimiento del modelo actual.
        
        Returns:
            Dict: Métricas de rendimiento
        """
        try:
            # Cargar datos de prueba recientes
            test_data = self._load_test_data()
            X, y = self._prepare_features(test_data)
            
            # Evaluar modelo
            performance = self._evaluate_model(X, y)
            
            # Verificar umbral de rendimiento
            if performance['accuracy'] < self.config['performance_threshold']:
                self._send_alert(
                    f"Rendimiento del modelo por debajo del umbral: {performance['accuracy']:.3f}"
                )
            
            return performance
            
        except Exception as e:
            logging.error(f"Error en evaluación: {str(e)}")
            return None
    
    def _load_recent_data(self) -> pd.DataFrame:
        """Cargar datos recientes para entrenamiento."""
        # TODO: Implementar carga de datos desde base de datos
        return pd.DataFrame()
    
    def _load_test_data(self) -> pd.DataFrame:
        """Cargar datos de prueba recientes."""
        # TODO: Implementar carga de datos desde base de datos
        return pd.DataFrame()
    
    def _prepare_features(self, data: pd.DataFrame) -> tuple:
        """Preparar características para el modelo."""
        # TODO: Implementar preparación de características
        return np.array([]), np.array([])
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluar rendimiento del modelo."""
        y_pred = self.model.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
    
    def _save_model(self):
        """Guardar modelo entrenado."""
        try:
            model_path = os.getenv('MODEL_PATH', 'ml-service/models/tennis_ensemble_model.joblib')
            joblib.dump(self.model, model_path)
            logging.info(f"Modelo guardado en {model_path}")
        except Exception as e:
            logging.error(f"Error al guardar modelo: {str(e)}")
            raise
    
    def _send_alert(self, message: str):
        """Enviar alerta por email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = "Alerta: Actualizador de Modelo de Tenis"
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['smtp_username'], self.email_config['smtp_password'])
                server.send_message(msg)
            
            logging.info("Alerta enviada exitosamente")
            
        except Exception as e:
            logging.error(f"Error al enviar alerta: {str(e)}")
    
    def start(self):
        """Iniciar el servicio de actualización."""
        try:
            # Programar tareas
            schedule.every().day.at(self.config.get('update_time', '00:00')).do(self.train_model)
            schedule.every().hour.do(self.evaluate_model)
            
            logging.info("Servicio de actualización iniciado")
            
            # Bucle principal
            while True:
                schedule.run_pending()
                time.sleep(60)
                
        except Exception as e:
            logging.error(f"Error en servicio de actualización: {str(e)}")
            self._send_alert(f"Error en servicio de actualización: {str(e)}")
    
    def stop(self):
        """Detener el servicio de actualización."""
        schedule.clear()
        logging.info("Servicio de actualización detenido")

if __name__ == "__main__":
    updater = TennisModelUpdater()
    updater.start() 