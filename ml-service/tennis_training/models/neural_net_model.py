# models/neural_net_model.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from joblib import dump, load

from .base_model import TennisModel

class TennisNeuralNetModel(TennisModel):
    """
    Neural Network model for tennis prediction using TensorFlow/Keras.
    Includes batch normalization, dropout, and L2 regularization for
    improved generalization.
    """
    
    def __init__(self, name="tennis_neural_net", version="1.0.0", random_state=42, 
                 model_params=None):
        """
        Initialize the Neural Network model with configurable architecture.
        
        Args:
            name (str): Name of the model
            version (str): Version string
            random_state (int): Random seed for reproducibility
            model_params (dict): Neural network architecture and training parameters
        """
        super().__init__(name, version)
        self.random_state = random_state
        
        # Set TensorFlow random seed
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Set default params if none provided
        if model_params is None:
            self.model_params = {
                'architecture': {
                    'hidden_layers': [100, 50, 25],
                    'dropout_rates': [0.3, 0.3, 0.3],
                    'activation': 'relu',
                    'output_activation': 'sigmoid',
                    'use_batch_norm': True
                },
                'regularization': {
                    'l2_lambda': 0.001,
                },
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'patience': 10,
                    'validation_split': 0.2
                }
            }
        else:
            self.model_params = model_params
            
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        # Setup logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def _build_model(self, input_dim):
        """
        Build the neural network architecture based on specified parameters.
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            keras.Model: Compiled Keras model
        """
        arch_params = self.model_params['architecture']
        reg_params = self.model_params['regularization']
        train_params = self.model_params['training']
        
        # Create sequential model
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,), name='input_layer'))
        
        # Hidden layers
        for i, (units, dropout_rate) in enumerate(
            zip(arch_params['hidden_layers'], arch_params['dropout_rates'])
        ):
            # Add dense layer with regularization
            model.add(layers.Dense(
                units, 
                activation=None,  # Activation will be after batch norm if used
                kernel_regularizer=regularizers.l2(reg_params['l2_lambda']),
                name=f'dense_{i}'
            ))
            
            # Add batch normalization if specified
            if arch_params.get('use_batch_norm', True):
                model.add(layers.BatchNormalization(name=f'batch_norm_{i}'))
                
            # Add activation
            model.add(layers.Activation(
                arch_params['activation'],
                name=f'activation_{i}'
            ))
            
            # Add dropout
            model.add(layers.Dropout(
                dropout_rate,
                name=f'dropout_{i}'
            ))
            
        # Output layer
        model.add(layers.Dense(
            1, 
            activation=arch_params['output_activation'],
            name='output'
        ))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=train_params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y, validation_data=None, verbose=1, **kwargs):
        """
        Train the Neural Network model on the provided data.
        
        Args:
            X: Features data
            y: Target data
            validation_data: Optional tuple of (X_val, y_val) for validation
            verbose (int): Verbosity mode for Keras fit
            **kwargs: Additional parameters passed to Keras fit method
            
        Returns:
            self: The trained model instance
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
        
        # Store training start time
        start_time = datetime.now()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Get training parameters
        train_params = self.model_params['training']
        
        # Build the model
        self.model = self._build_model(X_scaled.shape[1])
        
        # Set up callbacks
        callbacks_list = [
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=train_params['patience'],
                restore_best_weights=True
            ),
            # Reduce learning rate when plateau is reached
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=train_params['patience'] // 2,
                min_lr=0.00001
            )
        ]
        
        # Handle validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
                
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
            validation_split = 0.0  # Use provided validation data instead of split
        else:
            validation_split = train_params['validation_split']
            validation_data = None
        
        # Train the model
        history = self.model.fit(
            X_scaled, y,
            batch_size=train_params['batch_size'],
            epochs=train_params['epochs'],
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=verbose,
            **kwargs
        )
        
        # Store training history
        self.history = history.history
        
        # Update model metadata
        self.trained = True
        self.training_date = datetime.now()
        
        # Calculate training duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Store training and validation metrics
        self.metrics['training']['loss'] = history.history['loss'][-1]
        self.metrics['training']['accuracy'] = history.history['accuracy'][-1]
        
        if 'val_loss' in history.history:
            self.metrics['validation']['loss'] = history.history['val_loss'][-1]
            self.metrics['validation']['accuracy'] = history.history['val_accuracy'][-1]
        
        # Record training run in history
        training_record = {
            'date': self.training_date.isoformat(),
            'duration_seconds': duration,
            'params': self.model_params.copy(),
            'samples': len(X),
            'final_loss': float(history.history['loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1]),
            'epochs_completed': len(history.history['loss']),
            'early_stopped': len(history.history['loss']) < train_params['epochs']
        }
        
        if 'val_loss' in history.history:
            training_record['val_loss'] = float(history.history['val_loss'][-1])
            training_record['val_accuracy'] = float(history.history['val_accuracy'][-1])
            
        self.training_history.append(training_record)
        
        self.logger.info(f"Neural Network model training completed in {duration:.2f} seconds")
        self.logger.info(f"Final training loss: {training_record['final_loss']:.4f}, "
                        f"accuracy: {training_record['final_accuracy']:.4f}")
        
        if 'val_loss' in training_record:
            self.logger.info(f"Final validation loss: {training_record['val_loss']:.4f}, "
                            f"accuracy: {training_record['val_accuracy']:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make binary predictions with the trained Neural Network model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred_proba = self.model.predict(X_scaled, verbose=0)
        
        # Convert to binary predictions
        return (y_pred_proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Make probability predictions with the trained Neural Network model.
        
        Args:
            X: Features data for prediction
            
        Returns:
            numpy.ndarray: Probability predictions (between 0 and 1)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make probability predictions
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def feature_importances(self):
        """
        Compute feature importances using permutation importance.
        
        Since neural networks don't have built-in feature importance,
        we estimate it by permuting each feature and measuring the
        impact on model performance.
        
        Returns:
            numpy.ndarray: Feature importance scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importances")
        
        # Use a sample of training data for importance calculation
        if hasattr(self, '_X_sample') and hasattr(self, '_y_sample'):
            X = self._X_sample
            y = self._y_sample
        else:
            # We don't have stored training data, so we can't compute
            # feature importances this way
            self.logger.warning("No training data sample available for importance calculation")
            
            # Return a rough estimate based on first layer weights
            # This is not as accurate but at least provides something
            weights = self.model.layers[0].get_weights()[0]  # Input layer weights
            importances = np.mean(np.abs(weights), axis=1)
            return importances
            
        # Initialize importance array
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        # Get baseline score
        baseline_score = self.model.evaluate(X, y, verbose=0)[0]  # Loss
        
        # For each feature, permute its values and measure impact
        for i in range(n_features):
            # Copy the data
            X_permuted = X.copy()
            
            # Permute the feature
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Evaluate model on permuted data
            permuted_score = self.model.evaluate(X_permuted, y, verbose=0)[0]
            
            # Importance is the increase in loss when feature is permuted
            # Higher increase = more important feature
            importances[i] = permuted_score - baseline_score
            
        return importances
    
    def plot_training_history(self, save_path=None):
        """
        Plot the training history (loss and accuracy).
        
        Args:
            save_path (str): Path to save the plot image
            
        Returns:
            None
        """
        if not self.trained or self.history is None:
            raise ValueError("Model must be trained with history available")
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training and validation loss
        ax1.plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot training and validation accuracy
        ax2.plot(self.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history:
            ax2.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # Add main title
        plt.suptitle(f'Training History - {self.name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
    
    def _save_model(self, directory):
        """
        Save the model implementation to files.
        
        Args:
            directory (str): Directory to save the model to
            
        Returns:
            str: Path to the saved model directory
        """
        # Create a model-specific directory
        model_dir = os.path.join(directory, self.name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Keras model
        model_path = os.path.join(model_dir, "keras_model")
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        dump(self.scaler, scaler_path)
        
        # Save history and additional parameters
        params_path = os.path.join(model_dir, "params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump({
                'model_params': self.model_params,
                'random_state': self.random_state,
                'feature_names': self.feature_names,
                'history': self.history
            }, f)
            
        self.logger.info(f"Neural Network model saved to {model_dir}")
        
        return model_dir
    
    def load(self, directory):
        """
        Load a previously saved model.
        
        Args:
            directory (str): Directory containing the saved model
            
        Returns:
            self: The loaded model instance
        """
        # If directory is a specific model directory
        if os.path.basename(directory) == self.name:
            model_dir = directory
        else:
            # Otherwise, it's the parent directory
            model_dir = os.path.join(directory, self.name)
            
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist")
            
        # Load model parameters
        params_path = os.path.join(model_dir, "params.pkl")
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            
        self.model_params = params['model_params']
        self.random_state = params['random_state']
        self.feature_names = params['feature_names']
        self.history = params['history']
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.scaler = load(scaler_path)
        
        # Load Keras model
        model_path = os.path.join(model_dir, "keras_model")
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self.version = metadata['version']
            self.trained = metadata['trained']
            self.training_date = datetime.fromisoformat(metadata['training_date']) if metadata['training_date'] else None
            self.metrics = metadata['metrics']
            
            if 'training_history' in metadata:
                self.training_history = metadata['training_history']
                
        self.logger.info(f"Neural Network model loaded from {model_dir}")
        
        return self
    
    def save_for_api(self, directory):
        """
        Save the model in a format suitable for API deployment.
        This creates a more lightweight version focused on inference.
        
        Args:
            directory (str): Directory to save the model to
            
        Returns:
            str: Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save model in TensorFlow SavedModel format
        model_path = os.path.join(directory, f"{self.name}_tf")
        self.model.save(model_path, save_format='tf')
        
        # Save scaler separately
        scaler_path = os.path.join(directory, f"{self.name}_scaler.joblib")
        dump(self.scaler, scaler_path)
        
        # Save minimal metadata
        metadata = {
            'name': self.name,
            'version': self.version,
            'feature_names': self.feature_names,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'metrics': {
                'training': self.metrics['training'],
                'validation': self.metrics['validation']
            }
        }
        
        metadata_path = os.path.join(directory, f"{self.name}_api_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Model saved for API deployment to {directory}")
        
        return model_path