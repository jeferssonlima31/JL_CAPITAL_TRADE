# =============================================================================
# JL CAPITAL TRADE - MODELOS DE MACHINE LEARNING
# =============================================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import pickle
import os
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

class JLMLModels:
    """Modelos de ML com capacidade de aprendizado contínuo"""
    
    def __init__(self, config, continuous_learner=None):
        self.config = config
        self.continuous_learner = continuous_learner
        self.models_dir = config.models_dir
        
        # Cache de modelos em memória
        self.models = {
            'EUR_USD': {},
            'XAU_USD': {}
        }
        
        # Versões dos modelos
        self.model_versions = {}
        
        # Carrega modelos existentes
        self._load_all_models()
        
        logger.info("🤖 ML Models initialized")
    
    def _load_all_models(self):
        """Carrega todos os modelos salvos"""
        for symbol in ['EUR_USD', 'XAU_USD']:
            symbol_dir = self.models_dir / symbol.replace("/", "_")
            if symbol_dir.exists():
                for model_file in symbol_dir.glob("*.pkl"):
                    try:
                        model_name = model_file.stem.split('_v')[0]
                        with open(model_file, 'rb') as f:
                            self.models[symbol][model_name] = pickle.load(f)
                        logger.info(f"✅ Loaded {model_name} for {symbol}")
                    except Exception as e:
                        logger.error(f"Error loading {model_file}: {e}")
    
    def save_model(self, symbol: str, model_name: str, model):
        """Salva modelo"""
        
        symbol_dir = self.models_dir / symbol.replace("/", "_")
        symbol_dir.mkdir(exist_ok=True)
        
        # Versão baseada na data
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = symbol_dir / f"{model_name}_v{version}.pkl"
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.models[symbol][model_name] = model
            
            logger.info(f"✅ Model {model_name} saved for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def prepare_features(self, df: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
        """Prepara features para ML"""
        
        # Seleciona features relevantes
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position', 'atr', 'volume_ratio',
            'close_position', 'volatility', 'ema_cross',
            'momentum', 'roc'
        ]
        
        # Verifica se todas as colunas existem
        available_features = [f for f in feature_columns if f in df.columns]
        
        if not available_features:
            logger.error(f"No features available for {symbol}")
            return None
        
        # Extrai features
        features = df[available_features].values
        
        # Remove NaN
        features = pd.DataFrame(features).fillna(method='ffill').fillna(0).values
        
        return features
    
    def create_sequences(self, features: np.ndarray, lookback: int, 
                         horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cria sequências para LSTM"""
        
        X, y = [], []
        
        for i in range(lookback, len(features) - horizon):
            X.append(features[i-lookback:i])
            # Target será definido externamente
            
        return np.array(X), np.array([])
    
    def create_lstm_model(self, input_shape: Tuple[int, int], symbol: str) -> Sequential:
        """Cria modelo LSTM"""
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict_ensemble(self, symbol: str, X: np.ndarray, 
                         use_weights: bool = True) -> Dict:
        """Faz previsão ensemble"""
        
        predictions = {}
        
        # Obtém pesos dos modelos
        weights = {}
        if use_weights and self.continuous_learner:
            weights = self.continuous_learner.tracker.get_model_weights()
        
        if symbol in self.models:
            for name, model in self.models[symbol].items():
                try:
                    if name in ['lstm', 'gru']:
                        pred = model.predict(X, verbose=0).flatten()
                    elif name in ['xgboost', 'random_forest']:
                        if len(X.shape) == 3:
                            X_2d = X.reshape(X.shape[0], -1)
                        else:
                            X_2d = X
                        pred = model.predict_proba(X_2d)[:, 1]
                    else:
                        continue
                    
                    predictions[name] = pred
                    
                except Exception as e:
                    logger.error(f"Error predicting with {name}: {e}")
                    continue
        
        if predictions:
            # Calcula ensemble ponderado
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            total_weight = 0
            
            for name, pred in predictions.items():
                weight = weights.get(name, 1.0 / len(predictions))
                ensemble_pred += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
                predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def get_model_list(self, symbol: str) -> List[str]:
        """Retorna lista de modelos disponíveis"""
        return list(self.models.get(symbol, {}).keys())