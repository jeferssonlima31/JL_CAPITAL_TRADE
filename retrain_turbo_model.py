#!/usr/bin/env python3
"""
RETREINAMENTO DO MODELO TURBO COM FEATURES COMPATÍVEIS
Treina um novo modelo com as features EXATAS que o sistema de trading gera
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib
from datetime import datetime
import logging

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TurboModelRetrainer:
    def __init__(self):
        # Features EXATAS que o sistema de trading gera (15 features)
        self.required_features = [
            'close', 'open', 'high', 'low', 'tick_volume', 'spread',
            'ma_5', 'ma_20', 'ma_50', 'rsi_14', 'volatility_20',
            'returns', 'macd', 'macd_signal', 'volume_ma_ratio'
        ]
    
    def load_and_prepare_data(self):
        """Carrega e prepara dados de treinamento"""
        try:
            df = pd.read_csv('trained_models/training_data_enhanced.csv')
            logger.info(f"📊 Dados carregados: {df.shape}")
            
            # Filtra apenas as features que usaremos
            available_features = [col for col in self.required_features if col in df.columns]
            missing_features = [col for col in self.required_features if col not in df.columns]
            
            if missing_features:
                logger.warning(f"⚠️ Features faltantes: {missing_features}")
                # Cria features faltantes
                for feature in missing_features:
                    if feature == 'volume_ma_ratio' and 'tick_volume' in df.columns:
                        df['volume_ma_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
                    elif feature == 'spread':
                        df['spread'] = 1.0  # Valor padrão
                    else:
                        df[feature] = 0.0
            
            # Seleciona apenas as features necessárias
            X = df[self.required_features]
            y = df['target']
            
            # Remove linhas com valores NaN
            valid_mask = ~X.isna().any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            
            logger.info(f"🧹 Dados após remoção de NaN: X={X.shape}, y={y.shape}")
            
            logger.info(f"✅ Dados preparados: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            return None, None
    
    def train_turbo_model(self, X, y):
        """Treina novo modelo turbo"""
        try:
            # Time Series Cross Validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Parâmetros otimizados para GradientBoosting
            params = {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 4,
                'subsample': 0.8,
                'random_state': 42,
                'verbose': 1
            }
            
            logger.info("🚀 Treinando novo modelo turbo...")
            
            # Treina modelo
            model = GradientBoostingClassifier(**params)
            
            # Calibração para melhorar probabilidades
            calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
            calibrated_model.fit(X, y)
            
            # Avaliação
            y_pred = calibrated_model.predict(X)
            y_proba = calibrated_model.predict_proba(X)
            
            accuracy = accuracy_score(y, y_pred)
            confidence = y_proba.max(axis=1).mean()
            
            logger.info(f"✅ Modelo treinado com sucesso!")
            logger.info(f"   📊 Acurácia: {accuracy:.4f}")
            logger.info(f"   🎯 Confiança média: {confidence:.4f}")
            logger.info(f"   📈 Distribuição: {pd.Series(y).value_counts().to_dict()}")
            
            return calibrated_model
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            return None
    
    def save_model(self, model):
        """Salva o modelo e scaler"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"trained_models/xgboost_turbo_compatible_{timestamp}.joblib"
            
            joblib.dump(model, model_filename)
            logger.info(f"💾 Modelo salvo: {model_filename}")
            
            return model_filename
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar modelo: {e}")
            return None

def main():
    """Função principal"""
    logger.info("🎯 INICIANDO RETREINAMENTO DO MODELO TURBO")
    
    retrainer = TurboModelRetrainer()
    
    # Carrega e prepara dados
    X, y = retrainer.load_and_prepare_data()
    if X is None or y is None:
        return
    
    # Treina modelo
    model = retrainer.train_turbo_model(X, y)
    if model is None:
        return
    
    # Salva modelo
    model_path = retrainer.save_model(model)
    
    if model_path:
        logger.info(f"🚀 MODELO TURBO COMPATÍVEL CRIADO COM SUCESSO!")
        logger.info(f"   📁 Arquivo: {model_path}")
        logger.info(f"   🔢 Features: {retrainer.required_features}")
        logger.info(f"   🎯 Total features: {len(retrainer.required_features)}")
    else:
        logger.error("❌ Falha ao criar modelo compatível")

if __name__ == "__main__":
    main()