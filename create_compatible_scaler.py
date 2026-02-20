#!/usr/bin/env python3
"""
CRIADOR DE SCALER COMPATÍVEL
Cria um scaler que corresponde exatamente às features do novo modelo turbo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import logging

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_compatible_scaler():
    """Cria um scaler compatível com as 15 features do modelo turbo"""
    
    # Features exatas que o modelo turbo espera
    required_features = [
        'close', 'open', 'high', 'low', 'tick_volume', 'spread',
        'ma_5', 'ma_20', 'ma_50', 'rsi_14', 'volatility_20',
        'returns', 'macd', 'macd_signal', 'volume_ma_ratio'
    ]
    
    try:
        # Carrega dados de treinamento
        df = pd.read_csv('trained_models/training_data_enhanced.csv')
        logger.info(f"📊 Dados carregados: {df.shape}")
        
        # Cria features faltantes se necessário
        for feature in required_features:
            if feature not in df.columns:
                if feature == 'volume_ma_ratio' and 'tick_volume' in df.columns:
                    df['volume_ma_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
                elif feature == 'spread':
                    df['spread'] = 1.0
                elif feature == 'ma_5':
                    df['ma_5'] = df['close'].rolling(5).mean()
                elif feature == 'ma_20':
                    df['ma_20'] = df['close'].rolling(20).mean()
                elif feature == 'ma_50':
                    df['ma_50'] = df['close'].rolling(50).mean()
                elif feature == 'rsi_14':
                    # RSI simples
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    df['rsi_14'] = 100 - (100 / (1 + rs))
                elif feature == 'volatility_20':
                    df['volatility_20'] = df['close'].rolling(20).std()
                elif feature == 'returns':
                    df['returns'] = df['close'].pct_change()
                elif feature == 'macd':
                    # MACD simples
                    exp12 = df['close'].ewm(span=12).mean()
                    exp26 = df['close'].ewm(span=26).mean()
                    df['macd'] = exp12 - exp26
                elif feature == 'macd_signal':
                    exp12 = df['close'].ewm(span=12).mean()
                    exp26 = df['close'].ewm(span=26).mean()
                    macd = exp12 - exp26
                    df['macd_signal'] = macd.ewm(span=9).mean()
                else:
                    df[feature] = 0.0
        
        # Seleciona apenas as features necessárias
        X = df[required_features]
        
        # Remove NaN
        X = X.dropna()
        
        logger.info(f"✅ Dados preparados para scaler: {X.shape}")
        
        # Cria e treina o scaler
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Salva o scaler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scaler_filename = f"trained_models/scaler_turbo_compatible_{timestamp}.pkl"
        
        joblib.dump(scaler, scaler_filename)
        logger.info(f"💾 Scaler compatível salvo: {scaler_filename}")
        logger.info(f"🎯 Features do scaler: {len(scaler.feature_names_in_)}")
        logger.info(f"📊 Features: {list(scaler.feature_names_in_)}")
        
        return scaler_filename
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar scaler: {e}")
        return None

if __name__ == "__main__":
    logger.info("🎯 CRIANDO SCALER COMPATÍVEL")
    scaler_path = create_compatible_scaler()
    
    if scaler_path:
        logger.info(f"🚀 SCALER COMPATÍVEL CRIADO COM SUCESSO!")
        logger.info(f"   📁 Arquivo: {scaler_path}")
    else:
        logger.error("❌ Falha ao criar scaler compatível")