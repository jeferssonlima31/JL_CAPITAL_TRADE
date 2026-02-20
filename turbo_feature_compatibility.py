#!/usr/bin/env python3
"""
SISTEMA DE COMPATIBILIDADE ESPECÍFICO PARA MODELO TURBO
Converte features para o formato esperado pelo GradientBoostingClassifier turbo
"""

import pandas as pd
import numpy as np

class TurboFeatureCompatibility:
    def __init__(self):
        # Features EXATAS que o modelo turbo espera (15 features)
        self.required_features = [
            'close', 'open', 'high', 'low', 'tick_volume',
            'ma_5', 'ma_20', 'ma_50', 'rsi_14', 'volatility_20',
            'returns', 'macd', 'macd_signal', 'volume_ma_ratio', 'spread'
        ]
    
    def convert_to_turbo_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte DataFrame para formato EXATO do modelo turbo
        """
        if df is None or len(df) < 50:
            return None
            
        result_df = df.copy()
        
        # Garante que todas as features necessárias existem
        self._ensure_required_features(result_df)
        
        # Seleciona APENAS as 15 features que o modelo turbo espera
        available_features = [col for col in self.required_features if col in result_df.columns]
        
        # Pega apenas a última linha (dados mais recentes)
        latest_data = result_df[available_features].iloc[[-1]]
        
        return latest_data
    
    def _ensure_required_features(self, df: pd.DataFrame):
        """Garante que todas as 15 features necessárias existem no DataFrame"""
        
        # Features básicas (já devem existir)
        for basic_feature in ['close', 'open', 'high', 'low', 'tick_volume', 'spread']:
            if basic_feature not in df.columns:
                df[basic_feature] = 0.0
        
        # Médias móveis simples
        if 'ma_5' not in df.columns:
            df['ma_5'] = df['close'].rolling(window=5).mean()
        if 'ma_20' not in df.columns:
            df['ma_20'] = df['close'].rolling(window=20).mean()
        if 'ma_50' not in df.columns:
            df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        if 'rsi_14' not in df.columns:
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            df['rsi_14'] = df['rsi_14'].fillna(50)
        
        # Volatilidade
        if 'volatility_20' not in df.columns:
            df['volatility_20'] = df['close'].rolling(20).std()
            df['volatility_20'] = df['volatility_20'].fillna(0)
        
        # Returns
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            df['returns'] = df['returns'].fillna(0)
        
        # MACD
        if 'macd' not in df.columns:
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volume ratio
        if 'volume_ma_ratio' not in df.columns:
            if 'tick_volume' in df.columns:
                df['volume_ma_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
                df['volume_ma_ratio'] = df['volume_ma_ratio'].fillna(1.0)
            else:
                df['volume_ma_ratio'] = 1.0
        
        # Preenche valores NaN
        df.fillna(0, inplace=True)

def test_turbo_compatibility():
    """Testa a compatibilidade com o modelo turbo"""
    compat = TurboFeatureCompatibility()
    
    # Cria dados de exemplo
    sample_data = {
        'close': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1050] * 10,
        'open': [1.0990, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045] * 10,
        'high': [1.1010, 1.1020, 1.1030, 1.1040, 1.1050, 1.1060] * 10,
        'low': [1.0980, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035] * 10,
        'tick_volume': [100, 150, 200, 180, 220, 250] * 10,
        'spread': [1, 1, 1, 1, 1, 1] * 10
    }
    
    df = pd.DataFrame(sample_data)
    converted_df = compat.convert_to_turbo_format(df)
    
    print("📊 DataFrame original shape:", df.shape)
    print("🎯 DataFrame convertido shape:", converted_df.shape)
    print("✅ Features disponíveis:", list(converted_df.columns))
    print("🔢 Número de features:", len(converted_df.columns))
    
    # Verifica se tem exatamente 15 features
    assert len(converted_df.columns) == 15, f"Esperado 15 features, obtido {len(converted_df.columns)}"
    print("✅ Teste de compatibilidade passou!")

if __name__ == "__main__":
    test_turbo_compatibility()