#!/usr/bin/env python3
"""
SISTEMA DE COMPATIBILIDADE DE FEATURES
Converte features do sistema de trading para o formato do modelo turbo
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List

class FeatureCompatibility:
    def __init__(self):
        self.required_features = [
            'ma_5', 'ma_20', 'ma_50', 'rsi_14', 'volatility_20',
            'returns', 'macd', 'macd_signal', 'volume_ma_ratio'
        ]
    
    def convert_to_turbo_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte DataFrame com features complexas para formato simples do modelo turbo
        """
        result_df = df.copy()
        
        # Garante que todas as features necessárias existem
        self._ensure_required_features(result_df)
        
        # Seleciona apenas as features que o modelo turbo espera
        available_features = [col for col in self.required_features if col in result_df.columns]
        
        return result_df[available_features]
    
    def _ensure_required_features(self, df: pd.DataFrame):
        """Garante que todas as features necessárias existem no DataFrame"""
        
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
        
        # Volatilidade
        if 'volatility_20' not in df.columns:
            df['volatility_20'] = df['close'].rolling(20).std()
        
        # Returns
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # MACD
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volume ratio
        if 'volume_ma_ratio' not in df.columns and 'tick_volume' in df.columns:
            df['volume_ma_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        elif 'volume_ma_ratio' not in df.columns:
            df['volume_ma_ratio'] = 1.0  # Valor padrão
    
    def get_feature_mapping(self) -> Dict[str, str]:
        """Retorna mapeamento de features complexas para simples"""
        return {
            # Features básicas
            'close': 'close',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'tick_volume': 'tick_volume',
            
            # Médias móveis
            'ma_5': 'ma_5',
            'ma_20': 'ma_20', 
            'ma_50': 'ma_50',
            
            # Indicadores
            'rsi_14': 'rsi_14',
            'volatility_20': 'volatility_20',
            'returns': 'returns',
            'macd': 'macd',
            'macd_signal': 'macd_signal',
            'volume_ma_ratio': 'volume_ma_ratio'
        }

def test_compatibility():
    """Testa a compatibilidade de features"""
    compat = FeatureCompatibility()
    
    # Cria dados de exemplo
    sample_data = {
        'close': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1050],
        'open': [1.0990, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045],
        'high': [1.1010, 1.1020, 1.1030, 1.1040, 1.1050, 1.1060],
        'low': [1.0980, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035],
        'tick_volume': [100, 150, 200, 180, 220, 250]
    }
    
    df = pd.DataFrame(sample_data)
    converted_df = compat.convert_to_turbo_format(df)
    
    print("📊 DataFrame original:")
    print(df)
    print("\n🎯 DataFrame convertido:")
    print(converted_df)
    print("\n✅ Features disponíveis:", list(converted_df.columns))

if __name__ == "__main__":
    test_compatibility()