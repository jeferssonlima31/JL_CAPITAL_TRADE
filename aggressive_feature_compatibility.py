#!/usr/bin/env python3
"""
SISTEMA DE COMPATIBILIDADE PARA MODELO AGRESSIVO
Calcula as 43 features necessárias para o modelo de alta performance
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AggressiveFeatureCompatibility:
    def __init__(self):
        # Features ROBUSTAS selecionadas para evitar overfitting (6 features)
        self.required_features = [
            'ma_50', 'ma_100', 'ma_200', 'volatility_20', 'volatility_50', 'bb_std'
        ]
    
    def convert_to_aggressive_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte DataFrame para formato do modelo agressivo
        """
        if df is None or len(df) < 200:
            return None
            
        result_df = df.copy()
        
        # Garante que todas as features necessárias existem
        self._calculate_all_features(result_df)
        
        # Seleciona APENAS as features que o modelo espera
        available_features = [col for col in self.required_features if col in result_df.columns]
        
        # Pega apenas a última linha
        latest_data = result_df[available_features].iloc[[-1]]
        
        return latest_data
    
    def _calculate_all_features(self, df: pd.DataFrame):
        """Calcula todas as 43 features"""
        
        # Features de Tempo/Sessão
        if 'time' in df.columns:
            df['time_dt'] = pd.to_datetime(df['time'], unit='s')
            df['hour'] = df['time_dt'].dt.hour
            df['day_of_week'] = df['time_dt'].dt.dayofweek
            
            # Sessões (Horário de Portugal)
            df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
            df['is_usa'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
            df['is_overlay'] = (df['is_london'] & df['is_usa']).astype(int)
        
        # Features básicas
        for col in ['real_volume', 'spread']:
            if col not in df.columns:
                df[col] = 1.0 if col == 'spread' else 0.0
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Médias móveis
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        
        # Momentum e ROC
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
        df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # Volatilidade
        df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['volatility_50'] = df['close'].rolling(50).std() / df['close'].rolling(50).mean()
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        for fast, slow in [(12, 26), (8, 17), (5, 35)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
            df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume
        df['volume_ma_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        df['volume_roc'] = df['tick_volume'].pct_change()
        
        # Fill NaN
        df.fillna(0, inplace=True)
