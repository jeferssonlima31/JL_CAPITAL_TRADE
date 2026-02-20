# =============================================================================
# JL CAPITAL TRADE - COLETOR DE DADOS
# =============================================================================

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class DataCollector:
    """Coletor de dados e calculador de indicadores"""
    
    def __init__(self, config, mt5_connector):
        self.config = config
        self.mt5 = mt5_connector
        self.cache = None  # Será injetado depois
    
    def set_cache(self, cache_manager):
        """Injeta cache manager"""
        self.cache = cache_manager
    
    def get_historical_data(self, symbol: str, timeframe: str = "H1", 
                             count: int = 500) -> Optional[pd.DataFrame]:
        """Obtém dados históricos com cache"""
        
        # Verifica cache primeiro
        if self.cache and self.config.cache.enabled:
            cache_key = f"hist_{symbol}_{timeframe}_{count}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Using cached data for {symbol}")
                return cached
        
        # Busca do MT5
        df = self.mt5.get_historical_data(symbol, timeframe, count)
        
        # Salva no cache
        if df is not None and self.cache and self.config.cache.enabled:
            self.cache.set(df, cache_key, ttl=self.config.cache.historical_data_ttl)
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calcula indicadores técnicos"""
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        if symbol == "XAU_USD":
            df['atr'] = true_range.rolling(window=20).mean()
        else:
            df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Posição do preço
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volatilidade
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_cross'] = (df['ema_9'] - df['ema_21']) / df['ema_21']
        
        # Suporte e Resistência
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # ROC (Rate of Change)
        df['roc'] = df['close'].pct_change(periods=10) * 100
        
        return df
    
    def get_market_hours(self, symbol: str) -> Dict:
        """Retorna melhores horários para trading"""
        
        from datetime import datetime
        
        now = datetime.now()
        current_hour = now.hour
        
        # Sessões de mercado (UTC)
        sessions = {
            "london": {"start": 8, "end": 17},
            "new_york": {"start": 13, "end": 22},
            "asia": {"start": 0, "end": 9},
            "overlap": {"start": 13, "end": 17}
        }
        
        result = {
            "current_session": "off_hours",
            "is_optimal": False,
            "next_session": None
        }
        
        if symbol == "EUR_USD":
            # Melhor para EUR/USD é overlap Londres/NY
            if sessions["overlap"]["start"] <= current_hour < sessions["overlap"]["end"]:
                result["current_session"] = "london_ny_overlap"
                result["is_optimal"] = True
            elif sessions["london"]["start"] <= current_hour < sessions["london"]["end"]:
                result["current_session"] = "london"
                result["is_optimal"] = True
            elif sessions["new_york"]["start"] <= current_hour < sessions["new_york"]["end"]:
                result["current_session"] = "new_york"
                result["is_optimal"] = True
        
        elif symbol == "XAU_USD":
            # OURO tem melhor liquidez em Londres e NY
            if sessions["london"]["start"] <= current_hour < sessions["london"]["end"]:
                result["current_session"] = "london"
                result["is_optimal"] = True
            elif sessions["new_york"]["start"] <= current_hour < sessions["new_york"]["end"]:
                result["current_session"] = "new_york"
                result["is_optimal"] = True
            elif sessions["overlap"]["start"] <= current_hour < sessions["overlap"]["end"]:
                result["current_session"] = "overlap"
                result["is_optimal"] = True
        
        return result