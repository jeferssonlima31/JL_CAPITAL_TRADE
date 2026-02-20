#!/usr/bin/env python3
# =============================================================================
# JL CAPITAL TRADE - CRIAÇÃO DE DADOS DE TREINAMENTO
# =============================================================================

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente
load_dotenv()

def create_enhanced_features(df):
    """Cria features técnicas avançadas"""
    try:
        # Features básicas
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Médias móveis
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volatilidade
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['atr'] = calculate_atr(df)
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(5)
        df['roc'] = (df['close'] / df['close'].shift(5) - 1) * 100
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price patterns
        df['price_vs_ma'] = df['close'] / df['ma20'] - 1
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Target inteligente - movimento significativo futuro
        # Retorno futuro em 5 períodos (ajustável)
        future_periods = 5
        future_return = df['close'].shift(-future_periods) / df['close'] - 1
        
        # Target baseado em retornos positivos (balanceamento melhorado)
        return_threshold = 0.0005  # 0.05% (mais exemplos positivos)
        df['target'] = (future_return > return_threshold).astype(int)
        
        # Target alternativo: tendência de alta sustentada
        # df['target'] = ((df['close'].shift(-5) > df['close'] * 1.005) & 
        #                (df['close'].rolling(3).mean().shift(-3) > df['close'])).astype(int)
        
        # Remove NaN values
        df = df.dropna()
        
        logger.info(f"Features criadas: {df.shape[1]} colunas")
        return df
        
    except Exception as e:
        logger.error(f"Erro ao criar features: {e}")
        return df

def calculate_atr(df, period=14):
    """Calcula Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    return atr

def connect_mt5():
    """Conecta ao MT5"""
    try:
        login = int(os.getenv("MT5_LOGIN"))
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        
        if not mt5.initialize():
            logger.error("Falha ao inicializar MT5")
            return False
        
        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            logger.error("Falha no login MT5")
            return False
        
        logger.info(f"Conectado ao MT5: {mt5.account_info()}")
        return True
        
    except Exception as e:
        logger.error(f"Erro na conexão MT5: {e}")
        return False

def get_historical_data(symbol, timeframe, start_date, end_date):
    """Obtém dados históricos"""
    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            logger.warning(f"Nenhum dado para {symbol} {timeframe}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        logger.info(f"{symbol} {timeframe}: {len(df)} registros")
        return df
        
    except Exception as e:
        logger.error(f"Erro ao obter dados de {symbol}: {e}")
        return None

def main():
    """Função principal"""
    logger.info("=" * 60)
    logger.info("JL CAPITAL TRADE - CRIAÇÃO DE DADOS DE TREINAMENTO")
    logger.info("=" * 60)
    
    # Conecta ao MT5
    if not connect_mt5():
        return
    
    # Configurações
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    timeframe = mt5.TIMEFRAME_H1
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=2)
    
    all_data = []
    
    for symbol in symbols:
        logger.info(f"Coletando {symbol}...")
        
        # Obtém dados
        df = get_historical_data(symbol, timeframe, start_date, end_date)
        if df is None:
            continue
        
        # Adiciona símbolo
        df['symbol'] = symbol
        
        # Cria features
        df_enhanced = create_enhanced_features(df)
        if df_enhanced is not None:
            all_data.append(df_enhanced)
            logger.info(f"{symbol}: {len(df_enhanced)} registros processados")
    
    # Combina todos os dados
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Salva dados
        models_dir = "trained_models"
        os.makedirs(models_dir, exist_ok=True)
        
        output_path = os.path.join(models_dir, "training_data_enhanced.csv")
        combined_data.to_csv(output_path, index=False)
        
        logger.info(f"Dados salvos: {output_path}")
        logger.info(f"Total: {len(combined_data)} registros")
        logger.info(f"Features: {combined_data.shape[1]} colunas")
        
        # Estatísticas básicas
        logger.info(f"Distribuição do target: {combined_data['target'].value_counts().to_dict()}")
        
    else:
        logger.warning("Nenhum dado coletado")
    
    # Desconecta
    mt5.shutdown()
    logger.info("✅ Dados de treinamento criados com sucesso!")

if __name__ == "__main__":
    main()