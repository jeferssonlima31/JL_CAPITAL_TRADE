#!/usr/bin/env python3
# =============================================================================
# JL CAPITAL TRADE - COLETA DE DADOS PARA TREINAMENTO
# =============================================================================

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_data_collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente
load_dotenv()

class TrainingDataCollector:
    """Coletor especializado para dados de treinamento"""
    
    def __init__(self):
        self.login = int(os.getenv("MT5_LOGIN"))
        self.password = os.getenv("MT5_PASSWORD")
        self.server = os.getenv("MT5_SERVER")
        
        # Símbolos e timeframes para coleta
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        self.timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5, 
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        # Período de coleta (últimos 3 anos para mais dados)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=1095)  # 3 anos
        
    def connect_mt5(self):
        """Conecta ao MetaTrader 5"""
        try:
            if not mt5.initialize():
                logger.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
                return False
            
            authorized = mt5.login(
                login=self.login,
                password=self.password,
                server=self.server
            )
            
            if not authorized:
                logger.error(f"Login falhou: {mt5.last_error()}")
                return False
            
            logger.info("Conectado ao MT5 com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro na conexão MT5: {e}")
            return False
    
    def get_historical_data(self, symbol, timeframe, start_date, end_date):
        """Obtém dados históricos para um símbolo e timeframe"""
        try:
            # Converte datas para timestamp UTC
            from_date = int(start_date.timestamp())
            to_date = int(end_date.timestamp())
            
            # Obtém dados do MT5
            rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"Nenhum dado encontrado para {symbol} {timeframe}")
                return None
            
            # Converte para DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Adiciona informações do símbolo
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            logger.info(f"📊 {symbol} {timeframe}: {len(df)} registros de {df.index[0]} até {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter dados de {symbol} {timeframe}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calcula indicadores técnicos para o DataFrame"""
        try:
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
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Volume (se disponível)
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            else:
                # Cria volume simulado se não estiver disponível
                df['volume'] = 1000  # valor padrão
                df['volume_sma'] = 1000
                df['volume_ratio'] = 1.0
            
            # Posição do preço
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Volatilidade
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # EMAs
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            df['ema_cross'] = (df['ema_9'] - df['ema_21']) / df['ema_21']
            
            # Momentum
            df['momentum'] = df['close'] - df['close'].shift(10)
            
            # ROC (Rate of Change)
            df['roc'] = df['close'].pct_change(periods=10) * 100
            
            # Target: Previsão de tendência (1 se próximo candle for positivo, 0 se negativo)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Remove NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            return df
    
    def collect_all_data(self):
        """Coleta dados para todos os símbolos e timeframes"""
        all_data = []
        
        if not self.connect_mt5():
            return None
        
        try:
            for symbol in self.symbols:
                logger.info(f"\n🔍 Coletando dados para {symbol}")
                logger.info("-" * 50)
                
                # Verifica se o símbolo está disponível
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.warning(f"Símbolo {symbol} não encontrado, pulando...")
                    continue
                
                # Ativa o símbolo se não estiver visível
                if not symbol_info.visible:
                    logger.info(f"Ativando símbolo {symbol}...")
                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"Falha ao ativar {symbol}, pulando...")
                        continue
                
                for tf_name, tf_value in self.timeframes.items():
                    logger.info(f"📈 Coletando {symbol} {tf_name}...")
                    
                    # Coleta dados históricos
                    df = self.get_historical_data(symbol, tf_value, self.start_date, self.end_date)
                    
                    if df is not None and len(df) > 0:
                        # Calcula indicadores técnicos
                        df_with_indicators = self.calculate_technical_indicators(df.copy())
                        
                        if df_with_indicators is not None and len(df_with_indicators) > 0:
                            all_data.append(df_with_indicators)
                            logger.info(f"✅ {symbol} {tf_name}: {len(df_with_indicators)} registros com indicadores")
                        else:
                            logger.warning(f"⚠️  {symbol} {tf_name}: Falha ao calcular indicadores")
                    else:
                        logger.warning(f"⚠️  {symbol} {tf_name}: Nenhum dado coletado")
                    
                    # Pequena pausa para não sobrecarregar o MT5
                    import time
                    time.sleep(1)
            
            if all_data:
                # Combina todos os DataFrames
                combined_df = pd.concat(all_data, ignore_index=False)
                
                # Salva os dados
                self.save_training_data(combined_df)
                
                return combined_df
            else:
                logger.error("Nenhum dado foi coletado")
                return None
                
        finally:
            # Sempre desconecta
            mt5.shutdown()
            logger.info("🔌 MT5 desconectado")
    
    def save_training_data(self, df):
        """Salva os dados de treinamento"""
        try:
            # Cria diretório se não existir
            os.makedirs('training_data', exist_ok=True)
            
            # Data atual para nome do arquivo
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Salva em formato Parquet (eficiente)
            parquet_path = f"training_data/training_data_{current_date}.parquet"
            df.to_parquet(parquet_path)
            
            # Salva também em CSV (para fácil inspeção)
            csv_path = f"training_data/training_data_{current_date}.csv"
            df.to_csv(csv_path)
            
            logger.info(f"💾 Dados salvos em: {parquet_path}")
            logger.info(f"💾 Dados salvos em: {csv_path}")
            logger.info(f"📊 Total de registros: {len(df):,}")
            logger.info(f"📈 Símbolos: {df['symbol'].unique().tolist()}")
            logger.info(f"⏰ Timeframes: {df['timeframe'].unique().tolist()}")
            
            # Estatísticas básicas
            logger.info(f"📅 Período: {df.index.min()} até {df.index.max()}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {e}")
    
    def analyze_data_quality(self, df):
        """Analisa a qualidade dos dados coletados"""
        logger.info("\n🔍 ANALISE DE QUALIDADE DOS DADOS")
        logger.info("=" * 50)
        
        # Verifica valores NaN
        nan_counts = df.isnull().sum()
        logger.info(f"Valores NaN por coluna:\n{nan_counts[nan_counts > 0]}")
        
        # Estatísticas por símbolo
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            logger.info(f"\n📊 {symbol}:")
            logger.info(f"   Registros: {len(symbol_data):,}")
            logger.info(f"   Período: {symbol_data.index.min()} até {symbol_data.index.max()}")
            logger.info(f"   Timeframes: {symbol_data['timeframe'].unique().tolist()}")
        
        # Distribuição do target
        if 'target' in df.columns:
            target_dist = df['target'].value_counts()
            logger.info(f"\n🎯 Distribuição do Target:")
            logger.info(f"   Compra (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
            logger.info(f"   Venda (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")

def main():
    """Função principal"""
    logger.info("=" * 70)
    logger.info("🤖 JL CAPITAL TRADE - COLETA DE DADOS PARA TREINAMENTO")
    logger.info("=" * 70)
    
    collector = TrainingDataCollector()
    
    logger.info(f"📅 Período de coleta: {collector.start_date} até {collector.end_date}")
    logger.info(f"📊 Símbolos: {collector.symbols}")
    logger.info(f"⏰ Timeframes: {list(collector.timeframes.keys())}")
    
    # Coleta os dados
    training_data = collector.collect_all_data()
    
    if training_data is not None:
        # Analisa qualidade dos dados
        collector.analyze_data_quality(training_data)
        
        logger.info("\n✅ COLETA CONCLUÍDA COM SUCESSO!")
        logger.info("🚀 Os dados estão prontos para treinamento dos modelos!")
    else:
        logger.error("\n❌ FALHA NA COLETA DE DADOS!")
        
    logger.info("=" * 70)

if __name__ == "__main__":
    main()