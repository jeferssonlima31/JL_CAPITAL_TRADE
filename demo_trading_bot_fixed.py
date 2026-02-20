#!/usr/bin/env python3
"""
BOT DE TRADING CORRIGIDO - GERA FEATURES COMPLETAS
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from integration_system import IntegrationSystem
import logging
import time
from datetime import datetime

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedDemoTradingBot:
    def __init__(self):
        self.system = IntegrationSystem()
        self.connected = False
        self.account_info = None
        
        # Features esperadas pelo modelo
        self.expected_features = [
            'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
            'returns', 'log_returns', 'ma5', 'ma20', 'ma50', 'ma200', 'bb_middle',
            'bb_std', 'bb_upper', 'bb_lower', 'bb_width', 'rsi', 'macd', 'macd_signal',
            'macd_hist', 'volatility', 'atr', 'momentum', 'roc', 'price_vs_ma', 'high_low_ratio'
        ]
        
    def connect_to_demo(self):
        """Conecta à conta demo"""
        try:
            demo_config = {
                'login': 3263303,
                'password': '!rH5UiSb',
                'server': 'Just2Trade-MT5',
                'timeout': 30000
            }
            
            if not mt5.initialize():
                logger.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
                return False
            
            authorized = mt5.login(**demo_config)
            
            if authorized:
                self.account_info = mt5.account_info()
                self.connected = True
                logger.info("✅ CONTA DEMO CONECTADA!")
                return True
            else:
                logger.error(f"Falha no login: {mt5.last_error()}")
                return False
                
        except Exception as e:
            logger.error(f"Erro na conexão: {e}")
            return False
    
    def load_ml_system(self):
        """Carrega o sistema de ML"""
        logger.info("📦 Carregando sistema de ML...")
        
        if not self.system.load_best_model():
            logger.error("Falha ao carregar modelo de ML!")
            return False
        
        logger.info("✅ Sistema de ML carregado com sucesso!")
        return True
    
    def get_market_data(self, symbol, timeframe=mt5.TIMEFRAME_H1, count=200):
        """Obtém dados de mercado suficientes para calcular todas as features"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"Nenhum dado para {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter dados de {symbol}: {e}")
            return None
    
    def calculate_all_features(self, df):
        """Calcula TODAS as 28 features que o modelo espera"""
        if df is None or len(df) < 50:
            return None
        
        # Pega o candle mais recente
        latest = df.iloc[-1].copy()
        
        # Features básicas
        features = {
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'], 
            'close': latest['close'],
            'tick_volume': latest['tick_volume'],
            'spread': latest['spread'],
            'real_volume': latest.get('real_volume', 0),
            'returns': 0,  # Será calculado
            'log_returns': 0  # Será calculado
        }
        
        # Calcula returns
        if len(df) > 1:
            prev_close = df.iloc[-2]['close']
            features['returns'] = (latest['close'] / prev_close) - 1
            features['log_returns'] = np.log(latest['close'] / prev_close)
        
        # Médias móveis
        features['ma5'] = df['close'].tail(5).mean()
        features['ma20'] = df['close'].tail(20).mean()
        features['ma50'] = df['close'].tail(50).mean()
        features['ma200'] = df['close'].tail(200).mean() if len(df) >= 200 else features['ma50']
        
        # Bollinger Bands (simplificado)
        features['bb_middle'] = features['ma20']
        features['bb_std'] = df['close'].tail(20).std()
        features['bb_upper'] = features['bb_middle'] + 2 * features['bb_std']
        features['bb_lower'] = features['bb_middle'] - 2 * features['bb_std']
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        
        # RSI
        gains = df['close'].diff().clip(lower=0).tail(14).mean()
        losses = (-df['close'].diff().clip(upper=0)).tail(14).mean()
        features['rsi'] = 100 - (100 / (1 + gains/losses)) if losses != 0 else 50
        
        # MACD (simplificado)
        ema12 = df['close'].ewm(span=12).mean().iloc[-1]
        ema26 = df['close'].ewm(span=26).mean().iloc[-1]
        features['macd'] = ema12 - ema26
        features['macd_signal'] = pd.Series([features['macd']]).ewm(span=9).mean().iloc[-1]
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Volatilidade e indicadores
        features['volatility'] = df['close'].tail(20).std() / df['close'].tail(20).mean()
        features['atr'] = (df['high'].tail(14) - df['low'].tail(14)).mean()
        features['momentum'] = latest['close'] - df['close'].iloc[-4]  # 4 períodos atrás
        features['roc'] = (latest['close'] / df['close'].iloc[-10] - 1) * 100  # Rate of Change
        features['price_vs_ma'] = (latest['close'] / features['ma20'] - 1) * 100
        features['high_low_ratio'] = latest['high'] / latest['low'] if latest['low'] != 0 else 1
        
        # Garante que todas as features esperadas existem
        for feature in self.expected_features:
            if feature not in features:
                features[feature] = 0  # Valor padrão
        
        return pd.DataFrame([features])[self.expected_features]
    
    def execute_demo_signal(self, symbol, prediction):
        """Executa e exibe sinal de trading"""
        if not prediction:
            return None
            
        signal = prediction['prediction'][0]
        confidence = prediction['confidence'][0]
        
        action = "COMPRA" if signal == 1 else "VENDA"
        
        if confidence > 0.7:
            status = "🚀 ALTA CONFIANÇA"
            color = "GREEN"
        elif confidence > 0.6:
            status = "🎯 MÉDIA CONFIANÇA" 
            color = "YELLOW"
        else:
            status = "⚠️ BAIXA CONFIANÇA"
            color = "RED"
        
        # Obtém preço atual
        symbol_info = mt5.symbol_info(symbol)
        current_price = symbol_info.ask if symbol_info else 0
        
        logger.info(f"{color} {action} {symbol} | Conf: {confidence:.2%} | Preço: {current_price:.5f} | {status}")
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'price': current_price,
            'status': status,
            'timestamp': datetime.now()
        }
    
    def run_trading_session(self):
        """Executa sessão de trading com features completas"""
        if not self.connected:
            return []
        
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        results = []
        
        logger.info("\n" + "=" * 70)
        logger.info("🚀 SESSÃO DE TRADING - FEATURES COMPLETAS")
        logger.info("=" * 70)
        
        for symbol in symbols:
            try:
                # Obtém dados
                market_data = self.get_market_data(symbol, count=200)
                if market_data is None:
                    continue
                
                # Calcula TODAS as features
                features = self.calculate_all_features(market_data)
                if features is None:
                    continue
                
                # Faz previsão
                prediction = self.system.predict(features)
                
                # Executa sinal
                trade_result = self.execute_demo_signal(symbol, prediction)
                if trade_result:
                    results.append(trade_result)
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erro em {symbol}: {e}")
                continue
        
        # Resumo
        logger.info("\n" + "=" * 70)
        logger.info("📊 RESUMO DA SESSÃO")
        logger.info("=" * 70)
        
        high_conf = sum(1 for r in results if r['confidence'] > 0.7)
        medium_conf = sum(1 for r in results if 0.6 <= r['confidence'] <= 0.7)
        low_conf = sum(1 for r in results if r['confidence'] < 0.6)
        
        buys = sum(1 for r in results if r['action'] == 'COMPRA')
        sells = sum(1 for r in results if r['action'] == 'VENDA')
        
        logger.info(f"📈 Compras: {buys} | 📉 Vendas: {sells}")
        logger.info(f"🚀 Alta confiança: {high_conf} | 🎯 Média: {medium_conf} | ⚠️ Baixa: {low_conf}")
        
        return results
    
    def shutdown(self):
        """Fecha conexão"""
        if self.connected:
            mt5.shutdown()
            logger.info("✅ Conexão fechada")

def main():
    """Função principal"""
    bot = FixedDemoTradingBot()
    
    try:
        # Conecta e carrega sistema
        if not bot.connect_to_demo() or not bot.load_ml_system():
            return
        
        # Executa trading
        results = bot.run_trading_session()
        
        # Mostra recomendações fortes
        strong_signals = [r for r in results if r['confidence'] > 0.7]
        
        if strong_signals:
            logger.info("\n🎯 FORTES RECOMENDAÇÕES:")
            for signal in strong_signals:
                logger.info(f"   {'🟢' if signal['action']=='COMPRA' else '🔴'} {signal['symbol']}: "
                          f"{signal['action']} (Conf: {signal['confidence']:.2%})")
        
        logger.info("\n✅ Teste em conta demo concluído!")
        
    except Exception as e:
        logger.error(f"Erro: {e}")
    finally:
        bot.shutdown()

if __name__ == "__main__":
    main()