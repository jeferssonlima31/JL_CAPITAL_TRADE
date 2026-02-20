#!/usr/bin/env python3
"""
BOT DE TRADING PARA CONTA DEMO - JL CAPITAL TRADE
Usa sistema de ML para fazer operações automáticas
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

class DemoTradingBot:
    def __init__(self):
        self.system = IntegrationSystem()
        self.connected = False
        self.account_info = None
        
    def connect_to_demo(self):
        """Conecta à conta demo"""
        try:
            # Credenciais da conta demo Lime Trading
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
                logger.info(f"   👤 {self.account_info.name}")
                logger.info(f"   💰 Saldo: ${self.account_info.balance:.2f}")
                logger.info(f"   📈 Equity: ${self.account_info.equity:.2f}")
                
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
    
    def get_market_data(self, symbol, timeframe=mt5.TIMEFRAME_H1, count=100):
        """Obtém dados de mercado"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"Nenhum dado para {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Adiciona features básicas
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter dados de {symbol}: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepara features para o modelo"""
        if df is None or len(df) < 50:
            return None
        
        # Pega o candle mais recente
        latest = df.iloc[-1].copy()
        
        # Calcula médias móveis
        latest['ma5'] = df['close'].tail(5).mean()
        latest['ma20'] = df['close'].tail(20).mean()
        latest['ma50'] = df['close'].tail(50).mean()
        
        # Calcula RSI simples
        gains = df['close'].diff().clip(lower=0).tail(14).mean()
        losses = (-df['close'].diff().clip(upper=0)).tail(14).mean()
        latest['rsi'] = 100 - (100 / (1 + gains/losses)) if losses != 0 else 50
        
        # Features básicas
        features = {
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'close': latest['close'],
            'tick_volume': latest['tick_volume'],
            'spread': latest['spread'],
            'ma5': latest['ma5'],
            'ma20': latest['ma20'], 
            'ma50': latest['ma50'],
            'rsi': latest['rsi'],
            'returns': latest['returns'] if not pd.isna(latest['returns']) else 0,
            'log_returns': latest['log_returns'] if not pd.isna(latest['log_returns']) else 0
        }
        
        return pd.DataFrame([features])
    
    def execute_demo_trade(self, symbol, signal, confidence, price):
        """Simula uma operação (apenas logging)"""
        
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
        
        logger.info(f"{color} {action} {symbol} | Conf: {confidence:.2%} | Preço: {price:.5f} | {status}")
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'price': price,
            'status': status,
            'timestamp': datetime.now()
        }
    
    def run_trading_session(self):
        """Executa uma sessão de trading"""
        if not self.connected:
            logger.error("Não conectado à conta demo!")
            return
        
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        results = []
        
        logger.info("\n" + "=" * 70)
        logger.info("🚀 SESSÃO DE TRADING - CONTA DEMO")
        logger.info("=" * 70)
        
        for symbol in symbols:
            try:
                # Obtém dados de mercado
                market_data = self.get_market_data(symbol)
                if market_data is None:
                    continue
                
                # Prepara features
                features = self.prepare_features(market_data)
                if features is None:
                    continue
                
                # Faz previsão
                prediction = self.system.predict(features)
                
                if prediction:
                    signal = prediction['prediction'][0]
                    confidence = prediction['confidence'][0]
                    current_price = market_data.iloc[-1]['close']
                    
                    # Executa operação demo
                    trade_result = self.execute_demo_trade(symbol, signal, confidence, current_price)
                    results.append(trade_result)
                    
                    # Pequena pausa entre símbolos
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Erro ao processar {symbol}: {e}")
                continue
        
        # Resumo da sessão
        logger.info("\n" + "=" * 70)
        logger.info("📊 RESUMO DA SESSÃO DE TRADING")
        logger.info("=" * 70)
        
        high_confidence = sum(1 for r in results if r['confidence'] > 0.7)
        medium_confidence = sum(1 for r in results if 0.6 <= r['confidence'] <= 0.7)
        low_confidence = sum(1 for r in results if r['confidence'] < 0.6)
        
        buys = sum(1 for r in results if r['action'] == 'COMPRA')
        sells = sum(1 for r in results if r['action'] == 'VENDA')
        
        logger.info(f"📈 Operações de COMPRA: {buys}")
        logger.info(f"📉 Operações de VENDA: {sells}")
        logger.info(f"🚀 Alta confiança: {high_confidence}")
        logger.info(f"🎯 Média confiança: {medium_confidence}")
        logger.info(f"⚠️ Baixa confiança: {low_confidence}")
        logger.info(f"💰 Saldo atual: ${self.account_info.balance:.2f}")
        
        return results
    
    def shutdown(self):
        """Fecha conexão"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("✅ Conexão MT5 fechada")

def main():
    """Função principal"""
    bot = DemoTradingBot()
    
    try:
        # Conecta à conta demo
        if not bot.connect_to_demo():
            return
        
        # Carrega sistema ML
        if not bot.load_ml_system():
            bot.shutdown()
            return
        
        # Executa sessão de trading
        results = bot.run_trading_session()
        
        logger.info("\n✅ Sessão de trading demo concluída!")
        
        # Mostra recomendações
        strong_buys = [r for r in results if r['action'] == 'COMPRA' and r['confidence'] > 0.7]
        strong_sells = [r for r in results if r['action'] == 'VENDA' and r['confidence'] > 0.7]
        
        if strong_buys:
            logger.info("\n🎯 FORTES SINAIS DE COMPRA:")
            for trade in strong_buys:
                logger.info(f"   🟢 {trade['symbol']} - Conf: {trade['confidence']:.2%}")
        
        if strong_sells:
            logger.info("\n🎯 FORTES SINAIS DE VENDA:")
            for trade in strong_sells:
                logger.info(f"   🔴 {trade['symbol']} - Conf: {trade['confidence']:.2%}")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Sessão interrompida pelo usuário")
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
    finally:
        bot.shutdown()

if __name__ == "__main__":
    main()