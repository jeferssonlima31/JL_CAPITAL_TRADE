#!/usr/bin/env python3
"""
TESTE EM CONTA DEMO - JL CAPITAL TRADE
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from integration_system import IntegrationSystem
import logging
import time
from datetime import datetime

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_demo():
    """Conecta à conta demo"""
    logger.info("Conectando à conta demo...")
    
    # Credenciais da conta demo (substituir pelas suas)
    demo_account = {
        'login': 3263303,           # Seu login demo
        'password': '!rH5UiSb',     # Sua senha demo  
        'server': 'Just2Trade-MT5',
        'timeout': 60000,
        'portable': False
    }
    
    # Tenta conectar
    if not mt5.initialize():
        logger.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
        return False
    
    # Login na conta demo
    authorized = mt5.login(
        login=demo_account['login'],
        password=demo_account['password'],
        server=demo_account['server'],
        timeout=demo_account['timeout'],
        portable=demo_account['portable']
    )
    
    if authorized:
        account_info = mt5.account_info()
        logger.info(f"✅ CONECTADO À CONTA DEMO!")
        logger.info(f"   Login: {account_info.login}")
        logger.info(f"   Nome: {account_info.name}")
        logger.info(f"   Saldo: ${account_info.balance:.2f}")
        logger.info(f"   Equity: ${account_info.equity:.2f}")
        logger.info(f"   Server: {account_info.server}")
        return True
    else:
        logger.error(f"Falha no login: {mt5.last_error()}")
        return False

def get_current_market_data():
    """Obtém dados de mercado atuais"""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    
    current_data = {}
    for symbol in symbols:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            current_data[symbol] = df
            logger.info(f"{symbol}: {len(df)} candles coletados")
        else:
            logger.warning(f"Falha ao coletar {symbol}")
    
    return current_data

def simulate_trading(system, market_data):
    """Simula operações de trading"""
    logger.info("\n🎯 SIMULAÇÃO DE TRADING NA CONTA DEMO")
    logger.info("=" * 50)
    
    results = []
    
    for symbol, data in market_data.items():
        logger.info(f"\n📊 Analisando {symbol}...")
        
        # Pega o candle mais recente
        latest_candle = data.iloc[-1]
        
        # Prepara features para previsão
        features = {
            'open': latest_candle['open'],
            'high': latest_candle['high'], 
            'low': latest_candle['low'],
            'close': latest_candle['close'],
            'tick_volume': latest_candle['tick_volume'],
            'spread': latest_candle['spread'],
            'real_volume': latest_candle['real_volume'] if 'real_volume' in data.columns else 0
        }
        
        # Faz previsão
        prediction = system.predict(pd.DataFrame([features]))
        
        if prediction:
            signal = "COMPRA" if prediction['prediction'][0] == 1 else "VENDA"
            confidence = prediction['confidence'][0]
            
            logger.info(f"   📈 Previsão: {signal}")
            logger.info(f"   🎯 Confiança: {confidence:.2%}")
            logger.info(f"   💰 Preço atual: {latest_candle['close']:.5f}")
            
            results.append({
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'price': latest_candle['close'],
                'time': latest_candle['time']
            })
    
    return results

def main():
    """Função principal"""
    logger.info("=" * 60)
    logger.info("🚀 TESTE EM CONTA DEMO - JL CAPITAL TRADE")
    logger.info("=" * 60)
    
    # Conecta à conta demo
    if not connect_to_demo():
        return
    
    # Carrega sistema de integração
    logger.info("\n📦 Carregando sistema de ML...")
    system = IntegrationSystem()
    if not system.load_best_model():
        logger.error("Falha ao carregar modelo!")
        mt5.shutdown()
        return
    
    # Obtém dados de mercado
    logger.info("\n🌐 Coletando dados de mercado...")
    market_data = get_current_market_data()
    
    if not market_data:
        logger.error("Nenhum dado de mercado disponível!")
        mt5.shutdown()
        return
    
    # Simula trading
    trading_results = simulate_trading(system, market_data)
    
    # Exibe resumo
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMO DO TESTE DEMO")
    logger.info("=" * 60)
    
    for result in trading_results:
        status = "✅" if result['confidence'] > 0.6 else "⚠️"
        logger.info(f"{status} {result['symbol']}: {result['signal']} (Conf: {result['confidence']:.2%}, Preço: {result['price']:.5f})")