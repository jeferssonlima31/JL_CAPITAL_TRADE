#!/usr/bin/env python3
"""
SISTEMA DE TRADING AUTOMÁTICO - CONTA DEMO
JL CAPITAL TRADE
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from integration_system import IntegrationSystem
from aggressive_feature_compatibility import AggressiveFeatureCompatibility
import logging
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Carrega configurações do .env
load_dotenv()

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTradingSystem:
    def __init__(self):
        self.system = IntegrationSystem()
        self.connected = False
        self.account_info = None
        self.open_positions = []
        
        # Configurações do .env
        self.mt5_login = int(os.getenv('MT5_LOGIN', 3263303))
        self.mt5_password = os.getenv('MT5_PASSWORD', '!rH5UiSb')
        self.mt5_server = os.getenv('MT5_SERVER', 'Just2Trade-MT5')
        
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', 0.02))
        self.max_trades = int(os.getenv('MAX_TRADES', 5))
        self.stop_loss_pips = int(os.getenv('STOP_LOSS_PIPS', 50))
        self.take_profit_pips = int(os.getenv('TAKE_PROFIT_PIPS', 100))
        
        # Símbolos para trading (Limitado a EURUSD conforme pedido)
        self.symbols = ['EURUSD']
        
        # Sessões de Mercado (Horário de Portugal)
        self.sessions = {
            'LONDRES': {'start': 8, 'end': 17},
            'EUA': {'start': 13, 'end': 22}
        }
        
        # Sistema de compatibilidade de features (AGRESSIVO)
        self.feature_compat = AggressiveFeatureCompatibility()
        
        # Features esperadas
        self.expected_features = [
            'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
            'returns', 'log_returns', 'ma5', 'ma20', 'ma50', 'ma200', 'bb_middle',
            'bb_std', 'bb_upper', 'bb_lower', 'bb_width', 'rsi', 'macd', 'macd_signal',
            'macd_hist', 'volatility', 'atr', 'momentum', 'roc', 'price_vs_ma', 'high_low_ratio'
        ]
        
        logger.info("⚙️ Sistema de Trading Automático Inicializado")
        logger.info(f"   📊 Risk per Trade: {self.risk_per_trade*100}%")
        logger.info(f"   🔢 Max Trades: {self.max_trades}")
        logger.info(f"   ⚠️ Stop Loss: {self.stop_loss_pips} pips")
        logger.info(f"   🎯 Take Profit: {self.take_profit_pips} pips")
    
    def connect_to_mt5(self):
        """Conecta ao MT5"""
        try:
            if not mt5.initialize():
                logger.error(f"❌ Falha ao inicializar MT5: {mt5.last_error()}")
                return False
            
            authorized = mt5.login(
                login=self.mt5_login,
                password=self.mt5_password,
                server=self.mt5_server,
                timeout=30000
            )
            
            if authorized:
                self.account_info = mt5.account_info()
                self.connected = True
                
                logger.info("✅ CONECTADO À CONTA DEMO!")
                logger.info(f"   👤 {self.account_info.name}")
                logger.info(f"   💰 Saldo: ${self.account_info.balance:.2f}")
                logger.info(f"   📈 Equity: ${self.account_info.equity:.2f}")
                
                return True
            else:
                logger.error(f"❌ Falha no login: {mt5.last_error()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro na conexão: {e}")
            return False
    
    def load_ml_system(self):
        """Carrega sistema de ML"""
        logger.info("📦 Carregando sistema de ML...")
        
        if not self.system.load_best_model():
            logger.error("❌ Falha ao carregar modelo de ML!")
            return False
        
        logger.info("✅ Sistema de ML carregado com sucesso!")
        return True
    
    def get_market_data(self, symbol, timeframe=mt5.TIMEFRAME_H1, count=500):
        """Obtém dados de mercado"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter {symbol}: {e}")
            return None
    
    def calculate_features(self, df):
        """Calcula features para o modelo agressivo"""
        if df is None or len(df) < 200:
            return None
        
        # Usa o conversor de features agressivo
        features_df = self.feature_compat.convert_to_aggressive_format(df)
        
        if features_df is None or features_df.empty:
            return None
        
        # Pega a última linha (dados mais recentes) como DataFrame
        latest_features = features_df.iloc[[-1]]  # Mantém como DataFrame com índice
        
        return latest_features
    
    def calculate_position_size(self, symbol, stop_loss_pips):
        """Calcula tamanho da posição baseado no risco"""
        if not self.account_info:
            return 0.1  # tamanho padrão
        
        # Obtém informações do símbolo
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.1
        
        # Calcula risco por trade
        risk_amount = self.account_info.balance * self.risk_per_trade
        
        # Calcula valor do pip
        pip_value = symbol_info.trade_tick_value
        
        # Calcula tamanho do lote
        if pip_value > 0 and stop_loss_pips > 0:
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            # Ajusta para tamanhos padrão
            lot_size = max(0.01, min(lot_size, 10.0))  # entre 0.01 e 10 lotes
            lot_size = round(lot_size, 2)
            return lot_size
        
        return 0.1  # tamanho padrão
    
    def execute_trade(self, symbol, signal, confidence):
        """Executa uma ordem de trading"""
        if not self.connected:
            return False
        
        # Verifica se já tem posição aberta neste símbolo
        positions = mt5.positions_get(symbol=symbol)
        if positions and len(positions) > 0:
            logger.info(f"⏭️ Posição já aberta em {symbol}")
            return False
        
        # Obtém preço atual
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"❌ Símbolo {symbol} não encontrado")
            return False
        
        # Define direção da ordem
        if signal == 1:  # COMPRA
            order_type = mt5.ORDER_TYPE_BUY
            price = symbol_info.ask
            sl = price - self.stop_loss_pips * symbol_info.point
            tp = price + self.take_profit_pips * symbol_info.point
        else:  # VENDA
            order_type = mt5.ORDER_TYPE_SELL
            price = symbol_info.bid
            sl = price + self.stop_loss_pips * symbol_info.point
            tp = price - self.take_profit_pips * symbol_info.point
        
        # Calcula tamanho do lote
        lot_size = self.calculate_position_size(symbol, self.stop_loss_pips)
        
        # Prepara ordem
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 123456,
            "comment": f"AutoTrade Conf: {confidence:.2%}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Envia ordem
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            action = "COMPRA" if signal == 1 else "VENDA"
            logger.info(f"✅ {action} {symbol} | Lots: {lot_size} | Preço: {price:.5f}")
            logger.info(f"   ⚠️ SL: {sl:.5f} | 🎯 TP: {tp:.5f} | Conf: {confidence:.2%}")
            return True
        else:
            logger.error(f"❌ Erro na ordem: {result.comment}")
            return False
    
    def analyze_symbol(self, symbol):
        """Analisa um símbolo e retorna sinal"""
        try:
            # Obtém dados
            market_data = self.get_market_data(symbol)
            if market_data is None:
                return None
            
            # Calcula features
            features = self.calculate_features(market_data)
            if features is None:
                return None
            
            # Faz previsão
            prediction = self.system.predict(features)
            
            if prediction:
                return {
                    'symbol': symbol,
                    'signal': prediction['prediction'][0],
                    'confidence': prediction['confidence'][0],
                    'price': market_data.iloc[-1]['close']
                }
            
        except Exception as e:
            logger.error(f"Erro analisando {symbol}: {e}")
        
        return None
    
    def is_market_open(self):
        """Verifica se o mercado está em uma das sessões permitidas (Londres ou EUA)"""
        now = datetime.now()
        hour = now.hour
        
        # Verifica se é dia de semana (0=Segunda, 4=Sexta)
        if now.weekday() > 4:
            return False, "Final de semana"
            
        is_london = self.sessions['LONDRES']['start'] <= hour < self.sessions['LONDRES']['end']
        is_usa = self.sessions['EUA']['start'] <= hour < self.sessions['EUA']['end']
        
        if is_london and is_usa:
            return True, "Sobreposição Londres/EUA 🇬🇧🇺🇸"
        elif is_london:
            return True, "Sessão de Londres 🇬🇧"
        elif is_usa:
            return True, "Sessão dos EUA 🇺🇸"
            
        return False, "Fora das sessões operacionais"

    def run_trading_cycle(self):
        """Executa um ciclo completo de trading"""
        if not self.connected:
            return
            
        # Verifica horário de mercado
        market_open, session_name = self.is_market_open()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"🔄 CICLO DE TRADING - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"🌐 Status: {session_name}")
        logger.info("=" * 70)
        
        if not market_open:
            logger.info(f"💤 {session_name}. Aguardando abertura das sessões...")
            return 0
        
        signals = []
        
        # Analisa todos os símbolos
        for symbol in self.symbols:
            signal = self.analyze_symbol(symbol)
            if signal:
                signals.append(signal)
            time.sleep(0.5)  # Pequena pausa
        
        # Filtra sinais com boa confiança
        good_signals = [s for s in signals if s['confidence'] > 0.6]
        
        # Executa trades
        executed_trades = 0
        for signal in good_signals:
            if executed_trades >= self.max_trades:
                logger.info("⏹️ Limite máximo de trades atingido")
                break
                
            if self.execute_trade(signal['symbol'], signal['signal'], signal['confidence']):
                executed_trades += 1
                time.sleep(1)  # Pausa entre trades
        
        # Log de resumo
        logger.info(f"\n📊 RESUMO: {len(signals)} sinais | {len(good_signals)} bons | {executed_trades} executados")
        
        return executed_trades
    
    def monitor_positions(self):
        """Monitora posições abertas"""
        positions = mt5.positions_get()
        
        if positions:
            logger.info(f"\n📋 POSIÇÕES ABERTAS: {len(positions)}")
            for pos in positions:
                profit = pos.profit
                status = "🟢" if profit >= 0 else "🔴"
                logger.info(f"   {status} {pos.symbol} | Profit: ${profit:.2f}")
        
        return len(positions)
    
    def shutdown(self):
        """Fecha conexão"""
        if self.connected:
            mt5.shutdown()
            logger.info("✅ Conexão MT5 fechada")

def main():
    """Função principal - Inicia trading automático"""
    trader = AutoTradingSystem()
    
    try:
        # Conecta ao MT5
        if not trader.connect_to_mt5():
            return
        
        # Carrega sistema ML
        if not trader.load_ml_system():
            trader.shutdown()
            return
        
        logger.info("\n🎯 INICIANDO TRADING AUTOMÁTICO!")
        logger.info("💡 Pressione Ctrl+C para parar")
        
        # Loop principal de trading
        cycle_count = 0
        while True:
            cycle_count += 1
            
            # Executa ciclo de trading
            trades_executed = trader.run_trading_cycle()
            
            # Monitora posições
            open_positions = trader.monitor_positions()
            
            # Espera para próximo ciclo (5 minutos)
            logger.info(f"\n⏰ Próximo ciclo em 5 minutos... (Ciclo {cycle_count})")
            time.sleep(300)  # 5 minutos
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Trading automático interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro no trading automático: {e}")
    finally:
        trader.shutdown()

if __name__ == "__main__":
    main()