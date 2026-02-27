# =============================================================================
# JL CAPITAL TRADE - BOT PRINCIPAL
# =============================================================================

import time
import threading
from datetime import datetime
import logging
from typing import Dict, Optional
import numpy as np

from .config import config
from .security import SecurityManager, AuditLogger
from .mt5_connector import MT5Connector
from .data_collector import DataCollector
from .ml_models import JLMLModels
from .continuous_learning import ContinuousLearner
from .cache_manager import CacheManager
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)

class JLTradingBot:
    """Bot principal da JL Capital Trade"""
    
    def __init__(self):
        # Configuração
        self.config = config
        
        # Segurança
        self.security = SecurityManager(config)
        self.audit = AuditLogger()
        
        # Cache
        self.cache = CacheManager(config)
        
        # Componentes
        self.mt5 = MT5Connector(config)
        self.data_collector = DataCollector(config, self.mt5)
        self.data_collector.set_cache(self.cache)
        self.risk_manager = RiskManager(config)
        
        # ML e aprendizado
        self.continuous_learner = ContinuousLearner(
            config, None, self.data_collector
        )
        self.ml_models = JLMLModels(config, self.continuous_learner)
        
        # Atualiza referência circular
        self.continuous_learner.ml_models = self.ml_models
        
        # Estado
        self.is_running = False
        self.positions = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'last_update': datetime.now()
        }
        
        # Threads
        self.main_thread = None
        self.monitor_thread = None
        
        logger.info("🚀 JL Capital Trade Bot initialized")
        self.audit.log_action("system", "init", "bot", "success")
    
    def start(self):
        """Inicia o bot"""
        if self.is_running:
            logger.warning("Bot already running")
            return
        
        logger.info("▶️ Starting JL Capital Trade Bot...")
        
        # Conecta ao MT5
        if not self.mt5.connect():
            logger.error("Failed to connect to MT5")
            return
        
        # Inicia aprendizado contínuo
        self.continuous_learner.start_learning()
        
        # Inicia threads
        self.is_running = True
        self.main_thread = threading.Thread(target=self._main_loop)
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        
        self.main_thread.daemon = True
        self.monitor_thread.daemon = True
        
        self.main_thread.start()
        self.monitor_thread.start()
        
        logger.info("✅ Bot started successfully")
        self.audit.log_action("system", "start", "bot", "success")
    
    def stop(self):
        """Para o bot"""
        logger.info("⏹️ Stopping JL Capital Trade Bot...")
        
        self.is_running = False
        self.continuous_learner.stop_learning_process()
        
        # Fecha posições se necessário
        self._close_all_positions()
        
        # Desconecta
        self.mt5.disconnect()
        
        logger.info("✅ Bot stopped")
        self.audit.log_action("system", "stop", "bot", "success")
    
    def _main_loop(self):
        """Loop principal de trading"""
        
        while self.is_running:
            try:
                # Verifica sessões (Londres/EUA - Horário de Portugal)
                now = datetime.now()
                hour = now.hour
                is_london = 8 <= hour < 17
                is_usa = 13 <= hour < 22
                
                if not (is_london or is_usa):
                    if now.minute % 15 == 0: # Log a cada 15 min fora de hora
                        logger.info(f"💤 Fora das sessões operacionais ({hour:02d}:{now.minute:02d})")
                    time.sleep(60)
                    continue

                # Analisa EUR/USD
                if self.config.is_testing() or self._check_market_hours("EUR_USD"):
                    eurusd_signal = self._analyze_pair("EUR_USD", "H1")
                    if eurusd_signal and eurusd_signal['action'] != "HOLD":
                        self._execute_trade(eurusd_signal)
                
                # Analisa XAU/USD
                if self.config.is_testing() or self._check_market_hours("XAU_USD"):
                    xauusd_signal = self._analyze_pair("XAU_USD", "H1")
                    if xauusd_signal and xauusd_signal['action'] != "HOLD":
                        self._execute_trade(xauusd_signal)
                
                # Pausa entre análises
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(5)
    
    def _monitor_loop(self):
        """Loop de monitoramento de posições"""
        
        while self.is_running:
            try:
                self._monitor_positions()
                self._update_performance()
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(5)
    
    def _check_market_hours(self, symbol: str) -> bool:
        """Verifica se é bom horário para trading"""
        market_hours = self.data_collector.get_market_hours(symbol)
        return market_hours['is_optimal']
    
    def _analyze_pair(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Analisa par específico"""
        
        # Verifica cache
        cache_key = f"analysis_{symbol}_{timeframe}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Coleta dados (Aumentado para 500 para compatibilidade agressiva)
        df = self.data_collector.get_historical_data(symbol, timeframe, 500)
        if df is None:
            return None
        
        # Calcula indicadores
        df = self.data_collector.calculate_indicators(df, symbol)
        
        # Prepara features
        features = self.ml_models.prepare_features(df, symbol)
        if features is None:
            return None
        
        # Cria sequência para previsão (Lookback agressivo = 200)
        lookback = 200 if "aggressive" in self.ml_models.get_model_list(symbol) else (self.config.ml.eurusd_lookback if symbol == "EUR_USD" else self.config.ml.xauusd_lookback)
        
        if len(features) >= lookback:
            X_pred = features[-lookback:].reshape(1, lookback, features.shape[1])
            
            # Previsão ensemble
            predictions = self.ml_models.predict_ensemble(symbol, X_pred)
            
            if predictions and 'ensemble' in predictions:
                ensemble_pred = predictions['ensemble'][0]
                
                # Gera sinal
                signal = self._generate_signal(
                    symbol, ensemble_pred, predictions,
                    df['close'].iloc[-1], df['atr'].iloc[-1]
                )
                
                # Salva no cache (15 minutos)
                if signal['action'] != "HOLD":
                    self.cache.set(signal, cache_key, ttl=900)
                
                return signal
        
        return None
    
    def _generate_signal(self, symbol: str, ensemble_pred: float,
                         predictions: Dict, current_price: float,
                         atr: float) -> Dict:
        """Gera sinal de trading"""
        
        config = self.config.risk
        
        # Calcula confiança
        confidence = abs(ensemble_pred - 0.5) * 2
        
        # Decisão
        if ensemble_pred > self.config.ml.buy_threshold and confidence > self.config.ml.confidence_threshold:
            action = "BUY"
            strength = "STRONG" if ensemble_pred > 0.75 else "MODERATE"
            
            if symbol == "EUR_USD":
                sl = current_price - (config.default_sl_pips_eurusd * 0.0001)
                tp = current_price + (config.default_tp_pips_eurusd * 0.0001)
            else:
                sl = current_price - (config.default_sl_pips_xauusd * 0.1)
                tp = current_price + (config.default_tp_pips_xauusd * 0.1)
                
        elif ensemble_pred < self.config.ml.sell_threshold and confidence > self.config.ml.confidence_threshold:
            action = "SELL"
            strength = "STRONG" if ensemble_pred < 0.25 else "MODERATE"
            
            if symbol == "EUR_USD":
                sl = current_price + (config.default_sl_pips_eurusd * 0.0001)
                tp = current_price - (config.default_tp_pips_eurusd * 0.0001)
            else:
                sl = current_price + (config.default_sl_pips_xauusd * 0.1)
                tp = current_price - (config.default_tp_pips_xauusd * 0.1)
        else:
            action = "HOLD"
            strength = "WEAK"
            sl = 0
            tp = 0
        
        signal = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'strength': strength,
            'price': float(current_price),
            'stop_loss': float(sl),
            'take_profit': float(tp),
            'confidence': float(confidence),
            'ensemble_pred': float(ensemble_pred),
            'individual_predictions': {
                k: float(v[0]) for k, v in predictions.items() if k != 'ensemble'
            },
            'atr': float(atr)
        }
        
        logger.info(f"📊 Signal: {symbol} - {action} {strength} @ {current_price:.4f} | Conf: {confidence:.1%}")
        
        return signal
    
    def _execute_trade(self, signal: Dict):
        """Executa trade"""
        
        if signal['symbol'] in self.positions:
            logger.warning(f"Already have position in {signal['symbol']}")
            return
        
        # Verifica risco
        if not self.risk_manager.can_trade(signal['symbol']):
            logger.warning("Risk check failed")
            return
        
        # Obtém informações da conta
        account_info = self.mt5.get_account_info()
        if not account_info:
            logger.error("Could not get account info")
            return
        
        # Calcula tamanho da posição
        position_size = self.risk_manager.calculate_position_size(
            signal['symbol'],
            signal['price'],
            signal['atr'],
            account_info['balance']
        )
        
        # Executa via MT5
        order = {
            'symbol': signal['symbol'].replace('_', ''),
            'type': signal['action'],
            'volume': position_size,
            'price': signal['price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'comment': f"JL_{signal['strength']}_{int(signal['confidence']*100)}"
        }
        
        # Em modo teste, não executa de verdade
        if self.config.is_testing():
            logger.info(f"🧪 TEST MODE - Would execute: {order}")
            self.positions[signal['symbol']] = {
                **signal,
                'ticket': 123456,
                'open_time': datetime.now(),
                'open_price': signal['price']
            }
            self.risk_manager.update_after_trade(signal['symbol'])
            return
        
        # Executa real
        result = self.mt5.place_order(order)
        
        if result['success']:
            self.positions[signal['symbol']] = {
                **signal,
                'ticket': result['ticket'],
                'open_time': datetime.now(),
                'open_price': result['price']
            }
            
            self.risk_manager.update_after_trade(signal['symbol'])
            
            logger.info(f"✅ Trade executed: {signal['action']} {signal['symbol']} @ {result['price']}")
            self.audit.log_action("system", "trade", signal['symbol'], "success", signal)
        else:
            logger.error(f"❌ Trade failed: {result.get('error')}")
    
    def _monitor_positions(self):
        """Monitora posições abertas"""
        
        for symbol, position in list(self.positions.items()):
            # Obtém preço atual
            current_price = self.mt5.get_current_price(symbol)
            
            if current_price:
                # Calcula P&L
                if position['action'] == "BUY":
                    pnl = (current_price - position['open_price']) * 100000
                else:
                    pnl = (position['open_price'] - current_price) * 100000
                
                # Verifica stop loss
                if position['action'] == "BUY" and current_price <= position['stop_loss']:
                    self._close_position(symbol, "Stop Loss", pnl)
                elif position['action'] == "SELL" and current_price >= position['stop_loss']:
                    self._close_position(symbol, "Stop Loss", pnl)
                
                # Verifica take profit
                elif position['action'] == "BUY" and current_price >= position['take_profit']:
                    self._close_position(symbol, "Take Profit", pnl)
                elif position['action'] == "SELL" and current_price <= position['take_profit']:
                    self._close_position(symbol, "Take Profit", pnl)
    
    def _close_position(self, symbol: str, reason: str, pnl: float):
        """Fecha posição"""
        
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Fecha no MT5 (se não for teste)
            if not self.config.is_testing() and 'ticket' in position:
                self.mt5.close_position(position['ticket'])
            
            # Atualiza performance
            self.performance['total_trades'] += 1
            if pnl > 0:
                self.performance['winning_trades'] += 1
            self.performance['total_pnl'] += pnl
            
            # Atualiza risco
            self.risk_manager.update_pnl(pnl)
            self.risk_manager.remove_position()
            
            # Feedback para aprendizado
            self.continuous_learner.add_trade_outcome({
                'symbol': symbol,
                'prediction': position['ensemble_pred'],
                'confidence': position['confidence'],
                'model_used': 'ensemble',
                'pnl': pnl
            }, pnl)
            
            logger.info(f"✅ Position closed: {symbol} - {reason} - P&L: ${pnl:.2f}")
            
            # Remove da lista
            del self.positions[symbol]
    
    def _close_all_positions(self):
        """Fecha todas as posições"""
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, "Bot Shutdown", 0)
    
    def _update_performance(self):
        """Atualiza métricas de performance"""
        self.performance['last_update'] = datetime.now()
        
        if self.performance['total_trades'] > 0:
            win_rate = (self.performance['winning_trades'] / self.performance['total_trades']) * 100
            logger.info(f"📈 Performance - Trades: {self.performance['total_trades']}, "
                       f"Win Rate: {win_rate:.1f}%, Total P&L: ${self.performance['total_pnl']:.2f}")
    
    def get_status(self) -> Dict:
        """Retorna status atual do bot"""
        return {
            'running': self.is_running,
            'positions': len(self.positions),
            'performance': self.performance,
            'risk': self.risk_manager.get_status(),
            'mt5_connected': self.mt5.is_connected(),
            'models_loaded': len(self.ml_models.models['EUR_USD']) + len(self.ml_models.models['XAU_USD']),
            'cache_stats': self.cache.get_stats()
        }
    
    def analyze_pair(self, pair: str, timeframe: str = "H1") -> Optional[Dict]:
        """Método público para análise"""
        return self._analyze_pair(pair, timeframe)
    
    def execute_trade(self, trade_data: Dict) -> Dict:
        """Método público para executar trade"""
        
        signal = {
            'symbol': trade_data['symbol'],
            'action': trade_data['action'],
            'price': trade_data['price'],
            'stop_loss': trade_data.get('stop_loss', 0),
            'take_profit': trade_data.get('take_profit', 0),
            'confidence': trade_data.get('confidence', 0.5),
            'ensemble_pred': trade_data.get('prediction', 0.5),
            'atr': trade_data.get('atr', 0)
        }
        
        self._execute_trade(signal)
        
        return {
            'success': True,
            'message': f"Trade {trade_data['action']} executado para {trade_data['symbol']}",
            'ticket': self.positions.get(trade_data['symbol'], {}).get('ticket')
        }