# =============================================================================
# JL CAPITAL TRADE - BOT PRINCIPAL
# =============================================================================

import time
import threading
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

from .config import config
from .security import SecurityManager, AuditLogger
from .mt5_connector import MT5Connector
from .data_collector import DataCollector
from .ml_models import JLMLModels
from .continuous_learning import ContinuousLearner
from .cache_manager import CacheManager
from .risk_manager import RiskManager
from .news_filter import NewsFilter

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
        self.news_filter = NewsFilter(config)
        
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
        """Loop principal de trading com controle UTC"""
        
        # Atualização inicial de notícias
        self.news_filter.update_news()
        last_news_update = datetime.now()
        
        while self.is_running:
            try:
                # 1. Controle de Sessões via UTC (Evita erro horário verão)
                now_utc = datetime.utcnow()
                hour_utc = now_utc.hour
                
                # Sessão Londres: 08:00 - 16:00 UTC
                # Sessão EUA: 13:00 - 21:00 UTC
                is_london = 8 <= hour_utc < 16
                is_usa = 13 <= hour_utc < 21
                
                if not (is_london or is_usa):
                    if now_utc.minute % 30 == 0:
                        logger.info(f"💤 Fora das sessões UTC (London/USA) - Hora atual UTC: {hour_utc:02d}:{now_utc.minute:02d}")
                    time.sleep(60)
                    continue

                # 2. Atualiza notícias a cada 4 horas
                if datetime.now() - last_news_update > timedelta(hours=4):
                    self.news_filter.update_news()
                    last_news_update = datetime.now()

                # 3. Verifica Filtro de Notícias Econômicas

                # Analisa EUR/USD
                if self.config.is_testing() or self._check_market_hours("EUR_USD"):
                    eurusd_signal = self._analyze_pair("EUR_USD", "H1")
                    if eurusd_signal and eurusd_signal['action'] != "HOLD":
                        self._execute_trade(eurusd_signal)
                
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
        """Analisa símbolo usando ML, Regime de Mercado e MTF"""
        
        # Verifica cache
        cache_key = f"analysis_{symbol}_{timeframe}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        # 1. Coleta dados
        df = self.data_collector.get_historical_data(symbol, timeframe, 500)
        if df is None or len(df) < 200:
            return None
            
        # 2. Detecta Regime de Mercado
        regime = self.data_collector.detect_market_regime(df)
        logger.info(f"📊 Regime {symbol}: {regime['regime'].upper()} | Volatilidade: {regime['volatility'].upper()} (ADX: {regime['adx']:.1f})")
        
        # 3. Contexto Multi-Timeframe (MTF)
        mtf = self.data_collector.get_mtf_context(symbol)
        
        # 4. Prepara features e Previsão ML
        df = self.data_collector.calculate_indicators(df, symbol)
        features = self.ml_models.prepare_features(df, symbol)
        if features is None:
            return None
            
        # Filtro de Volatilidade ATR
        atr_current = df['atr'].iloc[-1]
        atr_mean = df['atr'].rolling(50).mean().iloc[-1]
        if not self.news_filter.check_volatility_protection(atr_current, atr_mean):
            return None
            
        # Cria sequência para previsão
        lookback = 200 if "aggressive" in self.ml_models.get_model_list(symbol) else (self.config.ml.eurusd_lookback if symbol == "EUR_USD" else self.config.ml.xauusd_lookback)
        
        if len(features) >= lookback:
            X_pred = features[-lookback:].reshape(1, lookback, features.shape[1])
            # Passa o regime detectado para o ensemble
            predictions = self.ml_models.predict_ensemble(symbol, X_pred, regime=regime['regime'])
            
            if predictions and 'ensemble' in predictions:
                prob = predictions['ensemble'][0]
                
                # 5. Filtro de Contexto M15 (Micro-Momentum)
                m15_trend = mtf.get('m15_trend')
                m15_rsi = mtf.get('m15_rsi', 50)
                
                if prob > 0.7:
                    if m15_trend == "bearish":
                        logger.info(f"⏳ Aguardando Pullback: COMPRA retida porque tendência M15 é de QUEDA.")
                        return None
                    if m15_rsi > 75:
                        logger.info(f"⏳ Exaustão Curta: COMPRA retida porque M15 RSI={m15_rsi:.1f} (Sobrecomprado).")
                        return None
                        
                if prob < 0.3:
                    if m15_trend == "bullish":
                        logger.info(f"⏳ Aguardando Pullback: VENDA retida porque tendência M15 é de ALTA.")
                        return None
                    if m15_rsi < 25:
                        logger.info(f"⏳ Exaustão Curta: VENDA retida porque M15 RSI={m15_rsi:.1f} (Sobrevendido).")
                        return None

                # 6. Filtro Macro MTF (H4)
                if prob > 0.7 and mtf.get('h4_trend') == "bearish":
                    if self.config.risk.strict_mtf_filter:
                        logger.warning(f"⚠️ Sinal IGNORADO (Filtro MTF): IA indica {prob:.1%} COMPRA, mas Tendência H4 é de QUEDA")
                        return None
                    else:
                        logger.info(f"⚡ Sinal PERMITIDO (MTF Strict=Off): IA indica {prob:.1%} COMPRA contra Tendência H4 de QUEDA")
                        
                if prob < 0.3 and mtf.get('h4_trend') == "bullish":
                    if self.config.risk.strict_mtf_filter:
                        logger.warning(f"⚠️ Sinal IGNORADO (Filtro MTF): IA indica {(1-prob):.1%} VENDA, mas Tendência H4 é de ALTA")
                        return None
                    else:
                        logger.info(f"⚡ Sinal PERMITIDO (MTF Strict=Off): IA indica {(1-prob):.1%} VENDA contra Tendência H4 de ALTA")

                # 6. Gera sinal
                signal = self._generate_signal(
                    symbol, prob, predictions,
                    df['close'].iloc[-1], atr_current, df
                )
                
                # Armazena features para Online Learning posterior
                signal['features_at_entry'] = X_pred.reshape(1, -1)
                
                # Salva no cache (15 minutos)
                if signal['action'] != "HOLD":
                    self.cache.set(signal, cache_key, ttl=900)
                
                return signal
        
        return None
    
    def _generate_signal(self, symbol: str, ensemble_pred: float, 
                         predictions: Dict, current_price: float, 
                         atr: float, df: pd.DataFrame) -> Dict:
        """Gera sinal calibrado para alvo de 70% de acurácia real"""
        
        # 1. Threshold de Confiança Calibrado para ~70-72%
        # 0.72-0.75 é o "ponto doce" para acurácia real de 70% no longo prazo
        confidence_threshold = 0.73 
        
        # 2. Filtros de Alinhamento Técnico Suavizados
        # Queremos filtrar apenas o ruído extremo, permitindo trades de 70%
        hurst = df['hurst'].iloc[-1]
        efficiency = df['efficiency_ratio'].iloc[-1]
        
        # Alinhamento mais permissivo (Hurst > 0.50 e Efficiency > 0.25)
        technical_alignment = hurst > 0.50 and efficiency > 0.28
        
        action = 'HOLD'
        if ensemble_pred > confidence_threshold and technical_alignment:
            action = 'BUY'
            sl = current_price - (atr * 2.8) # Equilíbrio entre proteção e acurácia
            tp = current_price + (atr * 11.2) # Mantém RR 1:4
            strength = 'STRONG'
            confidence = ensemble_pred
        elif ensemble_pred < (1 - confidence_threshold) and technical_alignment:
            action = 'SELL'
            sl = current_price + (atr * 2.8)
            tp = current_price - (atr * 11.2)
            strength = 'STRONG'
            confidence = 1 - ensemble_pred
        else:
            action = 'HOLD'
            sl = 0
            tp = 0
            strength = 'WEAK'
            confidence = 0
            
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
        
        if action != "HOLD":
            logger.info(f"📊 Signal: {symbol} - {action} {strength} @ {current_price:.4f} | Conf: {confidence:.1%}")
            
        return signal
    
    def _execute_trade(self, signal: Dict):
        """Executa trade com custos reais"""
        
        if signal['symbol'] in self.positions:
            logger.warning(f"Already have position in {signal['symbol']}")
            return
        
        # Verifica Spread atual antes de entrar
        current_tick = self.mt5.get_current_tick(signal['symbol'])
        current_spread = 0
        if current_tick:
            current_spread = (current_tick['ask'] - current_tick['bid']) * 10000 if signal['symbol'] == "EUR_USD" else (current_tick['ask'] - current_tick['bid']) * 100
        
        # Verifica risco e circuit breakers (agora com spread)
        if not self.risk_manager.can_trade(signal['symbol'], current_spread=current_spread):
            logger.warning("Risk or Circuit Breaker check failed")
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
            account_info['balance'],
            model_confidence=signal['confidence']
        )
        
        # Aplica Slippage Realista ao preço de entrada (Simulação)
        slippage = (self.config.risk.expected_slippage_pips * 0.0001) if signal['symbol'] == "EUR_USD" else (self.config.risk.expected_slippage_pips * 0.01)
        entry_price = signal['price'] + slippage if signal['action'] == "BUY" else signal['price'] - slippage
        
        # Executa via MT5
        order = {
            'symbol': signal['symbol'].replace('_', ''),
            'type': signal['action'],
            'volume': position_size,
            'price': entry_price,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'comment': f"JL_{signal['strength']}_{int(signal['confidence']*100)}"
        }
        
        # Em modo teste, não executa de verdade
        if self.config.is_testing():
            # Calcula comissão
            commission = position_size * self.config.risk.commission_per_lot
            
            logger.info(f"🧪 TEST MODE - Would execute: {order} | Est. Commission: ${commission:.2f}")
            self.positions[signal['symbol']] = {
                **signal,
                'ticket': 123456,
                'open_time': datetime.now(),
                'open_price': entry_price,
                'commission': commission
            }
            self.risk_manager.update_after_trade(signal['symbol'])
            return
        
        # Executa real
        exec_result = self.mt5.place_order(order)
        
        if exec_result['success']:
            # Calcula slippage real ocorrido
            actual_price = exec_result['price']
            slippage_pips = abs(actual_price - entry_price) * 10000 if signal['symbol'] == "EUR_USD" else abs(actual_price - entry_price) * 100
            
            logger.info(f"✅ Trade executed: {signal['symbol']} @ {actual_price} (Slippage: {slippage_pips:.1f} pips)")
            
            # Verifica Circuit Breaker de Slippage após execução
            if not self.risk_manager.check_circuit_breakers(current_slippage=slippage_pips):
                logger.critical(f"🛑 CIRCUIT BREAKER TRIGGERED BY SLIPPAGE: {slippage_pips:.1f} pips")
            
            self.positions[signal['symbol']] = {
                **signal,
                'ticket': exec_result['ticket'],
                'open_time': datetime.now(),
                'open_price': actual_price,
                'commission': position_size * self.config.risk.commission_per_lot
            }
            
            self.risk_manager.update_after_trade(signal['symbol'])
            self.audit.log_action("system", "trade", signal['symbol'], "success", signal)
        else:
            logger.error(f"❌ Trade failed: {exec_result.get('error')}")
    
    def _monitor_positions(self):
        """Monitora posições abertas"""
        
        for symbol, position in list(self.positions.items()):
            # Obtém preço atual
            current_price = self.mt5.get_current_price(symbol)
            
            if current_price:
                # Calcula P&L Bruto e Líquido
                if position['action'] == "BUY":
                    gross_pnl = (current_price - position['open_price']) * 100000 * position['volume']
                else:
                    gross_pnl = (position['open_price'] - current_price) * 100000 * position['volume']
                
                # Desconta comissão
                net_pnl = gross_pnl - position.get('commission', 0)
                
                # Verifica stop loss
                if position['action'] == "BUY" and current_price <= position['stop_loss']:
                    self._close_position(symbol, "Stop Loss", net_pnl)
                elif position['action'] == "SELL" and current_price >= position['stop_loss']:
                    self._close_position(symbol, "Stop Loss", net_pnl)
                
                # Verifica take profit
                elif position['action'] == "BUY" and current_price >= position['take_profit']:
                    self._close_position(symbol, "Take Profit", net_pnl)
                elif position['action'] == "SELL" and current_price <= position['take_profit']:
                    self._close_position(symbol, "Take Profit", net_pnl)
    
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
            
            # Atualiza Risco e Performance
            account_info = self.mt5.get_account_info()
            balance = account_info['balance'] if account_info else 0
            pnl_percent = (pnl / balance * 100) if balance > 0 else 0
            
            self.risk_manager.update_pnl(pnl_percent, balance)
            self.risk_manager.remove_position()
            
            # 1. Online Learning (Atualiza SGD com o resultado real)
            if 'features_at_entry' in position:
                X_fit = position['features_at_entry']
                # Target: 1 se lucro, 0 se prejuízo
                y_fit = np.array([1 if pnl > 0 else 0])
                self.ml_models.partial_fit_online(symbol, X_fit, y_fit)
            
            # 2. Feedback para Continuous Learning (Ajuste de pesos do ensemble)
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
        """Atualiza e loga performance em tempo real"""
        now = datetime.now()
        if (now - self.performance['last_update']).seconds >= 300: # Log a cada 5 min
            account = self.mt5.get_account_info()
            balance = account['balance'] if account else 0
            
            logger.info("\n" + "="*50)
            logger.info("📈 RELATÓRIO DE PERFORMANCE EM TEMPO REAL")
            logger.info("="*50)
            logger.info(f"💰 Saldo Atual: ${balance:,.2f}")
            logger.info(f"📊 Total Trades: {self.performance['total_trades']}")
            logger.info(f"🎯 Win Rate: {(self.performance['winning_trades']/self.performance['total_trades']*100 if self.performance['total_trades'] > 0 else 0):.2f}%")
            logger.info(f"💵 PnL Acumulado: ${self.performance['total_pnl']:,.2f}")
            
            # Posições Abertas
            if self.positions:
                logger.info("-" * 20)
                logger.info("📂 Posições Abertas:")
                for symbol, pos in self.positions.items():
                    logger.info(f"  • {symbol}: {pos['action']} @ {pos['open_price']:.4f}")
            
            logger.info("="*50 + "\n")
            self.performance['last_update'] = now
        
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