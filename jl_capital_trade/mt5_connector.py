# =============================================================================
# JL CAPITAL TRADE - CONECTOR METATRADER 5
# =============================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time
import threading
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

class MT5Connector:
    """Conector para MetaTrader 5"""
    
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.account_info = None
        self.heartbeat_thread = None
        self.stop_heartbeat = False
        
    def connect(self) -> bool:
        """Conecta ao MetaTrader 5 e inicia heartbeat"""
        try:
            # Inicializa MT5
            if not mt5.initialize(
                path=self.config.mt5.path,
                login=self.config.mt5.login,
                password=self.config.mt5.password,
                server=self.config.mt5.server,
                timeout=self.config.mt5.timeout
            ):
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False
            
            # Verifica conexão
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logger.error("MT5 account info not available")
                return False
            
            self.connected = True
            logger.info(f"✅ Connected to MT5 - Account: {self.account_info.login}")
            
            # Inicia Heartbeat
            self._start_heartbeat()
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def _start_heartbeat(self):
        """Inicia thread de verificação de conexão (Heartbeat)"""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
            
        self.stop_heartbeat = False
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        logger.info("💓 Heartbeat monitor started (1s check)")

    def _heartbeat_loop(self):
        """Loop de verificação de conexão"""
        while not self.stop_heartbeat:
            try:
                # Verifica se terminal ainda está respondendo
                if not mt5.terminal_info():
                    logger.critical("🚨 MT5 Terminal connection LOST!")
                    self.connected = False
                    # Tenta reconectar se configurado
                    self.connect()
                else:
                    self.connected = True
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self.connected = False
            
            time.sleep(1)

    def disconnect(self):
        """Desconecta do MT5 e para heartbeat"""
        self.stop_heartbeat = True
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
        except Exception as e:
            logger.error(f"MT5 disconnect error: {e}")
    
    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        return self.connected
    
    def is_connected(self) -> bool:
        """Verifica se está conectado ao MT5"""
        return self.connected and mt5.terminal_info() is not None

    def get_account_info(self) -> Optional[Dict]:
        """Retorna informações da conta"""
        if not self.connected:
            return None
        
        info = mt5.account_info()
        if info:
            return {
                'login': info.login,
                'balance': info.balance,
                'equity': info.equity,
                'margin': info.margin,
                'margin_free': info.margin_free,
                'currency': info.currency,
                'leverage': info.leverage
            }
        return None
    
    def get_historical_data(self, symbol: str, timeframe: str = "H1", 
                            count: int = 1000) -> Optional[pd.DataFrame]:
        """Obtém dados históricos"""
        if not self.connected:
            logger.error("MT5 not connected")
            return None
        
        # Mapeia timeframe
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        
        # Converte símbolo para formato MT5
        mt5_symbol = symbol.replace("_", "")
        
        try:
            rates = mt5.copy_rates_from_pos(mt5_symbol, tf_map[timeframe], 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {mt5_symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Garante que tick_volume e volume coexistam (MT5 usa tick_volume)
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            elif 'volume' in df.columns:
                df['tick_volume'] = df['volume']
            
            df.set_index('time', inplace=True)
            
            logger.info(f"✅ Got {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual"""
        if not self.connected:
            return None
        
        mt5_symbol = symbol.replace("_", "")
        tick = mt5.symbol_info_tick(mt5_symbol)
        
        if tick:
            return (tick.ask + tick.bid) / 2
        return None
    
    def get_current_spread(self, symbol: str) -> int:
        """Obtém spread atual em pontos"""
        if not self.connected:
            return 999
        
        mt5_symbol = symbol.replace("_", "")
        info = mt5.symbol_info(mt5_symbol)
        
        if info:
            return info.spread
        return 999
    
    def place_order(self, order: Dict) -> Dict:
        """Coloca ordem no mercado com verificações internas de Spread/Slippage"""
        if not self.connected:
            return {'success': False, 'error': 'MT5 not connected'}
        
        try:
            mt5_symbol = order['symbol'].replace("_", "")
            
            # 1. Verificação Interna de Spread (Proteção de Última Milha)
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick:
                spread = (tick.ask - tick.bid) * 10000 if "EURUSD" in mt5_symbol else (tick.ask - tick.bid) * 100
                max_allowed_spread = self.config.risk.max_spread_pips
                
                if spread > max_allowed_spread:
                    logger.warning(f"❌ Order Cancelled: Spread too high ({spread:.1f} > {max_allowed_spread})")
                    return {'success': False, 'error': f'High spread: {spread:.1f}'}

            # 2. Prepara request
            order_type = mt5.ORDER_TYPE_BUY if order['type'] == 'BUY' else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": order['volume'],
                "type": order_type,
                "price": order['price'],
                "sl": order.get('stop_loss', 0),
                "tp": order.get('take_profit', 0),
                "deviation": 10, # 10 points slippage tolerance
                "magic": 234000,
                "comment": order.get('comment', 'JL_Capital'),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # 3. Envia ordem
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f"Order failed: {result.comment} (Code: {result.retcode})",
                    'ticket': None
                }
            
            # 4. Monitoramento de Slippage Real
            actual_price = result.price
            requested_price = order['price']
            slippage = abs(actual_price - requested_price) * 10000 if "EURUSD" in mt5_symbol else abs(actual_price - requested_price) * 100
            
            logger.info(f"✅ Order executed: {order['type']} {order['symbol']} @ {actual_price} (Slippage: {slippage:.1f} pips)")
            
            return {
                'success': True,
                'error': None,
                'ticket': result.order,
                'price': actual_price,
                'volume': result.volume,
                'slippage': slippage
            }
            
        except Exception as e:
            logger.error(f"Order error: {e}")
            return {'success': False, 'error': str(e), 'ticket': None}
    
    def close_position(self, ticket: int) -> bool:
        """Fecha uma posição específica"""
        if not self.connected:
            return False
        
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            position = position[0]
            
            # Prepara ordem de fechamento
            order_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).bid,
                "deviation": 10,
                "magic": 234000,
                "comment": "Close by JL Capital",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            success = result.retcode == mt5.TRADE_RETCODE_DONE
            
            if success:
                logger.info(f"✅ Position {ticket} closed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def modify_position(self, ticket: int, stop_loss: float, 
                        take_profit: float) -> bool:
        """Modifica stop loss e take profit"""
        if not self.connected:
            return False
        
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": stop_loss,
                "tp": take_profit
            }
            
            result = mt5.order_send(request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False
    
    def get_open_positions(self) -> List[Dict]:
        """Retorna posições abertas"""
        if not self.connected:
            return []
        
        positions = mt5.positions_get()
        result = []
        
        if positions:
            for pos in positions:
                result.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == 0 else 'SELL',
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'current_price': pos.price_current,
                    'stop_loss': pos.sl,
                    'take_profit': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'comment': pos.comment
                })
        
        return result