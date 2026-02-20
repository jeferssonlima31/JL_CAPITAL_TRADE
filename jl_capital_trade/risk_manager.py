# =============================================================================
# JL CAPITAL TRADE - GERENCIADOR DE RISCO
# =============================================================================

from datetime import datetime, date
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class RiskManager:
    """Gerenciador de risco para trading"""
    
    def __init__(self, config):
        self.config = config
        self.daily_trades = {}
        self.daily_pnl = 0
        self.last_update = datetime.now()
        self.positions_count = 0
        
    def can_trade(self, symbol: str) -> bool:
        """Verifica se pode fazer nova operação"""
        
        today = date.today().isoformat()
        
        # 1. Verifica limite diário de perdas
        if self.daily_pnl < -self.config.risk.max_daily_loss * 100:  # Convertendo para USD
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        # 2. Verifica número de posições por símbolo
        symbol_positions = self.daily_trades.get(symbol, {}).get(today, 0)
        if symbol_positions >= self.config.risk.max_positions:
            logger.warning(f"Max positions reached for {symbol}")
            return False
        
        # 3. Verifica total de posições
        if self.positions_count >= self.config.risk.max_positions:
            logger.warning(f"Max total positions reached: {self.positions_count}")
            return False
        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, 
                                atr: float, account_balance: float) -> float:
        """Calcula tamanho da posição baseado em risco"""
        
        # Risco por trade (% da conta)
        risk_percent = self.config.risk.max_risk_per_trade / 100
        
        # Valor em risco
        risk_amount = account_balance * risk_percent
        
        # Distância do stop loss em pips
        if symbol == "EUR_USD":
            sl_pips = self.config.risk.default_sl_pips_eurusd
            pip_value = 10  # $10 por pip para 1 lote padrão
        else:
            sl_pips = self.config.risk.default_sl_pips_xauusd
            pip_value = 1  # $1 por pip para XAU/USD
        
        # Tamanho da posição
        position_size = risk_amount / (sl_pips * pip_value)
        
        # Ajuste para conta cent (Exness Standard Cent)
        if self._is_cent_account():
            position_size = position_size * 100  # Ajuste para lotes cent
        
        # Arredonda para tamanhos padronizados
        position_size = self._round_to_standard_lot(position_size)
        
        return position_size
    
    def _is_cent_account(self) -> bool:
        """Verifica se é conta cent (Exness)"""
        # Implementar detecção automática se possível
        return True  # Por padrão, assume conta cent para segurança
    
    def _round_to_standard_lot(self, size: float) -> float:
        """Arredonda para tamanhos de lote padronizados"""
        
        if size < 0.01:
            return 0.01  # Micro lote
        elif size < 0.1:
            return round(size * 100) / 100  # 0.01 increments
        elif size < 1:
            return round(size * 10) / 10  # 0.1 increments
        else:
            return round(size)  # Lotes inteiros
    
    def update_after_trade(self, symbol: str, pnl: float = 0):
        """Atualiza contadores após trade"""
        
        today = date.today().isoformat()
        
        if symbol not in self.daily_trades:
            self.daily_trades[symbol] = {}
        
        if today not in self.daily_trades[symbol]:
            self.daily_trades[symbol][today] = 0
        
        self.daily_trades[symbol][today] += 1
        self.positions_count += 1
        
        if pnl != 0:
            self.daily_pnl += pnl
    
    def update_pnl(self, profit_loss: float):
        """Atualiza P&L diário"""
        self.daily_pnl += profit_loss
        self.last_update = datetime.now()
    
    def remove_position(self):
        """Remove uma posição do contador"""
        self.positions_count = max(0, self.positions_count - 1)
    
    def calculate_risk_reward(self, entry: float, stop_loss: float, 
                              take_profit: float) -> float:
        """Calcula relação risco/retorno"""
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    def validate_stop_loss(self, symbol: str, stop_loss: float, 
                           current_price: float) -> bool:
        """Valida se stop loss está em nível aceitável"""
        
        # Calcula distância em pips
        if symbol == "EUR_USD":
            distance_pips = abs(current_price - stop_loss) * 10000
            min_pips = 10
            max_pips = 50
        else:
            distance_pips = abs(current_price - stop_loss) * 100
            min_pips = 30
            max_pips = 150
        
        if distance_pips < min_pips:
            logger.warning(f"Stop loss too tight: {distance_pips:.0f} pips")
            return False
        
        if distance_pips > max_pips:
            logger.warning(f"Stop loss too wide: {distance_pips:.0f} pips")
            return False
        
        return True
    
    def reset_daily(self):
        """Reseta contadores diários"""
        self.daily_pnl = 0
        self.daily_trades = {}
        logger.info("Daily risk counters reset")
    
    def get_status(self) -> Dict:
        """Retorna status do gerenciador de risco"""
        return {
            'daily_pnl': self.daily_pnl,
            'positions_count': self.positions_count,
            'daily_trades': self.daily_trades,
            'last_update': self.last_update.isoformat()
        }