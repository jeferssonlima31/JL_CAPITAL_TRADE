# =============================================================================
# JL CAPITAL TRADE - GERENCIADOR DE RISCO
# =============================================================================

from datetime import datetime, date
import logging
from typing import Dict, Optional
from .var_engine import VaREngine

logger = logging.getLogger(__name__)

class RiskManager:
    """Gerenciador de risco para trading"""
    
    def __init__(self, config):
        self.config = config
        self.daily_trades = {}
        self.daily_pnl = 0
        self.last_update = datetime.now()
        self.positions_count = 0
        
        # Engine Isolada de Risco
        self.var_engine = VaREngine(config)
        
        # Novos controles
        self.consecutive_losses = 0
        self.peak_balance = 0
        self.current_drawdown = 0
        self.circuit_broken = False
        self.breaker_reason = ""
        
    def check_circuit_breakers(self, current_spread: float = 0, current_slippage: float = 0) -> bool:
        """Consolidado de Circuit Breakers de Segurança"""
        
        # 1. Se o disjuntor já disparou, permanece travado até reset manual/diário
        if self.circuit_broken:
            return False

        # 2. Check Spread Alto
        if current_spread > self.config.risk.max_spread_pips:
            self.breaker_reason = f"Spread Alto: {current_spread:.1f} pips"
            return False

        # 3. Check Slippage Excessivo
        if current_slippage > self.config.risk.max_slippage_pips:
            self.circuit_broken = True
            self.breaker_reason = f"Slippage Excessivo: {current_slippage:.1f} pips"
            logger.critical(f"🚨 CIRCUIT BREAKER: {self.breaker_reason}")
            return False

        # 4. Check Perdas Consecutivas
        if self.consecutive_losses >= self.config.risk.max_consecutive_losses:
            self.circuit_broken = True
            self.breaker_reason = f"Limite de Perdas Consecutivas ({self.consecutive_losses})"
            logger.critical(f"🚨 CIRCUIT BREAKER: {self.breaker_reason}")
            return False

        # 5. Check Drawdown Intraday/Histórico
        if self.current_drawdown >= self.config.risk.max_drawdown:
            self.circuit_broken = True
            self.breaker_reason = f"Drawdown Crítico: {self.current_drawdown:.2f}%"
            logger.critical(f"🚨 CIRCUIT BREAKER: {self.breaker_reason}")
            return False

        # 6. Check Perda Diária
        if self.daily_pnl < -self.config.risk.max_daily_loss:
            self.circuit_broken = True
            self.breaker_reason = f"Limite de Perda Diária: {self.daily_pnl:.2f}%"
            logger.critical(f"🚨 CIRCUIT BREAKER: {self.breaker_reason}")
            return False

        return True

    def can_trade(self, symbol: str, current_spread: float = 0, 
                  historical_returns=None, proposed_volume: float = 0, balance: float = 0) -> bool:
        """Verifica se pode fazer nova operação considerando circuit breakers e Motor VaR"""
        
        # 1. Checa Circuit Breakers Tradicionais
        if not self.check_circuit_breakers(current_spread=current_spread):
            logger.warning(f"Trade bloqueado por Circuit Breaker: {self.breaker_reason}")
            return False
            
        # 2. Avaliação de Value at Risk (VaR) Institucional via Monte Carlo
        if historical_returns is not None and proposed_volume > 0 and balance > 0:
            var_result = self.var_engine.calculate_var(
                symbol, 
                current_price=1.0, # Preço simulado irrelevante para PnL %
                position_volume=proposed_volume, 
                balance=balance, 
                historical_returns=historical_returns
            )
            
            if var_result and not var_result.is_safe:
                self.breaker_reason = f"Projeção VaR Crítica: Risco Excede Limites"
                return False
            
        today = date.today().isoformat()
        
        # 3. Verifica número de posições globais intra-diárias
        symbol_positions = self.daily_trades.get(symbol, {}).get(today, 0)
        if symbol_positions >= self.config.risk.max_positions:
            logger.warning(f"Max positions reached for {symbol}")
            return False
        
        # 4. Verifica total de posições
        if self.positions_count >= self.config.risk.max_positions:
            logger.warning(f"Max total positions reached: {self.positions_count}")
            return False
        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, 
                                atr: float, account_balance: float, 
                                model_confidence: float = 0.5) -> float:
        """Calcula tamanho da posição com Gestão de Capital Dinâmica"""
        
        # 1. Risco Base (1.5%)
        risk_percent = self.config.risk.max_risk_per_trade
        
        # 2. Ajuste Dinâmico por Confiança do Modelo
        # Se confiança for muito alta (>0.85), podemos aumentar levemente o risco
        # Se for baixa (<0.75), mantemos o risco base ou reduzimos
        if model_confidence > 0.85:
            risk_percent *= 1.2 # +20% de risco
        elif model_confidence < 0.75:
            risk_percent *= 0.8 # -20% de risco
            
        # 3. Ajuste por Drawdown (Proteção de Capital)
        # Se drawdown > 10%, reduz o risco pela metade
        if self.current_drawdown > 10.0:
            risk_percent *= 0.5
            
        # Valor final em risco
        risk_amount = account_balance * (risk_percent / 100)
        
        # Distância do stop loss dinâmico (2.5x ATR)
        sl_distance = atr * 2.5
        
        if symbol == "EUR_USD":
            pip_value = 10  # Standard Lot
            # Converte distância de preço para pips
            sl_pips = sl_distance * 10000
        else:
            pip_value = 1
            sl_pips = sl_distance * 100
        
        # Evita divisão por zero
        if sl_pips == 0: sl_pips = 30
        
        # Tamanho da posição
        position_size = risk_amount / (sl_pips * pip_value)
        
        # Ajuste conta cent
        if self._is_cent_account():
            position_size *= 100
            
        return self._round_to_standard_lot(position_size)
    
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
    
    def update_pnl(self, profit_loss_percent: float, current_balance: float):
        """Atualiza P&L diário e métricas de risco"""
        self.daily_pnl += profit_loss_percent
        self.last_update = datetime.now()
        
        # Atualiza perdas consecutivas
        if profit_loss_percent < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # Atualiza Drawdown
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            
        if self.peak_balance > 0:
            self.current_drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100
            
        logger.info(f"Risk Update: Daily PnL: {self.daily_pnl:.2f}% | "
                   f"Consecutive Losses: {self.consecutive_losses} | "
                   f"Drawdown: {self.current_drawdown:.2f}%")
    
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