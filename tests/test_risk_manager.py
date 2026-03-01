import unittest
from jl_capital_trade.risk_manager import RiskManager

class MockRiskConfig:
    max_spread_pips = 2.0
    max_slippage_pips = 1.5
    max_consecutive_losses = 3
    max_drawdown = 15.0
    max_daily_loss = 5.0
    max_positions = 3
    max_risk_per_trade = 1.5

class MockConfig:
    def __init__(self):
        self.risk = MockRiskConfig()

class TestRiskManager(unittest.TestCase):
    
    def setUp(self):
        """Prepara o ambiente e o RiskManager mockado antes de cada teste."""
        self.config = MockConfig()
        self.risk_manager = RiskManager(self.config)

    def test_circuit_breaker_spread(self):
        """Testa o disparador de Spread do Circuit Breaker."""
        # Spread aceitável = 1.0 (limite 2.0)
        self.assertTrue(self.risk_manager.can_trade("EUR_USD", current_spread=1.0))
        
        # Spread muito alto = 2.5 (limite 2.0)
        self.assertFalse(self.risk_manager.can_trade("EUR_USD", current_spread=2.5))
        self.assertEqual(self.risk_manager.breaker_reason, "Spread Alto: 2.5 pips")
        self.assertFalse(self.risk_manager.circuit_broken) # Spread alto não trava o robô permanentemente
        
    def test_circuit_breaker_slippage(self):
        """Testa o disparador de Slippage do Circuit Breaker."""
        # Se ultrapassar o Max Slippage, ele trava o CB permanentemente (circuit_broken = True)
        self.assertFalse(self.risk_manager.check_circuit_breakers(current_slippage=3.0))
        self.assertTrue(self.risk_manager.circuit_broken)
        self.assertEqual(self.risk_manager.breaker_reason, "Slippage Excessivo: 3.0 pips")

    def test_circuit_breaker_consecutive_losses(self):
        """Testa se perdas consecutivas bloqueiam o sistema."""
        self.risk_manager.update_pnl(-0.5, 10000)
        self.risk_manager.update_pnl(-0.5, 10000)
        self.assertTrue(self.risk_manager.can_trade("EUR_USD", 1.0)) # 2 perdas
        
        self.risk_manager.update_pnl(-0.5, 10000) # 3 perdas = límite
        self.assertFalse(self.risk_manager.can_trade("EUR_USD", 1.0))
        self.assertTrue(self.risk_manager.circuit_broken)
        
    def test_position_sizing_confidence_boost(self):
        """Verifica o gerenciamento dinâmico de capital baseado em confiança na IA."""
        # Risco Base: 1.5% do balance (10000) = 150 
        base_size = self.risk_manager.calculate_position_size("EUR_USD", price=1.1000, atr=0.0010, account_balance=10000, model_confidence=0.80)
        
        # Confiança MÁXIMA (>0.85): Risco deve ser +20% (1.8% = 180) -> size maior
        high_conf_size = self.risk_manager.calculate_position_size("EUR_USD", price=1.1000, atr=0.0010, account_balance=10000, model_confidence=0.95)
        
        # Confiança BAIXA (<0.75): Risco deve ser -20% (1.2% = 120) -> size menor
        low_conf_size = self.risk_manager.calculate_position_size("EUR_USD", price=1.1000, atr=0.0010, account_balance=10000, model_confidence=0.60)
        
        self.assertGreater(high_conf_size, base_size)
        self.assertLess(low_conf_size, base_size)

if __name__ == '__main__':
    unittest.main()
