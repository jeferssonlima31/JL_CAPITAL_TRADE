import unittest
import os
import importlib
from unittest.mock import patch

class TestConfig(unittest.TestCase):
    
    @patch('jl_capital_trade.config.load_dotenv')
    def test_production_config_validation(self, mock_load_dotenv):
        """Testa validação de configuração em ambiente de produção."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "MT5_LOGIN": "123456",
            "MT5_PASSWORD": "secret_password",
            "MT5_SERVER": "Test-Server",
            "ENCRYPTION_KEY": "some-encryption-key",
            "JWT_SECRET": "some-jwt-secret",
            "DB_PASSWORD": "db-password",
            "OANDA_ENABLED": "true"
        }):
            import jl_capital_trade.config
            importlib.reload(jl_capital_trade.config)
            from jl_capital_trade.config import JLConfig, Environment
            
            config = JLConfig()
            self.assertEqual(config.environment, Environment.PRODUCTION)
            self.assertTrue(config.mt5.validate())
            self.assertTrue(config.validate())
            self.assertTrue(config.oanda.enabled)

    @patch('jl_capital_trade.config.load_dotenv')
    def test_risk_config_parsing(self, mock_load_dotenv):
        """Testa parsing de variáveis de ambiente para a classe de risco."""
        with patch.dict(os.environ, {
            "RISK_PER_TRADE": "2.0",
            "STRICT_MTF_FILTER": "false"
        }, clear=True):
            import jl_capital_trade.config
            importlib.reload(jl_capital_trade.config)
            from jl_capital_trade.config import JLConfig
            
            config = JLConfig()
            self.assertEqual(config.risk.max_risk_per_trade, 2.0)
            self.assertFalse(config.risk.strict_mtf_filter)

if __name__ == '__main__':
    unittest.main()
