# =============================================================================
# JL CAPITAL TRADE - CONECTOR METATRADER 5 (VERSÃO SIMPLIFICADA)
# =============================================================================

import MetaTrader5 as mt5
from datetime import datetime
import logging
import time
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

class MT5ConnectorSimple:
    """Conector simplificado para MetaTrader 5 (sem pandas)"""
    
    def __init__(self, config):
        self.config = config
        self.connected = False

    def connect(self) -> bool:
        """Conecta ao MetaTrader 5"""
        try:
            if not mt5.initialize():
                logger.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
                return False
            
            # Tenta login se credenciais estiverem disponíveis
            if hasattr(self.config, 'mt5_login') and self.config.mt5_login:
                authorized = mt5.login(
                    login=self.config.mt5_login,
                    password=self.config.mt5_password,
                    server=self.config.mt5_server
                )
                if not authorized:
                    logger.warning(f"Login MT5 falhou: {mt5.last_error()}")
                    # Continua mesmo sem login - pode ser demo
            
            self.connected = True
            logger.info("✅ Conectado ao MetaTrader 5")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao conectar ao MT5: {e}")
            return False

    def disconnect(self):
        """Desconecta do MetaTrader 5"""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("Desconectado do MetaTrader 5")
        except Exception as e:
            logger.error(f"Erro ao desconectar do MT5: {e}")

    def get_account_info(self) -> Optional[Dict]:
        """Obtém informações da conta"""
        if not self.connected:
            return None
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'profit': account_info.profit,
                'currency': account_info.currency,
                'leverage': account_info.leverage,
                'name': account_info.name
            }
        except Exception as e:
            logger.error(f"Erro ao obter informações da conta: {e}")
            return None

    def is_connected(self) -> bool:
        """Verifica se está conectado ao MT5"""
        return self.connected and mt5.initialize()