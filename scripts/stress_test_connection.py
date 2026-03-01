#!/usr/bin/env python3
"""
STRESS TEST - MT5 HEARTBEAT & CONNECTION
Testa a resiliência do sistema de Heartbeat e reconexão automática.
"""

import time
import logging
from jl_capital_trade.config import config
from jl_capital_trade.mt5_connector import MT5Connector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_heartbeat_resilience():
    logger.info("🚀 Iniciando Teste de Estresse do Heartbeat...")
    
    connector = MT5Connector(config)
    
    # 1. Testa Conexão Inicial
    if not connector.connect():
        logger.error("❌ Falha na conexão inicial. O MT5 está aberto?")
        return

    logger.info("✅ Conexão inicial estabelecida. Heartbeat ativo.")
    
    # 2. Monitora por 10 segundos em condições normais
    logger.info("⏳ Monitorando conexão estável por 10s...")
    for i in range(10):
        status = "CONECTADO" if connector.is_connected() else "DESCONECTADO"
        logger.info(f"[{i+1}/10] Status: {status}")
        time.sleep(1)

    # 3. Simula perda de conexão (Shutdown forçado no conector)
    logger.info("🔥 Simulando perda de conexão abrupta...")
    import MetaTrader5 as mt5
    mt5.shutdown() # Força o desligamento da biblioteca MT5
    
    logger.info("⏳ Aguardando detecção do Heartbeat (deve detectar em ~1s)...")
    
    detected = False
    for i in range(10):
        if not connector.is_connected():
            logger.warning(f"🎯 Heartbeat detectou queda na iteração {i+1}!")
            detected = True
            break
        time.sleep(1)
        
    if not detected:
        logger.error("❌ Falha: Heartbeat não detectou a queda da conexão!")
    else:
        logger.info("🔄 Testando tentativa de reconexão automática...")
        # O heartbeat já deve estar tentando reconectar. Vamos aguardar.
        time.sleep(5)
        if connector.is_connected():
            logger.info("✅ Reconexão automática bem-sucedida!")
        else:
            logger.warning("⚠️ Reconexão automática falhou ou MT5 ainda está fechado.")

    connector.disconnect()
    logger.info("🏁 Teste de estresse finalizado.")

if __name__ == "__main__":
    test_heartbeat_resilience()
