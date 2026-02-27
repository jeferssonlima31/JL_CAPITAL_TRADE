#!/usr/bin/env python3
"""
DRY RUN - FULL SYSTEM VALIDATION
Executa o sistema completo em modo teste para verificar a coerência de todos os novos módulos.
"""

import time
import logging
from jl_capital_trade.trading_bot import JLTradingBot
from jl_capital_trade.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_system_validation():
    logger.info("🚀 Iniciando Validação de Coerência do Sistema Completo (Dry Run)...")
    
    # Força modo de teste para não abrir ordens reais
    config.mt5.is_test_mode = True
    
    bot = JLTradingBot()
    
    try:
        # 1. Inicia o Bot
        bot.start()
        logger.info("✅ Bot iniciado. Aguardando ciclos de análise...")
        
        # 2. Monitora por 3 minutos para ver o fluxo de:
        # News -> Session -> Regime -> MTF -> ML -> Execution Simulation
        for i in range(180):
            if i % 30 == 0:
                status = bot.get_status()
                logger.info(f"📊 Status do Bot (T+{i}s): MT5: {status['mt5_connected']} | Positions: {status['positions']}")
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário.")
    except Exception as e:
        logger.error(f"❌ Erro durante validação: {e}")
    finally:
        bot.stop()
        logger.info("🏁 Validação finalizada.")

if __name__ == "__main__":
    run_full_system_validation()
