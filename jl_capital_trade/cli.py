# =============================================================================
# JL CAPITAL TRADE - INTERFACE DE LINHA DE COMANDO
# =============================================================================

import argparse
import logging
import sys

# Força UTF-8 para evitar UnicodeEncodeError no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from .trading_bot import JLTradingBot
from .config import config

def setup_logging():
    """Configura logging"""
    logging.basicConfig(
        level=config.log_level.value,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.logs_dir / 'cli.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Função principal da CLI"""
    parser = argparse.ArgumentParser(description='JL Capital Trade - Forex ML Bot')
    
    parser.add_argument('--mode', choices=['live', 'demo', 'test'], 
                       default='test', help='Modo de operação')
    parser.add_argument('--analyze', action='store_true', 
                       help='Analisar pares')
    parser.add_argument('--pair', default='EUR_USD', 
                       help='Par para análise')
    parser.add_argument('--timeframe', default='H1', 
                       help='Timeframe para análise')
    parser.add_argument('--test-connection', action='store_true',
                       help='Testar conexão MT5')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Testa conexão
    if args.test_connection:
        from .mt5_connector import MT5Connector
        mt5 = MT5Connector(config)
        if mt5.connect():
            logger.info("✅ Conexão MT5 OK")
            account = mt5.get_account_info()
            if account:
                logger.info(f"📊 Saldo: {account['balance']} {account['currency']}")
            mt5.disconnect()
        else:
            logger.error("❌ Falha na conexão MT5")
        return
    
    # Inicia bot
    bot = JLTradingBot()
    
    if args.analyze:
        if not bot.mt5.connect():
            logger.error("Não foi possível conectar ao MT5")
            return
        
        logger.info(f"🔍 Analisando {args.pair} {args.timeframe}...")
        signal = bot.analyze_pair(args.pair, args.timeframe)
        
        if signal:
            logger.info(f"📊 Sinal: {signal}")
        else:
            logger.info("Nenhum sinal gerado")
        
        bot.mt5.disconnect()
        return
    
    # Modo bot
    if args.mode == 'test':
        logger.info("🧪 Modo teste - trades não serão executados")
    
    try:
        bot.start()
        
        # Mantém executando
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Parando bot...")
        bot.stop()
    except Exception as e:
        logger.error(f"Erro: {e}")
        bot.stop()

if __name__ == '__main__':
    main()