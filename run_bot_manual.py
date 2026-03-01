import sys
import os
import logging

# Força UTF-8 para evitar UnicodeEncodeError no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Configura logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

# Adiciona diretório atual ao path
sys.path.append('.')

print('Iniciando JL Capital Trading Bot...')

try:
    from jl_capital_trade.trading_bot import JLTradingBot
    from jl_capital_trade.config import config
    
    print('✅ Módulos importados com sucesso')
    
    print('✅ Configuração carregada')
    
    bot = JLTradingBot()
    print('✅ Bot instanciado com sucesso')
    
    print('🔌 Tentando conectar ao MT5 via Bot...')
    # Vamos tentar iniciar o bot, mas em uma thread separada ou apenas conectar
    if bot.mt5.connect():
        print('✅ Bot conectado ao MT5!')
        
        info = bot.mt5.get_account_info()
        if info:
            print(f"💰 Conta logada: {info['login']}")
            print(f"💵 Saldo: ${info['balance']:.2f}")
        else:
            print("⚠️ Não foi possível obter as infos da conta.")

        # Coleta dados de teste
        print('🧪 Testando coleta de dados...')
        symbol = 'EUR_USD'
        data = bot.data_collector.get_historical_data(symbol, timeframe='M1', count=10)
        if data is not None and not data.empty:
            print(f'✅ Dados coletados para {symbol}: {len(data)} candles')
            print(data.head())
        else:
            print(f'⚠️  Não foi possível coletar dados para {symbol}')
            
        bot.mt5.disconnect()
    else:
        print('⚠️  Falha ao conectar ao MT5 via Bot')

except ImportError as e:
    print(f'❌ Erro de importação: {e}')
    print('Verifique se todos os arquivos estão na pasta correta.')
except Exception as e:
    print(f'❌ Erro ao executar bot: {e}')
    import traceback
    traceback.print_exc()

print('\n🏁 Teste concluído')