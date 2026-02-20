import sys
import os
import logging

# Configura logging
logging.basicConfig(level=logging.INFO)

# Adiciona diretório atual ao path
sys.path.append('.')

print('Iniciando JL Capital Trading Bot...')

try:
    from jl_capital_trade.trading_bot import JLTradingBot
    from jl_capital_trade.config import Config
    
    print('✅ Módulos importados com sucesso')
    
    config = Config()
    print('✅ Configuração carregada')
    
    bot = JLTradingBot()
    print('✅ Bot instanciado com sucesso')
    
    print('🔌 Tentando conectar ao MT5 via Bot...')
    # Vamos tentar iniciar o bot, mas em uma thread separada ou apenas conectar
    if bot.mt5.connect():
        print('✅ Bot conectado ao MT5!')
        print('📊 Símbolos disponíveis:', len(bot.mt5.get_symbols()))
        
        # Coleta dados de teste
        print('🧪 Testando coleta de dados...')
        symbol = 'EURUSD'
        data = bot.data_collector.get_latest_data(symbol, timeframe='M1', n=10)
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