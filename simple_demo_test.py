#!/usr/bin/env python3
"""
TESTE SIMPLES DA CONTA DEMO
"""

import MetaTrader5 as mt5
import logging

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_demo_connection():
    """Testa conexão com a conta demo"""
    print("=" * 60)
    print("🔗 TESTANDO CONEXÃO COM CONTA DEMO")
    print("=" * 60)
    
    # Credenciais da sua conta demo Lime Trading
    demo_config = {
        'login': 3263303,
        'password': '!rH5UiSb', 
        'server': 'Just2Trade-MT5',
        'timeout': 30000,
        'portable': False
    }
    
    # Inicializa MT5
    if not mt5.initialize():
        print(f"❌ Falha ao inicializar MT5: {mt5.last_error()}")
        return False
    
    print("✅ MT5 inicializado com sucesso")
    
    # Tenta login
    authorized = mt5.login(
        login=demo_config['login'],
        password=demo_config['password'], 
        server=demo_config['server'],
        timeout=demo_config['timeout']
    )
    
    if authorized:
        # Sucesso!
        account_info = mt5.account_info()
        print("\n🎉 CONTA DEMO CONECTADA COM SUCESSO!")
        print(f"   📋 Login: {account_info.login}")
        print(f"   👤 Nome: {account_info.name}")
        print(f"   💰 Saldo: ${account_info.balance:.2f}")
        print(f"   📈 Equity: ${account_info.equity:.2f}")
        print(f"   🏢 Broker: {account_info.company}")
        print(f"   🌐 Server: {account_info.server}")
        print(f"   💵 Moeda: {account_info.currency}")
        
        # Verifica símbolos disponíveis
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        print("\n📊 SÍMBOLOS DISPONÍVEIS:")
        for symbol in symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                print(f"   ✅ {symbol}: {symbol_info.ask:.5f}/{symbol_info.bid:.5f}")
            else:
                print(f"   ❌ {symbol}: Não disponível")
        
        mt5.shutdown()
        return True
    else:
        print(f"❌ FALHA NO LOGIN: {mt5.last_error()}")
        mt5.shutdown()
        return False

if __name__ == "__main__":
    test_demo_connection()