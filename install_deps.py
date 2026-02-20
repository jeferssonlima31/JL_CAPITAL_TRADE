#!/usr/bin/env python3
"""
Instalador de dependências para JL Capital Forex Bot
Usa o Python disponível no sistema
"""

import sys
import subprocess
import os

def run_command(cmd):
    """Executa comando e retorna resultado"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("=" * 50)
    print("   JL CAPITAL - INSTALADOR DE DEPENDÊNCIAS")
    print("=" * 50)
    print()
    
    # Verifica Python
    print("[1/6] 🔍 Verificando Python...")
    success, stdout, stderr = run_command("python --version")
    
    if success:
        print(f"   ✅ {stdout.strip()}")
        python_cmd = "python"
    else:
        # Tenta python3
        success, stdout, stderr = run_command("python3 --version")
        if success:
            print(f"   ✅ {stdout.strip()}")
            python_cmd = "python3"
        else:
            print("   ❌ Python não encontrado!")
            print("   💡 Instale Python 3.8+ em: https://python.org/downloads/")
            print("   💡 Lembre-se de marcar 'Add Python to PATH'")
            return
    
    # Dependências essenciais para Forex
    dependencies = [
        "pandas",           # Manipulação de dados
        "numpy",            # Computação numérica
        "matplotlib",       # Gráficos
        "scikit-learn",     # Machine Learning
        "xgboost",          # Gradient Boosting
        "tensorflow",       # Deep Learning
        "MetaTrader5",      # Conexão com MT5
        "python-dotenv",    # Variáveis de ambiente
        "ta-lib",           # Indicadores técnicos
    ]
    
    print("[2/6] 📦 Instalando dependências Forex...")
    
    for dep in dependencies:
        print(f"   📥 Instalando {dep}...")
        success, stdout, stderr = run_command(f"{python_cmd} -m pip install {dep}")
        
        if not success:
            print(f"   ⚠️  Falha ao instalar {dep}: {stderr}")
        else:
            print(f"   ✅ {dep} instalado")
    
    print("[3/6] ✅ Dependências instaladas!")
    
    # Testa importações básicas
    print("[4/6] 🧪 Testando importações...")
    
    test_imports = [
        "import pandas as pd",
        "import numpy as np", 
        "import MetaTrader5 as mt5",
        "from sklearn.ensemble import RandomForestClassifier"
    ]
    
    for import_stmt in test_imports:
        test_code = f"{import_stmt}; print('✅ {import_stmt.split()[1]} ok')"
        success, stdout, stderr = run_command(f"{python_cmd} -c \"{test_code}\"")
        
        if success:
            print(f"   {stdout.strip()}")
        else:
            print(f"   ⚠️  Falha: {import_stmt.split()[1]}")
    
    # Verifica MT5
    print("[5/6] 🔌 Testando MetaTrader5...")
    mt5_test = """
import MetaTrader5 as mt5
if mt5.initialize():
    print('✅ MT5 conectado!')
    print(f'Símbolos: {mt5.symbols_total()}')
    mt5.shutdown()
else:
    print('⚠️  MT5 não conectado (certifique-se que o MT5 está aberto)')
"""
    
    success, stdout, stderr = run_command(f"{python_cmd} -c \"{mt5_test}\"")
    if success:
        for line in stdout.strip().split('\n'):
            if line.strip():
                print(f"   {line}")
    else:
        print("   ⚠️  Erro ao testar MT5")
    
    print("[6/6] 🎯 Instalação concluída!")
    print()
    print("=" * 50)
    print("   PRÓXIMOS PASSOS:")
    print("=" * 50)
    print("1. 💻 Abra o MetaTrader 5")
    print("2. 🚀 Execute: python -m jl_capital_trade.trading_bot")
    print("3. 📊 Configure seu .env com as credenciais MT5")
    print("=" * 50)

if __name__ == "__main__":
    main()