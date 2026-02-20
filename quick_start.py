#!/usr/bin/env python3
"""
Inicializador Rápido do JL Capital Forex Bot
"""

import os
import sys

def check_python():
    """Verifica se o Python está disponível"""
    print("🔍 Procurando Python...")
    
    # Tenta encontrar python executável
    python_paths = [
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Python', 'Python311', 'python.exe'),
        os.path.join('C:\\', 'Python311', 'python.exe'),
        os.path.join('C:\\', 'Python39', 'python.exe'),
        os.path.join('C:\\', 'Python38', 'python.exe'),
    ]
    
    for path in python_paths:
        if os.path.exists(path):
            print(f"✅ Python encontrado em: {path}")
            return path
    
    print("❌ Python não encontrado nos locais comuns")
    return None

def main():
    print("=" * 60)
    print("           JL CAPITAL FOREX AI BOT - INICIALIZADOR")
    print("=" * 60)
    print()
    
    # Verifica Python
    python_path = check_python()
    
    if not python_path:
        print("""
🚨 PROBLEMA ENCONTRADO:
O Python não está instalado corretamente ou não está no PATH.

🔧 SOLUÇÃO:
1. Desinstale qualquer Python da Microsoft Store
2. Baixe o Python oficial: https://python.org/downloads/
3. Durante a instalação, MARQUE: 'Add Python to PATH'
4. Reinicie o terminal após a instalação

💡 Enquanto isso, você pode:
- Usar o Google Colab para testar o código
- Instalar o Python manualmente em C:\\Python39
""")
        return
    
    print("📦 Dependências necessárias para Forex Trading:")
    print("   • pandas           - Análise de dados")
    print("   • numpy            - Computação numérica") 
    print("   • matplotlib       - Gráficos")
    print("   • scikit-learn     - Machine Learning")
    print("   • xgboost          - Algoritmos de trading")
    print("   • MetaTrader5      - Conexão com MT5")
    print("   • python-dotenv    - Configurações")
    print()
    
    print("🎯 PRÓXIMOS PASSOS:")
    print("1. Instale o Python corretamente (veja instruções acima)")
    print("2. Execute: pip install -r requirements.txt")
    print("3. Configure o arquivo .env com suas credenciais MT5")
    print("4. Execute: python -m jl_capital_trade.trading_bot")
    print()
    
    print("💡 DICA: Abra o MetaTrader 5 antes de executar o bot!")
    print("=" * 60)

if __name__ == "__main__":
    main()