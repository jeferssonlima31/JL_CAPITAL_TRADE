#!/bin/bash
echo "========================================"
echo "   JL CAPITAL FOREX AI BOT - SETUP"
echo "========================================"
echo

# Verifica Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERRO] Python não encontrado!"
    echo "Instale Python 3.8+ em: https://python.org/downloads/"
    exit 1
fi

echo "[1/5] ✅ Python encontrado: $($PYTHON_CMD --version)"

# Cria ambiente virtual
echo "[2/5] Criando ambiente virtual..."
$PYTHON_CMD -m venv venv

if [ $? -ne 0 ]; then
    echo "[ERRO] Falha ao criar ambiente virtual"
    exit 1
fi

# Ativa ambiente virtual
echo "[3/5] Ativando ambiente virtual..."
source venv/bin/activate

# Instala dependências
echo "[4/5] Instalando dependências Forex..."
pip install pandas numpy matplotlib scikit-learn xgboost tensorflow MetaTrader5 python-dotenv ta-lib

# Instalação básica se houver erro
if [ $? -ne 0 ]; then
    echo "[AVISO] Instalando dependências básicas..."
    pip install pandas numpy MetaTrader5 python-dotenv
fi

echo "[5/5] ✅ Setup completo!"
echo
echo "Para ativar o ambiente virtual:"
echo "source venv/bin/activate"
echo
echo "Para executar o trading bot:"
echo "python -m jl_capital_trade.trading_bot"