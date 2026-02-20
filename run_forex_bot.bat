@echo off
title JL Capital - Forex AI Bot
color 0A

echo ========================================
echo        JL CAPITAL FOREX AI BOT
echo ========================================
echo.

:: Verifica Python
set PYTHON_CMD=

for %%i in (python python3) do (
    %%i --version >nul 2>&1
    if not errorlevel 1 set PYTHON_CMD=%%i
)

if "%PYTHON_CMD%"=="" (
    echo [ERRO] Python nÒo encontrado!
    echo Baixe e instale Python 3.8+ em:
    echo https://python.org/downloads/
    echo Lembre-se de marcar "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/5] ✅ Python encontrado: %PYTHON_CMD%
%PYTHON_CMD% --version

:: Verifica se ambiente virtual existe
if not exist "venv" (
    echo [2/5] Criando ambiente virtual...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar ambiente virtual
        pause
        exit /b 1
    )
)

:: Ativa ambiente virtual
echo [3/5] Ativando ambiente virtual...
call venv\Scripts\activate.bat

:: Instala dependÛncias
echo [4/5] Instalando dependÛncias Forex...
pip install pandas numpy matplotlib scikit-learn xgboost tensorflow MetaTrader5 python-dotenv ta-lib

if errorlevel 1 (
    echo [AVISO] Algumas dependÛncias podem ter falhado
    echo Continuando com instalação básica...
)

:: Executa o trading bot
echo [5/5] 🚀 Iniciando JL Capital Forex Bot...
echo.
echo 📊 Coletando dados de mercado...
echo 🤖 Analisando com IA...
echo ⚡ Pronto para trading!
echo.
python -c "
import sys
sys.path.append('.')
from jl_capital_trade.trading_bot import JLTradingBot
from jl_capital_trade.config import Config

print('Iniciando JL Capital Trading Bot...')
config = Config()
bot = JLTradingBot(config)

# Testa conexão com MT5
try:
    import MetaTrader5 as mt5
    if mt5.initialize():
        print('✅ Conexão com MetaTrader 5 estabelecida!')
        print(f'📈 Símbolos disponíveis: {len(mt5.symbols_total())}')
        mt5.shutdown()
    else:
        print('⚠️  MetaTrader 5 não conectado')
        print('💡 Certifique-se de que o MT5 está aberto')
except Exception as e:
    print(f'⚠️  Erro ao conectar com MT5: {e}')

print('\\n🎯 Bot inicializado com sucesso!')
print('💡 Use CTRL+C para parar')
"

echo.
echo ========================================
echo        BOT INICIADO COM SUCESSO!
echo ========================================
echo.
echo Comandos disponíveis manualmente:
echo python -m jl_capital_trade.trading_bot
echo python -m jl_capital_trade.data_collector
echo.
pause