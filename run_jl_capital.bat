@echo off
title JL Capital + OpenClaw Integration
color 0B

echo ========================================
echo    JL CAPITAL TRADE + OPENCLAW
echo ========================================
echo.

:: Verifica Python e define caminho
set PYTHON_EXE=

:: Tenta encontrar Python
for %%i in (python python3) do (
    %%i --version >nul 2>&1
    if not errorlevel 1 set PYTHON_EXE=%%i
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python not found! Please install Python 3.8+
    echo Download from: https://python.org/downloads/
    pause
    exit /b 1
)

:: Ativa ambiente virtual
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: Inicia API Bridge em segundo plano
echo [1/4] Starting JL Capital API Bridge...
start /B python -m jl_capital_trade.api_bridge

:: Aguarda inicialização
timeout /t 3 /nobreak >nul

:: Verifica se está rodando
curl http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo [ERROR] API Bridge did not start
    pause
    exit /b 1
)

echo [2/4] ✅ API Bridge running on port 5000

:: Configura OpenClaw
echo [3/4] Configuring OpenClaw...

:: Verifica OpenClaw
where openclaw >nul 2>&1
if errorlevel 1 (
    echo [WARNING] OpenClaw not found in PATH
    echo Please install OpenClaw first
    pause
    exit /b 1
)

:: Instala skill
echo Installing JL Capital skill...
copy skills\jl_capital_skill.js "%USERPROFILE%\.openclaw\skills\" >nul 2>&1

:: Configura token JWT (opcional)
set JWT_TOKEN=test_token_123

:: Inicia OpenClaw Gateway
echo [4/4] Starting OpenClaw Gateway...
start /B openclaw gateway run --bind 0.0.0.0 --port 18789

echo.
echo ========================================
echo    ✅ SISTEMA INTEGRADO COM SUCESSO!
echo ========================================
echo.
echo JL Capital API: http://localhost:5000
echo OpenClaw UI:    http://localhost:18789
echo.
echo Comandos disponíveis no OpenClaw:
echo   /jl analyze --pair EUR_USD
echo   /jl trade --action BUY --symbol EUR_USD --price 1.0892
echo   /jl status
echo   /jl positions
echo   /jl risk
echo.
pause