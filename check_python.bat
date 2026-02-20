@echo off
echo ========================================
echo    VERIFICADOR DE PYTHON - JL CAPITAL
echo ========================================
echo.

echo 🔍 Procurando Python instalado...
echo.

:: Verifica locais comuns do Python
set FOUND=0

for %%d in (C D E) do (
    if exist "%%d:\Python39\python.exe" (
        echo ✅ Python 3.9 encontrado em: %%d:\Python39\
        set FOUND=1
    )
    if exist "%%d:\Python310\python.exe" (
        echo ✅ Python 3.10 encontrado em: %%d:\Python310\
        set FOUND=1
    )
    if exist "%%d:\Python311\python.exe" (
        echo ✅ Python 3.11 encontrado em: %%d:\Python311\
        set FOUND=1
    )
    if exist "%%d:\Python312\python.exe" (
        echo ✅ Python 3.12 encontrado em: %%d:\Python312\
        set FOUND=1
    )
)

:: Verifica AppData Local
if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    echo ✅ Python 3.9 encontrado em: %LOCALAPPDATA%\Programs\Python\Python39\
    set FOUND=1
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    echo ✅ Python 3.10 encontrado em: %LOCALAPPDATA%\Programs\Python\Python310\
    set FOUND=1
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    echo ✅ Python 3.11 encontrado em: %LOCALAPPDATA%\Programs\Python\Python311\
    set FOUND=1
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    echo ✅ Python 3.12 encontrado em: %LOCALAPPDATA%\Programs\Python\Python312\
    set FOUND=1
)

if %FOUND% equ 0 (
    echo ❌ Nenhum Python encontrado!
    echo.
    echo 📥 INSTALE O PYTHON CORRETAMENTE:
    echo 1. Baixe de: https://python.org/downloads/
    echo 2. Durante a instalação, MARQUE: "Add Python to PATH"
    echo 3. NÃO use a versão da Microsoft Store!
    echo.
    echo 💡 Dica: Instale em C:\Python39 para facilitar
) else (
    echo.
    echo ✅ Python encontrado! Agora execute:
    echo pip install -r requirements.txt
    echo python -m jl_capital_trade.trading_bot
)

echo.
echo ========================================
pause