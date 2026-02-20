@echo off
echo ================================
echo    JL CAPITAL - SETUP VENV
echo ================================
echo.

:: Tenta encontrar Python
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

echo [1/3] Usando Python: %PYTHON_CMD%
%PYTHON_CMD% --version

:: Cria ambiente virtual
echo [2/3] Criando ambiente virtual...
%PYTHON_CMD% -m venv venv

if errorlevel 1 (
    echo [ERRO] Falha ao criar ambiente virtual
    echo Tente executar manualmente:
    echo %PYTHON_CMD% -m venv venv
    pause
    exit /b 1
)

:: Ativa ambiente virtual
echo [3/3] Ativando ambiente virtual...
call venv\Scripts\activate.bat

:: Instala dependÛncias
echo Instalando dependÛncias...
pip install -r requirements.txt

if errorlevel 1 (
    echo [AVISO] Algumas dependÛncias podem falhar
    echo Tente instalar manualmente:
    echo pip install -r requirements.txt
)

echo.
echo ✅ Ambiente virtual configurado!
echo Para ativar manualmente:
echo venv\Scripts\activate.bat
echo.
pause