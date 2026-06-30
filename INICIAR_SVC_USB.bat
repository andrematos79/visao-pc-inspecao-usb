@echo off
setlocal
title SVC USB - Production

set "BASE_DIR=C:\SVC_INSPECAO_USB_GIT"

cd /d "%BASE_DIR%" || (
    echo ERRO: Nao foi possivel acessar "%BASE_DIR%"
    pause
    exit /b 1
)

echo ==========================================
echo      INICIANDO SVC USB - PRODUCAO
echo ==========================================
echo.

echo Pasta atual:
cd
echo.

echo [CHECK] Conferindo arquivos principais...
if not exist "svc_core_usb_external.py" (
    echo ERRO: Arquivo svc_core_usb_external.py nao encontrado.
    pause
    exit /b 1
)

if not exist "app_svc_usb_producao.py" (
    echo ERRO: Arquivo app_svc_usb_producao.py nao encontrado.
    pause
    exit /b 1
)

echo [1/2] Iniciando CORE...
start "SVC USB CORE" cmd /k "cd /d ""%BASE_DIR%"" && python svc_core_usb_external.py"

timeout /t 5 /nobreak >nul

echo [2/2] Iniciando APP...
start "SVC USB APP" cmd /k "cd /d ""%BASE_DIR%"" && python -m streamlit run app_svc_usb_producao.py"

echo.
echo SVC USB iniciado. Aguarde a interface abrir no navegador.
echo.
pause