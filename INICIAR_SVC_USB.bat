@echo off
title SVC USB - Production

cd /d C:\SVC_INSPECAO_USB

echo ==========================================
echo      INICIANDO SVC USB - PRODUCAO
echo ==========================================
echo.

echo [1/2] Iniciando CORE...
start "SVC USB CORE" cmd /k ".venv_usb\Scripts\python.exe svc_core_usb_external.py"

timeout /t 5 /nobreak >nul

echo [2/2] Iniciando APP...
start "SVC USB APP" cmd /k ".venv_usb\Scripts\activate.bat && python -m streamlit run app_svc_usb_producao.py"

echo.
echo SVC USB iniciado. Aguarde a interface abrir no navegador.
echo.

exit