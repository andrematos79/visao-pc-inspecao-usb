@echo off
title SVC USB
cd /d C:\SVC_INSPECAO_USB
call .\.venv_svc\Scripts\activate.bat
python -m streamlit run app_camera_infer_usb.py --server.port 8501