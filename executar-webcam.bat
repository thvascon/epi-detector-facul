@echo off
echo ============================================================
echo DETECTOR DE CAPACETE - WEBCAM
echo ============================================================
echo.
echo Verificando Python...
python --version
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    echo Instale Python em: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo.
echo Instalando dependencias...
pip install opencv-python supervision inference-sdk
echo.
echo Iniciando detector...
python detector-EPI-webcam.py
pause
