@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM PM2-start-all.bat puede pasar "nopause" para no bloquear el resto.
if /I "%~1"=="nopause" set "NO_PAUSE=1"

set "PYTHONW=%CD%\.venv\Scripts\pythonw.exe"
set "PYTHON=%CD%\.venv\Scripts\python.exe"
set "PORT=3036"
set "PM2_APP=API-VoiceClone"

if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if /I "%%a"=="PORT" if not "%%b"=="" set "PORT=%%b"
    )
)

if exist "%PYTHONW%" (
    set "INTERPRETER=%PYTHONW%"
) else if exist "%PYTHON%" (
    set "INTERPRETER=%PYTHON%"
) else (
    goto NO_VENV
)

if not exist "%PYTHON%" goto NO_VENV

"%PYTHON%" -c "import omnivoice" >nul 2>&1
if errorlevel 1 goto NO_PKG

where pm2 >nul 2>&1
if errorlevel 1 goto NO_PM2

call pm2 describe %PM2_APP% >nul 2>&1
if not errorlevel 1 (
    echo Recreando %PM2_APP% en puerto %PORT%...
    call pm2 delete %PM2_APP%
)

echo Liberando puerto %PORT% si quedo un proceso huerfano...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "0.0.0.0:%PORT%" ^| findstr LISTENING') do (
    echo Cerrando PID %%a que ocupa el puerto %PORT%...
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:%PORT%" ^| findstr LISTENING') do (
    echo Cerrando PID %%a que ocupa el puerto %PORT%...
    taskkill /F /PID %%a >nul 2>&1
)

echo Iniciando %PM2_APP% en PM2...
set "PORT=%PORT%"
set "PYTHONUNBUFFERED=1"
set "PYTHONPATH=%CD%"
REM API FastAPI de OmniVoice: clonacion, diseno de voz y Swagger en /docs.
call pm2 start "%INTERPRETER%" --name %PM2_APP% --interpreter none --cwd "%CD%" -- -m omnivoice.api --host 0.0.0.0 --port %PORT%
if errorlevel 1 goto PM2_FAIL

echo.
call pm2 status %PM2_APP%
echo.
echo Servicio listo en PM2 como %PM2_APP%
echo API:     http://localhost:%PORT%
echo Swagger: http://localhost:%PORT%/docs
echo Logs:    pm2 logs %PM2_APP%
echo.
echo La primera vez descarga el modelo de HuggingFace; puede tardar varios minutos.
echo.
if not defined NO_PAUSE pause
endlocal
exit /b 0

:NO_VENV
echo [ERROR] No existe .venv\Scripts\python.exe
echo Crea el entorno en esta carpeta:
echo   uv sync
echo o:
echo   python -m venv .venv
echo   .venv\Scripts\activate
echo   pip install -e .
echo.
if not defined NO_PAUSE pause
endlocal
exit /b 1

:NO_PKG
echo [ERROR] El paquete omnivoice no esta instalado en .venv
echo Ejecuta en esta carpeta:
echo   uv sync
echo o:
echo   .venv\Scripts\activate
echo   pip install -e .
echo.
if not defined NO_PAUSE pause
endlocal
exit /b 1

:NO_PM2
echo [ERROR] PM2 no esta instalado o no esta en el PATH.
echo Instala con: npm install -g pm2
echo.
if not defined NO_PAUSE pause
endlocal
exit /b 1

:PM2_FAIL
echo [ERROR] PM2 no pudo iniciar el servicio.
echo Revisa logs: pm2 logs %PM2_APP% --lines 50
echo.
if not defined NO_PAUSE pause
endlocal
exit /b 1
