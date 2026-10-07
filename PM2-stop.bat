@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PM2_APP=API-VoiceClone"

where pm2 >nul 2>&1
if errorlevel 1 goto NO_PM2

call pm2 describe %PM2_APP% >nul 2>&1
if errorlevel 1 goto NOT_FOUND

echo Deteniendo %PM2_APP%...
call pm2 stop %PM2_APP%
if errorlevel 1 goto STOP_FAIL

echo Eliminando %PM2_APP% de PM2...
call pm2 delete %PM2_APP%
if errorlevel 1 goto DELETE_FAIL

echo %PM2_APP% detenido y eliminado de PM2.
endlocal
exit /b 0

:NOT_FOUND
echo %PM2_APP% no esta registrado en PM2.
endlocal
exit /b 0

:NO_PM2
echo [ERROR] PM2 no esta instalado o no esta en el PATH.
endlocal
exit /b 1

:STOP_FAIL
echo [ERROR] PM2 no pudo detener el servicio.
endlocal
exit /b 1

:DELETE_FAIL
echo [ERROR] PM2 no pudo eliminar el servicio.
endlocal
exit /b 1
