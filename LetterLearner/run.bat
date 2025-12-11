@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

set "SKIP_PAUSE=0"
:parse_flags
if /I "%~1"=="--no-pause" (
    set "SKIP_PAUSE=1"
    shift
    goto parse_flags
)

if not exist "%VENV_PY%" (
    echo Vytvarim virtualni prostredi...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 goto fail
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto fail

echo Kontroluji zavislosti...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto fail
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail

echo(
set "MODE="
set "ACTION=%~1"
if "%ACTION%"=="" goto run_pipeline
if "%ACTION:~0,1%"=="-" goto run_pipeline
if /I "%ACTION%"=="train" goto do_train
if /I "%ACTION%"=="marshall" goto do_marshall

echo Neznamy prikaz: %ACTION%
echo Pouzijte bez parametru pro marshall -> train, pripadne "train" nebo "marshall" pro jednotlive kroky.
set "APP_EXIT=1"
goto result

:do_train
shift
set "MODE=train"
call :filter_args TRAIN_ARGS %*
echo(
echo Spoustim trenink (train.py)...
"%VENV_PY%" train.py %TRAIN_ARGS%
set "APP_EXIT=%ERRORLEVEL%"
if "%APP_EXIT%"=="0" (
    echo(
    echo Vypravuji model ^(vizualizace^)...
    "%VENV_PY%" dispatch.py
    if errorlevel 1 (
        echo VAROVANI: Vizualizace modelu selhala
        echo Pokracuji dale...
    )
)
goto result

:do_marshall
shift
set "MODE=marshall"
echo(
echo Spoustim editor konfigurace (marshall.py)...
"%VENV_PY%" marshall.py %*
set "APP_EXIT=%ERRORLEVEL%"
goto result

:run_pipeline
set "MODE=pipeline"
call :filter_args TRAIN_ARGS %*
echo(
echo Nejprve oteviram editor konfigurace (marshall.py)...
"%VENV_PY%" marshall.py
if errorlevel 1 (
    echo Editor konfigurace skoncil s chybou nebo byl zavren.
    set "APP_EXIT=%ERRORLEVEL%"
    goto result
)

echo(
echo Spoustim trenink (train.py)...
"%VENV_PY%" train.py %TRAIN_ARGS%
set "APP_EXIT=%ERRORLEVEL%"
if "%APP_EXIT%"=="0" (
    echo(
    echo Vypravuji model ^(vizualizace^)...
    "%VENV_PY%" dispatch.py
    if errorlevel 1 (
        echo VAROVANI: Vizualizace modelu selhala
        echo Pokracuji dale...
    )
)
goto result

:fail
echo(
echo Spusteni se nezdarilo.
set "APP_EXIT=1"
goto result

:result
if "%APP_EXIT%"=="0" goto success
goto failure

:success
if /I "%MODE%"=="marshall" goto msg_marshall
set "RESULT_MSG=Trenink dokoncen. Souhrn je vypsany vyse."
goto after_result

:msg_marshall
set "RESULT_MSG=Editor konfigurace byl ukoncen."
goto after_result

:failure
set "RESULT_MSG=Skript skoncil s chybou (kod %APP_EXIT%)."
goto after_result

:after_result
echo(
echo %RESULT_MSG%

if "%SKIP_PAUSE%"=="0" (
    echo(
    echo Stisknete libovolnou klavesu pro zavreni okna.
    pause >nul
)

:end
endlocal & exit /b %APP_EXIT%

:filter_args
setlocal EnableDelayedExpansion
set "DEST=%~1"
shift
set "OUT="
:filter_loop
if "%~1"=="" goto filter_done
if /I "%~1"=="--no-pause" (
    shift
    goto filter_loop
)
set "OUT=!OUT! %~1"
shift
goto filter_loop
:filter_done
endlocal & set "%DEST%=%OUT%"
goto :eof
