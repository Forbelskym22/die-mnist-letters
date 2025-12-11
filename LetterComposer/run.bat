@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

set "SKIP_PAUSE="
if /I "%~1"=="--no-pause" (
    set "SKIP_PAUSE=1"
    shift
)

set "PY_ARGS="
:collect_args
if "%~1"=="" goto args_ready
if defined PY_ARGS (
    set "PY_ARGS=%PY_ARGS% %1"
) else (
    set "PY_ARGS=%1"
)
shift
goto collect_args

:args_ready

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
echo Spoustim LetterComposer...
"%VENV_PY%" main.py %PY_ARGS%
set "APP_EXIT=%ERRORLEVEL%"
goto result

:fail
set "APP_EXIT=1"
echo(
echo Spusteni se nezdarilo.
goto result

:result
if "%APP_EXIT%"=="0" goto success

echo(
echo Aplikace skoncila s chybou (kod %APP_EXIT%).
goto status_done

:success
echo(
echo Komponovani datasetu bylo dokonceno.

:status_done

if defined SKIP_PAUSE goto end

echo(
echo Stisknete libovolnou klavesu pro zavreni okna.
pause >nul

:end
endlocal & exit /b %APP_EXIT%
