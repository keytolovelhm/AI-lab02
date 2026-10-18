@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>nul
if errorlevel 1 goto use_python
py -3 run.py %*
goto done
:use_python
where python >nul 2>nul
if errorlevel 1 goto missing
python -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>nul
if errorlevel 1 goto missing
python run.py %*
goto done
:missing
echo Python 3.10+ was not found. Install Python and enable its launcher or PATH.
exit /b 1
:done
set "lab2_exit=%errorlevel%"
if not "%lab2_exit%"=="0" echo Lab 2 failed. See the error above.
exit /b %lab2_exit%
