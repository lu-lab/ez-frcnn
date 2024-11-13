@echo off
REM Navigate to the directory containing the batch script
cd /d "%~dp0"

REM Check for GPU availability
echo Checking for GPU availability...
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo GPU is available. Using GPU configuration.
    set COMPOSE_FILE=./docker/docker-compose.gpuPane.yml
) else (
    echo GPU is not available. Using CPU configuration.
    set COMPOSE_FILE=./docker/docker-compose.cpuPane.yml
)

REM Start docker-compose with the appropriate file
echo Starting docker-compose...
docker-compose -f %COMPOSE_FILE% up -d

REM Wait for the server to start by pinging it
echo Waiting for the server to be available...
set /a "count=0"
:wait_loop
timeout /t 2 >nul
set /a "count+=1"
echo Attempt %count%: Checking if server is online...
if %count% geq 30 (
    echo Server did not start within the expected time.
    pause
    exit /b 1
)

REM Ping the server
curl -s http://localhost:5006 >nul 2>&1
if %errorlevel% neq 0 (
    goto wait_loop
)
echo Server is up and running.

REM Set the predefined token (make sure this matches the token in docker-compose.yml)
set TOKEN=351

REM Construct the Jupyter Notebook URL
set JUPYTER_URL=http://localhost:5006

REM Open the URL in the default browser using PowerShell and capture the process ID
echo Opening Jupyter Notebook in the default browser...
for /f %%i in ('powershell -Command "Start-Process 'chrome' -ArgumentList '%JUPYTER_URL%' -PassThru | Select-Object -ExpandProperty Id"') do set BROWSER_PID=%%i

REM Output the browser process ID
echo Browser process ID: %BROWSER_PID%

REM Ensure we have a process ID
if "%BROWSER_PID%"=="" (
    echo Failed to detect the browser process ID.
    pause
    exit /b 1
)

REM Wait for the browser process to close
echo Waiting for Jupyter Notebook browser window to close...
:check_browser
tasklist /fi "pid eq %BROWSER_PID%" | find "%BROWSER_PID%" >nul
if %errorlevel%==0 (
    timeout /t 2 /nobreak
    goto check_browser
)

REM Stop and remove the container
echo Stopping and removing the container...
docker-compose -f %COMPOSE_FILE% down
echo Jupyter Notebook closed and container stopped.
pause