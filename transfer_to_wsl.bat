@echo off
REM SeqTrack Assignment 3 - Transfer to WSL Script
REM Run this script from Windows to transfer files to WSL

echo === SeqTrack Assignment 3 - Transfer to WSL ===
echo.

REM Get current directory
set CURRENT_DIR=%cd%
echo Current directory: %CURRENT_DIR%

REM Check if WSL is available
wsl --list >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL is not installed or not available
    echo Please install WSL2 with Ubuntu first
    pause
    exit /b 1
)

echo.
echo Available WSL distributions:
wsl --list

echo.
set /p WSL_DISTRO="Enter WSL distribution name (e.g., Ubuntu-20.04): "

REM Check if distribution exists
wsl -d %WSL_DISTRO% -- echo "Testing connection" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL distribution '%WSL_DISTRO%' not found
    echo Please check the distribution name
    pause
    exit /b 1
)

echo.
echo ✅ WSL distribution '%WSL_DISTRO%' found

REM Get username in WSL
for /f "tokens=*" %%i in ('wsl -d %WSL_DISTRO% whoami') do set WSL_USER=%%i
echo WSL username: %WSL_USER%

REM Set WSL destination path
set WSL_DEST=/home/%WSL_USER%/assignment_3

echo.
echo 📁 Creating directory in WSL: %WSL_DEST%
wsl -d %WSL_DISTRO% mkdir -p %WSL_DEST%

echo.
echo 📋 Copying assignment files to WSL...

REM Copy assignment files (excluding SeqTrack directory)
wsl -d %WSL_DISTRO% cp -r /mnt/c/Users/%USERNAME%/AppData/Local/Temp/assignment_3_temp/* %WSL_DEST%/ 2>nul
if %errorlevel% neq 0 (
    echo Using alternative copy method...
    REM Copy files directly
    for /f "delims=" %%i in ('dir /b /s assignment_3\*.*') do (
        echo Copying %%i
        wsl -d %WSL_DISTRO% cp "%%i" %WSL_DEST%/
    )
)

echo.
echo 🎯 Copying SeqTrack repository...
if exist "SeqTrack" (
    echo Copying SeqTrack directory...
    wsl -d %WSL_DISTRO% cp -r "%CURRENT_DIR%/SeqTrack" %WSL_DEST%/SeqTrack
) else (
    echo ⚠️ SeqTrack directory not found in current location
    echo Please ensure SeqTrack repository is in: %CURRENT_DIR%/SeqTrack
)

echo.
echo 🔧 Setting up permissions in WSL...
wsl -d %WSL_DISTRO% chmod +x %WSL_DEST%/seqtrack_train.py
wsl -d %WSL_DISTRO% chmod +x %WSL_DEST%/dataset_loader.py
wsl -d %WSL_DISTRO% chmod +x %WSL_DEST%/setup_wsl.sh

echo.
echo 📁 Creating checkpoint directory...
wsl -d %WSL_DISTRO% mkdir -p %WSL_DEST%/checkpoints

echo.
echo 🧪 Testing WSL setup...
wsl -d %WSL_DISTRO% ls -la %WSL_DEST%/

echo.
echo ✅ Transfer completed successfully!
echo.
echo 📋 Next steps:
echo 1. Open WSL terminal: wsl -d %WSL_DISTRO%
echo 2. Navigate to project: cd %WSL_DEST%
echo 3. Run setup script: bash setup_wsl.sh
echo 4. Activate environment: source seqtrack_env/bin/activate
echo 5. Run training: python3 seqtrack_train.py
echo.
echo 📖 See PYCHARM_WSL_DEPLOYMENT.md for PyCharm integration
echo.
pause
