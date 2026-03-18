@echo off
:: ──────────────────────────────────────────────────────────────────────────
:: install.bat  —  Windows wrapper for install.ps1
:: Run from the repository root:
::   setup\windows\install.bat
:: Optional args are forwarded to the PowerShell script, e.g.:
::   setup\windows\install.bat -CudaVersion 12.1
::   setup\windows\install.bat -Reinstall
:: ──────────────────────────────────────────────────────────────────────────

echo.
echo  Unsloth Qwen3 Advanced GRPO — Windows Isolated Setup
echo  =====================================================
echo.

:: Check for admin (not required, but inform the user)
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo  [INFO] Running as standard user. Miniconda will be installed
    echo         to %%USERPROFILE%%\miniconda3 if not already present.
    echo.
)

:: Bypass execution policy only for this invocation
powershell.exe -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0install.ps1" %*

if %errorLevel% NEQ 0 (
    echo.
    echo  [ERROR] Setup encountered errors. Review the output above.
    pause
    exit /b 1
)

echo.
echo  Setup completed successfully.
echo  Activate the environment with:  conda activate unsloth-qwen3-grpo
echo.
pause
