@echo off
setlocal enabledelayedexpansion

set MIN_DRIVER_VERSION=31.0.101.5122
set "DEFAULT_CONDA_ENV=intel_gpu_llm"

:: Store the original path
set "ORIGINAL_PATH=%cd%"

:: Ask the user if it is the first time running the script
set /p FIRST_TIME="Is this the first time running this script? (yes/no): "

if /i "%FIRST_TIME%"=="yes" (
    goto :FirstTimeSetup
) else (
    goto :SubsequentRun
)

:FirstTimeSetup
:: Get GPU information and save to gpuinfo.txt
wmic path win32_videocontroller get caption,driverversion /format:csv > gpuinfo.txt

:: Remove the first two lines from the output file (header and empty line)
more +2 gpuinfo.txt > temp.txt
move /y temp.txt gpuinfo.txt

set /a count=1
set "GPU_TYPE=none"

:: Read from gpuinfo.txt and print the caption and driver version with numbering
for /f "tokens=2,3 delims=," %%i in (gpuinfo.txt) do (
    set "caption=%%i"
    set "driverversion=%%j"
    rem Remove leading spaces from DriverVersion
    for /f "tokens=* delims= " %%k in ("!driverversion!") do (
        set "driverversion=%%k"
    )
    rem Check if the device is Intel dGPU or iGPU
    set "type="
    echo !caption! | findstr /i "Intel" >nul
    if !errorlevel! == 0 (
        echo !caption! | findstr /i "Arc" >nul
        if !errorlevel! == 0 (
            set "type=Intel dGPU"
        ) else (
            set "type=Intel iGPU"
        )
        set "GPU_TYPE=!type!"
        set "CURRENT_DRIVER_VERSION=!driverversion!"
        goto :IntelGPUFound
    )
    set /a count+=1
)

:IntelGPUFound
if "%GPU_TYPE%"=="none" (
    echo No Intel GPU found. Exiting...
    exit /b 1
)

echo Device !count!: !caption!
echo Type: !GPU_TYPE!
echo Driver Version: !CURRENT_DRIVER_VERSION!

echo Please ensure your Intel GPU driver version is higher than %MIN_DRIVER_VERSION%.
echo If not, please update your driver to the recommended version.

pause

:CheckMiniforge
:: Check if Miniforge is installed
echo Checking for Miniforge installation...
where miniforge >nul 2>nul
if !errorlevel! neq 0 (
    echo Miniforge not found. Installing Miniforge...
    curl -L -o Miniforge3-Windows-x86_64.exe https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe
    start /wait Miniforge3-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /SILENT
    if !errorlevel! neq 0 (
        echo Failed to install Miniforge. Exiting...
        exit /b 1
    )
    del Miniforge3-Windows-x86_64.exe

    :: Ask for Miniforge installation path
    set /p MINIFORGE_PATH="Enter the path where Miniforge is installed: "
    set "PATH=%MINIFORGE_PATH%;%PATH%"
) else (
    where miniforge > temp_miniforge_path.txt
    set /p MINIFORGE_PATH=<temp_miniforge_path.txt
    del temp_miniforge_path.txt
)

:: Create and activate Conda environment
echo Creating and activating Conda environment...
call "%MINIFORGE_PATH%\Scripts\conda.exe" create -n intel_gpu_llm python=3.11 libuv -y
if !errorlevel! neq 0 (
    echo Failed to create Conda environment. Exiting...
    exit /b 1
)
call "%MINIFORGE_PATH%\Scripts\activate" intel_gpu_llm
if !errorlevel! neq 0 (
    echo Failed to activate Conda environment. Exiting...
    exit /b 1
)

:: Install ipex-llm
echo Installing ipex-llm...
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
if !errorlevel! neq 0 (
    echo Failed to install ipex-llm. Exiting...
    exit /b 1
)

:: Install tiktoken
echo Installing tiktoken...
pip install tiktoken transformers_stream_generator einops
if !errorlevel! neq 0 (
    echo Failed to install tiktoken. Exiting...
    exit /b 1
)

:: Install Streamlit
echo Installing Streamlit...
pip install streamlit
if !errorlevel! neq 0 (
    echo Failed to install ipex-llm. Exiting...
    exit /b 1
)

:: Set environment variables based on GPU type
if "%GPU_TYPE%"=="Intel dGPU" (
    echo Setting environment variables for Intel dGPU...
    set "SYCL_CACHE_PERSISTENT=1"
) else (
    echo Setting environment variables for Intel iGPU...
    set "SYCL_CACHE_PERSISTENT=1"
    set "BIGDL_LLM_XMX_DISABLED=1"
)


:: Ask the user for the path to the Streamlit application
set /p STREAMLIT_APP_PATH="Enter the path to the Streamlit application (leave empty to use default: %ORIGINAL_PATH%\intel_llm_chat.py): "
if "%STREAMLIT_APP_PATH%"=="" (
    set "STREAMLIT_APP_PATH=%ORIGINAL_PATH%\intel_llm_chat.py"
)

:: Run Streamlit application
echo Running Streamlit application...
streamlit run "%STREAMLIT_APP_PATH%"
if !errorlevel! neq 0 (
    echo Failed to run Streamlit application. Exiting...
    exit /b 1
)

:: Cleanup
del gpuinfo.txt

echo All tasks completed successfully.
endlocal
pause
exit /b 0

:SubsequentRun
echo Please confirm the following:
echo 1. You have an Intel GPU.
echo 2. Your driver version is higher than %MIN_DRIVER_VERSION%.
set /p CONFIRM_GPU="Do you confirm the above details? (yes/no): "
if /i "%CONFIRM_GPU%"=="no" (
    echo Please ensure your Intel GPU and driver version meet the requirements. Exiting...
    exit /b 1
)

set /p CONDA_PATH="Enter the path to your Miniforge installation (leave empty to use default): "
if "%CONDA_PATH%"=="" (
    set "CONDA_PATH=%MINIFORGE_PATH%"
)

set /p CONDA_ENV="Enter the name of your Conda environment (leave empty to use default: %DEFAULT_CONDA_ENV%): "
if "%CONDA_ENV%"=="" (
    set "CONDA_ENV=%DEFAULT_CONDA_ENV%"
)

cd %CONDA_PATH%
call "%CONDA_PATH%\Scripts\activate" %CONDA_ENV%
if !errorlevel! neq 0 (
    echo Failed to activate Conda environment. Exiting...
    exit /b 1
)

:: Ask the user for the path to the Streamlit application
set /p STREAMLIT_APP_PATH="Enter the path to the Streamlit application (leave empty to use default: %ORIGINAL_PATH%\intel_llm_chat.py): "
if "%STREAMLIT_APP_PATH%"=="" (
    set "STREAMLIT_APP_PATH=%ORIGINAL_PATH%\intel_llm_chat.py"
)

:: Run Streamlit application
echo Running Streamlit application...
streamlit run "%STREAMLIT_APP_PATH%"
if !errorlevel! neq 0 (
    echo Failed to run Streamlit application. Exiting...
    exit /b 1
)

:: Deactivate environment and revert to original path
call conda deactivate
cd %ORIGINAL_PATH%

echo All tasks completed successfully.
endlocal
pause
exit /b 0
