@echo off
REM Disable command echoing for cleaner output

setlocal enabledelayedexpansion
REM Enable delayed environment variable expansion

REM Change to script directory
cd /d "%~dp0"

REM Check if the drive letter is mapped
@REM echo Checking if drive letter T: is already mapped...
@REM if exist T:\ (
@REM     echo Drive letter T: is already mapped.
@REM     echo If you want to remap it, please unmap it first.
@REM     echo Use the command: subst T: /d
@REM     exit /b 1
@REM )

REM Map drive letter to the current directory to avoid long paths
echo Mapping drive letter T: to current directory...
subst T: "%~dp0"

REM Check if the drive letter is mapped
if not exist T:\ (
    echo Failed to map drive letter T:
    exit /b 1
)

REM Go to the mapped drive
T:

REM Go to onnxruntime-1.21.1 directory
cd onnxruntime-1.21.1 || exit /b 1

REM Remove and recreate build directory
if exist build rmdir /s /q build

REM Set the CUDA and cuDNN environment variables
set "CUDA_PATH=C:\CUDA\v11.8"
set "CUDNN_PATH=C:\cuDNN\cudnn-windows-x86_64-8.9.7.29_cuda11-archive"

REM Configure the build with CMake
.\build.bat ^
    --config Release ^
    --parallel ^
    --compile_no_warning_as_error ^
    --skip_submodule_sync ^
    --skip_tests ^
    --build_shared_lib ^
    --use_cuda ^
    --cuda_home %CUDA_PATH% ^
    --cudnn_home %CUDNN_PATH%
 
REM Go to root directory
cd ..

REM Remove and recreate output directories
if exist X86_64_WINDOWS_CUDA_11 rmdir /s /q X86_64_WINDOWS_CUDA_11
mkdir X86_64_WINDOWS_CUDA_11
mkdir X86_64_WINDOWS_CUDA_11\include

REM Go to Release build output directory
cd .\onnxruntime-1.21.1\build\Windows\Release

REM Copy artifacts
@echo off
for /R %%f in (*.lib) do (
    REM Copy file to X86_64_WINDOWS_CUDA_11
    copy "%%f" "..\..\..\..\X86_64_WINDOWS_CUDA_11\"
)
for /R %%f in (*.dll) do (
    REM Copy file to X86_64_WINDOWS_CUDA_11
    copy "%%f" "..\..\..\..\X86_64_WINDOWS_CUDA_11\"
)
REM Go back to the root of the dependency
cd ..\..\..\
REM Copy include files to output directory
xcopy .\include\onnxruntime ..\X86_64_WINDOWS_CUDA_11\include\onnxruntime /E /I