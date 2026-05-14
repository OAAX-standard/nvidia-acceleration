@echo off
REM Disable command echoing for cleaner output

setlocal enabledelayedexpansion
REM Enable delayed environment variable expansion

REM Change to script directory
cd /d "%~dp0"

REM Check if the drive letter is mapped
echo Checking if drive letter T: is already mapped...
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

REM Go to onnxruntime-1bbd0a80 directory
cd onnxruntime-bbd0a80 || exit /b 1

REM Remove and recreate build directory
if exist build rmdir /s /q build

REM Set the CUDA and cuDNN environment variables
set "CUDA_PATH=C:\CUDA\v13.0"
set "CUDNN_PATH=C:\cuDNN\cudnn-windows-x86_64-9.14.0.64_cuda13-archive"

set "VCVER=14.44.35207"
set "VCDIR=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\%VCVER%"
set "PATH=%VCDIR%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%VCDIR%\include;%INCLUDE%"
set "LIB=%VCDIR%\lib\x64;%LIB%"

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
    --cudnn_home %CUDNN_PATH% ^
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=75;80;86;89;90;100;120
    --parallel 4
    --nvcc_threads 1
 
REM Go to root directory
cd ..

REM Remove and recreate output directories
if exist X86_64_WINDOWS_CUDA_13 rmdir /s /q X86_64_WINDOWS_CUDA_13
mkdir X86_64_WINDOWS_CUDA_13
mkdir X86_64_WINDOWS_CUDA_13\include

REM Go to Release build output directory
cd .\onnxruntime-bbd0a80\build\Windows\Release

REM Copy artifacts
@echo off
for /R %%f in (*.lib) do (
    REM Copy file to X86_64_WINDOWS_CUDA_13
    copy "%%f" "..\..\..\..\X86_64_WINDOWS_CUDA_13\"
)
for /R %%f in (*.dll) do (
    REM Copy file to X86_64_WINDOWS_CUDA_13
    copy "%%f" "..\..\..\..\X86_64_WINDOWS_CUDA_13\"
)
REM Go back to the root of the dependency
cd ..\..\..\
REM Copy include files to output directory
xcopy .\include\onnxruntime ..\X86_64_WINDOWS_CUDA_13\include\onnxruntime /E /I