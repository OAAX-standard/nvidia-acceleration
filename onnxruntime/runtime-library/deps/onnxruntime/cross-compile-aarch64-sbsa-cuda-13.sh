#!/bin/bash
# Cross-compile ONNX Runtime for aarch64 SBSA (Server Base System Architecture).
# Target: NVIDIA DGX Spark / Grace CPU + Blackwell GPU (SM 100/120/121).
# Prerequisites: CUDA 13.1 + cuDNN installed on the build machine.

set -e

# ARM GCC 13.2 toolchain — supports -mcpu=neoverse-v2 for NVIDIA Grace CPU
export CROSS_ROOT="/opt/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu"
export COMPILER_PREFIX="aarch64-none-linux-gnu-"
export PATH=$CROSS_ROOT/bin:/snap/bin:/bin:/usr/bin:/usr/local/bin
export SYSROOT="${CROSS_ROOT}/aarch64-none-linux-gnu/libc/"

export PATH="/usr/local/cuda-13.1/bin:$PATH"
export CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
# sbsa-linux = Server Base System Architecture (aarch64 server/workstation target)
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/targets/sbsa-linux/lib"

export BUILD_CONFIG=Release
export CPU_ARCHITECTURE=aarch64

export onnxruntime_CUDNN_HOME=/opt/cudnn-linux-aarch64-9.14.0.64_cuda13-archive/
export onnxruntime_CUDA_HOME=/usr/local/cuda-13.1/targets/sbsa-linux

# Blackwell server GPUs: B100/B200 (SM 100), GB10 (SM 120), GB10 DGX Spark (SM 121)
export CUDA_ARCHITECTURES="100;120;121"

export CPPFLAGS="-I${CROSS_ROOT}/include "
export CFLAGS="-I${CROSS_ROOT}/include "
export AR=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar
export AS=${CROSS_ROOT}/bin/${COMPILER_PREFIX}as
export LD=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld
export RANLIB=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib
export CC=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc
export CXX=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++
export NM=${CROSS_ROOT}/bin/${COMPILER_PREFIX}nm

BASE_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "$BASE_DIR"
cd onnxruntime-bbd0a80

rm -rf build
export VERBOSE=1
./build.sh --config Release --skip_tests --build_shared_lib --parallel 8 --nvcc_threads 2 --allow_running_as_root \
--cmake_extra_defines \
  CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  CMAKE_CUDA_COMPILER_FORCED=1 \
  CUDA_HOST_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc \
  CMAKE_CUDA_HOST_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++ \
  CMAKE_CUDA_FLAGS="-ccbin ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc --target-directory sbsa-linux" \
  CUDA_NVCC_FLAGS="-ccbin ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc --target-directory sbsa-linux" \
  CMAKE_POSITION_INDEPENDENT_CODE=ON \
  CUDA_TOOLKIT_TARGET_NAME="sbsa-linux" \
  CUDA_TOOLKIT_ROOT_DIR="${onnxruntime_CUDA_HOME}" \
  CPUINFO_TARGET_PROCESSOR=arm64 \
  CMAKE_SYSTEM_PROCESSOR=aarch64 \
  CMAKE_FIND_ROOT_PATH=${SYSROOT} \
  CMAKE_SYSROOT=${SYSROOT} \
  CPUINFO_BUILD_BENCHMARKS=OFF \
  CPUINFO_RUNTIME_TYPE=static \
  CMAKE_SYSTEM_NAME=Linux \
  CMAKE_CROSSCOMPILING=1 \
  CMAKE_BUILD_TYPE=Release \
  CMAKE_C_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc \
  CMAKE_CXX_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++ \
  CMAKE_LINKER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld \
  CMAKE_LIBRARY_ARCHITECTURE=sbsa-linux \
  CMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  CMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  CMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
  CMAKE_CXX_FLAGS="-Wno-psabi -Wno-unused-parameter" \
  CMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
  onnxruntime_CUDNN_HOME=${onnxruntime_CUDNN_HOME} \
  onnxruntime_CUDA_HOME=${onnxruntime_CUDA_HOME} \
  onnxruntime_BUILD_UNIT_TESTS=OFF \
  onnxruntime_USE_NSYNC=ON \
  onnxruntime_ENABLE_CPUINFO=ON \
  onnxruntime_CROSS_COMPILING=ON \
  CUDNN_INCLUDE_DIR=${onnxruntime_CUDNN_HOME}/include \
  cudnn_LIBRARY=${onnxruntime_CUDNN_HOME}/lib/libcudnn.so \
  --use_cuda --cudnn_home ${onnxruntime_CUDNN_HOME} \
  --cuda_home ${onnxruntime_CUDA_HOME}

printf "\n\n\n\n\n\n"

cd ..

rm -rf AARCH64_SBSA_CUDA_13 || true
mkdir AARCH64_SBSA_CUDA_13  || true

cp -rf ./onnxruntime-bbd0a80/include ./AARCH64_SBSA_CUDA_13 || true
RELEASE="./onnxruntime-bbd0a80/build/Linux/Release"
cp $RELEASE/*.a ./AARCH64_SBSA_CUDA_13 2>>/dev/null || true
cp $RELEASE/*.so ./AARCH64_SBSA_CUDA_13 2>>/dev/null || true
shopt -s globstar || true
cp -rf $RELEASE/_deps/*-build/**/*.a ./AARCH64_SBSA_CUDA_13 2>>/dev/null || true
cp -rf $RELEASE/_deps/*-build/**/*.so ./AARCH64_SBSA_CUDA_13 2>>/dev/null || true

printf "\nDone :) \n"
