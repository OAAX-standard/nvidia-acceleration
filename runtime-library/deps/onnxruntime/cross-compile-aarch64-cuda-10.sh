#!/bin/bash

set -e

export CROSS_ROOT="/opt/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu"
export COMPILER_PREFIX="aarch64-linux-gnu-"
export PATH=$CROSS_ROOT/bin:/snap/bin:/bin:/usr/bin:/usr/local/bin
export SYSROOT="/opt/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/aarch64-linux-gnu/libc"

export PATH="/opt/cuda-10.2/bin:$PATH"
export CUDA_COMPILER=/opt/cuda-10.2/bin/nvcc
export LD_LIBRARY_PATH=/opt/aarch64/libcudnn8.2.1.32/usr/lib:/opt/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/aarch64-linux-gnu/libc/lib:/opt/cuda-10.2/targets/aarch64-linux/lib:/opt/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/aarch64-linux-gnu/libc

export BUILD_CONFIG=Release
export CPU_ARCHITECTURE=aarch64

export  onnxruntime_CUDNN_HOME=/opt/aarch64/libcudnn8.2.1.32/usr
export  onnxruntime_CUDA_HOME=/opt/cuda-10.2/targets/aarch64-linux

#all but jetson nano
export CUDA_ARCHITECTURES="53"

export CPPFLAGS="-I${CROSS_ROOT}/include -I/opt/cuda-10.2/targets/aarch64-linux/incude -L/opt/aarch64/libcudnn8.2.1.32/usr/lib -latomic "
export CFLAGS="-I${CROSS_ROOT}/include -I/opt/cuda-10.2/targets/aarch64-linux/include -L/opt/aarch64/libcudnn8.2.1.32/usr/lib -latomic "
export AR=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar
export AS=${CROSS_ROOT}/bin/${COMPILER_PREFIX}as
export LD=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld
export RANLIB=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib
export CC=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc
export CXX=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++
export NM=${CROSS_ROOT}/bin/${COMPILER_PREFIX}nm

BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
cd "$BASE_DIR"

cd onnxruntime-1.11.1

rm -rf build

./build.sh --config Release  --skip_tests --build_shared_lib  --parallel --arm64 --cmake_extra_defines \
  CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  CMAKE_CUDA_COMPILER_FORCED=1 \
  onnxruntime_ONNX_CUSTOM_PROTOC_EXECUTABLE=$BASE_DIR/__protoc-3.18.0__/bin/protoc \
  ONNX_CUSTOM_PROTOC_EXECUTABLE=$BASE_DIR/__protoc-3.18.0__/bin/protoc \
  CUDA_HOME=/opt/aarch64/cuda-10.2 \
  CUDA_HOST_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc \
  CMAKE_CUDA_FLAGS="-ccbin ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc --target-directory aarch64-linux -L/opt/aarch64/libcudnn8.2.1.32/usr/lib -Xcompiler -latomic" \
  CUDA_NVCC_FLAGS="-ccbin ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc --target-directory aarch64-linux -L/opt/aarch64/libcudnn8.2.1.32/usr/lib -Xcompiler -latomic" \
  CMAKE_POSITION_INDEPENDENT_CODE=ON \
  CUDA_TOOLKIT_TARGET_NAME="aarch64-linux" \
  CPUINFO_TARGET_PROCESSOR=arm64 \
  CMAKE_SYSTEM_PROCESSOR=aarch64 \
  CMAKE_CUDA_COMPILER=nvcc \
  CMAKE_FIND_ROOT_PATH=${SYSROOT} \
  CMAKE_SYSROOT=${SYSROOT} \
  CPUINFO_BUILD_BENCHMARKS=OFF \
  CPUINFO_RUNTIME_TYPE=static \
  CMAKE_SYSTEM_NAME=Linux \
  CMAKE_SYSTEM_PROCESSOR=aarch64 \
  CMAKE_CROSSCOMPILING=1 \
  CMAKE_BUILD_TYPE=Release \
  CMAKE_C_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc \
  CMAKE_CXX_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++ \
  CMAKE_LIBRARY_ARCHITECTURE=aarch64-linux \
  CMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  CMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  CMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
  CMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
  onnxruntime_CUDNN_HOME=${onnxruntime_CUDNN_HOME} \
  onnxruntime_CUDA_HOME=${onnxruntime_CUDA_HOME} \
  onnxruntime_BUILD_UNIT_TESTS=OFF \
  onnxruntime_USE_NSYNC=ON  \
  onnxruntime_ENABLE_CPUINFO=ON \
  onnxruntime_CROSS_COMPILING=ON \
  CMAKE_CXX_FLAGS="-latomic" \
  CMAKE_C_FLAGS="-latomic" \
--use_cuda --cudnn_home ${onnxruntime_CUDNN_HOME} \
--cuda_home ${onnxruntime_CUDA_HOME}

printf "\n\n\n\n\n\n"

cd ..

rm -rf AARCH64_CUDA_10 || true
mkdir AARCH64_CUDA_10  || true

cp -rf ./onnxruntime-1.11.1/include ./AARCH64_CUDA_10   || true
RELEASE="./onnxruntime-1.11.1/build/Linux/Release"
cp $RELEASE/*.a ./AARCH64_CUDA_10 2>>/dev/null  || true
cp $RELEASE/*.so ./AARCH64_CUDA_10 2>>/dev/null  || true
shopt -s globstar
cp -rf $RELEASE/external/**/*.a ./AARCH64_CUDA_10 2>>/dev/null  || true
cp -rf $RELEASE/external/**/*.so ./AARCH64_CUDA_10 2>>/dev/null  || true


printf "\nDone :) \n"
