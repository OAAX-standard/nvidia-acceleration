#!/bin/bash

# https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
# https://developer.download.nvidia.com/compute/redist/cudnn/
# https://developer.nvidia.com/rdp/cudnn-download

set -e

export CPATH="$BASE_DIR/../nlohmann/single_include"
export CROSS_ROOT="/opt/x86_64-unknown-linux-gnu-gcc-9.5.0"
export COMPILER_PREFIX="x86_64-unknown-linux-gnu-"
export PATH=$CROSS_ROOT/bin:/snap/bin:/bin:/usr/bin:/opt/cmake-3.28.1/bin:/usr/local/bin
export SYSROOT="/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot"

export PATH="/opt/cuda-12.3/bin:$PATH"
export LD_LIBRARY_PATH="/opt/cuda-12.3/targets/x86_64-linux/lib:/opt/cuda-12.3/lib:"
export CUDA_INSTALL_PATH="/opt/cuda-12.3"

export NVCC_LD_LIBRARY_PATH="/opt/cuda-12.3/targets/x86_64-linux/lib,/opt/cuda-12.3/lib,/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot/lib,/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot/lib64"
export LD_LIBRARY_PATH_EXT="/opt/cuda-12.3/targets/x86_64-linux/lib;/opt/cuda-12.3/lib;/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot/lib;/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot/lib64"

export  onnxruntime_CUDNN_HOME=/opt/cudnn-linux-x86_64-8.9.5.29_cuda12-archive
export  onnxruntime_CUDA_HOME=/opt/cuda-12.3/targets/x86_64-linux

export CUDA_ARCHITECTURES="52;60;61;70;75;80;86;87;89;90"

export CPPFLAGS="-I${CROSS_ROOT}/include "
export CFLAGS="-I${CROSS_ROOT}/include "
export AR=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar
export AS=${CROSS_ROOT}/bin/${COMPILER_PREFIX}as
export LD=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld
export RANLIB=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib
export CC=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc
export CXX=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++
export NM=${CROSS_ROOT}/bin/${COMPILER_PREFIX}nm

BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
cd "$BASE_DIR"
ort_foldername="onnxruntime-1.17.3"
cd $ort_foldername

rm -rf build

########################################################################################################################
# TODO: RVE: dangerous, but found no way to fix in other way (expect needs changes in onnxruntime python build script)
sudo mkdir /lib64 || true
sudo mkdir /usr/lib64 || true
sudo mv /lib64/libm.so.6 /lib64/libm.so.6.bak || true
sudo mv /lib64/libmvec.so.1 /lib64/libmvec.so.1.bak || true
sudo mv /lib64/libc.so.6 //lib64/libc.so.6.bak || true
sudo mv /usr/lib64/libc_nonshared.a /usr/lib64/libc_nonshared.a.bak || true
sudo cp $SYSROOT/lib64/libmvec.so.1 /lib64 || true
sudo cp $SYSROOT/lib64/libm.so.6 /lib64 || true
sudo cp $SYSROOT/usr/lib64/libc_nonshared.a /usr/lib64 || true
sudo cp $SYSROOT/lib64/libc.so.6 /lib64 || true
########################################################################################################################

#--disable_contrib_ops
./build.sh --config Release --skip_tests --skip_onnx_tests --build_shared_lib  --parallel --x86 --allow_running_as_root \
--cmake_extra_defines \
  CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  CMAKE_POSITION_INDEPENDENT_CODE=ON \
  CMAKE_SYSTEM_PROCESSOR=x86_64 \
  CMAKE_FIND_ROOT_PATH=${SYSROOT} \
  CMAKE_SYSROOT=${SYSROOT} \
  CPUINFO_TARGET_PROCESSOR=x86_64 \
  CPUINFO_BUILD_BENCHMARKS=OFF \
  CPUINFO_RUNTIME_TYPE=static \
  CMAKE_SYSTEM_NAME=Linux \
  CMAKE_BUILD_TYPE=Release \
  CUDA_NVCC_FLAGS="--library-path ${NVCC_LD_LIBRARY_PATH}" \
  CMAKE_LIBRARY_PATH=${NVCC_LD_LIBRARY_PATH_EXT} \
  CMAKE_C_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc \
  CMAKE_CXX_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++ \
  CMAKE_LINKER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld \
  CMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  CMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  CMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
  CMAKE_CUDA_COMPILER=/opt/cuda-12.3/bin/nvcc \
  onnxruntime_CUDNN_HOME=${onnxruntime_CUDNN_HOME} \
  onnxruntime_CUDA_HOME=${onnxruntime_CUDA_HOME} \
  onnxruntime_BUILD_UNIT_TESTS=OFF \
  onnxruntime_USE_NSYNC=OFF  \
  onnxruntime_ENABLE_CPUINFO=ON \
  onnxruntime_CROSS_COMPILING=ON \
--use_cuda --cudnn_home ${onnxruntime_CUDNN_HOME} \
--cuda_home ${onnxruntime_CUDA_HOME}

########################################################################################################################
# TODO: RVE: dangerous, but found no way to fix in other way (expect needs changes in onnxruntime python build script)
sudo rm /lib64/libm.so.6 || true
sudo rm /lib64/libmvec.so.1 || true
sudo rm /lib64/libc.so.6 || true
sudo rm /usr/lib64/libc_nonshared.a || true
sudo mv /lib64/libm.so.6.bak /lib64/libm.so.6 || true
sudo mv /lib64/libmvec.so.1.bak /lib64/libmvec.so || true
sudo mv /lib64/libc.so.6.bak /lib64/libc.so.6 || true
sudo mv /usr/lib64/libc_nonshared.a.bak /usr/lib64/libc_nonshared.a || true
sudo rm /lib64/*.bak || true
sudo rm /usr/lib64/*.bak || true
######################################################################################################################

printf "\n\n\n\n\n\n"

cd ..

rm -rf X86_64_CUDA_12 || true
mkdir X86_64_CUDA_12  || true

cp -rf ./$ort_foldername/include ./X86_64_CUDA_12  || true
RELEASE="./$ort_foldername/build/Linux/Release"
cp $RELEASE/*.a ./X86_64_CUDA_12 2>>/dev/null  || true
cp $RELEASE/*.so ./X86_64_CUDA_12 2>>/dev/null  || true
shopt -s globstar
cp -rf $RELEASE/_deps/*-build/**/*.a ./X86_64_CUDA_12 2>>/dev/null  || true
cp -rf $RELEASE/_deps/*-build/**/*.so ./X86_64_CUDA_12 2>>/dev/null  || true

printf "\nDone :) \n"
