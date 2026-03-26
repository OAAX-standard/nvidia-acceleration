#!/bin/bash

set -e

cd "$(dirname "$0")"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y wget curl build-essential

# ── Cross-compilation toolchains ──────────────────────────────────────────────
toolchain_urls=(
    "https://oaax.nbg1.your-objectstorage.com/toolchains/x86-64-v2--glibc--stable-2022.08-1.tar.bz2"
    "https://oaax.nbg1.your-objectstorage.com/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu.tar.xz"
    "https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"
)

function get_filename_from_url() {
    local url=$1
    echo "${url##*/}"
}

for url in "${toolchain_urls[@]}"; do
    filename=$(get_filename_from_url "$url")
    echo "Downloading: $url"
    wget -nv -c "$url"
    tar xzf "$filename" -C /opt 2>/dev/null || tar xf "$filename" -C /opt
    rm -f "$filename"
    echo "Extracted: $filename"
done

# ── CMake ─────────────────────────────────────────────────────────────────────
host_platform=$(uname -m)
wget -q "https://cmake.org/files/v3.31/cmake-3.31.7-linux-${host_platform}.sh" \
    -O /tmp/cmake-install.sh
chmod u+x /tmp/cmake-install.sh
mkdir -p /opt/cmake-3.31.7
/tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.7
rm /tmp/cmake-install.sh
ln -fs /opt/cmake-3.31.7/bin/* /usr/local/bin

# ── TensorRT and CUDA ─────────────────────────────────────────────────────────
# TensorRT and CUDA must be installed manually on the runner before building.
# They must be downloaded directly from NVIDIA:
#
#   TensorRT: https://developer.nvidia.com/tensorrt
#     - Download the Linux tar.gz archive for your platform (x86_64 and/or aarch64)
#     - Extract to a directory; the directory must contain include/ and lib/
#
#   CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
#     - Install the CUDA toolkit for the host platform
#     - For cross-compilation to aarch64, also install the cross-compilation packages
#       so that targets/aarch64-linux/ and targets/sbsa-linux/ exist under the CUDA root
#
# In CI, set the following secrets per CUDA version:
#   TRT_DIR_X86_CUDA12, TRT_DIR_X86_CUDA13
#   TRT_DIR_AARCH64_CUDA12, TRT_DIR_AARCH64_CUDA13
#   CUDA_ROOT_12, CUDA_ROOT_13
#
# TRT_DIR and CUDA_DIR are validated by build-runtimes.sh at build time.

echo "Environment setup complete. TensorRT and CUDA paths are validated per build."
