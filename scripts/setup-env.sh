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
    "https://oaax.nbg1.your-objectstorage.com/toolchains/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu.tar.xz"
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
# TensorRT and CUDA must be installed manually before running builds.
# They must be downloaded directly from NVIDIA:
#
#   TensorRT: https://developer.nvidia.com/tensorrt
#     - Download the Linux tar.gz archive for your platform (x86_64 and/or aarch64)
#     - Extract to a directory and set TRT_DIR to that path
#     - The directory must contain include/ and lib/
#
#   CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
#     - Install the CUDA toolkit for your host platform
#     - For cross-compilation to aarch64, also install the cross-compilation packages
#       (targets/aarch64-linux/ and targets/sbsa-linux/ under the CUDA install root)
#     - Set CUDA_DIR to the appropriate target directory
#
# Expected paths per platform:
#   X86_64:        TRT_DIR=<trt-x86_64>/  CUDA_DIR=/usr/local/cuda
#   AARCH64:       TRT_DIR=<trt-aarch64>/ CUDA_DIR=/usr/local/cuda/targets/aarch64-linux
#   AARCH64_SERVER:TRT_DIR=<trt-aarch64>/ CUDA_DIR=/usr/local/cuda/targets/sbsa-linux
#
# In CI, set TRT_DIR_X86, TRT_DIR_AARCH64, and CUDA_ROOT as environment variables
# or GitHub Actions secrets pointing to where you have installed these packages.

ok=true

if [ -z "$TRT_DIR_X86" ] || [ ! -d "$TRT_DIR_X86/lib" ]; then
    echo ""
    echo "WARNING: TRT_DIR_X86 is not set or invalid."
    echo "  Download the TensorRT Linux x86_64 tar.gz from:"
    echo "    https://developer.nvidia.com/tensorrt"
    echo "  Extract it and set TRT_DIR_X86 to the extracted directory."
    ok=false
fi

if [ -z "$TRT_DIR_AARCH64" ] || [ ! -d "$TRT_DIR_AARCH64/lib" ]; then
    echo ""
    echo "WARNING: TRT_DIR_AARCH64 is not set or invalid."
    echo "  Download the TensorRT Linux aarch64 tar.gz from:"
    echo "    https://developer.nvidia.com/tensorrt"
    echo "  Extract it and set TRT_DIR_AARCH64 to the extracted directory."
    ok=false
fi

if [ -z "$CUDA_ROOT" ] || [ ! -d "$CUDA_ROOT/include" ]; then
    echo ""
    echo "WARNING: CUDA_ROOT is not set or invalid."
    echo "  Download and install the CUDA Toolkit from:"
    echo "    https://developer.nvidia.com/cuda-downloads"
    echo "  Set CUDA_ROOT to the CUDA installation root (e.g. /usr/local/cuda)."
    ok=false
fi

if [ "$ok" = false ]; then
    echo ""
    echo "Build will fail until the above dependencies are installed."
    echo "Re-run this script or set the variables before building."
    exit 1
fi

echo "TensorRT and CUDA dependencies found. Ready to build."
