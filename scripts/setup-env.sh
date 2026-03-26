#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change the working directory to the directory of the script
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

# ── TensorRT archives ─────────────────────────────────────────────────────────
# TRT archives are large (~2 GB each). Host them on your own storage and set
# the URLs below (or override with TRT_X86_URL / TRT_AARCH64_URL env vars).
TRT_X86_URL="${TRT_X86_URL:-https://oaax.nbg1.your-objectstorage.com/toolchains/TensorRT-10.16.0.72.Linux.x86_64-gnu.cuda-13.2.tar.gz}"
TRT_AARCH64_URL="${TRT_AARCH64_URL:-https://oaax.nbg1.your-objectstorage.com/toolchains/TensorRT-10.16.0.72.Linux.aarch64-gnu.cuda-13.2.tar.gz}"

for url in "$TRT_X86_URL" "$TRT_AARCH64_URL"; do
    filename=$(get_filename_from_url "$url")
    echo "Downloading: $url"
    wget -nv -c "$url"
    # Extract to a versioned directory so x86 and aarch64 don't overwrite each other
    if echo "$filename" | grep -q "x86_64"; then
        mkdir -p /opt/TensorRT-x86_64
        tar xf "$filename" --strip-components=1 -C /opt/TensorRT-x86_64
    else
        mkdir -p /opt/TensorRT-aarch64
        tar xf "$filename" --strip-components=1 -C /opt/TensorRT-aarch64
    fi
    rm -f "$filename"
    echo "Extracted: $filename"
done

# ── CUDA toolkit (headers + cross-compilation stubs) ─────────────────────────
# Install CUDA 13 from NVIDIA's apt repository for both native and cross builds.
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
apt-get update -y
# Native x86_64 CUDA runtime + headers
apt-get install -y cuda-cudart-dev-13-2 cuda-libraries-dev-13-2
# aarch64 cross-compilation stubs (sbsa = server base system architecture)
apt-get install -y cuda-cross-aarch64-13-2 || true

# ── CMake ─────────────────────────────────────────────────────────────────────
host_platform=$(uname -m)
wget -q "https://cmake.org/files/v3.31/cmake-3.31.7-linux-${host_platform}.sh" \
    -O /tmp/cmake-install.sh
chmod u+x /tmp/cmake-install.sh
mkdir -p /opt/cmake-3.31.7
/tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.7
rm /tmp/cmake-install.sh
ln -fs /opt/cmake-3.31.7/bin/* /usr/local/bin
