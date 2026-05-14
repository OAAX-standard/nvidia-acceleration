#!/bin/bash
# Sets up the build environment for the runtime library.
# Downloads cross-compilation toolchains and CUDA into .deps/,
# and unpacks TensorRT archives from tensorrt-archives/ into .deps/.
#
# TensorRT archives must be downloaded manually before running this script.
# Run scripts/unpack-trt.sh for instructions on which archives are needed.
#
# Usage: bash setup-env.sh [--force]
#   --force   Ignore the setup cache and re-run all steps.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_DIR="${REPO_ROOT}/.deps"
STAMP_FILE="${DEPS_DIR}/.setup-complete"

FORCE=false
for arg in "$@"; do
    [ "$arg" = "--force" ] && FORCE=true
done

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y wget curl build-essential python3 xz-utils

# ── CMake ─────────────────────────────────────────────────────────────────────
if ! command -v cmake &>/dev/null || [ "$(cmake --version | head -1 | awk '{print $3}')" \< "3.14" ]; then
    host_platform=$(uname -m)
    wget -q "https://cmake.org/files/v3.31/cmake-3.31.7-linux-${host_platform}.sh" \
        -O /tmp/cmake-install.sh
    chmod u+x /tmp/cmake-install.sh
    mkdir -p /opt/cmake-3.31.7
    /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.7
    rm /tmp/cmake-install.sh
    ln -fs /opt/cmake-3.31.7/bin/* /usr/local/bin
    echo "CMake installed."
else
    echo "CMake already available: $(cmake --version | head -1)"
fi

if [ "$FORCE" = false ] && [ -f "$STAMP_FILE" ]; then
    echo "Environment already set up (remove .deps/.setup-complete or use --force to re-run)."
    exit 0
fi


mkdir -p "${DEPS_DIR}/toolchains"

# ── Cross-compilation toolchains ──────────────────────────────────────────────
declare -A TOOLCHAINS=(
    ["x86-64-v2--glibc--stable-2022.08-1"]="https://oaax.nbg1.your-objectstorage.com/toolchains/x86-64-v2--glibc--stable-2022.08-1.tar.bz2"
    ["gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu"]="https://oaax.nbg1.your-objectstorage.com/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu.tar.xz"
    ["arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu"]="https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"
)

for name in "${!TOOLCHAINS[@]}"; do
    dest="${DEPS_DIR}/toolchains/${name}"
    if [ -d "$dest" ]; then
        echo "Toolchain already present: ${name} — skipping"
        continue
    fi
    url="${TOOLCHAINS[$name]}"
    filename="${url##*/}"
    echo "Downloading toolchain: ${name}"
    wget -nv -c "$url" -O "/tmp/${filename}"
    mkdir -p "$dest"
    tar xf "/tmp/${filename}" --strip-components=1 -C "$dest"
    rm -f "/tmp/${filename}"
    echo "Extracted: ${name}"
done

# ── CUDA ──────────────────────────────────────────────────────────────────────
echo "Downloading CUDA components..."
bash "${SCRIPT_DIR}/download-cuda.sh" --output "${DEPS_DIR}/cuda"

# ── TensorRT ──────────────────────────────────────────────────────────────────
echo "Unpacking TensorRT archives..."
bash "${SCRIPT_DIR}/unpack-trt.sh"

touch "$STAMP_FILE"
echo ""
echo "Environment setup complete. All dependencies are in .deps/"
