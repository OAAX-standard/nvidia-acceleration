#!/bin/bash
# Runs the runtime library build inside a clean ubuntu:22.04 container,
# exactly mirroring the CI environment. This catches issues that are masked
# by system-installed CUDA/toolchains on the host machine.
#
# Usage:
#   bash scripts/build-in-docker.sh [PLATFORM=...] [CUDA_VERSION=...] [MARCH=...]
#
# Environment variables are forwarded to build-runtimes.sh:
#   PLATFORM      e.g. X86_64, AARCH64_JETSON, AARCH64_SBSA (default: all)
#   CUDA_VERSION  e.g. 12, 13 (default: all)
#   MARCH         e.g. x86-64-v3 (default: all for the given platform)
#   BUILD_TYPE    Release or Debug (default: Release)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve .deps — follow symlink if present
DEPS_DIR="${REPO_ROOT}/.deps"
if [ -L "$DEPS_DIR" ]; then
    DEPS_DIR="$(readlink -f "$DEPS_DIR")"
fi

if [ ! -d "$DEPS_DIR" ]; then
    echo "Error: .deps directory not found at ${DEPS_DIR}"
    echo "  Run bash scripts/setup-env.sh first, or create a .deps symlink."
    exit 1
fi

# Forward build matrix env vars if set
ENV_ARGS=()
[ -n "$PLATFORM" ]     && ENV_ARGS+=(-e "PLATFORM=$PLATFORM")
[ -n "$CUDA_VERSION" ] && ENV_ARGS+=(-e "CUDA_VERSION=$CUDA_VERSION")
[ -n "$MARCH" ]        && ENV_ARGS+=(-e "MARCH=$MARCH")
[ -n "$BUILD_TYPE" ]   && ENV_ARGS+=(-e "BUILD_TYPE=$BUILD_TYPE")

echo "Building in clean ubuntu:22.04 container..."
echo "  Repo:  ${REPO_ROOT}"
echo "  Deps:  ${DEPS_DIR}"
[ ${#ENV_ARGS[@]} -gt 0 ] && echo "  Env:   ${ENV_ARGS[*]}"
echo ""

docker run --rm \
    -v "${REPO_ROOT}":/workspace \
    -v "${DEPS_DIR}":/workspace/.deps \
    -w /workspace \
    "${ENV_ARGS[@]}" \
    ubuntu:22.04 \
    bash -c '
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -q
        apt-get install -y -q wget curl build-essential python3 xz-utils git

        # Install CMake (same logic as setup-env.sh)
        if ! command -v cmake &>/dev/null; then
            host_platform=$(uname -m)
            wget -q "https://cmake.org/files/v3.31/cmake-3.31.7-linux-${host_platform}.sh" \
                -O /tmp/cmake-install.sh
            chmod u+x /tmp/cmake-install.sh
            mkdir -p /opt/cmake-3.31.7
            /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.7
            ln -fs /opt/cmake-3.31.7/bin/* /usr/local/bin
        fi

        bash runtime-library/build-runtimes.sh
    '

echo ""
echo "Build complete. Artifacts in runtime-library/build/NVIDIA/"
