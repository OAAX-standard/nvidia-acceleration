#!/bin/bash
# Cross-compile system_scanner and optionally run it on the target.
#
# Usage:
#   PLATFORM=X86_64 bash build-scanner.sh
#   PLATFORM=AARCH64_JETSON SSH_TARGET=user@jetson bash build-scanner.sh
#   PLATFORM=AARCH64_SBSA   SSH_TARGET=user@grace  bash build-scanner.sh
#
# Environment variables:
#   PLATFORM     X86_64 | AARCH64_JETSON | AARCH64_SBSA  (default: X86_64)
#   SSH_TARGET   user@host — copy and run on a remote machine after building

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_DIR="${REPO_ROOT}/.deps"

PLATFORM="${PLATFORM:-X86_64}"

case "$PLATFORM" in
    X86_64)
        CROSS_ROOT="${DEPS_DIR}/toolchains/x86-64-v2--glibc--stable-2022.08-1"
        COMPILER_PREFIX="x86_64-linux-"
        SYSTEM_PROCESSOR="x86_64"
        MARCH="x86-64"
        ;;
    AARCH64_JETSON)
        CROSS_ROOT="${DEPS_DIR}/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu"
        COMPILER_PREFIX="aarch64-none-linux-gnu-"
        SYSTEM_PROCESSOR="aarch64"
        MARCH="armv8-a"
        ;;
    AARCH64_SBSA)
        CROSS_ROOT="${DEPS_DIR}/toolchains/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu"
        COMPILER_PREFIX="aarch64-none-linux-gnu-"
        SYSTEM_PROCESSOR="aarch64"
        MARCH="armv8-a"
        ;;
    *)
        echo "Error: Unknown PLATFORM='$PLATFORM'"
        echo "  Supported: X86_64, AARCH64_JETSON, AARCH64_SBSA"
        exit 1
        ;;
esac

if [ ! -f "${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc" ]; then
    echo "Error: toolchain not found at ${CROSS_ROOT}"
    echo "  Run scripts/setup-env.sh to download toolchains."
    exit 1
fi

BUILD_DIR="${SCRIPT_DIR}/build/${PLATFORM}"
BINARY="${BUILD_DIR}/system_scanner"

echo "Building: PLATFORM=$PLATFORM  MARCH=$MARCH"
echo "          Output: ${BINARY}"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR="${SYSTEM_PROCESSOR}" \
    -DCMAKE_C_COMPILER="${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc" \
    -DCMAKE_CXX_COMPILER="${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++" \
    -DCMAKE_CXX_FLAGS="-march=${MARCH}" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_BUILD_TYPE=Release \
    "${SCRIPT_DIR}"

make -j"$(nproc)"
echo "Build complete: ${BINARY}"

# Run
if [ -n "$SSH_TARGET" ]; then
    echo "Copying to ${SSH_TARGET} and running..."
    scp "$BINARY" "${SSH_TARGET}:/tmp/system_scanner"
    ssh "$SSH_TARGET" "/tmp/system_scanner; rm -f /tmp/system_scanner"
elif [ "$(uname -m)" = "$SYSTEM_PROCESSOR" ]; then
    echo ""
    "$BINARY"
else
    echo ""
    echo "To run on target:"
    echo "  scp ${BINARY} <user@host>:/tmp/system_scanner"
    echo "  ssh <user@host> /tmp/system_scanner"
    echo ""
    echo "Or set SSH_TARGET=user@host and re-run this script."
fi
