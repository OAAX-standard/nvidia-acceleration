#!/bin/bash
# Build the C++ runtime tests against the prebuilt TRT runtime library.
# Run tensorrt/runtime-library/build-runtimes.sh first to populate the build dir.
set -e

cd "$(dirname "$0")"

# Auto-detect the directory containing libRuntimeLibrary.so if not set
if [ -z "$RUNTIME_LIB_DIR" ]; then
    RUNTIME_LIB_DIR=$(find "$(pwd)/../../tensorrt/runtime-library/build" -name "libRuntimeLibrary.so" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    RUNTIME_LIB_DIR="${RUNTIME_LIB_DIR:-$(pwd)/../../tensorrt/runtime-library/build}"
fi
BUILD_DIR="$(pwd)/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DRUNTIME_LIB_DIR="$RUNTIME_LIB_DIR" -DCMAKE_BUILD_TYPE=Release
make -j "$(nproc)"

# Symlink all shared libs from the runtime lib directory for $ORIGIN RPATH
for lib in "$RUNTIME_LIB_DIR"/*.so*; do
    [ -e "$lib" ] || continue
    ln -sf "$lib" "$(basename "$lib")"
done

echo "Tests built in: $BUILD_DIR"
