set -e

cd "$(dirname "$0")"

VERSION_FILE="../../VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

# Auto-detect platform if not provided
if [ -z "$PLATFORM" ]; then
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        PLATFORM="AARCH64_SERVER"
    else
        PLATFORM="X86_64"
    fi
    echo "Auto-detected PLATFORM=$PLATFORM"
fi

BUILD_TYPE="${BUILD_TYPE:-Release}"

# TRT_DIR: path to TensorRT installation (e.g. TensorRT-10.x.y.z/ from the archive)
# CUDA_DIR: path to CUDA toolkit (e.g. /usr/local/cuda or /opt/cuda-13.0)
if [ -z "$TRT_DIR" ]; then
    echo "Error: TRT_DIR is not set."
    echo "  Set it to the extracted TensorRT archive directory (must contain include/ and lib/)."
    echo "  Example: TRT_DIR=/opt/TensorRT-10.16.0.72 PLATFORM=X86_64 ./build-runtimes.sh"
    exit 1
fi
if [ -z "$CUDA_DIR" ]; then
    echo "Error: CUDA_DIR is not set."
    echo "  Set it to the CUDA toolkit directory (must contain include/ and lib/ or lib64/)."
    echo "  Example: CUDA_DIR=/usr/local/cuda PLATFORM=X86_64 ./build-runtimes.sh"
    exit 1
fi

echo "Building: PLATFORM=$PLATFORM  VERSION=$VERSION  BUILD_TYPE=$BUILD_TYPE"
echo "          TRT_DIR=$TRT_DIR"
echo "          CUDA_DIR=$CUDA_DIR"

BUILD_DIR="build/${PLATFORM}"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake \
    -DPLATFORM="$PLATFORM" \
    -DRUNTIME_VERSION="$VERSION" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DTRT_DIR="$TRT_DIR" \
    -DCUDA_DIR="$CUDA_DIR" \
    ../..

make -j"$(nproc)"

echo "Build complete. Output: ${BUILD_DIR}/bin/"
