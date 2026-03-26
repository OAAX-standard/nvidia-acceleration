set -e

cd "$(dirname "$0")"

VERSION_FILE="../VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

# Auto-detect platform if not provided
if [ -z "$PLATFORM" ]; then
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        PLATFORM="AARCH64_GLIBC2_38"
    else
        PLATFORM="X86_64"
    fi
    echo "Auto-detected PLATFORM=$PLATFORM"
fi

BUILD_TYPE="${BUILD_TYPE:-Release}"

if [ -z "$TRT_DIR" ]; then
    echo "Error: TRT_DIR is not set."
    echo ""
    echo "  TRT_DIR must point to an extracted TensorRT archive (contains include/ and lib/)."
    echo "  Download the Linux tar.gz for your platform from the NVIDIA developer portal:"
    echo "    https://developer.nvidia.com/tensorrt"
    echo ""
    echo "  Example:"
    echo "    TRT_DIR=/opt/TensorRT-10.16.0.72-x86_64 PLATFORM=X86_64 ./build-runtimes.sh"
    exit 1
fi

if [ ! -d "$TRT_DIR/include" ] || [ ! -d "$TRT_DIR/lib" ]; then
    echo "Error: TRT_DIR=$TRT_DIR does not look like a valid TensorRT directory."
    echo "  Expected include/ and lib/ subdirectories."
    echo "  Download from: https://developer.nvidia.com/tensorrt"
    exit 1
fi

if [ -z "$CUDA_DIR" ]; then
    echo "Error: CUDA_DIR is not set."
    echo ""
    echo "  CUDA_DIR must point to a CUDA toolkit directory (contains include/ and lib/ or lib64/)."
    echo "  Download and install the CUDA Toolkit from:"
    echo "    https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "  For cross-compilation to aarch64, use the target-specific subdirectory:"
    echo "    AARCH64_GLIBC2_34:        CUDA_DIR=/usr/local/cuda/targets/aarch64-linux"
    echo "    AARCH64_GLIBC2_38: CUDA_DIR=/usr/local/cuda/targets/sbsa-linux"
    echo ""
    echo "  Example:"
    echo "    CUDA_DIR=/usr/local/cuda PLATFORM=X86_64 ./build-runtimes.sh"
    exit 1
fi

if [ ! -d "$CUDA_DIR/include" ]; then
    echo "Error: CUDA_DIR=$CUDA_DIR does not look like a valid CUDA directory."
    echo "  Expected an include/ subdirectory."
    echo "  Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

if [ -z "$CUDA_VERSION" ]; then
    echo "Error: CUDA_VERSION is not set."
    echo ""
    echo "  Set CUDA_VERSION to the major version of the CUDA toolkit pointed to by CUDA_DIR."
    echo "  Example: CUDA_VERSION=12 or CUDA_VERSION=13"
    exit 1
fi

echo "Building: PLATFORM=$PLATFORM  VERSION=$VERSION  BUILD_TYPE=$BUILD_TYPE"
echo "          CUDA_VERSION=$CUDA_VERSION  MARCH=${MARCH:-<default>}"
echo "          TRT_DIR=$TRT_DIR"
echo "          CUDA_DIR=$CUDA_DIR"

# MARCH defaults are applied by CMakeLists.txt if not set here
MARCH_CMAKE_ARG=""
if [ -n "$MARCH" ]; then
    MARCH_CMAKE_ARG="-DMARCH=${MARCH}"
fi

BUILD_DIR="build/${PLATFORM}/cuda-${CUDA_VERSION}/${MARCH:-default}"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake \
    -DPLATFORM="$PLATFORM" \
    -DRUNTIME_VERSION="$VERSION" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DTRT_DIR="$TRT_DIR" \
    -DCUDA_DIR="$CUDA_DIR" \
    $MARCH_CMAKE_ARG \
    ../../../..

make -j"$(nproc)"

echo "Build complete. Output: runtime-library/${BUILD_DIR}/bin/"
