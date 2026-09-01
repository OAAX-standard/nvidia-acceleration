set -e

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"
REPO_ROOT="$(cd "../.." && pwd)"
DEPS_DIR="${REPO_ROOT}/.deps"

VERSION_FILE="${REPO_ROOT}/VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

BUILD_TYPE="${BUILD_TYPE:-Release}"

# Full build matrix: "PLATFORM:CUDA_VERSION:TRT_VERSION:march"
# One march per platform — inference runs on GPU so CPU micro-arch tuning is unnecessary.
# AARCH64_JETSON (Jetson): TRT 10 + CUDA 12 only — JetPack does not ship TRT 11.
# AARCH64_SBSA (DGX Spark/Grace): TRT 10 (CUDA 12+13), TRT 11 (CUDA 13 only — no SBSA/CUDA 12 archive).
MATRIX=(
    # TRT 10
    "X86_64:12:10:x86-64"
    "X86_64:13:10:x86-64"
    "AARCH64_JETSON:12:10:armv8-a"
    "AARCH64_SBSA:12:10:armv8-a"
    "AARCH64_SBSA:13:10:armv8-a"
    # TRT 11
    "X86_64:12:11:x86-64"
    "X86_64:13:11:x86-64"
    "AARCH64_SBSA:13:11:armv8-a"
)

# If PLATFORM, CUDA_VERSION, or TRT_VERSION is unset, iterate over the matrix.
if [ -z "$PLATFORM" ] || [ -z "$CUDA_VERSION" ] || [ -z "$TRT_VERSION" ]; then
    for entry in "${MATRIX[@]}"; do
        plat="${entry%%:*}"; rest="${entry#*:}"
        cuda="${rest%%:*}"; rest="${rest#*:}"
        trt="${rest%%:*}";  march="${rest#*:}"
        [ -n "$PLATFORM" ]    && [ "$PLATFORM"    != "$plat" ] && continue
        [ -n "$CUDA_VERSION" ] && [ "$CUDA_VERSION" != "$cuda" ] && continue
        [ -n "$TRT_VERSION" ]  && [ "$TRT_VERSION"  != "$trt"  ] && continue
        PLATFORM=$plat CUDA_VERSION=$cuda TRT_VERSION=$trt MARCH=$march bash "$SCRIPT_PATH"
    done
    exit 0
fi

# Default TRT_DIR from .deps/tensorrt/ based on TRT_VERSION, CUDA_VERSION and PLATFORM
if [ -z "$TRT_DIR" ]; then
    if [ -n "$TRT_VERSION" ] && [ -n "$CUDA_VERSION" ]; then
        case "$PLATFORM" in
            X86_64)         trt_platform="x86_64" ;;
            AARCH64_JETSON) trt_platform="aarch64-linux" ;;
            AARCH64_SBSA)   trt_platform="sbsa-linux" ;;
        esac
        TRT_DIR="${DEPS_DIR}/tensorrt/trt${TRT_VERSION}/cuda-${CUDA_VERSION}/${trt_platform}"
    fi
fi

if [ -z "$TRT_DIR" ]; then
    echo "Error: TRT_DIR is not set."
    echo ""
    echo "  TRT_DIR must point to an extracted TensorRT archive (contains include/ and lib/)."
    echo "  Run scripts/unpack-trt.sh to extract archives into .deps/tensorrt/, or set TRT_DIR manually."
    echo ""
    echo "  Example:"
    echo "    TRT_DIR=.deps/tensorrt/trt10/cuda-12/x86_64 PLATFORM=X86_64 CUDA_VERSION=12 TRT_VERSION=10 ./build-runtimes.sh"
    exit 1
fi

if [ ! -d "$TRT_DIR/include" ] || [ ! -d "$TRT_DIR/lib" ]; then
    echo "Error: TRT_DIR=$TRT_DIR does not look like a valid TensorRT directory."
    echo "  Expected include/ and lib/ subdirectories."
    echo "  Download from: https://developer.nvidia.com/tensorrt"
    exit 1
fi

# Default CUDA_DIR from .deps/cuda/ based on CUDA_VERSION and PLATFORM
if [ -z "$CUDA_DIR" ]; then
    if [ -n "$CUDA_VERSION" ]; then
        case "$PLATFORM" in
            X86_64)             cuda_platform="x86_64" ;;
            AARCH64_JETSON)  cuda_platform="aarch64-linux" ;;
            AARCH64_SBSA)  cuda_platform="sbsa-linux" ;;
        esac
        CUDA_DIR="${DEPS_DIR}/cuda/${CUDA_VERSION}/${cuda_platform}"
    fi
fi

if [ -z "$CUDA_DIR" ]; then
    echo "Error: CUDA_DIR is not set."
    echo ""
    echo "  CUDA_DIR must point to a CUDA toolkit directory (contains include/ and lib/ or lib64/)."
    echo "  Run scripts/setup-env.sh to download CUDA into .deps/cuda/, or set CUDA_DIR manually."
    echo ""
    echo "  For cross-compilation to aarch64, use the target-specific subdirectory:"
    echo "    AARCH64_JETSON:  CUDA_DIR=.deps/cuda/12/aarch64-linux"
    echo "    AARCH64_SBSA:  CUDA_DIR=.deps/cuda/13/sbsa-linux"
    echo ""
    echo "  Example:"
    echo "    CUDA_VERSION=12 PLATFORM=X86_64 ./build-runtimes.sh"
    exit 1
fi

if [ ! -d "$CUDA_DIR/include" ]; then
    echo "Error: CUDA_DIR=$CUDA_DIR does not look like a valid CUDA directory."
    echo "  Expected an include/ subdirectory."
    echo "  Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Map PLATFORM to artifact path components
case "$PLATFORM" in
    X86_64)         arch="x86_64";        ubuntu_version="22.04" ;;
    AARCH64_JETSON) arch="aarch64-jetson"; ubuntu_version="22.04" ;;
    AARCH64_SBSA)   arch="aarch64-sbsa";   ubuntu_version="24.04" ;;
esac

# MARCH defaults are applied by CMakeLists.txt if not set here
MARCH_CMAKE_ARG=""
if [ -n "$MARCH" ]; then
    MARCH_CMAKE_ARG="-DMARCH=${MARCH}"
fi

ARTIFACT="library-trt${TRT_VERSION}-cuda_${CUDA_VERSION}"
BUILD_DIR="build/NVIDIA_TENSORRT/${arch}/Ubuntu/${ubuntu_version}/${ARTIFACT}"

echo "Building: PLATFORM=$PLATFORM  VERSION=$VERSION  BUILD_TYPE=$BUILD_TYPE"
echo "          TRT_VERSION=$TRT_VERSION  CUDA_VERSION=$CUDA_VERSION  MARCH=${MARCH:-<default>}"
echo "          TRT_DIR=$TRT_DIR"
echo "          CUDA_DIR=$CUDA_DIR"
echo "          Ubuntu: ${ubuntu_version}  Output: tensorrt/runtime-library/${BUILD_DIR}.tar.gz"

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
    "${REPO_ROOT}/tensorrt/runtime-library"

make -j"$(nproc)"

# Package libraries directly into the archive (no bin/ subdirectory)
cd ..
tar czf "${ARTIFACT}.tar.gz" -C "${ARTIFACT}/bin" .
echo "Build complete. Archive: tensorrt/runtime-library/${BUILD_DIR}.tar.gz"
