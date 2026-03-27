set -e

cd "$(dirname "$0")"
REPO_ROOT="$(cd ".." && pwd)"
DEPS_DIR="${REPO_ROOT}/.deps"

VERSION_FILE="../VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

BUILD_TYPE="${BUILD_TYPE:-Release}"

# Full build matrix: "PLATFORM:CUDA_VERSION:march1 march2 ..."
# AARCH64_JETSON (Jetson): CUDA 12 only — no linux-aarch64 in CUDA 13 redistrib.
# AARCH64_SBSA (DGX Spark/Grace): CUDA 12 and 13 both have SBSA TRT packages.
MATRIX=(
    "X86_64:12:x86-64 x86-64-v3 x86-64-v4"
    "X86_64:13:x86-64 x86-64-v3 x86-64-v4"
    "AARCH64_JETSON:12:armv8-a armv8.2-a+fp16+dotprod"
    "AARCH64_SBSA:12:armv8-a armv9-a"
    "AARCH64_SBSA:13:armv8-a armv9-a"
)

# If PLATFORM or CUDA_VERSION is unset, iterate over the matrix.
if [ -z "$PLATFORM" ] || [ -z "$CUDA_VERSION" ]; then
    for entry in "${MATRIX[@]}"; do
        plat="${entry%%:*}"; rest="${entry#*:}"
        cuda="${rest%%:*}"; marches="${rest#*:}"
        [ -n "$PLATFORM" ] && [ "$PLATFORM" != "$plat" ] && continue
        [ -n "$CUDA_VERSION" ] && [ "$CUDA_VERSION" != "$cuda" ] && continue
        for march in $marches; do
            PLATFORM=$plat CUDA_VERSION=$cuda MARCH=$march bash "$0"
        done
    done
    exit 0
fi

# Default TRT_DIR from .deps/tensorrt/ based on CUDA_VERSION and PLATFORM
if [ -z "$TRT_DIR" ]; then
    if [ -n "$CUDA_VERSION" ]; then
        case "$PLATFORM" in
            X86_64)             trt_platform="x86_64" ;;
            AARCH64_JETSON)  trt_platform="aarch64-linux" ;;
            AARCH64_SBSA)  trt_platform="sbsa-linux" ;;
        esac
        TRT_DIR="${DEPS_DIR}/tensorrt/cuda-${CUDA_VERSION}/${trt_platform}"
    fi
fi

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
