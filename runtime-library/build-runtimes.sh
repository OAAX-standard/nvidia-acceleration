set -e

cd "$(dirname "$0")" || exit 1

BUILD_DIR="$(pwd)/build"
ARTIFACTS_DIR="$(pwd)/artifacts"
ROOT_DIR="$(pwd)/.."

mkdir -p $BUILD_DIR
rm -rf $ARTIFACTS_DIR
mkdir -p $ARTIFACTS_DIR

VERSION_FILE="$ROOT_DIR/VERSION"
RUNTIME_VERSION="$(cat $VERSION_FILE)"

echo "Building for runtime version: $RUNTIME_VERSION"

# Accept platform and CUDA version as arguments or set defaults
ALL=0
if [ $# -lt 2 ]; then
    ALL=1
else
    PLATFORM=$1
    CUDA_VERSION=$2
fi

function build_runtime {
    platform=$1
    cuda_version=$2

    # Go to the build directory
    cd "$BUILD_DIR"

    echo "Building runtime for platform: $platform with CUDA version: $cuda_version"
    rm -rf *
    cmake .. -DPLATFORM=$platform -DCUDA_VERSION=$cuda_version -DRUNTIME_VERSION=$RUNTIME_VERSION
    make -j

    mkdir -p "${ARTIFACTS_DIR}/$platform/$cuda_version"
    echo "Copying shared libraries to artifacts directory..."
    cp ./bin/* "${ARTIFACTS_DIR}/$platform/$cuda_version/"
    # Bundle the shared libraries into a tarball
    cd "${ARTIFACTS_DIR}/$platform/$cuda_version/"
    echo "Creating tarball of shared libraries..."
    tar czf "../../runtime-library-${platform}-cuda_${cuda_version}.tar.gz" ./*

    echo "Shared libraries for $platform have been copied to ${ARTIFACTS_DIR}/$platform/$cuda_version/"
}

# X86_64 builds
if [[ "$PLATFORM" == "X86_64" && "$CUDA_VERSION" == "11" || "$ALL" == "1" ]]; then
    build_runtime "X86_64" "11"
fi
if [[ "$PLATFORM" == "X86_64" && "$CUDA_VERSION" == "12" || "$ALL" == "1" ]]; then
    build_runtime "X86_64" "12"
fi
if [[ "$PLATFORM" == "X86_64" && "$CUDA_VERSION" == "13" || "$ALL" == "1" ]]; then
    build_runtime "X86_64" "13"
fi

# AARCH64 builds (Jetson and other embedded aarch64 devices)
if [[ "$PLATFORM" == "AARCH64" && "$CUDA_VERSION" == "11" || "$ALL" == "1" ]]; then
    build_runtime "AARCH64" "11"
fi
if [[ "$PLATFORM" == "AARCH64" && "$CUDA_VERSION" == "12" || "$ALL" == "1" ]]; then
    build_runtime "AARCH64" "12"
fi
if [[ "$PLATFORM" == "AARCH64" && "$CUDA_VERSION" == "13" || "$ALL" == "1" ]]; then
    build_runtime "AARCH64" "13"
fi

# AARCH64_SERVER builds (aarch64 server machines, e.g. NVIDIA DGX Spark with Grace CPU)
if [[ "$PLATFORM" == "AARCH64_SERVER" && "$CUDA_VERSION" == "13" || "$ALL" == "1" ]]; then
    build_runtime "AARCH64_SERVER" "13"
fi