set -e

cd "$(dirname "$0")" || exit 1

BUILD_DIR="$(pwd)/build"
ARTIFACTS_DIR="$(pwd)/artifacts"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
rm -rf $ARTIFACTS_DIR
mkdir -p $ARTIFACTS_DIR

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
    cmake .. -DPLATFORM=$platform -DCUDA_VERSION=$cuda_version
    make -j

    mkdir -p "${ARTIFACTS_DIR}/$platform/$cuda_version"
    cp ./bin/* "${ARTIFACTS_DIR}/$platform/$cuda_version/"
    # Bundle the shared libraries into a tarball
    cd "${ARTIFACTS_DIR}/$platform/$cuda_version/"
    tar czf "../../runtime-library-${platform}-cuda_${cuda_version}.tar.gz" ./*

    echo "Shared libraries for $platform have been copied to ${ARTIFACTS_DIR}/$platform/$cuda_version/"
}

if [[ "$PLATFORM" == "X86_64" && "$CUDA_VERSION" == "11" || "$ALL" == "1" ]]; then
    build_runtime "X86_64" "11"
fi
if [[ "$PLATFORM" == "X86_64" && "$CUDA_VERSION" == "12" || "$ALL" == "1" ]]; then
    build_runtime "X86_64" "12"
fi
if [[ "$PLATFORM" == "AARCH64" && "$CUDA_VERSION" == "11" || "$ALL" == "1" ]]; then
    build_runtime "AARCH64" "11"
fi
if [[ "$PLATFORM" == "AARCH64" && "$CUDA_VERSION" == "12" || "$ALL" == "1" ]]; then
    build_runtime "AARCH64" "12"
fi
