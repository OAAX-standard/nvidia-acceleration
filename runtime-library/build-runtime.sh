set -e

cd "$(dirname "$0")" || exit 1

rm -rf build 2&> /dev/null || true
mkdir build

cd build

PLATFORM=X86_64 # or AAARCH64
CUDA_VERSION=12 # or 10 or 11
cmake .. -DPLATFORM=$PLATFORM -DCUDA_VERSION=$CUDA_VERSION
make -j