set -e

cd "$(dirname "$0")"

VERSION_FILE="../../VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

rm -rf artifacts 2>/dev/null || true
mkdir artifacts

docker build -t oaax-nvidia-tensorrt-toolchain:$VERSION .
docker save oaax-nvidia-tensorrt-toolchain:$VERSION -o ./artifacts/oaax-nvidia-tensorrt-toolchain.tar
