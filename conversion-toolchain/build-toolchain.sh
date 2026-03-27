set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERSION_FILE="$REPO_ROOT/VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

mkdir -p "$SCRIPT_DIR/artifacts"

docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t oaax-nvidia-tensorrt-toolchain:$VERSION \
    "$REPO_ROOT"
docker save oaax-nvidia-tensorrt-toolchain:$VERSION -o "$SCRIPT_DIR/artifacts/oaax-nvidia-tensorrt-toolchain.tar"
