set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION_FILE="$REPO_ROOT/VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

mkdir -p "$SCRIPT_DIR/artifacts"

# Build with repo root as context so COPY VERSION works
docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t oaax-nvidia-toolchain:$VERSION \
    "$REPO_ROOT"

docker save oaax-nvidia-toolchain:$VERSION -o "$SCRIPT_DIR/artifacts/oaax-nvidia-toolchain.tar"
