#!/bin/bash
# Downloads the CUDA cudart component (headers + libcudart.so) for all platforms
# needed to cross-compile the runtime library. Uses NVIDIA's public redistrib
# repository — no authentication required.
#
# Usage:
#   bash download-cuda.sh [--output <dir>] [<version> ...]
#
# Arguments:
#   --output <dir>   Destination directory (default: /opt/cuda)
#   <version>        One or more CUDA versions to download (default: 12.9.1 13.0.0)
#
# Output structure (mirrors the CUDA toolkit targets/ layout):
#   <output>/<major>/x86_64/        CUDA_DIR for X86_64
#   <output>/<major>/sbsa-linux/    CUDA_DIR for AARCH64_SBSA
#   <output>/<major>/aarch64-linux/ CUDA_DIR for AARCH64_JETSON (CUDA 12 only)
#
# Note: CUDA 13 does not provide a linux-aarch64 (Jetson) package.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_URL="https://developer.download.nvidia.com/compute/cuda/redist"
OUTPUT_DIR="${REPO_ROOT}/.deps/cuda"
VERSIONS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *)        VERSIONS+=("$1"); shift ;;
    esac
done

if [ ${#VERSIONS[@]} -eq 0 ]; then
    VERSIONS=("12.9.1" "13.0.0")
fi

# Map redistrib platform name to our directory name
platform_dir() {
    case "$1" in
        linux-x86_64)  echo "x86_64" ;;
        linux-sbsa)    echo "sbsa-linux" ;;
        linux-aarch64) echo "aarch64-linux" ;;
    esac
}

for version in "${VERSIONS[@]}"; do
    major="${version%%.*}"
    echo "=== CUDA ${version} ==="

    manifest=$(curl -fsSL "${BASE_URL}/redistrib_${version}.json")

    for platform in linux-x86_64 linux-sbsa linux-aarch64; do
        dir_name=$(platform_dir "$platform")
        dest="${OUTPUT_DIR}/${major}/${dir_name}"

        relative_path=$(echo "$manifest" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('cuda_cudart', {}).get('$platform', {}).get('relative_path', ''))
")
        if [ -z "$relative_path" ]; then
            echo "  [$platform] not available for CUDA ${version} — skipping"
            continue
        fi

        if [ -d "${dest}/include" ] && [ -d "${dest}/lib" ]; then
            echo "  [$platform] already downloaded at ${dest} — skipping"
            continue
        fi

        url="${BASE_URL}/${relative_path}"
        echo "  [$platform] downloading $(basename "$url") ..."
        mkdir -p "$dest"
        curl -fsSL "$url" | tar -xJ --strip-components=1 -C "$dest"
        echo "  [$platform] extracted to ${dest}"
    done

    echo ""
done

echo "Done. Available CUDA directories:"
find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type d | sort | while read -r dir; do
    major=$(basename "$(dirname "$dir")")
    platform=$(basename "$dir")
    lib_count=$(find "$dir/lib" -name "*.so*" 2>/dev/null | wc -l)
    echo "  cuda-${major}/${platform}  (${lib_count} shared libs)"
done
