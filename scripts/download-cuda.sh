#!/bin/bash
# Downloads the CUDA headers and libcudart for all platforms needed to
# cross-compile the runtime library. Uses NVIDIA's public redistrib
# repository — no authentication required.
#
# Packages downloaded:
#   cuda_cudart — runtime headers + libcudart.so
#   cuda_crt    — crt/ headers (CUDA 13+; in cuda_nvcc for CUDA 12)
#
# Usage:
#   bash download-cuda.sh [--output <dir>] [<version> ...]
#
# Arguments:
#   --output <dir>   Destination directory (default: .deps/cuda)
#   <version>        One or more CUDA versions to download (default: 12.9.1 13.0.0)
#
# Output structure:
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

get_relative_path() {
    local manifest="$1" package="$2" platform="$3"
    echo "$manifest" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('$package', {}).get('$platform', {}).get('relative_path', ''))
"
}

for version in "${VERSIONS[@]}"; do
    major="${version%%.*}"
    echo "=== CUDA ${version} ==="

    manifest=$(curl -fsSL "${BASE_URL}/redistrib_${version}.json")

    # ── Per-platform packages ─────────────────────────────────────────────────
    for platform in linux-x86_64 linux-sbsa linux-aarch64; do
        dir_name=$(platform_dir "$platform")
        dest="${OUTPUT_DIR}/${major}/${dir_name}"

        # Check for crt/ specifically — that's what was previously missing
        if [ -d "${dest}/include/crt" ] && [ -d "${dest}/lib" ]; then
            echo "  [$platform] already complete at ${dest} — skipping"
            continue
        fi

        mkdir -p "$dest"

        for package in cuda_cudart cuda_crt; do
            rel=$(get_relative_path "$manifest" "$package" "$platform")
            [ -z "$rel" ] && continue
            echo "  [$platform] $package ..."
            curl -fsSL "${BASE_URL}/${rel}" | tar -xJ --strip-components=1 -C "$dest"
        done

        echo "  [$platform] extracted to ${dest}"
    done

    # ── crt/ headers from cuda_nvcc (CUDA 12 only — no cuda_crt package) ─────
    # For CUDA 13+, cuda_crt above covers this. For CUDA 12, the crt/ headers
    # live inside cuda_nvcc. We extract only include/crt/ from it (no binaries).
    if [ "$major" = "12" ]; then
        rel=$(get_relative_path "$manifest" "cuda_nvcc" "linux-x86_64")
        if [ -n "$rel" ]; then
            echo "  [crt headers] extracting from cuda_nvcc ..."
            tmpdir=$(mktemp -d)
            curl -fsSL "${BASE_URL}/${rel}" \
                | tar -xJ --strip-components=2 -C "$tmpdir" \
                    --wildcards '*/include/crt/*'
            # Copy crt/ into every platform dir that exists for this major version
            for dest in "${OUTPUT_DIR}/${major}"/*/; do
                [ -d "$dest" ] || continue
                mkdir -p "${dest}include"
                cp -r "${tmpdir}/crt" "${dest}include/"
            done
            rm -rf "$tmpdir"
            echo "  [crt headers] done"
        fi
    fi

    echo ""
done

echo "Done. Available CUDA directories:"
find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type d | sort | while read -r dir; do
    major=$(basename "$(dirname "$dir")")
    platform=$(basename "$dir")
    lib_count=$(find "$dir/lib" -name "*.so*" 2>/dev/null | wc -l)
    crt_ok=$([ -d "$dir/include/crt" ] && echo "crt ✓" || echo "crt ✗")
    echo "  cuda-${major}/${platform}  (${lib_count} shared libs, ${crt_ok})"
done
