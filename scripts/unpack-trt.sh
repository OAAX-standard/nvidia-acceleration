#!/bin/bash
# Unpacks TensorRT archives from tensorrt-archives/ into .deps/tensorrt/.
#
# Download the following archives and place them in tensorrt-archives/:
#
#   From https://developer.nvidia.com/tensorrt/download/10x
#     TensorRT 10.16.0 GA for Linux x86_64 and CUDA 12.0 to 12.9 TAR Package
#     TensorRT 10.16.0 GA for Linux x86_64 and CUDA 13.0 to 13.2 TAR Package
#     TensorRT 10.16.0 GA for Linux SBSA and CUDA 13.2 TAR Package
#
#   From https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/
#     TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz   (Jetson / AARCH64_JETSON)
#     TensorRT-10.7.0.23.Linux.aarch64-gnu.cuda-12.6.tar.gz  (SBSA / AARCH64_SBSA + CUDA 12)
#
# Output:
#   .deps/tensorrt/cuda-12/x86_64/
#   .deps/tensorrt/cuda-12/aarch64-linux/
#   .deps/tensorrt/cuda-12/sbsa-linux/
#   .deps/tensorrt/cuda-13/x86_64/
#   .deps/tensorrt/cuda-13/sbsa-linux/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARCHIVES_DIR="${REPO_ROOT}/tensorrt-archives"
DEPS_DIR="${REPO_ROOT}/.deps"

DOWNLOAD_URL="https://developer.nvidia.com/tensorrt/download/10x"

# Each entry: "glob_pattern|dest_subdir|description"
TARGETS=(
    "TensorRT-*.Linux.x86_64*.cuda-12*.tar.gz|cuda-12/x86_64|TensorRT 10.16.0 GA for Linux x86_64 and CUDA 12.0 to 12.9 TAR Package — https://developer.nvidia.com/tensorrt/download/10x"
    "TensorRT-*.Linux.x86_64*.cuda-13*.tar.gz|cuda-13/x86_64|TensorRT 10.16.0 GA for Linux x86_64 and CUDA 13.0 to 13.2 TAR Package — https://developer.nvidia.com/tensorrt/download/10x"
    "TensorRT-*.Linux.aarch64*.cuda-13*.tar.gz|cuda-13/sbsa-linux|TensorRT 10.16.0 GA for Linux SBSA and CUDA 13.2 TAR Package — https://developer.nvidia.com/tensorrt/download/10x"
    "TensorRT-*.l4t.aarch64*.cuda-12*.tar.gz|cuda-12/aarch64-linux|TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz — https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/"
    "TensorRT-*.Linux.aarch64*.cuda-12*.tar.gz|cuda-12/sbsa-linux|TensorRT-10.7.0.23.Linux.aarch64-gnu.cuda-12.6.tar.gz — https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/"
)

if [ ! -d "$ARCHIVES_DIR" ]; then
    mkdir -p "$ARCHIVES_DIR"
fi

missing=false

for entry in "${TARGETS[@]}"; do
    pattern="${entry%%|*}";  rest="${entry#*|}"
    dest_subdir="${rest%%|*}"; desc="${rest#*|}"
    dest="${DEPS_DIR}/tensorrt/${dest_subdir}"

    if [ -d "${dest}/include" ] && [ -d "${dest}/lib" ]; then
        echo "  [${dest_subdir}] already unpacked — skipping"
        continue
    fi

    archive=$(find "$ARCHIVES_DIR" -maxdepth 1 -name "$pattern" 2>/dev/null | head -1)

    if [ -z "$archive" ]; then
        echo "  [${dest_subdir}] archive not found"
        echo "    Download: ${desc}"
        echo "    From:     ${DOWNLOAD_URL}"
        echo "    Place in: tensorrt-archives/"
        missing=true
        continue
    fi

    echo "  [${dest_subdir}] unpacking $(basename "$archive") ..."
    mkdir -p "$dest"
    tar -xzf "$archive" --strip-components=1 -C "$dest"
    echo "  [${dest_subdir}] extracted to ${dest}"
done

if [ "$missing" = true ]; then
    echo ""
    echo "Re-run this script after placing the missing archives in tensorrt-archives/"
    exit 1
fi

echo ""
echo "Done. TensorRT directories:"
find "${DEPS_DIR}/tensorrt" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort | while read -r dir; do
    echo "  ${dir}"
done
