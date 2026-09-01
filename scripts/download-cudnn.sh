#!/bin/bash
# Downloads and unpacks the cuDNN shared libraries needed to bundle into an
# ORT runtime package for one (platform, CUDA version) combination. Uses
# pinned builds from NVIDIA's public cudnn redist repository — no
# authentication required. CUDA 12/13 use cuDNN 9.25.0.15; CUDA 11 (X86_64 and
# WINDOWS only) uses cuDNN 8.9.7.29, the last cuDNN release with a CUDA 11
# build at all.
#
# Usage:
#   bash download-cudnn.sh <PLATFORM> <cuda_version> [--output <dir>]
#
#   PLATFORM       X86_64, AARCH64_SBSA, or WINDOWS
#   cuda_version   11, 12, or 13
#
# Output:
#   Linux:   <output>/<PLATFORM>/<cuda_version>/lib/libcudnn*.so*
#   Windows: <output>/WINDOWS/<cuda_version>/bin/*.dll (nesting under bin/x64
#            varies by cuDNN version -- callers should search recursively)
#
# Note: cuDNN has no linux-aarch64 (Jetson) build at any version compatible
# with CUDA 11/12 (JetPack-era cuDNN 8.6/8.9), and no linux-sbsa build for
# CUDA 11. Those combos exit 0 without downloading anything so the runtime
# build isn't blocked; JetPack devices already ship with a matching cuDNN
# preinstalled.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/.deps/cudnn"

PLATFORM="$1"
CUDA_VERSION="$2"
shift 2 || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$PLATFORM" ] || [ -z "$CUDA_VERSION" ]; then
    echo "Usage: $0 <PLATFORM> <cuda_version> [--output <dir>]" >&2
    exit 1
fi

BASE_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn"
case "${PLATFORM}:${CUDA_VERSION}" in
    "X86_64:11")       url="${BASE_URL}/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz" ;;
    "X86_64:12")       url="${BASE_URL}/linux-x86_64/cudnn-linux-x86_64-9.25.0.15_cuda12-archive.tar.xz" ;;
    "X86_64:13")       url="${BASE_URL}/linux-x86_64/cudnn-linux-x86_64-9.25.0.15_cuda13-archive.tar.xz" ;;
    "AARCH64_SBSA:12") url="${BASE_URL}/linux-sbsa/cudnn-linux-sbsa-9.25.0.15_cuda12-archive.tar.xz" ;;
    "AARCH64_SBSA:13") url="${BASE_URL}/linux-sbsa/cudnn-linux-sbsa-9.25.0.15_cuda13-archive.tar.xz" ;;
    "WINDOWS:11")      url="${BASE_URL}/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip" ;;
    "WINDOWS:12")      url="${BASE_URL}/windows-x86_64/cudnn-windows-x86_64-9.25.0.15_cuda12-archive.zip" ;;
    "WINDOWS:13")      url="${BASE_URL}/windows-x86_64/cudnn-windows-x86_64-9.25.0.15_cuda13-archive.zip" ;;
    *)
        echo "No pinned cuDNN package available for ${PLATFORM}/cuda_${CUDA_VERSION} — skipping (package will not have cuDNN bundled)" >&2
        exit 0
        ;;
esac

dest="${OUTPUT_DIR}/${PLATFORM}/${CUDA_VERSION}"
marker="${dest}/lib"
[ "$PLATFORM" = "WINDOWS" ] && marker="${dest}/bin"
if [ -d "$marker" ] && [ -n "$(ls -A "$marker" 2>/dev/null)" ]; then
    echo "[${PLATFORM}/cuda_${CUDA_VERSION}] cuDNN already present at ${dest} — skipping download"
    exit 0
fi

mkdir -p "$dest"
echo "[${PLATFORM}/cuda_${CUDA_VERSION}] downloading ${url} ..."

if [[ "$url" == *.zip ]]; then
    archive="${dest}/cudnn.zip"
    curl -fsSL "$url" -o "$archive"
    if command -v unzip >/dev/null 2>&1; then
        # GNU tar (Linux) can't extract zip; prefer unzip when present.
        tmpdir=$(mktemp -d)
        unzip -q "$archive" -d "$tmpdir"
        extracted_dir=$(find "$tmpdir" -mindepth 1 -maxdepth 1 -type d)
        cp -a "$extracted_dir"/. "$dest"/
        rm -rf "$tmpdir"
    else
        # No unzip (e.g. Windows) -- tar there is bsdtar-based and reads zip natively.
        tar -xf "$archive" --strip-components=1 -C "$dest"
    fi
    rm -f "$archive"
else
    curl -fsSL "$url" | tar -xJ --strip-components=1 -C "$dest"
fi

echo "[${PLATFORM}/cuda_${CUDA_VERSION}] extracted to ${dest}"
