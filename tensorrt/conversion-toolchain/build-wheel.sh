#!/bin/bash
set -e
cd "$(dirname "$0")"

uv build --wheel --out-dir artifacts/ .
echo ""
echo "Wheel written to artifacts/:"
ls -lh artifacts/*.whl
