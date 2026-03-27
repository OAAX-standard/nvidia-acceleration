#!/bin/bash
set -e
cd "$(dirname "$0")"

python -m build --wheel --outdir artifacts/ .
echo ""
echo "Wheel written to artifacts/:"
ls -lh artifacts/*.whl
