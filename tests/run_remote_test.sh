#!/bin/bash
# Run conversion + inference tests on remote deployment machines.
#
# Usage:
#   bash tests/run_remote_test.sh [device-1|device-2|device-3|device-4|all]
#   bash tests/run_remote_test.sh device-4          # single device
#   bash tests/run_remote_test.sh device-2 device-3 # subset
#   bash tests/run_remote_test.sh                   # defaults to all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

ALL_DEVICES=(device-1 device-2 device-3 device-4)
RUNTIME_LIB="/opt/oaax/bin/libRuntimeLibrary.so"
REMOTE_DIR="/tmp/oaax-test"

# ── Parse arguments ────────────────────────────────────────────────────────────
if [ $# -eq 0 ] || { [ $# -eq 1 ] && [ "$1" = "all" ]; }; then
    DEVICES=("${ALL_DEVICES[@]}")
else
    DEVICES=("$@")
fi

# ── Local setup ────────────────────────────────────────────────────────────────
echo "==> Generating test model..."
cd "$REPO_ROOT"
python3 tests/generate_model.py

echo ""
echo "==> Building wheel..."
bash conversion-toolchain/build-wheel.sh
WHEEL=$(ls conversion-toolchain/artifacts/*.whl | sort | tail -1)
WHEEL_NAME=$(basename "$WHEEL")
echo "    $WHEEL_NAME"

# ── Per-device test ────────────────────────────────────────────────────────────
declare -A RESULTS

run_device() {
    local device="$1"
    echo ""
    echo "══════════════════════════════════════════"
    echo " Testing: $device"
    echo "══════════════════════════════════════════"

    # Copy artifacts
    echo "  → Copying artifacts..."
    ssh "$device" "mkdir -p $REMOTE_DIR/output"
    rsync -q \
        "$WHEEL" \
        "tests/artifacts/bundle.zip" \
        "tests/test_inference.py" \
        "${device}:${REMOTE_DIR}/"

    # Install wheel and run conversion
    echo "  → Installing wheel and converting model..."
    if ! ssh "$device" bash <<EOF
set -e
cd $REMOTE_DIR
rm -rf .venv output
curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv
uv pip install --quiet "$WHEEL_NAME"
.venv/bin/conversion_toolchain --input-path bundle.zip --output-dir output
EOF
    then
        echo -e "  ${RED}[FAIL] Conversion${NC}"
        RESULTS[$device]="FAIL (conversion)"
        return
    fi
    echo "  [OK] Conversion → output/model.trt"

    # Check runtime lib exists
    if ! ssh "$device" "test -f $RUNTIME_LIB"; then
        echo -e "  ${RED}[FAIL] Runtime lib not found: $RUNTIME_LIB${NC}"
        RESULTS[$device]="FAIL (runtime lib missing)"
        return
    fi

    # Run inference test
    echo "  → Running inference test..."
    if ssh "$device" \
        "python3 $REMOTE_DIR/test_inference.py \
            --lib   $RUNTIME_LIB \
            --model $REMOTE_DIR/output/model.trt \
            --input-name  input \
            --input-shape 1,4"; then
        echo -e "  ${GREEN}PASS${NC}"
        RESULTS[$device]="PASS"
    else
        echo -e "  ${RED}FAIL (inference)${NC}"
        RESULTS[$device]="FAIL (inference)"
    fi
}

for device in "${DEVICES[@]}"; do
    run_device "$device"
done

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Summary"
echo "══════════════════════════════════════════"
all_pass=true
for device in "${DEVICES[@]}"; do
    result="${RESULTS[$device]:-SKIP}"
    if [ "$result" = "PASS" ]; then
        echo -e "  $device : ${GREEN}${result}${NC}"
    else
        echo -e "  $device : ${RED}${result}${NC}"
        all_pass=false
    fi
done
echo ""
$all_pass && exit 0 || exit 1
