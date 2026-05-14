#!/usr/bin/env python3
"""Stage 1 (ORT): Convert ONNX models via the ORT simplification toolchain.

Runs conversion tests and caches simplified ONNX models to
tests/ort/compiled_models/.  No GPU required.

Requirements:
  - Docker
  - oaax-nvidia-toolchain Docker image (or set NVIDIA_ORT_TOOLCHAIN_IMAGE)
  - onnx + ultralytics (uv sync --group dev)  ← for YOLO models

Usage:
    python tests/ort/stage1.py
    python tests/ort/stage1.py --smoke-only   # only test identity model
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def header(title: str) -> None:
    print(f"\n\033[34m=== {title} ===\033[0m")


def run_pytest(*args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-v", "--tb=short"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke-only", action="store_true",
                   help="Only run the identity-model smoke test (no YOLO)")
    args = p.parse_args()

    header("Step 1: Identity-model conversion smoke test (ORT)")
    run_pytest("tests/ort/test_conversion.py")

    if args.smoke_only:
        print("\nStage 1 (ORT) complete (smoke only)")
        return

    header("Step 2: YOLO 640×640 conversion tests (ORT)")
    run_pytest("tests/ort/test_yolo.py", "-k", "TestOrtYoloConversion640")

    header("Step 3: YOLO 320×320 conversion tests (ORT)")
    run_pytest("tests/ort/test_yolo.py", "-k", "TestOrtYoloConversion320")

    print("\nStage 1 (ORT) complete — models saved to tests/ort/compiled_models/")


if __name__ == "__main__":
    main()
