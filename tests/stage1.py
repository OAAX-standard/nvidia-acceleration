#!/usr/bin/env python3
"""Stage 1: Convert ONNX models to TensorRT engines via Docker toolchain.

Runs conversion tests and caches engines to tests/compiled_models/.
Stage 2 reads from this cache without re-converting.

Requirements:
  - NVIDIA GPU with CUDA
  - Docker with --gpus all support
  - oaax-nvidia-tensorrt-toolchain image (bash tensorrt/conversion-toolchain/build-toolchain.sh)
  - ultralytics (uv sync --group dev)  ← for YOLO models only

Usage:
    python tests/stage1.py
    python tests/stage1.py --smoke-only   # skip YOLO, only test identity model
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


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

    header("Step 1: Identity-model conversion smoke test")
    run_pytest("tests/test_conversion.py")

    if args.smoke_only:
        print("\nStage 1 complete (smoke only) — no YOLO models converted")
        return

    header("Step 2: YOLO 640×640 integration tests (FP16 / INT8)")
    run_pytest("tests/test_yolo.py", "-k", "TestYoloConversion640")

    header("Step 3: YOLO 320×320 integration tests (FP16)")
    run_pytest("tests/test_yolo.py", "-k", "TestYoloConversion320")

    print("\nStage 1 complete — compiled models saved to tests/compiled_models/")


if __name__ == "__main__":
    main()
