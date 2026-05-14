#!/usr/bin/env python3
"""Stage 1: Convert ONNX models via the Docker toolchain.

Runs conversion tests and caches output models to tests/compiled_models/<backend>/.

Requirements:
  - Docker
  - TRT backend: oaax-nvidia-tensorrt-toolchain image + GPU
  - ORT backend: oaax-nvidia-toolchain image (or set NVIDIA_ORT_TOOLCHAIN_IMAGE)
  - ultralytics (uv sync --group dev)  ← for YOLO models

Usage:
    python -m tests.stage1                       # both backends
    python -m tests.stage1 --backend trt         # TRT only
    python -m tests.stage1 --backend ort         # ORT only
    python -m tests.stage1 --smoke-only          # identity model only, both backends
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


def run_for_backend(backend_name: str, smoke_only: bool) -> None:
    label = backend_name.upper()
    header(f"Step 1: Identity-model smoke test ({label})")
    run_pytest("tests/test_conversion.py", "-k", backend_name)

    if smoke_only:
        print(f"\nStage 1 ({label}) complete (smoke only)")
        return

    header(f"Step 2: YOLO 640×640 ({label})")
    run_pytest("tests/test_yolo.py", "-k", f"TestYoloConversion640 and {backend_name}")

    header(f"Step 3: YOLO 320×320 ({label})")
    run_pytest("tests/test_yolo.py", "-k", f"TestYoloConversion320 and {backend_name}")

    print(f"\nStage 1 ({label}) complete — models saved to tests/compiled_models/{backend_name}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["trt", "ort", "all"], default="all",
                   help="Which backend to run (default: all)")
    p.add_argument("--smoke-only", action="store_true",
                   help="Only run the identity-model smoke test")
    args = p.parse_args()

    backends = ["trt", "ort"] if args.backend == "all" else [args.backend]
    for b in backends:
        run_for_backend(b, args.smoke_only)


if __name__ == "__main__":
    main()
