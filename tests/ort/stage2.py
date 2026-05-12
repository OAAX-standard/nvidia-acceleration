#!/usr/bin/env python3
"""Stage 2 (ORT): Benchmark simplified ONNX models via the OAAX ORT runtime.

Requires tests/ort/compiled_models/ to be populated by stage1 first.
Reuses the same inference_test binary from tests/runtime/.

Usage:
    python tests/ort/stage2.py [options]

Options:
    --runtime-lib    PATH   path to ORT libRuntimeLibrary.so (or directory)
    --runs           N      benchmark iterations (default: 100)
    --warmup         N      warmup iterations (default: 10)
    --models         LIST   comma-separated model names to run (default: all)
    --csv            PATH   write results to CSV
    --skip-build            skip building inference_test binary
    --log-level      N      runtime log level 0-4 (default: 3)
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT             = Path(__file__).parent.parent.parent
ORT_COMPILED_DIR = ROOT / "tests" / "ort" / "compiled_models"
TEST_BUILD_DIR   = ROOT / "tests" / "runtime" / "build"
ORT_RUNTIME_DIR  = ROOT / "onnxruntime" / "runtime-library"

# YOLO input shapes by model name pattern
_YOLO_SHAPES: dict[str, str] = {
    "_320": "1,3,320,320",
}
_DEFAULT_YOLO_SHAPE = "1,3,640,640"

# Input name for identity model and YOLO
_YOLO_INPUT_NAME    = "images"
_IDENTITY_INPUT_NAME = "input"


def header(title: str) -> None:
    print(f"\n\033[34m=== {title} ===\033[0m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-lib", default="",
                   help="Path to ORT libRuntimeLibrary.so or its directory")
    p.add_argument("--runs",       type=int, default=100)
    p.add_argument("--warmup",     type=int, default=10)
    p.add_argument("--models",     default="")
    p.add_argument("--csv",        default="")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--log-level",  type=int, default=3)
    return p.parse_args()


def find_ort_lib_dir(lib_arg: str) -> str:
    """Return the directory containing the ORT libRuntimeLibrary.so."""
    if lib_arg:
        p = Path(lib_arg)
        if p.is_file() and "RuntimeLibrary" in p.name:
            return str(p.parent)
        if p.is_dir() and any(p.glob("libRuntimeLibrary.so*")):
            return str(p)
    # Auto-detect from artifacts directory
    candidates = list((ORT_RUNTIME_DIR / "artifacts").glob("**/libRuntimeLibrary.so"))
    if candidates:
        return str(candidates[0].parent)
    return str(ORT_RUNTIME_DIR / "build" / "bin")


def inference_test_path() -> Path:
    return TEST_BUILD_DIR / "inference_test"


def build_inference_test() -> bool:
    cmake = shutil.which("cmake") or "cmake"
    TEST_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([cmake, "..", "-DCMAKE_BUILD_TYPE=Release"],
                       cwd=TEST_BUILD_DIR, check=True)
        subprocess.run([cmake, "--build", ".", "--target", "inference_test",
                        "-j", str(os.cpu_count() or 4)],
                       cwd=TEST_BUILD_DIR, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Build failed: {e}")
        return False


def get_compiled_models(model_filter: set[str] | None) -> list[tuple[Path, str]]:
    results = []
    for onnx_path in sorted(ORT_COMPILED_DIR.glob("*/model.onnx")):
        model_name = onnx_path.parent.name
        if model_filter and model_name not in model_filter:
            continue
        results.append((onnx_path, model_name))
    return results


def _input_config(model_name: str) -> tuple[str, str]:
    """Return (input_name, input_shape) for a model."""
    if model_name == "identity":
        return _IDENTITY_INPUT_NAME, "1,4"
    shape = _DEFAULT_YOLO_SHAPE
    for pattern, s in _YOLO_SHAPES.items():
        if pattern in model_name:
            shape = s
    return _YOLO_INPUT_NAME, shape


def parse_field(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def run_inference_test(
    onnx_path: Path,
    model_name: str,
    runs: int,
    warmup: int,
    log_level: int,
    ort_lib_dir: str,
) -> tuple | None:
    binary = inference_test_path()
    if not binary.exists():
        print(f"  [error] inference_test not found: {binary}")
        return None

    input_name, input_shape = _input_config(model_name)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{TEST_BUILD_DIR}:{ort_lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    )

    try:
        r = subprocess.run(
            [
                str(binary),
                str(onnx_path),
                "--input-name",  input_name,
                "--input-shape", input_shape,
                "--runs",        str(runs),
                "--warmup",      str(warmup),
                "--log-level",   str(log_level),
                "--no-validate",
            ],
            capture_output=True, text=True, env=env,
            timeout=300 + runs * 3,
        )
        text = r.stdout + r.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [error] {e}")
        return None

    if "=== Results ===" not in text:
        if text:
            print(f"  [output] {text[:400]}")
        return None
    return (
        parse_field(text, r"Avg latency:\s+([\d.]+)"),
        parse_field(text, r"Min latency:\s+([\d.]+)"),
        parse_field(text, r"p95 latency:\s+([\d.]+)"),
        parse_field(text, r"Throughput\s*:\s*([\d.]+)"),
    )


_HDR = "  {:<15}  {:>8}  {:>8}  {:>8}  {:>12}"
_ROW = "  {:<15}  {:>7}ms  {:>7}ms  {:>7}ms  {:>10} FPS"


def main() -> None:
    args = parse_args()
    model_filter = {m.strip() for m in args.models.split(",") if m.strip()} or None

    if not ORT_COMPILED_DIR.exists() or not any(ORT_COMPILED_DIR.rglob("model.onnx")):
        print("ERROR: tests/ort/compiled_models/ not found or empty — run stage1 first")
        sys.exit(1)

    models = get_compiled_models(model_filter)
    if not models:
        print("No compiled models found matching filters.")
        sys.exit(1)

    ort_lib_dir = find_ort_lib_dir(args.runtime_lib)

    if not args.skip_build and not inference_test_path().exists():
        print("  inference_test not found, building...")
        if not build_inference_test():
            print("  Build failed — aborting")
            sys.exit(1)
        # Symlink ORT shared libs for $ORIGIN RPATH
        for so in Path(ort_lib_dir).glob("*.so*"):
            dst = TEST_BUILD_DIR / so.name
            if not dst.exists():
                dst.symlink_to(so)

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["timestamp", "model", "avg_ms", "min_ms", "p95_ms", "throughput_fps"],
        )
        csv_writer.writeheader()

    try:
        header("NVIDIA ORT Runtime Benchmark")
        print(f"  Runtime lib: {ort_lib_dir}")
        print(f"  {'Model':<15}  {'avg_ms':>8}  {'min_ms':>8}  {'p95_ms':>8}  {'Throughput':>12}")
        print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")

        pass_count = fail_count = 0
        for onnx_path, model in models:
            r = run_inference_test(onnx_path, model, args.runs, args.warmup,
                                   args.log_level, ort_lib_dir)
            if r:
                print(_ROW.format(model, *r))
                if csv_writer:
                    csv_writer.writerow({
                        "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "model":          model,
                        "avg_ms":         r[0],
                        "min_ms":         r[1],
                        "p95_ms":         r[2],
                        "throughput_fps": r[3],
                    })
                pass_count += 1
            else:
                print(f"  {model:<15}  FAILED")
                fail_count += 1

        header("Stage 2 (ORT) results")
        if args.csv:
            print(f"  CSV saved to: {args.csv}")
        print(f"  Passed: {pass_count}  Failed: {fail_count}")

        if fail_count > 0:
            sys.exit(1)

    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
