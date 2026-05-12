#!/usr/bin/env python3
"""Stage 2: Benchmark compiled TRT models using the OAAX runtime.

Requires tests/compiled_models/ to be populated by stage1 first.
Runs inference_test for each compiled model and reports latency / throughput.

Usage:
    python tests/stage2.py [options]

Options:
    --runtime-lib    PATH   path to libRuntimeLibrary.so (or directory)
    --runs           N      benchmark iterations (default: 100)
    --warmup         N      warmup iterations (default: 10)
    --models         LIST   comma-separated model names to run (default: all)
    --precisions     LIST   comma-separated precisions: FP16,INT8 (default: all)
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

ROOT         = Path(__file__).parent.parent
COMPILED_DIR = ROOT / "tests" / "compiled_models"
TEST_BUILD_DIR = ROOT / "tests" / "runtime" / "build"
RUNTIME_BUILD_DIR = ROOT / "tensorrt" / "runtime-library" / "build"

# YOLO input shapes by model name pattern
_YOLO_SHAPES: dict[str, str] = {
    "_320": "1,3,320,320",
}
_DEFAULT_SHAPE = "1,3,640,640"

# YOLO input tensor name
_INPUT_NAME = "images"


def header(title: str) -> None:
    print(f"\n\033[34m=== {title} ===\033[0m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-lib", default="",
                   help="Path to libRuntimeLibrary.so or its directory")
    p.add_argument("--runs",      type=int, default=100)
    p.add_argument("--warmup",    type=int, default=10)
    p.add_argument("--models",    default="", help="Comma-separated model names")
    p.add_argument("--precisions",default="", help="Comma-separated precisions: FP16,INT8")
    p.add_argument("--csv",       default="")
    p.add_argument("--skip-build",action="store_true")
    p.add_argument("--log-level", type=int, default=3)
    return p.parse_args()


# ── Binary discovery ────────────────────────────────────────────────────────────


def inference_test_path() -> Path:
    return TEST_BUILD_DIR / "inference_test"


def build_inference_test() -> bool:
    cmake = shutil.which("cmake") or "cmake"
    TEST_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [cmake, "..", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=TEST_BUILD_DIR, check=True,
        )
        subprocess.run(
            [cmake, "--build", ".", "--target", "inference_test", "-j", str(os.cpu_count() or 4)],
            cwd=TEST_BUILD_DIR, check=True,
        )
        # Symlink shared libs for $ORIGIN RPATH
        lib_dir = Path(find_lib_dir(""))
        for so in lib_dir.glob("*.so*"):
            dst = TEST_BUILD_DIR / so.name
            if not dst.exists():
                dst.symlink_to(so)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Build failed: {e}")
        return False


def find_lib_dir(lib_arg: str) -> str:
    """Return the directory containing libRuntimeLibrary.so."""
    if lib_arg:
        p = Path(lib_arg)
        if p.is_file() and "RuntimeLibrary" in p.name:
            return str(p.parent)
        if p.is_dir() and any(p.glob("libRuntimeLibrary.so*")):
            return str(p)
    candidates = list(RUNTIME_BUILD_DIR.glob("**/libRuntimeLibrary.so"))
    return str(candidates[0].parent) if candidates else str(RUNTIME_BUILD_DIR)


# ── Model discovery ─────────────────────────────────────────────────────────────


def get_compiled_models(
    model_filter: set[str] | None = None,
    precision_filter: set[str] | None = None,
) -> list[tuple[Path, str, str]]:
    allowed_precisions = precision_filter or {"FP16", "INT8"}
    results = []
    for variant in ("FP16", "INT8"):
        if variant not in allowed_precisions:
            continue
        for trt_path in sorted(COMPILED_DIR.glob(f"*/{variant}/model.trt")):
            model_name = trt_path.parent.parent.name
            if model_filter and model_name not in model_filter:
                continue
            results.append((trt_path, model_name, variant))
    return results


def _input_shape(model_name: str) -> str:
    for pattern, shape in _YOLO_SHAPES.items():
        if pattern in model_name:
            return shape
    return _DEFAULT_SHAPE


# ── Output parsing ──────────────────────────────────────────────────────────────


def parse_field(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def run_process(cmd: list, env=None, timeout: int = 300) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        if r.returncode != 0:
            print(f"  [exit {r.returncode}] {Path(cmd[0]).name}")
        return r.stdout + r.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [error] {e}")
        return None


def run_inference_test(
    trt_path: Path,
    model_name: str,
    runs: int,
    warmup: int,
    log_level: int,
    runtime_lib_dir: str,
) -> tuple | None:
    binary = inference_test_path()
    if not binary.exists():
        print(f"  [error] inference_test not found: {binary}")
        return None

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{TEST_BUILD_DIR}:{runtime_lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    text = run_process(
        [
            str(binary),
            str(trt_path),
            "--input-name",  _INPUT_NAME,
            "--input-shape", _input_shape(model_name),
            "--runs",        str(runs),
            "--warmup",      str(warmup),
            "--log-level",   str(log_level),
        ],
        env=env,
        # 600s overhead covers TRT engine deserialization before warmup
        timeout=600 + runs * 5,
    )
    if not text or "=== Results ===" not in text:
        if text:
            print(f"  [output] {text[:400]}")
        return None
    return (
        parse_field(text, r"Avg latency:\s+([\d.]+)"),
        parse_field(text, r"Min latency:\s+([\d.]+)"),
        parse_field(text, r"p95 latency:\s+([\d.]+)"),
        parse_field(text, r"Throughput\s*:\s*([\d.]+)"),
    )


# ── Formatting ──────────────────────────────────────────────────────────────────

_HDR = "  {:<15}  {:<6}  {:>8}  {:>8}  {:>8}  {:>12}"
_ROW = "  {:<15}  {:<6}  {:>7}ms  {:>7}ms  {:>7}ms  {:>10} FPS"


def print_table_header() -> None:
    print(_HDR.format("Model", "Prec", "avg_ms", "min_ms", "p95_ms", "Throughput"))
    print(_HDR.format("-" * 15, "-" * 6, "-" * 8, "-" * 8, "-" * 8, "-" * 12))


def write_csv_row(writer, model: str, variant: str, r: tuple) -> None:
    if writer:
        writer.writerow({
            "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model":          model,
            "variant":        variant,
            "avg_ms":         r[0],
            "min_ms":         r[1],
            "p95_ms":         r[2],
            "throughput_fps": r[3],
        })


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    model_filter = {m.strip() for m in args.models.split(",") if m.strip()} or None
    precision_filter = {p.strip() for p in args.precisions.split(",") if p.strip()} or None

    if not COMPILED_DIR.exists() or not any(COMPILED_DIR.rglob("*.trt")):
        print("ERROR: tests/compiled_models/ not found or empty — run stage1 first")
        sys.exit(1)

    models = get_compiled_models(model_filter, precision_filter)
    if not models:
        print("No compiled models found matching the specified filters.")
        sys.exit(1)

    lib_dir = find_lib_dir(args.runtime_lib)

    # Build inference_test binary if needed
    if not args.skip_build and not inference_test_path().exists():
        print("  inference_test not found, building...")
        if not build_inference_test():
            print("  Build failed — aborting")
            sys.exit(1)

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["timestamp", "model", "variant", "avg_ms", "min_ms", "p95_ms", "throughput_fps"],
        )
        csv_writer.writeheader()

    try:
        header("NVIDIA TRT Runtime Benchmark")
        print_table_header()

        pass_count = fail_count = 0
        for trt_path, model, variant in models:
            r = run_inference_test(trt_path, model, args.runs, args.warmup, args.log_level, lib_dir)
            if r:
                print(_ROW.format(model, variant, *r))
                write_csv_row(csv_writer, model, variant, r)
                pass_count += 1
            else:
                print(f"  {model:<15}  {variant:<6}  FAILED")
                fail_count += 1

        header("Stage 2 results")
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
