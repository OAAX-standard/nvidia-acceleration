#!/usr/bin/env python3
"""Stage 2: Benchmark compiled models using the OAAX runtime library.

Requires tests/compiled_models/<backend>/ to be populated by stage1 first.

Usage:
    python -m tests.stage2 [options]

Options:
    --backend        trt|ort     which backend to benchmark (default: trt)
    --runtime-lib    PATH        path to libRuntimeLibrary.so or its directory
    --runs           N           benchmark iterations (default: 100)
    --warmup         N           warmup iterations (default: 10)
    --models         LIST        comma-separated model names (default: all)
    --csv            PATH        write results to CSV
    --skip-build                 skip building inference_test binary
    --log-level      N           runtime log level 0-4 (default: 3)
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

ROOT           = Path(__file__).parent.parent
TEST_BUILD_DIR = ROOT / "tests" / "runtime" / "build"

_IDENTITY_CONFIG = ("input", "1,4")


def header(title: str) -> None:
    print(f"\n\033[34m=== {title} ===\033[0m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend",     choices=["trt", "ort", "trt_int8"], default="trt")
    p.add_argument("--runtime-lib", default="")
    p.add_argument("--runs",        type=int, default=100)
    p.add_argument("--warmup",      type=int, default=10)
    p.add_argument("--models",      default="")
    p.add_argument("--csv",         default="")
    p.add_argument("--skip-build",  action="store_true")
    p.add_argument("--log-level",   type=int, default=3)
    return p.parse_args()


def default_runtime_dir(backend_name: str) -> Path:
    base = ROOT / f"{backend_name}runtime"[3:] if backend_name == "ort" else ROOT / "tensorrt"
    # onnxruntime or tensorrt
    name_map = {"trt": "tensorrt", "ort": "onnxruntime"}
    return ROOT / name_map[backend_name] / "runtime-library"


def find_lib_dir(lib_arg: str, backend_name: str) -> str:
    if lib_arg:
        p = Path(lib_arg)
        if p.is_file() and "RuntimeLibrary" in p.name:
            return str(p.parent)
        if p.is_dir() and any(p.glob("libRuntimeLibrary.so*")):
            return str(p)
    runtime_dir = default_runtime_dir(backend_name)
    candidates = sorted(runtime_dir.glob("**/libRuntimeLibrary.so"))
    x86 = [c for c in candidates if "x86_64" in str(c)]
    chosen = x86[0] if x86 else (candidates[0] if candidates else None)
    return str(chosen.parent) if chosen else str(runtime_dir / "build" / "bin")


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


def get_compiled_models(backend_name: str, model_filter: set | None) -> list[tuple[Path, str]]:
    from tests.backends import BACKENDS
    backend = BACKENDS[backend_name]
    results = []
    for model_path in sorted(backend.cache_dir.glob(f"*/{backend.output_filename}")):
        model_name = model_path.parent.name
        if model_filter and model_name not in model_filter:
            continue
        results.append((model_path, model_name))
    return results


def _input_config(model_name: str) -> tuple[str, str]:
    if model_name == "identity":
        return _IDENTITY_CONFIG
    from tests.models import MODEL_INPUT_CONFIGS
    if model_name in MODEL_INPUT_CONFIGS:
        return MODEL_INPUT_CONFIGS[model_name]
    raise ValueError(f"No input config for model: {model_name!r}")


def parse_field(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def run_inference_test(
    model_path: Path, model_name: str,
    runs: int, warmup: int, log_level: int, lib_dir: str,
) -> tuple | None:
    binary = inference_test_path()
    if not binary.exists():
        print(f"  [error] inference_test not found: {binary}")
        return None

    input_name, input_shape = _input_config(model_name)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{TEST_BUILD_DIR}:{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        r = subprocess.run(
            [str(binary), str(model_path),
             "--input-name",  input_name,
             "--input-shape", input_shape,
             "--runs",        str(runs),
             "--warmup",      str(warmup),
             "--log-level",   str(log_level),
             "--no-validate"],
            capture_output=True, text=True, env=env,
            timeout=600 + runs * 5,
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

    from tests.backends import BACKENDS
    backend = BACKENDS[args.backend]

    if not backend.cache_dir.exists() or not any(backend.cache_dir.rglob(backend.output_filename)):
        print(f"ERROR: {backend.cache_dir} empty — run stage1 --backend {args.backend} first")
        sys.exit(1)

    models = get_compiled_models(args.backend, model_filter)
    if not models:
        print("No compiled models found.")
        sys.exit(1)

    lib_dir = find_lib_dir(args.runtime_lib, args.backend)

    if not args.skip_build and not inference_test_path().exists():
        print("  Building inference_test...")
        if not build_inference_test():
            sys.exit(1)

    # Symlink runtime libs for $ORIGIN RPATH; always update symlinks so switching
    # backends picks up the correct libRuntimeLibrary.so.
    for so in Path(lib_dir).glob("*.so*"):
        dst = TEST_BUILD_DIR / so.name
        if dst.is_symlink():
            dst.unlink()
        if not dst.exists():
            dst.symlink_to(so)

    csv_file = csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["timestamp", "model", "avg_ms", "min_ms", "p95_ms", "throughput_fps"],
        )
        csv_writer.writeheader()

    try:
        header(f"NVIDIA {args.backend.upper()} Runtime Benchmark")
        print(_HDR.format("Model", "avg_ms", "min_ms", "p95_ms", "Throughput"))
        print(_HDR.format("-" * 15, "-" * 8, "-" * 8, "-" * 8, "-" * 12))

        pass_count = fail_count = 0
        for model_path, model_name in models:
            r = run_inference_test(model_path, model_name, args.runs, args.warmup,
                                   args.log_level, lib_dir)
            if r:
                print(_ROW.format(model_name, *r))
                if csv_writer:
                    csv_writer.writerow({
                        "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "model":          model_name,
                        "avg_ms":         r[0], "min_ms": r[1],
                        "p95_ms":         r[2], "throughput_fps": r[3],
                    })
                pass_count += 1
            else:
                print(f"  {model_name:<15}  FAILED")
                fail_count += 1

        header("Stage 2 results")
        if args.csv:
            print(f"  CSV saved to: {args.csv}")
        print(f"  Passed: {pass_count}  Failed: {fail_count}")
        if fail_count:
            sys.exit(1)

    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
