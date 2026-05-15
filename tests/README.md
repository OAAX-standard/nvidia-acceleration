# Tests

Two-stage test pipeline covering both the TRT and ORT backends.

## Prerequisites

```bash
uv sync --group dev   # installs pytest, ultralytics, onnx, torchvision
```

Docker must be running. The TRT backend additionally requires `--gpus all` access.

| Backend | Docker image env var | Default image |
|---|---|---|
| TRT | `NVIDIA_TRT_TOOLCHAIN_IMAGE` | `oaax-nvidia-tensorrt-toolchain:1.3.0` |
| ORT | `NVIDIA_ORT_TOOLCHAIN_IMAGE` | `oaax-nvidia-toolchain:1.3.0` |

Override the image with the env var if you have a different version locally, e.g.:
```bash
export NVIDIA_ORT_TOOLCHAIN_IMAGE=oaax-nvidia-toolchain:1.2.0
```

---

## Stage 1 — Model conversion

Converts ONNX models via the Docker toolchain and caches results in `tests/compiled_models/<backend>/`.

```bash
# Both backends
uv run python -m tests.stage1

# One backend only
uv run python -m tests.stage1 --backend trt
uv run python -m tests.stage1 --backend ort

# Smoke test only (identity model, fast)
uv run python -m tests.stage1 --smoke-only
```

Converted models are cached — re-running stage1 skips models that already exist.

---

## Stage 2 — Benchmarking

Benchmarks compiled models using the OAAX v2 runtime library. Requires stage1 to have run first.

```bash
uv run python -m tests.stage2 --backend trt \
  --runtime-lib tensorrt/runtime-library/build/NVIDIA_TENSORRT/x86_64/Ubuntu/22.04/library-cuda_13/bin/libRuntimeLibrary.so

uv run python -m tests.stage2 --backend ort \
  --runtime-lib onnxruntime/runtime-library/build/bin/libRuntimeLibrary.so
```

The `inference_test` binary is built automatically on first run from `tests/runtime/`. Pass `--skip-build` to skip this if it is already built.

### Options

| Flag | Default | Description |
|---|---|---|
| `--backend trt\|ort` | `trt` | Which backend to benchmark |
| `--runtime-lib PATH` | auto-detected | Path to `libRuntimeLibrary.so` or its directory |
| `--runs N` | `100` | Number of benchmark iterations |
| `--warmup N` | `10` | Number of warmup iterations |
| `--models LIST` | all | Comma-separated subset of model names |
| `--csv PATH` | — | Write results to a CSV file |
| `--skip-build` | off | Skip rebuilding `inference_test` |
| `--log-level N` | `3` | Runtime log verbosity (0 = silent, 4 = trace) |

---

## Running pytest directly

```bash
# All tests, both backends
uv run pytest tests/

# One backend only
uv run pytest tests/ -k trt
uv run pytest tests/ -k ort

# Specific test class
uv run pytest tests/test_yolo.py::TestModelConversion -k ort
```

Tests that require a backend not available on the machine are automatically skipped.
