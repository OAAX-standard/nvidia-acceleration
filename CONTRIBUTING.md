# Contributing to OAAX NVIDIA Implementation

Thank you for your interest in contributing! We welcome improvements to the runtime library, conversion toolchains, tests, and documentation.

Before you start, please read the [OAAX general CONTRIBUTING guide](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md) for commit message format, code style, and other project-wide guidelines.

## How to contribute

1. **Open an issue** to discuss your idea or bug report.
2. **Fork the repository** and create a branch for your changes.
3. **Make your changes**, add or update tests where relevant.
4. **Push** to your fork and **submit a pull request**.

---

## Development environment

Tested on Ubuntu 20.04+ (x86_64 and aarch64). Docker is required for the conversion toolchains; the TRT backend additionally requires an NVIDIA GPU.

### Python dependencies

```bash
uv sync --group dev
```

This installs `pytest`, `ultralytics`, `onnx`, and `torchvision` into a local `.venv`.

### Runtime library

Build dependencies (TensorRT, CUDA toolchains) are set up via:

```bash
sudo bash scripts/setup-env.sh
```

See `tensorrt/runtime-library/README.md` and `onnxruntime/runtime-library/` for backend-specific build instructions.

### Running tests

See [`tests/README.md`](tests/README.md) for the full test pipeline (stage1 model conversion + stage2 benchmarking).

Quick start:

```bash
# Convert models (requires Docker + toolchain image)
uv run python -m tests.stage1 --backend ort

# Benchmark compiled models
uv run python -m tests.stage2 --backend ort --runtime-lib <path-to-libRuntimeLibrary.so>
```
