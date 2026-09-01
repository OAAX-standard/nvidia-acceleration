# Changelog

All notable changes to this project are documented here. Versions are listed newest first.

---

## Unreleased

### Fixed
- ORT runtime builds again on Linux and Windows: C++17 is now set explicitly, and hardware detection uses the CUDA driver API (`nvcuda.dll` / `libcuda.so.1`) loaded dynamically — no CUDA headers/libs needed at build time, no hard CUDA dependency at load time, no CUDA context created (and no GPU memory reserved) during `runtime_init`
- MSVC compile error on the OAAX API exports: functions are now exported via a `.def` file instead of `__declspec(dllexport)` (conflicted with the plain declarations in `interface.h`)
- Runtime no longer terminates the host process when the log file cannot be created (e.g. Windows services with an unwritable working directory); it falls back to console-only logging with an unconditional console notice
- On Windows the default log file is created in the temp directory with a per-process name instead of `runtime.log` in the working directory
- Crash of the host process at exit (static destruction of the ONNX Runtime environment, inference threads, and async logger); process-lifetime globals are now intentionally leaked and released only by `runtime_cleanup()`
- C++ exceptions can no longer escape the OAAX C API boundary or the inference thread (including non-numeric config values, which now return an error status)
- A failed `runtime_init` no longer leaks settings into the next attempt, and `runtime_cleanup` resets all state even if model teardown fails
- `build-runtimes.bat` now propagates build failures, so the Windows CI job fails at the compile step instead of at artifact upload

### Added
- `tests/runtime/robustness_test.cpp` covering the failure modes above

---

## 1.4.0 — 2026-05-14

### Added
- ONNX Runtime (ORT) backend alongside native TensorRT: conversion toolchain and runtime library for Linux and Windows x86_64
- Windows x86_64 CI build job for the ORT runtime
- MobileNetV2 model added to the test suite
- `uv` adopted for Python package management across the repo

### Changed
- Unified TRT and ORT test suites into a single parameterised suite (`tests/stage1.py`, `tests/stage2.py`, `tests/conftest.py`)
- S3 artifact path prefixes renamed to `NVIDIA_TRT` and `NVIDIA_ORT`

### Removed
- Bundled CUDA libraries from ORT runtime build deps (sourced from the system at build time)

---

## 2.0.0 — 2026-03-26

### Changed
- Replaced ONNX Runtime + CUDA EP implementation with native TensorRT inference
- Engines are now compiled ahead-of-time by the conversion toolchain; GPU required at conversion time
- Runtime library loads `.trt` engine files directly — no ONNX Runtime dependency
- TRT and CUDA are external build-time dependencies; no large binaries stored in the repo

### Added
- Support for DGX Spark / Grace CPU (`AARCH64_SBSA` platform)

### Removed
- Windows support (Linux x86_64, aarch64, aarch64-server only)
- Per-CUDA-version build variants; runtime now bundles TensorRT 10 and CUDA 13

---

## 1.2.0 — 2025-10-30

### Added
- Support for CUDA 13 on Linux and Windows
- Support for JetPack SDK 7.0

---

## 1.1.2 — 2025-10-14

### Fixed
- Removed executable flag from library ELFs to fix compatibility with Ubuntu 25
- Included `msvcp140_1.dll` in Windows artifacts to resolve runtime errors on some systems

---

## 1.1.1 — 2025-09-03

### Fixed
- Inference stopping after a few iterations for some models

---

## 1.1.0 — 2025-07-15

### Added
- Windows support

---

## 1.0.0 — 2025-06-10

Initial release.

- OAAX runtime and conversion toolchain for NVIDIA GPUs
- Linux x86_64 and aarch64 platforms
- CUDA 11 and 12 support
- Conversion toolchain runs on x86_64
