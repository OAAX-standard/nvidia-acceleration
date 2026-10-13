# Changelog

All notable changes to this project are documented here. Versions are listed newest first.

---

## 2.0.0 — 2026-10-13

### Changed
- Native TensorRT backend (`tensorrt/`): engines are compiled ahead of time by the conversion toolchain on the deployment GPU, and the runtime loads `.trt` files directly (no ONNX Runtime); adds DGX Spark / Grace (`AARCH64_SBSA`) support. The ONNX Runtime backend (`onnxruntime/`) remains alongside it
- Both runtimes implement the OAAX v2 C API (`runtime_init(Config)`, `runtime_load_models`, `runtime_enqueue_input`, `runtime_retrieve_output`, `runtime_cleanup`, `runtime_get_*`), including multi-model loading
- TRT runtime runs inference on a concurrent slot pool with configurable in-flight requests
- ORT runtime exports only the OAAX v2 API (linker version script / `.def` file); ORT-internal symbols no longer leak into the host process
- Test suite uses Apache-2.0 YOLOX models (`yolox_nano`, `yolox_tiny`, `yolox_s`) and MobileNetV2 instead of Ultralytics YOLOv8/YOLO11 (AGPL-3.0); INT8 calibration uses synthetic images unless `OAAX_CALIB_DIR` points to real ones
- Test toolchain image tags follow the repo `VERSION` file

### Added
- TensorRT 11 support in the TRT conversion toolchain and runtime build matrix (TRT 10 and 11 packages)
- ORT runtime packages bundle cuDNN (CUDA 11/12/13, Linux and Windows); ORT `AARCH64_SBSA` (DGX Spark / Grace, CUDA 13) platform
- `perf_mode` and `log_stdout` runtime config keys; `optimization_level` and `avg_timing` toolchain config keys

### Documentation
- README for the ONNX Runtime runtime library

### Removed
- `system-scanner/` utility
- `ultralytics` dev dependency and the YOLOv8/YOLO11 test models

### Fixed
- ORT: `libonnxruntime_providers_cuda.so` RUNPATH rewritten to `$ORIGIN` so its cuDNN dependency resolves once deployed
- ORT: output names are freed with the CRT allocator (fixes a Windows host crash), dangling `Ort::TypeInfo` use-after-free in `runtime_load_models`, spdlog async thread pool joined on cleanup (DLL could not be unloaded on Windows)
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
