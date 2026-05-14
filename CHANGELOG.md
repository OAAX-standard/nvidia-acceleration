# Changelog

All notable changes to this project are documented here. Versions are listed newest first.

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
