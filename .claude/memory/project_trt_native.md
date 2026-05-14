---
name: nvidia-acceleration project status
description: Current state, structure, and key design decisions for the OAAX NVIDIA TensorRT runtime project
type: project
---

Project is at `/home/ayoub/nvidia-acceleration` on branch `native-tensorrt-implementation`. Version: **1.2.0**.

**Why:** Native TensorRT implementation of the OAAX standard for NVIDIA GPUs — AoT kernel compilation, no ONNX Runtime, no CPU fallback.

**How to apply:** Both components are complete and tested across 4 devices. Deployment tooling (system scanner, Python wheel) added this session.

---

## Conversion Toolchain (`conversion-toolchain/`)

- Docker image: `nvcr.io/nvidia/tensorrt:26.03-py3` base — bundles **TRT 10.16**, CUDA 13
- Python wheel: `oaax_nvidia_tensorrt_conversion-1.2.0-py3-none-any.whl` — Docker-free fallback
  - Built with `bash build-wheel.sh` → `artifacts/`
  - Package name: `oaax-nvidia-tensorrt-conversion`
  - Entry point: `conversion_toolchain --input-path bundle.zip --output-dir ./output`
  - `[int8]` extra for tensorrt Python bindings
- GPU architecture **auto-detected** via `nvidia-smi` at conversion time (removed `gpu_architecture` from config.json)
- Dependency checker (`check_nvidia_dependencies(precision)` in `utils.py`):
  - Hard errors: missing `nvidia-smi`, `trtexec`, `tensorrt` (int8), `pycuda` (int8)
  - Warnings to stderr: TRT version mismatch between system trtexec/tensorrt and runtime TRT 10.16
  - Expected runtime TRT constant: `_RUNTIME_TRT_VERSION = (10, 16, 0)` in `utils.py` — update when runtime TRT changes
  - trtexec banner parsing: `[TensorRT v101600]` → (10,16,0); `[TensorRT v8601]` → (8,6,0)
- Key fixes (from earlier implementation):
  - Use `--skipInference` not `--buildOnly` (deprecated in TRT 10, causes SIGSEGV)
  - `_ImageCalibrator` must be defined INSIDE `build_int8_engine()` so `trt` is in scope for inheritance
  - ONNX opset<9 includes weight initializers in `graph.input`; filter by `model.graph.initializer` names

## Runtime Library (`runtime-library/`)

- Cross-compiled for 3 platforms: `X86_64`, `AARCH64_JETSON`, `AARCH64_SBSA`
- Bundled TRT: **10.16.0**, bundled CUDA: **13.0.48** (x86/SBSA) or **12.9.79** (Jetson)
- Build tarballs: `build/NVIDIA/{arch}/{march}/Ubuntu/{glibc}/library-cuda_{N}.tar.gz`
- `-Wl,--allow-shlib-undefined` required for Jetson cross-compile (DLA libs only on-device)
- Key TRT 10 API fixes: `getIOTensorName(i)` not `getTensorName(i)`; manual output `tensors_struct` allocation

## System Scanner (`system-scanner/`)

- Standalone C++ CLI, built natively on the target (`cmake .. && make -j`)
- Detects: arch, platform, cpu_march, glibc, CUDA major, GPU SM; outputs JSON
- Platform detection for aarch64:
  - Jetson: `tegra` in `/proc/device-tree/compatible` or `/etc/nv_tegra_release`
  - SBSA: Neoverse CPU parts (0xd4f V2, 0xd85/0xd87 V3) in `/proc/cpuinfo`, or ACPI fallback (`/sys/firmware/acpi/` present + ARM implementer 0x41)
  - DGX Spark B200 uses Neoverse V3 (0xd85/0xd87) and ACPI — no device-tree
- x86 march: reads `/proc/cpuinfo` flags → v4 / v3 / v2 / baseline
- Exit 0 = supported, exit 1 = unsupported or error

## Devices

| Device | SSH alias | Platform | SM | Runtime tarball |
|---|---|---|---|---|
| DGX Spark B200 | device-1 | sbsa / armv9-a | 12.1 | `aarch64/armv9-a/Ubuntu/glibc2.38/library-cuda_13.tar.gz` |
| Jetson Orin | device-2 | jetson / armv8.2-a+fp16+dotprod | 8.7 | `aarch64/armv8.2-a+fp16+dotprod/Ubuntu/glibc2.34/library-cuda_12.tar.gz` |
| Jetson Orin | device-3 | jetson / armv8.2-a+fp16+dotprod | 8.7 | same as device-2 |
| ThinkStation (i7-12700K) | device-4 | x86_64 / x86-64-v3 | 8.6 | `x86_64/x86-64-v3/Ubuntu/glibc2.35/library-cuda_13.tar.gz` |

SSH key: `~/.ssh/id_nvidia_devices`. Inference results: `/home/ayoub/inference-results/{device-1..4}/`.

## Key Design Decisions

- Conversion runs on the deployment machine — TRT requires the target GPU physically present
- Engines are NOT portable across TRT versions or GPU architectures
- Runtime bundles its own TRT/CUDA libs (`$ORIGIN` rpath) — no system TRT needed at inference time
- Docker conversion guarantees TRT version match; wheel requires system TRT == TRT 10.16
