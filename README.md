# nvidia-acceleration

NVIDIA GPU implementation of the [OAAX standard](https://github.com/OAAX-standard/OAAX) (Open AI Accelerator eXchange), using native TensorRT for best-in-class inference performance.

## How it works

An ONNX model is compiled into a TensorRT engine at conversion time, targeting the specific GPU architecture where inference will run. The runtime library then loads and executes that engine directly — no ONNX Runtime, no JIT compilation at inference time.

```
[model.onnx]  →  conversion toolchain  →  [model.trt]  →  runtime library  →  inference
                  (GPU required)                            (TRT + CUDA only)
```

## Components

| Component | Location | Details |
|---|---|---|
| Conversion toolchain | [`conversion-toolchain/`](conversion-toolchain/README.md) | Compiles an ONNX model into a `.trt` engine |
| Runtime library | [`runtime-library/`](runtime-library/README.md) | Loads and runs the `.trt` engine; implements the OAAX C API |

## Supported platforms

| Platform | Architecture | Typical hardware |
|---|---|---|
| `X86_64` | Linux x86_64 | Any x86 server or workstation with NVIDIA GPU |
| `AARCH64_JETSON` | Linux aarch64 | Jetson (JetPack 7.0+) |
| `AARCH64_SBSA` | Linux aarch64 | DGX Spark, Grace-Hopper, Grace-based servers |

## Development machine

Two activities happen on the development machine: building the toolchain image and cross-compiling the runtime library. No GPU is required.

| Activity | Requirements |
|---|---|
| Build toolchain Docker image | Linux, Docker |
| Cross-compile runtime library | Linux x86_64, CMake ≥ 3.14, cross-compilation toolchains, TensorRT and CUDA from [developer.nvidia.com](https://developer.nvidia.com) |

See each component's README for full setup instructions.

## Deployment machine

Model conversion and inference both run on the deployment machine. TensorRT profiles and compiles kernels against the actual GPU at conversion time, so the engine is always optimal for the hardware it runs on.

| Activity | Requirements |
|---|---|
| Convert ONNX → `.trt` | NVIDIA GPU + driver ≥ 525; Docker + nvidia-container-toolkit **or** Python wheel (see [conversion-toolchain/README](conversion-toolchain/README.md)) |
| Run inference | NVIDIA GPU + driver ≥ 525 |

The runtime library is self-contained — the `bin/` directory bundles all required shared libraries. No separate TensorRT or CUDA installation is needed for inference.

| Requirement | Notes |
|---|---|
| Linux (x86_64 or aarch64) | Must match the build `PLATFORM` |
| NVIDIA GPU | Any SM ≥ 7.5 supported by TensorRT 10 |
| NVIDIA driver ≥ 525 | Required by TensorRT 10 |
| glibc ≥ 2.17 | Satisfied by any modern Linux distribution |

## Utilities

| Tool | Location | Description |
|---|---|---|
| System scanner | [`system-scanner/`](system-scanner/) | Scans a machine, checks runtime support, and identifies the correct tarball to deploy |

---

## User flow

Complete end-to-end walkthrough from ONNX model to running inference on a deployment machine.

### Step 1 — Build the artifacts (development machine, one-time)

**Runtime library** — cross-compile for the target platform:
```bash
bash runtime-library/build-runtimes.sh
# Output: runtime-library/build/NVIDIA/<arch>/<march>/Ubuntu/<glibc>/library-cuda_<N>.tar.gz
```

**Conversion toolchain** — build the Docker image (recommended) or the Python wheel:
```bash
# Docker image (recommended):
bash conversion-toolchain/build-toolchain.sh
# Output: conversion-toolchain/artifacts/oaax-nvidia-tensorrt-toolchain.tar

# Python wheel (Docker-free fallback):
bash conversion-toolchain/build-wheel.sh
# Output: conversion-toolchain/artifacts/oaax_nvidia_tensorrt_conversion-<version>-py3-none-any.whl
```

---

### Step 2 — Check the deployment machine

Build and run the system scanner **on the deployment machine** to confirm it is supported and identify which runtime tarball to use:

```bash
# Build (on the deployment machine):
cd system-scanner && mkdir -p build && cd build && cmake .. && make -j

# Run:
./system_scanner --output report.json
cat report.json
```

Example output:
```json
{
  "supported": true,
  "architecture": "x86_64",
  "platform": "x86_64",
  "cpu_march": "x86-64-v3",
  "glibc_version": "2.35",
  "cuda_version_major": 13,
  "gpu_compute_capability": "8.6",
  "gpu_sm": 86,
  "recommended_runtime": "x86_64/x86-64-v3/Ubuntu/glibc2.35/library-cuda_13.tar.gz"
}
```

If `"supported": false`, the `unsupported_reason` field explains what is missing.

---

### Step 3 — Deploy the runtime library

Copy and extract the tarball identified by the scanner:

```bash
scp runtime-library/build/NVIDIA/<recommended_runtime> user@deployment-machine:/tmp/
ssh user@deployment-machine "mkdir -p /opt/oaax && tar -xzf /tmp/<tarball>.tar.gz -C /opt/oaax"
# Result: /opt/oaax/bin/libRuntimeLibrary.so  (+ bundled TRT/CUDA libs)
```

---

### Step 4 — Convert the model

Run the conversion toolchain **on the deployment machine** (GPU required). The engine is compiled specifically for the GPU that is physically present.

**Prepare the input bundle:**
```bash
# config.json
echo '{"precision": "fp16", "workspace_gb": 4}' > config.json
zip bundle.zip model.onnx config.json
```

**Convert — Docker (recommended):**
```bash
# Load the image first (one-time, from dev machine):
docker load -i oaax-nvidia-tensorrt-toolchain.tar

docker run --rm --gpus all \
  -v $(pwd)/bundle.zip:/input/bundle.zip \
  -v $(pwd)/output:/output \
  oaax-nvidia-tensorrt-toolchain:<version> \
  /input/bundle.zip /output
```

**Convert — Python wheel (when Docker GPU access is unavailable):**
```bash
pip install oaax_nvidia_tensorrt_conversion-<version>-py3-none-any.whl
conversion_toolchain --input-path bundle.zip --output-dir ./output
```

Output: `output/model.trt` and `output/logs.json`.

> The wheel checks all NVIDIA dependencies (presence and TRT version) before starting and reports exactly what is missing.

---

### Step 5 — Run inference

Load `libRuntimeLibrary.so` via the [OAAX C API](https://github.com/OAAX-standard/OAAX) and pass it `model.trt`:

```c
void *lib = dlopen("/opt/oaax/bin/libRuntimeLibrary.so", RTLD_LAZY);
// resolve: runtime_initialization, runtime_load_model, send_input, receive_output, ...

runtime_initialization(2, (char*[]){"log_level=2", "log_file=runtime.log"});
runtime_load_model("/path/to/model.trt", model_config_json);
send_input(tensors);          // non-blocking
receive_output(&out_tensors); // polls until ready
runtime_destruct();
```

For a complete working example see [OAAX-standard/examples](https://github.com/OAAX-standard/examples).

---

### Quick-reference checklist

```
[ ] Dev machine:   bash runtime-library/build-runtimes.sh
[ ] Dev machine:   bash conversion-toolchain/build-toolchain.sh  (or build-wheel.sh)
[ ] Target:        system_scanner --output report.json  →  check "supported": true
[ ] Target:        extract runtime tarball to /opt/oaax/bin/
[ ] Target:        convert model  →  model.trt
[ ] App:           dlopen libRuntimeLibrary.so, run OAAX C API
```
