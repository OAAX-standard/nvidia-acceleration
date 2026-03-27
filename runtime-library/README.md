# Runtime Library

C++ shared library (`libRuntimeLibrary.so`) that implements the OAAX C API using native TensorRT. It deserializes a `.trt` engine produced by the conversion toolchain and runs inference on an NVIDIA GPU.

## Build the runtime

**Requirements:** Linux x86_64, CMake ≥ 3.14, cross-compilation toolchains, TensorRT and CUDA from [developer.nvidia.com](https://developer.nvidia.com). No GPU needed.

**1. Run the environment setup script**

This downloads all toolchains, CUDA, and unpacks TensorRT archives into `.deps/`:
```bash
# First place TRT archives in tensorrt-archives/ (see scripts/unpack-trt.sh for details)
bash scripts/setup-env.sh
```

**2. Build**

```bash
# All platforms and CUDA versions (uses built-in matrix)
bash build-runtimes.sh

# Single combination
PLATFORM=X86_64 CUDA_VERSION=13 MARCH=x86-64-v3 bash build-runtimes.sh
```

| Variable | Default | Description |
|---|---|---|
| `PLATFORM` | all platforms | `X86_64`, `AARCH64_JETSON`, or `AARCH64_SBSA` |
| `CUDA_VERSION` | all versions | Major CUDA version (`12` or `13`) |
| `MARCH` | per-platform default (see below) | Target CPU architecture flag passed to `-march=` |
| `BUILD_TYPE` | `Release` | `Release`, `Debug`, `RelWithDebInfo` |
| `TRT_DIR` | from `.deps/tensorrt/` | Override path to extracted TensorRT archive |
| `CUDA_DIR` | from `.deps/cuda/` | Override path to CUDA toolkit |

**Available `MARCH` values per platform:**

| Platform | `MARCH` | Target hardware |
|---|---|---|
| `X86_64` | `x86-64` | Baseline (SSE2 only) |
| `X86_64` | `x86-64-v3` *(default)* | Haswell+ (AVX2, FMA) |
| `X86_64` | `x86-64-v4` | Skylake-X+ (AVX-512) |
| `AARCH64_JETSON` | `armv8-a` | Baseline aarch64 |
| `AARCH64_JETSON` | `armv8.2-a+fp16+dotprod` *(default)* | Jetson Orin (Cortex-A78AE) |
| `AARCH64_SBSA` | `armv8-a` | Baseline aarch64 |
| `AARCH64_SBSA` | `armv9-a` *(default)* | Grace / Neoverse V2 |

Output: `build/NVIDIA/<arch>/<march>/Ubuntu/<libc>/library-cuda_<N>.tar.gz`

```
build/NVIDIA/x86_64/x86-64-v3/Ubuntu/glibc2.35/
├── library-cuda_13/          # build tree
│   └── bin/
│       ├── libRuntimeLibrary.so
│       ├── libnvinfer.so.10
│       ├── libnvinfer_plugin.so.10
│       └── libcudart.so.13
└── library-cuda_13.tar.gz    # deployable archive (bin/ contents)
```

## Deploy the runtime

**Requirements:**

| Requirement | Notes |
|---|---|
| Linux (x86_64 or aarch64) | Must match the build `PLATFORM` |
| NVIDIA GPU | Must match the `gpu_architecture` used at conversion time |
| NVIDIA driver ≥ 525 | Required by TensorRT 10 |
| glibc ≥ 2.17 | Satisfied by any modern Linux distribution |

No separate TensorRT or CUDA installation is needed — the `bin/` directory bundles all required shared libraries. Copy it as-is to the deployment machine.

**Install the NVIDIA driver:**
```bash
sudo apt-get install -y nvidia-driver-<version> && sudo reboot
```
Verify: `nvidia-smi`

## Test

A standalone C test is provided in `test/test_runtime.c`. It exercises the full OAAX API cycle. Run it inside the toolchain container (which has TensorRT installed):

```bash
docker run --rm --gpus all \
  -v $(pwd)/build/NVIDIA/x86_64/x86-64-v3/Ubuntu/glibc2.35/library-cuda_13/bin:/runtime_lib \
  -v $(pwd)/test/test_runtime.c:/tmp/test_runtime.c \
  -v /path/to/model.trt:/tmp/model.trt \
  --entrypoint bash \
  oaax-nvidia-tensorrt-toolchain:<version> \
  -c "gcc /tmp/test_runtime.c -L/runtime_lib -lRuntimeLibrary -Wl,-rpath,/runtime_lib -o /tmp/test_runtime && /tmp/test_runtime"
```
