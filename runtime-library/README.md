# Runtime Library

C++ shared library (`libRuntimeLibrary.so`) that implements the OAAX C API using native TensorRT. It deserializes a `.trt` engine produced by the conversion toolchain and runs inference on an NVIDIA GPU.

## Build the runtime

**Requirements:** Linux x86_64, CMake ≥ 3.14, cross-compilation toolchains, TensorRT and CUDA from [developer.nvidia.com](https://developer.nvidia.com). No GPU needed.

**1. Install build tools**
```bash
sudo apt-get install -y cmake make git
```

**2. Install cross-compilation toolchains** (extract to `/opt/`):

| Platform | Toolchain | Source |
|---|---|---|
| `X86_64` | `x86-64-v2--glibc--stable-2022.08-1` | [Bootlin](https://toolchains.bootlin.com) |
| `AARCH64_GLIBC2_34` | `gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu` | [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) |
| `AARCH64_GLIBC2_38` | `arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu` | [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) |

**3. Download TensorRT and CUDA from NVIDIA**

- **TensorRT**: download the Linux tar.gz archive from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt) and extract it. The directory must contain `include/` and `lib/`.
- **CUDA Toolkit**: install from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). For cross-compilation to aarch64, the CUDA install also provides target-specific directories under `targets/`.

**4. Build**

```bash
# X86_64
TRT_DIR=/path/to/TensorRT-10.x.y.z-x86_64 \
CUDA_DIR=/usr/local/cuda \
CUDA_VERSION=13 \
PLATFORM=X86_64 \
MARCH=x86-64-v3 \
bash build-runtimes.sh

# AARCH64_GLIBC2_34 (Jetson / JetPack)
TRT_DIR=/path/to/TensorRT-10.x.y.z-aarch64 \
CUDA_DIR=/usr/local/cuda/targets/aarch64-linux \
CUDA_VERSION=13 \
PLATFORM=AARCH64_GLIBC2_34 \
MARCH=armv8.2-a+fp16+dotprod \
bash build-runtimes.sh

# AARCH64_GLIBC2_38 (DGX Spark / Grace)
TRT_DIR=/path/to/TensorRT-10.x.y.z-aarch64 \
CUDA_DIR=/usr/local/cuda/targets/sbsa-linux \
CUDA_VERSION=13 \
PLATFORM=AARCH64_GLIBC2_38 \
MARCH=armv9-a \
bash build-runtimes.sh
```

| Variable | Default | Description |
|---|---|---|
| `PLATFORM` | auto-detected from `uname -m` | `X86_64`, `AARCH64_GLIBC2_34`, or `AARCH64_GLIBC2_38` |
| `TRT_DIR` | *(required)* | Path to extracted TensorRT archive |
| `CUDA_DIR` | *(required)* | Path to CUDA toolkit (or target-specific subdirectory) |
| `CUDA_VERSION` | *(required)* | Major CUDA version (e.g. `12` or `13`) — used to name the output |
| `MARCH` | per-platform default (see below) | Target CPU architecture flag passed to `-march=` |
| `BUILD_TYPE` | `Release` | `Release`, `Debug`, `RelWithDebInfo` |

**Available `MARCH` values per platform:**

| Platform | `MARCH` | Target hardware |
|---|---|---|
| `X86_64` | `x86-64` | Baseline (SSE2 only) |
| `X86_64` | `x86-64-v3` *(default)* | Haswell+ (AVX2, FMA) |
| `X86_64` | `x86-64-v4` | Skylake-X+ (AVX-512) |
| `AARCH64_GLIBC2_34` | `armv8-a` | Baseline aarch64 |
| `AARCH64_GLIBC2_34` | `armv8.2-a+fp16+dotprod` *(default)* | Jetson Orin (Cortex-A78AE) |
| `AARCH64_GLIBC2_38` | `armv8-a` | Baseline aarch64 |
| `AARCH64_GLIBC2_38` | `armv9-a` *(default)* | Grace / Neoverse V2 |

Output: `build/<PLATFORM>/cuda-<CUDA_VERSION>/<MARCH>/bin/`

```
build/X86_64/cuda-13/x86-64-v3/bin/
├── libRuntimeLibrary.so      # OAAX runtime
├── libnvinfer.so.10          # }
├── libnvinfer_plugin.so.10   # } bundled TRT libs
└── libcudart.so.13           # bundled CUDA runtime
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
  -v $(pwd)/build/X86_64/cuda-13/x86-64-v3/bin:/runtime_lib \
  -v $(pwd)/test/test_runtime.c:/tmp/test_runtime.c \
  -v /path/to/model.trt:/tmp/model.trt \
  --entrypoint bash \
  oaax-nvidia-tensorrt-toolchain:<version> \
  -c "gcc /tmp/test_runtime.c -L/runtime_lib -lRuntimeLibrary -Wl,-rpath,/runtime_lib -o /tmp/test_runtime && /tmp/test_runtime"
```
