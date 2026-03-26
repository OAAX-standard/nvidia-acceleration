# TensorRT-Native OAAX Implementation

Native TensorRT inference runtime for NVIDIA GPUs. Engines are compiled ahead-of-time by the conversion toolchain and loaded directly by the runtime library — no ONNX Runtime dependency.

## Components

| Component | Location | Purpose |
|---|---|---|
| Conversion toolchain | `conversion-toolchain/` | Compiles an ONNX model into a `.trt` engine |
| Runtime library | `runtime-library/` | Loads and runs the `.trt` engine; implements the OAAX C API |

---

## Conversion Toolchain

### Host machine setup

The conversion toolchain runs inside Docker and requires GPU access at build time because TensorRT profiles and compiles kernels against real hardware.

**1. Install the NVIDIA driver**

```bash
sudo apt-get install -y nvidia-driver-<version>
sudo reboot
```

Verify: `nvidia-smi` should show your GPU.

**2. Install Docker**

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```

**3. Install nvidia-container-toolkit**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`

### Build the image

```bash
cd conversion-toolchain
bash build-toolchain.sh
```

Produces `artifacts/oaax-nvidia-tensorrt-toolchain.tar` and loads it into Docker as `oaax-nvidia-tensorrt-toolchain:<version>`.

### Run a conversion

Prepare a zip archive:

```
bundle.zip
├── model.onnx
└── config.json
```

`config.json`:
```json
{
  "gpu_architecture": "sm_86",
  "precision": "fp16",
  "workspace_gb": 4
}
```

| Field | Required | Description |
|---|---|---|
| `gpu_architecture` | Yes | Compute capability of the **target** GPU (e.g. `sm_86`) — see table below |
| `precision` | Yes | `fp32`, `fp16`, or `int8` |
| `workspace_gb` | No | TensorRT workspace memory limit in GB (default: 4) |

**Finding your GPU architecture:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# e.g. "NVIDIA RTX A4000, 8.6"  →  sm_86
```

| Compute cap | `gpu_architecture` | Example GPUs |
|---|---|---|
| 7.5 | `sm_75` | T4, RTX 20xx |
| 8.0 | `sm_80` | A100 |
| 8.6 | `sm_86` | RTX 30xx, A10, RTX A4000 |
| 8.7 | `sm_87` | Jetson AGX Orin |
| 8.9 | `sm_89` | RTX 40xx, L40S |
| 9.0 | `sm_90` | H100, GH200 |
| 10.0 | `sm_100` | B100, B200 |
| 12.1 | `sm_121` | DGX Spark (GB10) |

```bash
docker run --rm --gpus all \
  -v /path/to/bundle.zip:/input/bundle.zip \
  -v /path/to/output:/output \
  oaax-nvidia-tensorrt-toolchain:<version> \
  /input/bundle.zip /output
```

Output:
- `model.trt` — compiled TensorRT engine
- `logs.json` — conversion trace and MD5 checksums

### int8 calibration

For `"precision": "int8"`, include a `calib/` directory of representative images in the zip:

```
bundle.zip
├── model.onnx
├── config.json
└── calib/
    ├── img_001.jpg
    └── ...
```

Additional `config.json` fields for int8:
```json
{
  "calibration_data": "calib/",
  "preprocessing": {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225]
  }
}
```

Use 100–500 images representative of real inference inputs. Supported formats: JPG, PNG, BMP, WebP.

---

## Runtime Library

### Host machine setup (build machine)

The runtime is cross-compiled on a Linux x86_64 machine. No GPU is required at build time.

**1. Install build tools**

```bash
sudo apt-get install -y cmake make git
```

**2. Install cross-compilation toolchains**

Download and extract the following toolchains to `/opt/`:

| Platform | Toolchain | Source |
|---|---|---|
| `X86_64` | `x86-64-v2--glibc--stable-2022.08-1` | [Bootlin toolchains](https://toolchains.bootlin.com) |
| `AARCH64` | `gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu` | [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) |
| `AARCH64_SERVER` | `arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu` | [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) |

Expected paths after extraction:
```
/opt/x86-64-v2--glibc--stable-2022.08-1/
/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/
/opt/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu/
```

**3. Download TensorRT and CUDA**

- **TensorRT**: Download the Linux tar archive from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt) and extract it. The directory must contain `include/` and `lib/`.
- **CUDA Toolkit**: Install from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) or use an existing installation (e.g. `/usr/local/cuda`).

For cross-compiling to aarch64, use the CUDA cross-compilation target libraries (e.g. `/opt/cuda/targets/sbsa-linux/` for server aarch64, or `/opt/cuda/targets/aarch64-linux/` for Jetson).

### Build

```bash
cd runtime-library

TRT_DIR=/path/to/TensorRT-10.x.y.z \
CUDA_DIR=/usr/local/cuda \
PLATFORM=X86_64 \
bash build-runtimes.sh
```

Supported `PLATFORM` values: `X86_64`, `AARCH64`, `AARCH64_SERVER`.
If `PLATFORM` is not set, it is auto-detected from `uname -m`.

Output: `build/<PLATFORM>/bin/`

```
build/X86_64/bin/
├── libRuntimeLibrary.so      # OAAX runtime
├── libnvinfer.so.10          # } bundled TRT libs
├── libnvinfer_plugin.so.10   # }
└── libcudart.so.13           # bundled CUDA runtime
```

### Build options

| Variable | Default | Description |
|---|---|---|
| `PLATFORM` | auto-detected | `X86_64`, `AARCH64`, or `AARCH64_SERVER` |
| `TRT_DIR` | *(required)* | Path to extracted TensorRT archive |
| `CUDA_DIR` | *(required)* | Path to CUDA toolkit |
| `BUILD_TYPE` | `Release` | CMake build type (`Release`, `Debug`, `RelWithDebInfo`) |

### Target machine setup

The `bin/` directory produced by the build is self-contained — it includes `libRuntimeLibrary.so` alongside the TRT and CUDA shared libraries it depends on. No separate TensorRT or CUDA installation is needed on the target machine.

**Minimum requirements:**

| Requirement | Notes |
|---|---|
| Linux (x86_64 or aarch64) | Matching the build `PLATFORM` |
| NVIDIA GPU | Matching the `gpu_architecture` used at conversion time |
| NVIDIA driver ≥ 525 | Required by TensorRT 10 |
| glibc ≥ 2.17 | Satisfied by any modern Linux distribution |

**Driver installation on the target machine:**

```bash
sudo apt-get install -y nvidia-driver-<version>
sudo reboot
```

Verify: `nvidia-smi`

### Test

A standalone C test is provided in `runtime-library/test/test_runtime.c`. It exercises the full OAAX API cycle and requires TRT at runtime, so run it inside the toolchain container:

```bash
docker run --rm --gpus all \
  -v $(pwd)/runtime-library/build/X86_64/bin:/runtime_lib \
  -v $(pwd)/runtime-library/test/test_runtime.c:/tmp/test_runtime.c \
  -v /path/to/model.trt:/tmp/model.trt \
  --entrypoint bash \
  oaax-nvidia-tensorrt-toolchain:<version> \
  -c "gcc /tmp/test_runtime.c -L/runtime_lib -lRuntimeLibrary -Wl,-rpath,/runtime_lib -o /tmp/test_runtime && /tmp/test_runtime"
```
