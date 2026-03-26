# nvidia-acceleration

NVIDIA GPU implementation of the [OAAX standard](https://github.com/OAAX-standard/OAAX) (Open AI Accelerator eXchange), using native TensorRT for best-in-class inference performance.

| Component | Location | Purpose |
|---|---|---|
| Conversion toolchain | `conversion-toolchain/` | Compiles an ONNX model into a `.trt` engine |
| Runtime library | `runtime-library/` | Loads and runs the `.trt` engine; implements the OAAX C API |

Both components expose the same [OAAX C API](https://github.com/OAAX-standard/OAAX).

---

## How it works

The conversion toolchain compiles an ONNX model into a TensorRT engine (`.trt`) at conversion time, targeting a specific GPU architecture. The runtime library deserializes and runs that engine directly via the TensorRT C++ API — no ONNX Runtime, no JIT compilation at inference time.

```
[model.onnx]  →  conversion toolchain  →  [model.trt]  →  runtime library  →  inference
                  (GPU required)                            (TRT + CUDA only)
```

---

## Conversion Toolchain

### Host machine setup

**Requirements:** Linux, NVIDIA GPU, Docker, nvidia-container-toolkit.

```bash
# 1. Install NVIDIA driver
sudo apt-get install -y nvidia-driver-<version> && sudo reboot
# Verify: nvidia-smi

# 2. Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# 3. Install nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Build the toolchain image

```bash
cd conversion-toolchain
bash build-toolchain.sh
```

Output: `artifacts/oaax-nvidia-tensorrt-toolchain.tar`

### Convert a model

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

| Field | Required | Values | Description |
|---|---|---|---|
| `gpu_architecture` | Yes | e.g. `sm_86` | Compute capability of the **inference** GPU |
| `precision` | Yes | `fp32`, `fp16`, `int8` | Inference precision |
| `workspace_gb` | No | integer | TensorRT workspace limit in GB (default: 4) |

**Finding your GPU architecture:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# "NVIDIA RTX A4000, 8.6"  →  sm_86
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

Output: `model.trt` + `logs.json`

### int8 calibration

Include a `calib/` directory of representative images in the zip and extend `config.json`:

```json
{
  "gpu_architecture": "sm_86",
  "precision": "int8",
  "workspace_gb": 4,
  "calibration_data": "calib/",
  "preprocessing": {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225]
  }
}
```

---

## Runtime Library

### Host machine setup (build machine)

Cross-compilation runs on a Linux x86_64 machine. No GPU needed at build time.

**1. Install build tools**
```bash
sudo apt-get install -y cmake make git
```

**2. Install cross-compilation toolchains** (to `/opt/`):

| Platform | Toolchain | Source |
|---|---|---|
| `X86_64` | `x86-64-v2--glibc--stable-2022.08-1` | [Bootlin](https://toolchains.bootlin.com) |
| `AARCH64` | `gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu` | [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) |
| `AARCH64_SERVER` | `arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu` | [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) |

**3. Download TensorRT and CUDA**

- **TensorRT**: download the Linux tar archive from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt) and extract it. The directory must contain `include/` and `lib/`.
- **CUDA**: install from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), or for cross-compilation targets use `/usr/local/cuda/targets/aarch64-linux/` (Jetson) or `/usr/local/cuda/targets/sbsa-linux/` (server aarch64).

The CI environment setup is automated via `scripts/setup-env.sh`.

### Build

```bash
cd runtime-library

# X86_64
TRT_DIR=/path/to/TensorRT-10.x.y.z-x86_64 \
CUDA_DIR=/usr/local/cuda \
PLATFORM=X86_64 \
bash build-runtimes.sh

# AARCH64 (Jetson / JetPack)
TRT_DIR=/path/to/TensorRT-10.x.y.z-aarch64 \
CUDA_DIR=/usr/local/cuda/targets/aarch64-linux \
PLATFORM=AARCH64 \
bash build-runtimes.sh

# AARCH64_SERVER (DGX Spark / Grace)
TRT_DIR=/path/to/TensorRT-10.x.y.z-aarch64 \
CUDA_DIR=/usr/local/cuda/targets/sbsa-linux \
PLATFORM=AARCH64_SERVER \
bash build-runtimes.sh
```

Output: `build/<PLATFORM>/bin/` — self-contained directory including `libRuntimeLibrary.so` and bundled TRT + CUDA shared libraries.

| Variable | Default | Description |
|---|---|---|
| `PLATFORM` | auto-detected | `X86_64`, `AARCH64`, or `AARCH64_SERVER` |
| `TRT_DIR` | *(required)* | Path to extracted TensorRT archive |
| `CUDA_DIR` | *(required)* | Path to CUDA toolkit |
| `BUILD_TYPE` | `Release` | `Release`, `Debug`, `RelWithDebInfo` |

### Target machine setup

The `bin/` directory is self-contained — deploy it as-is. No separate TensorRT or CUDA installation is required on the target machine.

**Minimum requirements:**

| Requirement | Notes |
|---|---|
| Linux (x86_64 or aarch64) | Must match build `PLATFORM` |
| NVIDIA GPU | Must match `gpu_architecture` used at conversion time |
| NVIDIA driver ≥ 525 | Required by TensorRT 10 |
| glibc ≥ 2.17 | Any modern Linux distribution |

```bash
# Install NVIDIA driver on target
sudo apt-get install -y nvidia-driver-<version> && sudo reboot
```

### Test

```bash
docker run --rm --gpus all \
  -v $(pwd)/runtime-library/build/X86_64/bin:/runtime_lib \
  -v $(pwd)/runtime-library/test/test_runtime.c:/tmp/test_runtime.c \
  -v /path/to/model.trt:/tmp/model.trt \
  --entrypoint bash \
  oaax-nvidia-tensorrt-toolchain:<version> \
  -c "gcc /tmp/test_runtime.c -L/runtime_lib -lRuntimeLibrary -Wl,-rpath,/runtime_lib -o /tmp/test_runtime && /tmp/test_runtime"
```
