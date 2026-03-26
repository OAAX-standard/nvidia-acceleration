# nvidia-acceleration

This repository contains two OAAX implementations for NVIDIA GPUs:

| Implementation | Location | Engine format | Notes |
|---|---|---|---|
| **ORT + CUDA EP** | `conversion-toolchain/` `runtime-library/` | `.onnx` | ONNX Runtime with CUDA execution provider; supports Windows |
| **TensorRT-native** | `tensorrt-native/` | `.trt` | Native TRT inference; Linux only (x86_64 + aarch64) |

The TensorRT-native implementation is recommended for production — it produces smaller, faster engines with ahead-of-time kernel compilation and no CPU fallback.

Both implementations expose the same [OAAX C API](https://github.com/OAAX-standard/OAAX).

---

## TensorRT-native

### Artifacts

- **Conversion toolchain** (`tensorrt-native/conversion-toolchain/`) — a Docker image that accepts a zip archive and produces a compiled `.trt` engine.
- **Runtime library** (`tensorrt-native/runtime-library/`) — a shared library (`libRuntimeLibrary.so`) that loads the `.trt` engine and runs inference.

### Machine setup (conversion machine)

The machine that runs the toolchain needs a GPU present at conversion time because TensorRT compiles and benchmarks kernels against the actual hardware.

**Requirements**
- Linux (x86_64 or aarch64)
- NVIDIA GPU with compute capability ≥ sm_75 (Turing or newer)
- NVIDIA driver ≥ 525
- Docker
- nvidia-container-toolkit (allows Docker containers to access the GPU)

**1. Install the NVIDIA driver**

Follow the [official driver installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/) for your distribution, or on Ubuntu:

```bash
sudo apt-get install -y nvidia-driver-<version>
sudo reboot
```

Verify: `nvidia-smi` should show your GPU and driver version.

**2. Install Docker**

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # allow running Docker without sudo
newgrp docker
```

**3. Install nvidia-container-toolkit**

This is the component that makes `--gpus all` work.

```bash
# Add NVIDIA's package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify the full setup with:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

You should see your GPU listed inside the container output.

**Finding your GPU architecture**

Pass the correct `sm_XX` value in `config.json` to ensure the engine targets your GPU:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# e.g. "NVIDIA RTX A4000, 8.6"  →  gpu_architecture: "sm_86"
```

Supported architectures (TensorRT 10):

| Architecture | Compute cap | `gpu_architecture` | Example GPUs |
|---|---|---|---|
| Turing | 7.5 | `sm_75` | T4, RTX 20xx |
| Ampere | 8.0 | `sm_80` | A100 |
| Ampere | 8.6 | `sm_86` | RTX 30xx, A10, RTX A4000 |
| Ampere | 8.7 | `sm_87` | Jetson AGX Orin |
| Ada Lovelace | 8.9 | `sm_89` | RTX 40xx, L40S |
| Hopper | 9.0 | `sm_90` | H100, GH200 |
| Blackwell | 10.0 | `sm_100` | B100, B200 |
| Blackwell | 12.1 | `sm_121` | DGX Spark (GB10) |

### 1. Convert a model

#### fp32 / fp16

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

#### int8 (post-training quantization)

Include a `calib/` directory of representative images alongside the model:
```
bundle.zip
├── model.onnx
├── config.json
└── calib/
    ├── img_001.jpg
    └── ...
```

`config.json`:
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

- `calibration_data` — path to the image directory inside the zip (required for int8).
- `preprocessing` — per-channel mean and std for normalisation; defaults to ImageNet values if omitted.
- Supported image formats: JPG, PNG, BMP, WebP. Use 100–500 images representative of real inference inputs for best accuracy.
- A calibration cache (`model.trt.calib`) is written alongside the engine and reused on subsequent builds.

`gpu_architecture` is metadata only — the engine always targets the GPU present on the build machine.

#### Run the toolchain

```bash
docker run --rm --gpus all \
  -v /path/to/bundle.zip:/input/bundle.zip \
  -v /path/to/output:/output \
  oaax-nvidia-tensorrt-toolchain:<version> \
  /input/bundle.zip /output
```

**Requires:** NVIDIA GPU + driver on the host, and `nvidia-container-toolkit` installed.

Output:
- `model.trt` — compiled TensorRT engine
- `logs.json` — full conversion trace including trtexec output and MD5 checksums

#### Build the toolchain image

```bash
cd tensorrt-native/conversion-toolchain
bash build-toolchain.sh
```

### 2. Build the runtime library

The runtime uses **cross-compilation**: it builds on any Linux x86_64 machine (no GPU required).

**Prerequisites**

Cross-compilation toolchains at `/opt/`:

| Platform | Toolchain path |
|---|---|
| X86_64 | `/opt/x86-64-v2--glibc--stable-2022.08-1` |
| AARCH64 | `/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu` |
| AARCH64_SERVER | `/opt/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu` |

TensorRT and CUDA installations (not bundled in the repo):

| Variable | Description | Example |
|---|---|---|
| `TRT_DIR` | Path to extracted TensorRT archive (must contain `include/` and `lib/`) | `/opt/TensorRT-10.16.0.72` |
| `CUDA_DIR` | Path to CUDA toolkit (must contain `include/` and `lib/` or `lib64/`) | `/usr/local/cuda` |

**Build**

```bash
cd tensorrt-native/runtime-library

# X86_64
TRT_DIR=/opt/TensorRT-10.16.0.72-x86_64 CUDA_DIR=/usr/local/cuda \
  PLATFORM=X86_64 bash build-runtimes.sh

# AARCH64 (Jetson / JetPack)
TRT_DIR=/opt/TensorRT-10.16.0.72-aarch64 CUDA_DIR=/opt/cuda-aarch64 \
  PLATFORM=AARCH64 bash build-runtimes.sh

# AARCH64_SERVER (DGX Spark / Grace)
TRT_DIR=/opt/TensorRT-10.16.0.72-aarch64 CUDA_DIR=/opt/cuda-aarch64 \
  PLATFORM=AARCH64_SERVER bash build-runtimes.sh
```

Output: `tensorrt-native/runtime-library/build/<PLATFORM>/bin/`

Each platform directory contains `libRuntimeLibrary.so` and the bundled TRT and CUDA shared libraries needed for deployment.

### 3. Test the runtime library

A standalone C test is provided at `tensorrt-native/runtime-library/test/test_runtime.c`. It exercises the full OAAX API cycle: `runtime_initialization` → `runtime_model_loading` → `send_input` → `receive_output` → `runtime_destruction`.

The test requires TRT 10 at runtime, so run it inside the toolchain Docker container (which has TRT 10 installed):

```bash
# 1. Build a .trt engine first (see step 1 above)
# 2. Build the runtime library (see step 2 above)

docker run --rm --gpus all \
  -v $(pwd)/tensorrt-native/runtime-library/build/X86_64/bin:/runtime_lib \
  -v $(pwd)/tensorrt-native/runtime-library/test/test_runtime.c:/tmp/test_runtime.c \
  -v /path/to/output:/tmp/trt-output \
  --entrypoint bash \
  oaax-nvidia-tensorrt-toolchain:<version> \
  -c "
    gcc /tmp/test_runtime.c \
      -L/runtime_lib -lRuntimeLibrary \
      -Wl,-rpath,/runtime_lib \
      -o /tmp/test_runtime && \
    /tmp/test_runtime
  "
```

The test uses SqueezeNet as the reference model (input `[1,3,224,224]` float32). Expected output:

```
=== TensorRT Runtime Test ===
Runtime: OAAX NVIDIA TensorRT Runtime
Version: x.x.x
[OK] runtime_initialization
[OK] runtime_model_loading
[OK] send_input
[OK] receive_output: 1 output tensor(s)
  [0] name=softmaxout_1  rank=4  shape=[1,1000,1,1]  dtype=1
  first values: 0.0000 0.0034 0.0001 0.0011 0.0012
[OK] runtime_destruction
=== PASS ===
```

### 4. Use the runtime

Pass the `.trt` engine path to `runtime_model_loading()`. See the [OAAX examples](https://github.com/oaax-standard/examples) repository for usage patterns.

Requirements on the **target machine** (not the build machine):
- NVIDIA driver
- CUDA 13 runtime (`libcudart.so.13`)
- TensorRT 10 runtime (`libnvinfer.so.10`, `libnvinfer_plugin.so.10`)

---

## ORT + CUDA EP

See [`conversion-toolchain/README.md`](conversion-toolchain/README.md) and [`runtime-library/`](runtime-library/) for the original ONNX Runtime-based implementation.