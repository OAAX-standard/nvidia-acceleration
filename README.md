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
| `AARCH64` | Linux aarch64 | Jetson (JetPack 7.0+) |
| `AARCH64_SERVER` | Linux aarch64 | DGX Spark, Grace-Hopper, Grace-based servers |

## Target machine requirements

The runtime library output directory is self-contained. Deploying it to a target machine requires only:

| Requirement | Notes |
|---|---|
| Linux (x86_64 or aarch64) | Must match the build platform |
| NVIDIA GPU | Must match the `gpu_architecture` used at conversion time |
| NVIDIA driver ≥ 525 | Required by TensorRT 10 |
| glibc ≥ 2.17 | Satisfied by any modern Linux distribution |

No separate TensorRT or CUDA installation is needed on the target — the runtime bundles all required shared libraries.

## Development environment

The **conversion toolchain** build machine needs a GPU at image-build time:
- Linux (x86_64), NVIDIA GPU + driver, Docker, nvidia-container-toolkit

The **runtime library** build machine is GPU-free (cross-compilation):
- Linux x86_64, CMake ≥ 3.14, platform-specific cross-compilation toolchains
- TensorRT Linux tar archive — download from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)
- CUDA Toolkit — download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

See each component's README for full setup and build instructions.
