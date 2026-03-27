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
| Convert ONNX → `.trt` | NVIDIA GPU + driver ≥ 525, Docker, nvidia-container-toolkit, toolchain image |
| Run inference | NVIDIA GPU + driver ≥ 525 |

The runtime library is self-contained — the `bin/` directory bundles all required shared libraries. No separate TensorRT or CUDA installation is needed.

| Requirement | Notes |
|---|---|
| Linux (x86_64 or aarch64) | Must match the build `PLATFORM` |
| NVIDIA GPU | Any SM ≥ 7.5 supported by TensorRT 10 |
| NVIDIA driver ≥ 525 | Required by TensorRT 10 |
| glibc ≥ 2.17 | Satisfied by any modern Linux distribution |
