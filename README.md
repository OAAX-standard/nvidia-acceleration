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
| `AARCH64_GLIBC2_34` | Linux aarch64 | Jetson (JetPack 7.0+) |
| `AARCH64_GLIBC2_38` | Linux aarch64 | DGX Spark, Grace-Hopper, Grace-based servers |

## Development machine

Two activities happen on the development machine: building the toolchain image and cross-compiling the runtime library. Running a model conversion also happens here and additionally requires a GPU.

| Activity | Requirements |
|---|---|
| Build toolchain Docker image | Linux, Docker |
| Run model conversion | Linux, NVIDIA GPU + driver, Docker, nvidia-container-toolkit |
| Cross-compile runtime library | Linux x86_64, CMake ≥ 3.14, cross-compilation toolchains, TensorRT and CUDA from [developer.nvidia.com](https://developer.nvidia.com) |

See each component's README for full setup instructions.

## Deployment machine

The runtime library output is self-contained — the `bin/` directory bundles all required shared libraries. No TensorRT or CUDA installation is needed on the deployment machine.

| Requirement | Notes |
|---|---|
| Linux (x86_64 or aarch64) | Must match the build platform |
| NVIDIA GPU | Must match the `gpu_architecture` used at conversion time |
| NVIDIA driver ≥ 525 | Required by TensorRT 10 |
| glibc ≥ 2.17 | Satisfied by any modern Linux distribution |
