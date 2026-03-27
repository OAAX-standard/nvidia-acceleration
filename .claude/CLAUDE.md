# nvidia-acceleration — OAAX Runtime for NVIDIA GPUs (TensorRT-native)

## What This Repo Is

NVIDIA GPU implementation of the [OAAX standard](https://github.com/OAAX-standard/OAAX) using native TensorRT inference. Two deliverables:

1. **Conversion Toolchain** (`conversion-toolchain/`) — Docker image that compiles an ONNX model into a `.trt` engine at conversion time (GPU required).
2. **Runtime Library** (`runtime-library/`) — C++ shared library (`libRuntimeLibrary.so`) implementing the OAAX C API using TensorRT directly.

---

## Repository Structure

```
nvidia-acceleration/
├── conversion-toolchain/
│   ├── Dockerfile                        # nvcr.io/nvidia/tensorrt:26.03-py3 base
│   ├── build-toolchain.sh                # builds and saves Docker image
│   ├── conversion_toolchain/
│   │   ├── main.py                       # unzip → validate → trtexec
│   │   ├── utils.py                      # int8 calibration helpers
│   │   └── logger.py                     # structured JSON logging
│   └── scripts/convert.sh               # Docker entrypoint
├── runtime-library/
│   ├── CMakeLists.txt                    # cross-compilation build; requires TRT_DIR, CUDA_DIR
│   ├── build-runtimes.sh                 # build script; requires TRT_DIR, CUDA_DIR, PLATFORM
│   ├── src/
│   │   ├── runtime_core.cpp             # OAAX API: init, model_loading, send_input, receive_output
│   │   └── runtime_utils.cpp            # logging, TRT→OAAX type mapping, queue helpers
│   ├── include/
│   │   ├── runtime_core.hpp
│   │   └── runtime_utils.hpp
│   └── test/test_runtime.c              # standalone C test (run inside toolchain container)
├── scripts/
│   └── setup-env.sh                     # CI env setup: toolchains, TRT archives, CUDA
├── .github/workflows/
│   ├── build-runtime.yml                # builds X86_64, AARCH64_JETSON, AARCH64_SBSA; uploads to S3
│   ├── build-toolchain.yml              # builds Docker image; uploads to S3
│   └── delete-temporary-artifacts.yml   # cleans S3 on PR close
└── VERSION                              # semver, read by build scripts
```

---

## How the Runtime Works

### Inference Pipeline

```
send_input()  ──► input_queue ──► inference_thread ──► output_queue ──► receive_output()
                  (ConcurrentQueue)  (TRT enqueueV3)   (ConcurrentQueue)
```

- `send_input()` enqueues `tensors_struct*` non-blocking
- Inference thread copies inputs to GPU, runs `context->enqueueV3(stream)`, copies outputs to host
- `receive_output()` polls output queue; returns -1 (100ms sleep) if not ready
- One model per runtime instance; hardcoded `device_id = 0`

### Execution Engine

Direct TensorRT deserialization — no ONNX Runtime:
```cpp
nvinfer1::IRuntime* trt_runtime = nvinfer1::createInferRuntime(logger);
engine = trt_runtime->deserializeCudaEngine(engine_data, size);
context = engine->createExecutionContext();
for (int i = 0; i < engine->getNbIOTensors(); i++)
    cudaMalloc(&gpu_bufs[i], tensor_byte_size(engine, i));
```

### Memory Model

- Input `tensors_struct*` is owned and freed by the runtime after inference
- Output `tensors_struct*` is `malloc`'d by the runtime; caller must free it
- All data in `tensors_struct.data[]` is host memory (`cudaMemcpy` to/from GPU internally)

---

## Conversion Toolchain

- Input: zip archive with `model.onnx` + `config.json` (`gpu_architecture`, `precision`, `workspace_gb`)
- Processing: `trtexec --onnx=model.onnx --saveEngine=model.trt [--fp16|--int8] --buildOnly`
- Output: `model.trt` + `logs.json`
- For int8: calibration images in `calib/` + `preprocessing` config (mean/std)
- GPU required at conversion time; engine is **not** portable across GPU architectures

---

## Build System

### Platforms

| PLATFORM | Target | Toolchain |
|---|---|---|
| `X86_64` | Linux x86_64 | `.deps/toolchains/x86-64-v2--glibc--stable-2022.08-1` |
| `AARCH64_JETSON` | Jetson / JetPack | `.deps/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu` |
| `AARCH64_SBSA` | DGX Spark / Grace | `.deps/toolchains/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu` |

### External Dependencies (not in repo)

- `TRT_DIR`: defaults to `.deps/tensorrt/cuda-<N>/<platform>/` (set by `setup-env.sh`)
- `CUDA_DIR`: defaults to `.deps/cuda/<N>/<platform>/` (set by `setup-env.sh`)

### Build Output

`runtime-library/build/NVIDIA/<arch>/<march>/Ubuntu/<libc>/library-cuda_<N>.tar.gz` — archive of `bin/` containing `libRuntimeLibrary.so` + bundled TRT + CUDA libs

### CI Artifacts (S3)

- Runtime: `s3://oaax/runtimes/<version>/NVIDIA/<arch>/<march>/Ubuntu/<libc>/library-cuda_<N>.tar.gz`
- Toolchain: `s3://oaax/conversion-toolchain/<version>/NVIDIA/oaax-nvidia-tensorrt-toolchain.tar`

---

## Known Limitations

1. **GPU required at conversion time** — TensorRT compiles kernels against real hardware.
2. **Engine not portable** — must rebuild for each target GPU architecture.
3. **Fixed batch size** — dynamic shapes not supported in v1; batch size is fixed at 1.
4. **Single GPU only** — hardcoded `device_id = 0`.
5. **Polling latency** — inference thread sleeps 10ms when idle; `receive_output` sleeps 100ms when no output is ready.
6. **`runtime_error_message()` is a stub** — returns a static string; no structured error state.
7. **Output always on CPU** — output is always `cudaMemcpy`'d to host; no GPU pointer exposure.
