# nvidia-acceleration — OAAX Runtime for NVIDIA GPUs

## What This Repo Is

This is the NVIDIA GPU implementation of the [OAAX standard](https://github.com/OAAX-standard/OAAX) (Open AI Accelerator eXchange). It consists of two deliverables:

1. **Conversion Toolchain** (`conversion-toolchain/`) — a Docker image that takes an ONNX model and produces an OAAX-compatible model bundle.
2. **Runtime Library** (`runtime-library/`) — a C++ shared library (`libRuntimeLibrary.so` / `.dll`) that implements the OAAX C API and runs inference on an NVIDIA GPU.

The OAAX C API (`send_input` / `receive_output` etc.) is defined in the [OAAX spec repo](https://github.com/OAAX-standard/OAAX). This implementation must conform to it exactly.

---

## Repository Structure

```
nvidia-acceleration/
├── conversion-toolchain/
│   ├── Dockerfile                        # Python 3.8 base, no GPU needed
│   ├── conversion_toolchain/
│   │   ├── main.py                       # CLI entrypoint: --onnx-path, --output-dir
│   │   └── utils.py                      # simplify_onnx() using onnxsim
│   └── requirements.txt                  # onnx, onnxsim, onnxruntime
└── runtime-library/
    ├── CMakeLists.txt                     # Cross-compilation build, CUDA_VERSION param
    ├── src/
    │   ├── runtime_core.cpp              # All OAAX API implementations
    │   └── runtime_utils.cpp             # Helpers: logging, type mapping, queue utils
    ├── include/                           # runtime_core.hpp, runtime_utils.hpp
    └── deps/
        ├── onnxruntime/                   # Pre-built ORT libs (CUDA 11/12/13, x86/aarch64)
        ├── cuda/                          # cuBLAS, cuDNN, cuFFT shared libs
        ├── spdlog/                        # Async rotating logger
        ├── concurrentqueue/               # moodycamel lock-free MPMC queue
        └── tools/c-utilities/             # tensors_struct definition + helpers
```

---

## How the Runtime Works

### Inference Pipeline

```
send_input()  ──► input_tensors_queue ──► inference_thread ──► output_tensors_queue ──► receive_output()
                   (ConcurrentQueue)        (ORT session.Run)     (ConcurrentQueue)
```

- `send_input()` enqueues `tensors_struct*` and returns immediately (non-blocking)
- A dedicated `std::thread` polls the input queue (10ms sleep when idle), runs `session->Run()`, enqueues output
- `receive_output()` returns -1 with a 100ms sleep if no output is ready yet (non-blocking semantics for caller)
- One model per runtime instance; single GPU (hardcoded `device_id = 0`)

### Execution Engine

ONNX Runtime with the **CUDA Execution Provider**:
```cpp
OrtCUDAProviderOptions cuda_options;
cuda_options.device_id = 0;
cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
session_options->AppendExecutionProvider_CUDA(cuda_options);
session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
```

ORT partitions the graph at model load time — CUDA-supported ops go to GPU, the rest fall back to CPU silently.

### Memory Model

- `send_input` takes ownership of the input `tensors_struct*` (freed after inference)
- `receive_output` allocates output `tensors_struct*` via `malloc`; caller is responsible for freeing it
- Output tensor data is `memcpy`'d from ORT's memory to fresh `malloc` buffers (host memory only — no GPU pointers exposed)
- The OAAX API `void* data` fields are always host memory

### Configurable Parameters (`runtime_initialization_with_args`)

| Key | Default | Notes |
|---|---|---|
| `log_level` | 2 (info) | spdlog levels 0–6 |
| `log_file` | `runtime.log` | Rotating, 5MB × 3 |
| `num_threads` | 4 | ORT intra-op threads, capped at 8 |

---

## Conversion Toolchain

The toolchain is intentionally minimal — it only runs ONNX Simplifier on the input model:

- Input: `.onnx` file
- Output: `model-simplified.onnx` + `logs.json`
- MIME type set to `application/x-onnx; device=cpu`

No GPU compilation happens at toolchain time. All CUDA optimization is done lazily by ORT at the first `session.Run()` call.

---

## Build Matrix

| Platform | CUDA 11 | CUDA 12 | CUDA 13 |
|---|---|---|---|
| x86_64 Linux | ORT 1.17.3 | ORT 1.17.3 | ORT bbd0a80 |
| aarch64 Linux (Jetson/JetPack 7.0) | ORT 1.17.3 | ORT 1.17.3 | ORT bbd0a80 |
| x86_64 Windows | ORT 1.17.3 | ORT 1.17.3 | ORT bbd0a80 |

CUDA architectures: `sm_52, 60, 61, 70, 75, 80, 86, 87, 89`

Build flags: `-DPLATFORM=X86_64|AARCH64|AARCH64_SERVER`, `-DCUDA_VERSION=11|12|13`, `-DRUNTIME_VERSION=<version>`

> **Note**: Several `.so` files in `deps/` are empty placeholders due to GitHub file size limits. Run `find runtime-library/deps/ -iname "*.so*" -type f -empty` to find them and replace with real files from the CUDA Toolkit before building.

---

## Known Limitations

1. **No ahead-of-time GPU compilation** — ORT's CUDA EP does JIT optimization at first run; no kernel auto-tuning or layer fusion targeting specific GPU architectures.
2. **CPU fallback is silent** — ops not supported by CUDA EP silently run on CPU; there's no warning or visibility into this.
3. **Polling latency** — the inference thread sleeps 10ms when idle; `receive_output` sleeps 100ms when no output is ready. This creates worst-case ~110ms added latency in low-throughput scenarios.
4. **Single GPU only** — hardcoded `device_id = 0`.
5. **No dynamic shape profiles** — dynamic axes in ONNX models are handled passively; there's no API to configure min/opt/max shapes.
6. **`runtime_error_message()` is a stub** — always returns a static string; no structured error state.
7. **Output always copied to CPU** — even though ORT may produce GPU tensors internally, output is always memcpy'd to host.

---

## Next Steps: TensorRT-Native Implementation

Replacing ORT+CUDA EP with **native TensorRT** for best-in-class NVIDIA GPU performance.

### Design Decisions (finalised)

| Decision | Choice |
|---|---|
| Engine build location | **Toolchain** — engine compiled at conversion time, not at runtime |
| Engine portability | Caller provides target GPU arch in `config.json` |
| Dynamic shapes | **Not supported in v1** — batch size is fixed at 1 |
| Op coverage | Hard fail if TRT can't compile the model (no CPU fallback) |
| Runtime input | Raw `.trt` engine file |
| MIME type | `application/x-tensorrt` |
| Platforms | Linux only (x86_64 + aarch64); no Windows in v1 |

### Toolchain Changes

**Docker base image**: `nvcr.io/nvidia/tensorrt:XX.XX-py3` (replaces `python:3.8`)

> A GPU is required at conversion time (`docker run --gpus all`).

**Input**: zip archive containing:
- `model.onnx`
- `config.json`:
  ```json
  {
    "gpu_architecture": "sm_86",
    "precision": "fp16",
    "workspace_gb": 4
  }
  ```

**Processing**: unzip → validate config → run `trtexec`:
```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --memPoolSize=workspace:4g \
        --buildOnly
```

**Output**: `model.trt` + `logs.json`

### Runtime Changes

Drop ORT entirely. New dependencies: `libnvinfer.so` + `libcudart.so` only.

**`runtime_model_loading()`**:
```cpp
nvinfer1::IRuntime* trt_runtime = nvinfer1::createInferRuntime(logger);
engine = trt_runtime->deserializeCudaEngine(engine_data, size);
context = engine->createExecutionContext();
// Allocate persistent GPU buffers per I/O tensor
for (int i = 0; i < engine->getNbIOTensors(); i++)
    cudaMalloc(&gpu_bufs[i], tensor_byte_size(engine, i));
```

**`inference_thread_func()`** (replaces `session->Run()`):
```cpp
for each input tensor:
    cudaMemcpyAsync(gpu_bufs[i], input->data[i], size, cudaMemcpyHostToDevice, stream);
for each tensor:
    context->setTensorAddress(engine->getTensorName(i), gpu_bufs[i]);
context->enqueueV3(stream);
for each output tensor:
    cudaMemcpyAsync(out->data[i], gpu_bufs[i], size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

**`runtime_destruction()`**:
```cpp
delete context; delete engine; delete trt_runtime;
for each gpu_buf: cudaFree(gpu_buf);
cudaStreamDestroy(stream);
```

**Output tensor metadata** (names, shapes, types) is read from the engine via TRT API:
- `engine->getTensorName(i)`, `engine->getTensorIOMode(i)` → distinguish inputs from outputs
- `engine->getTensorShape(i)` → shapes
- `engine->getTensorDataType(i)` → map to `tensor_data_type`

**Queues, threading, logging, OAAX C API surface — unchanged.**

### `runtime_initialization_with_args` Parameters

No new parameters needed for v1 (no dynamic shapes, precision set at build time).
