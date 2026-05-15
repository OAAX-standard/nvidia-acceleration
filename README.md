# nvidia-acceleration

NVIDIA GPU implementation of the [OAAX standard](https://github.com/OAAX-standard/OAAX) (Open AI Accelerator eXchange), with two inference backends:

| Backend | How it works | Best for |
|---|---|---|
| **TensorRT** | Compiles ONNX → `.trt` engine at conversion time; runs with native TRT | Maximum inference performance |
| **ONNX Runtime** | Simplifies ONNX graph; runs with ORT + CUDA Execution Provider | Broader model compatibility, no GPU at conversion time |

---

## Repository structure

```
nvidia-acceleration/
├── tensorrt/
│   ├── conversion-toolchain/   # Compiles ONNX → .trt engine (GPU required)
│   └── runtime-library/        # Loads and runs .trt engines via OAAX C API
├── onnxruntime/
│   ├── conversion-toolchain/   # Simplifies ONNX for ORT+CUDA (no GPU needed)
│   └── runtime-library/        # Runs simplified ONNX via ORT+CUDA EP
├── scripts/
│   └── setup-env.sh            # Dev machine setup: toolchains, TRT/CUDA archives
└── tests/                      # Stage1 (conversion) + Stage2 (benchmark) test suite
```

---

## Supported platforms

| Platform | Architecture | Typical hardware |
|---|---|---|
| `X86_64` | Linux x86_64 | Any x86 server or workstation with NVIDIA GPU |
| `AARCH64_JETSON` | Linux aarch64 | Jetson (JetPack 7.0+) |
| `AARCH64_SBSA` | Linux aarch64 | DGX Spark, Grace-Hopper, Grace-based servers |

Windows x86_64 is supported by the ORT backend only.

---

## TensorRT backend

### How it works

```
[model.onnx]  →  conversion toolchain  →  [model.trt]  →  runtime library  →  inference
                  (GPU required)                            (TRT + CUDA only)
```

The conversion toolchain runs `trtexec` on the deployment machine, compiling and profiling kernels against the GPU that is physically present. The `.trt` engine is then loaded directly by the runtime library — no JIT compilation at inference time.

### Development machine (build once)

```bash
# Cross-compile the runtime library for the target platform
bash tensorrt/runtime-library/build-runtimes.sh

# Build the Docker conversion toolchain image
bash tensorrt/conversion-toolchain/build-toolchain.sh
```

No GPU required on the development machine.

### Deployment machine (per model)

**1. Convert the model** (GPU required):
```bash
# Prepare the input bundle
echo '{"precision": "fp16", "workspace_gb": 4}' > config.json
zip bundle.zip model.onnx config.json

# Load the toolchain image (first time only)
docker load -i oaax-nvidia-tensorrt-toolchain.tar

docker run --rm --gpus all \
  -v $(pwd)/bundle.zip:/input/bundle.zip \
  -v $(pwd)/output:/output \
  oaax-nvidia-tensorrt-toolchain:<version> \
  /input/bundle.zip /output
# Output: output/model.trt + output/logs.json
```

**2. Run inference** — load `libRuntimeLibrary.so` and use the [OAAX v2 C API](https://github.com/OAAX-standard/OAAX):
```c
Config cfg = {0, NULL, NULL};
runtime_init(cfg);
ModelConfig mc = {"/path/to/model.trt", NULL, 0, {0, NULL, NULL}};
runtime_load_models(1, &mc);
runtime_enqueue_input(0, input_tensors);
int model_id; Tensors *out;
runtime_retrieve_output(&model_id, &out, 5000 /* ms */);
runtime_cleanup();
```

See [`tensorrt/conversion-toolchain/README.md`](tensorrt/conversion-toolchain/README.md) and [`tensorrt/runtime-library/README.md`](tensorrt/runtime-library/README.md) for full details.

---

## ONNX Runtime backend

### How it works

```
[model.onnx]  →  conversion toolchain  →  [model.onnx]  →  runtime library  →  inference
                  (no GPU needed)           (simplified)      (ORT + CUDA EP)
```

The conversion toolchain runs ONNX Simplifier to fuse and clean up the graph. The simplified model is then loaded by the runtime library using ONNX Runtime with the CUDA Execution Provider.

### Development machine (build once)

```bash
# Build the runtime library
bash onnxruntime/runtime-library/build-runtimes.sh    # Linux
.\onnxruntime\runtime-library\build-runtimes.bat       # Windows

# Build the Docker conversion toolchain image
bash onnxruntime/conversion-toolchain/build-toolchain.sh
```

### Deployment machine (per model)

**1. Convert the model** (no GPU required):
```bash
docker load -i oaax-nvidia-toolchain.tar

docker run --rm \
  -v $(pwd)/model.onnx:/input/model.onnx \
  -v $(pwd)/output:/output \
  oaax-nvidia-toolchain:<version> \
  /input/model.onnx /output
# Output: output/*-simplified.onnx + output/logs.json
```

**2. Run inference** — same OAAX C API, pointing at the simplified `.onnx` file.

See [`onnxruntime/conversion-toolchain/README.md`](onnxruntime/conversion-toolchain/README.md) for full details.

---

## Utilities

| Tool | Location | Description |
|---|---|---|
| System scanner | [`system-scanner/`](system-scanner/) | Checks runtime support and identifies the correct tarball for the target machine |
| Test suite | [`tests/`](tests/README.md) | Stage1 (model conversion) + Stage2 (benchmark) for both backends |

---

## Quick reference

```
[ ] Dev:     bash tensorrt/runtime-library/build-runtimes.sh
[ ] Dev:     bash tensorrt/conversion-toolchain/build-toolchain.sh
[ ] Target:  zip bundle.zip model.onnx config.json
[ ] Target:  docker run ... oaax-nvidia-tensorrt-toolchain → model.trt
[ ] App:     dlopen libRuntimeLibrary.so, OAAX C API with model.trt
```

For a complete working example see [OAAX-standard/examples](https://github.com/OAAX-standard/examples).
