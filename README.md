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
│   └── runtime-library/        # Loads and runs .trt engines via OAAX v2 C API
├── onnxruntime/
│   ├── conversion-toolchain/   # Simplifies ONNX for ORT+CUDA (no GPU needed)
│   └── runtime-library/        # Runs simplified ONNX via ORT+CUDA EP
├── scripts/                    # setup-env.sh (toolchains, TRT/CUDA), download-cuda/cudnn.sh, build-in-docker.sh
└── tests/                      # Stage1 (conversion) + Stage2 (benchmark) test suite
```

---

## Supported platforms

| Platform | Architecture | Typical hardware |
|---|---|---|
| `X86_64` | Linux x86_64 | Any x86 server or workstation with NVIDIA GPU |
| `AARCH64_JETSON` | Linux aarch64 | Jetson (JetPack 7.0+) |
| `AARCH64_SBSA` | Linux aarch64 | DGX Spark, Grace-Hopper, Grace-based servers |

Windows x86_64 is supported by the ORT backend only. Runtime packages are built per CUDA major version and, for TensorRT, per TRT major version (10/11). The ORT backend names its aarch64 Jetson platform `AARCH64` (CUDA 11/12/13) and builds `AARCH64_SBSA` for CUDA 13 only.

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
// Initialize runtime with key-value config
const char *keys[]   = {"log_level", "log_file"};
const char *values[] = {"2", "runtime.log"};
Config cfg = {2, keys, values};
runtime_init(cfg);

// Load model(s) — supports multiple models at once
ModelConfig mc = {.file_path = "/path/to/model.trt"};
runtime_load_models(1, &mc);

// Enqueue input (model_id=0 for first model)
runtime_enqueue_input(0, input_tensors);

// Retrieve output (blocks up to 1000ms)
int model_id;
Tensors *out = NULL;
runtime_retrieve_output(&model_id, &out, 1000);

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

**2. Run inference** — same OAAX v2 C API, pointing at the simplified `.onnx` file.

See [`onnxruntime/conversion-toolchain/README.md`](onnxruntime/conversion-toolchain/README.md) and [`onnxruntime/runtime-library/README.md`](onnxruntime/runtime-library/README.md) for full details.

---

## Utilities

| Tool | Location | Description |
|---|---|---|
| Test suite | [`tests/`](tests/README.md) | Stage1 (model conversion) + Stage2 (benchmark) for both backends |

---

## Quick reference

```
[ ] Dev:     bash tensorrt/runtime-library/build-runtimes.sh
[ ] Dev:     bash tensorrt/conversion-toolchain/build-toolchain.sh
[ ] Target:  zip bundle.zip model.onnx config.json
[ ] Target:  docker run ... oaax-nvidia-tensorrt-toolchain → model.trt
[ ] App:     dlopen libRuntimeLibrary.so, OAAX v2 C API with model.trt
```

For a complete working example see [OAAX-standard/examples](https://github.com/OAAX-standard/examples).

## Contributing

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Every commit needs a DCO sign-off (`git commit -s`).

## License

Apache 2.0, see [LICENSE](LICENSE).
