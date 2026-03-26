# TensorRT-Native OAAX Implementation

Native TensorRT inference runtime for NVIDIA GPUs. Engines are compiled ahead-of-time by the conversion toolchain and loaded directly by the runtime library — no ONNX Runtime dependency.

## Components

| Component | Location | Purpose |
|---|---|---|
| Conversion toolchain | `conversion-toolchain/` | Compiles an ONNX model into a `.trt` engine |
| Runtime library | `runtime-library/` | Loads and runs the `.trt` engine; implements the OAAX C API |

---

## Conversion Toolchain

### Requirements

- Docker
- NVIDIA GPU + driver (required at conversion time — TensorRT profiles kernels against real hardware)
- `nvidia-container-toolkit`

### Build the image

```bash
cd conversion-toolchain
bash build-toolchain.sh
```

This produces `artifacts/oaax-nvidia-tensorrt-toolchain.tar` (the image archive) and loads the image into Docker as `oaax-nvidia-tensorrt-toolchain:<version>`.

### Run a conversion

Prepare a zip archive containing your model and a config file:

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
| `gpu_architecture` | Yes | Compute capability of the target GPU (e.g. `sm_86`) |
| `precision` | Yes | `fp32`, `fp16`, or `int8` |
| `workspace_gb` | No | TensorRT workspace memory limit in GB (default: 4) |

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

---

## Runtime Library

### Requirements

- Linux x86_64 build machine (cross-compilation; no GPU needed at build time)
- CMake ≥ 3.14
- Cross-compilation toolchains at `/opt/`:

| Platform | Toolchain |
|---|---|
| `X86_64` | `/opt/x86-64-v2--glibc--stable-2022.08-1` |
| `AARCH64` | `/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu` |
| `AARCH64_SERVER` | `/opt/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu` |

- Extracted TensorRT archive (`TRT_DIR`) — must contain `include/` and `lib/`
- CUDA toolkit (`CUDA_DIR`) — must contain `include/` and `lib/` or `lib64/`

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

The output directory is self-contained — deploy `bin/` as-is to the target machine.

### Build options

| Variable | Default | Description |
|---|---|---|
| `PLATFORM` | auto-detected | `X86_64`, `AARCH64`, or `AARCH64_SERVER` |
| `TRT_DIR` | *(required)* | Path to extracted TensorRT archive |
| `CUDA_DIR` | *(required)* | Path to CUDA toolkit |
| `BUILD_TYPE` | `Release` | CMake build type (`Release`, `Debug`, `RelWithDebInfo`) |

### Test

A standalone C test is provided in `runtime-library/test/test_runtime.c`. It exercises the full OAAX API cycle using a SqueezeNet engine and requires TRT at runtime, so run it inside the toolchain container:

```bash
docker run --rm --gpus all \
  -v $(pwd)/runtime-library/build/X86_64/bin:/runtime_lib \
  -v $(pwd)/runtime-library/test/test_runtime.c:/tmp/test_runtime.c \
  -v /path/to/squeezenet.trt:/tmp/model.trt \
  --entrypoint bash \
  oaax-nvidia-tensorrt-toolchain:<version> \
  -c "gcc /tmp/test_runtime.c -L/runtime_lib -lRuntimeLibrary -Wl,-rpath,/runtime_lib -o /tmp/test_runtime && /tmp/test_runtime"
```
