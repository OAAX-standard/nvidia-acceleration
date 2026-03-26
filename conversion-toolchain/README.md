# OAAX NVIDIA Conversion Toolchain

This toolchain compiles an ONNX model into a TensorRT engine (`.trt`) optimised for a specific NVIDIA GPU architecture.

> **A GPU is required at conversion time.** TensorRT engine building performs hardware-specific kernel profiling and auto-tuning; the Docker container must be run with `--gpus all`.

## Input format

The toolchain accepts a **zip archive** containing:

- `model.onnx` — the ONNX model to compile
- `config.json` — build configuration:

```json
{
  "gpu_architecture": "sm_86",
  "precision": "fp16",
  "workspace_gb": 4
}
```

| Field | Required | Values | Description |
|---|---|---|---|
| `gpu_architecture` | Yes | e.g. `sm_80`, `sm_86`, `sm_89`, `sm_90` | Target GPU compute capability |
| `precision` | Yes | `fp32`, `fp16`, `int8` | Inference precision |
| `workspace_gb` | No | integer (default: `4`) | TensorRT workspace memory limit in GB |

> The engine is not portable across GPU architectures. Match `gpu_architecture` to the GPU where inference will run.

## Output

- `model.trt` — serialised TensorRT engine
- `logs.json` — conversion logs including input MD5, config used, and output MD5

MIME type: `application/x-tensorrt`

## Building the Docker image

```bash
bash build-toolchain.sh
```

This builds the image and saves it to `artifacts/`.

## Running the conversion

```bash
docker run --gpus all \
  -v /path/to/model-directory:/model \
  oaax-tensorrt-toolchain:latest \
  /model/model.zip /model/output
```

The converted engine will be written to `/path/to/model-directory/output/model.trt`.