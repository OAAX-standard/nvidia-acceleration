# Conversion Toolchain

Compiles an ONNX model into a TensorRT engine (`.trt`) optimised for the GPU it runs on. TensorRT profiles and compiles kernels against the actual hardware at conversion time, so **the toolchain must run on the deployment machine** (or a machine with an identical GPU). This guarantees the engine is fully optimised for the target hardware with no portability trade-offs.

Two delivery formats are available — choose based on what's available on the deployment machine:

| Format | Requirements | When to use |
|---|---|---|
| **Docker image** | Docker + nvidia-container-toolkit | Recommended — pinned TRT version, reproducible |
| **Python wheel** | TensorRT installed natively + Python 3.8+ | When Docker GPU access is unavailable (e.g. some aarch64 servers) |

> **Important:** The TRT version used for conversion must match the TRT version bundled in the runtime library (currently **TRT 10.16**). The Docker image guarantees this. The wheel uses whatever TRT is installed on the system — the dependency checker will warn if there is a mismatch.

---

## Docker (recommended)

### Build the image

**Requirements:** Linux, Docker.

```bash
bash build-toolchain.sh
```

Output: `artifacts/oaax-nvidia-tensorrt-toolchain.tar`

### Install nvidia-container-toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

Verify: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`

### Run a conversion

```bash
docker run --rm --gpus all \
  -v /path/to/bundle.zip:/input/bundle.zip \
  -v /path/to/output:/output \
  oaax-nvidia-tensorrt-toolchain:<version> \
  /input/bundle.zip /output
```

---

## Python wheel (Docker-free fallback)

### Build the wheel

```bash
bash build-wheel.sh
```

Output: `artifacts/oaax_nvidia_tensorrt_conversion-<version>-py3-none-any.whl`

### Install on the deployment machine

```bash
# fp32 / fp16:
pip install oaax_nvidia_tensorrt_conversion-<version>-py3-none-any.whl

# int8 (also needs the tensorrt Python bindings):
pip install "oaax_nvidia_tensorrt_conversion-<version>-py3-none-any.whl[int8]" \
    --extra-index-url https://pypi.nvidia.com
```

**System prerequisites** (must be installed separately — not bundled in the wheel):

| Dependency | Required for | How to install |
|---|---|---|
| NVIDIA driver + `nvidia-smi` | All | `apt-get install nvidia-driver-<version>` |
| `trtexec` in `$PATH` | fp32, fp16 | Install TensorRT; add `bin/` to PATH |
| `tensorrt` Python package | int8 | `pip install tensorrt --extra-index-url https://pypi.nvidia.com` |
| `pycuda` | int8 | `pip install pycuda` |

The wheel checks all of these at startup and reports exactly what is missing or version-mismatched before attempting any conversion.

### Run a conversion

```bash
conversion_toolchain --input-path bundle.zip --output-dir ./output
```

---

## Input bundle format

```
bundle.zip
├── model.onnx
└── config.json
```

`config.json`:
```json
{
  "precision": "fp16",
  "workspace_gb": 4
}
```

| Field | Required | Values | Description |
|---|---|---|---|
| `precision` | Yes | `fp32`, `fp16`, `int8` | Inference precision |
| `workspace_gb` | No | integer | TensorRT workspace memory limit in GB (default: 4) |

The GPU architecture is auto-detected from the machine running the conversion via `nvidia-smi`. The engine is always optimised for the GPU that is physically present.

### Output

- `model.trt` — compiled TensorRT engine
- `logs.json` — conversion trace and MD5 checksums

---

## int8 calibration

For `"precision": "int8"`, include a `calib/` directory of representative images in the zip:

```
bundle.zip
├── model.onnx
├── config.json
└── calib/
    ├── img_001.jpg
    └── ...
```

Additional `config.json` fields:
```json
{
  "calibration_data": "calib/",
  "preprocessing": {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225]
  }
}
```

Use 100–500 images representative of real inference inputs. Supported formats: JPG, PNG, BMP, WebP.
