# Conversion Toolchain

Compiles an ONNX model into a TensorRT engine (`.trt`) optimised for a specific NVIDIA GPU architecture. The engine is produced at conversion time on a machine with a physical GPU — TensorRT profiles and compiles kernels against real hardware.

## Host machine setup

**Requirements:** Linux, NVIDIA GPU + driver, Docker, nvidia-container-toolkit.

**1. Install the NVIDIA driver**
```bash
sudo apt-get install -y nvidia-driver-<version> && sudo reboot
```
Verify: `nvidia-smi`

**2. Install Docker**
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
```

**3. Install nvidia-container-toolkit**
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

## Build the image

```bash
bash build-toolchain.sh
```

Output: `artifacts/oaax-nvidia-tensorrt-toolchain.tar`

## Run a conversion

Prepare a zip archive:

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

| Field | Required | Values | Description |
|---|---|---|---|
| `gpu_architecture` | Yes | e.g. `sm_86` | Compute capability of the **inference** GPU |
| `precision` | Yes | `fp32`, `fp16`, `int8` | Inference precision |
| `workspace_gb` | No | integer | TensorRT workspace memory limit in GB (default: 4) |

**Finding your GPU architecture:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# "NVIDIA RTX A4000, 8.6"  →  sm_86
```

| Compute cap | `gpu_architecture` | Example GPUs |
|---|---|---|
| 7.5 | `sm_75` | T4, RTX 20xx |
| 8.0 | `sm_80` | A100 |
| 8.6 | `sm_86` | RTX 30xx, A10, RTX A4000 |
| 8.7 | `sm_87` | Jetson AGX Orin |
| 8.9 | `sm_89` | RTX 40xx, L40S |
| 9.0 | `sm_90` | H100, GH200 |
| 10.0 | `sm_100` | B100, B200 |
| 12.1 | `sm_121` | DGX Spark (GB10) |

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
