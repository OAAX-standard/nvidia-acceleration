# OAAX NVIDIA ORT Conversion Toolchain

Converts an ONNX model to a simplified ONNX model optimised for inference with ONNX Runtime + CUDA Execution Provider. The simplification step fuses operators, removes unused nodes, and validates the graph — the output model is functionally identical but leaner.

---

## Inputs and outputs

| | |
|---|---|
| **Input** | Raw `.onnx` file mounted at `/input/<model>.onnx` |
| **Output** | Simplified `*-simplified.onnx` + `logs.json` written to `/output/` |

No GPU is required to run this toolchain.

---

## Build the Docker image

```bash
bash build-toolchain.sh
```

This builds the image and saves it as a `.tar` archive in `artifacts/`.

---

## Run the conversion

```bash
docker run --rm \
  -v /path/to/model.onnx:/input/model.onnx \
  -v /path/to/output:/output \
  oaax-nvidia-toolchain:<version> \
  /input/model.onnx /output
```

The simplified model and `logs.json` will appear in `/path/to/output/`.
