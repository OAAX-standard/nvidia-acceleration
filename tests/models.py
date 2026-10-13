"""
Model download and ONNX export utilities for the NVIDIA OAAX test suite.

Detection models are Apache-2.0 YOLOX ONNX exports (pinned, sha256-verified);
MobileNetV2 is exported from torchvision (BSD).
Install:  uv sync --group dev
"""

import hashlib
import os
import urllib.request


# Input name + shape for each model — used by stage2 benchmarking.
MODEL_INPUT_CONFIGS: dict[str, tuple[str, str]] = {
    "yolox_nano":      ("images", "1,3,416,416"),
    "yolox_nano_int8": ("images", "1,3,416,416"),
    "yolox_tiny":      ("images", "1,3,416,416"),
    "yolox_s":         ("images", "1,3,640,640"),
    "mobilenetv2":     ("input",  "1,3,224,224"),
}

# Official YOLOX release assets (Apache-2.0): name -> sha256.
_YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{}.onnx"
_YOLOX_SHA256 = {
    "yolox_nano": "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    "yolox_tiny": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    "yolox_s":    "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
}


def get_calibration_images(calib_dir: str | None = None, n: int = 50, size: int = 416) -> list[str]:
    """Return JPEG paths for INT8 calibration.

    Uses real images from `calib_dir` (or $OAAX_CALIB_DIR) when set; otherwise
    generates deterministic synthetic images, which is enough to exercise the
    calibration pipeline but not to produce an accurate INT8 model.
    """
    import glob
    import tempfile

    source = calib_dir or os.environ.get("OAAX_CALIB_DIR")
    if source:
        images = sorted(glob.glob(os.path.join(source, "*.jpg")))
        if not images:
            raise FileNotFoundError(f"No .jpg calibration images in {source}")
        return images

    import numpy as np
    from PIL import Image

    out_dir = os.path.join(tempfile.gettempdir(), f"oaax_calib_{size}")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        path = os.path.join(out_dir, f"synthetic_{i:03d}.jpg")
        if not os.path.exists(path):
            pixels = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path, quality=90)
        paths.append(path)
    return paths


def download_model(model_name: str, onnx_dir: str) -> str:
    """Fetch/export a model as ONNX and return the path to the .onnx file."""
    if model_name in _YOLOX_SHA256:
        return _download_yolox(model_name, onnx_dir)
    if model_name == "mobilenetv2":
        return _export_mobilenetv2(onnx_dir)
    raise ValueError(f"Unknown model: {model_name!r}")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_yolox(model_name: str, onnx_dir: str) -> str:
    onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
    expected = _YOLOX_SHA256[model_name]
    if os.path.exists(onnx_path) and _sha256(onnx_path) == expected:
        return onnx_path

    os.makedirs(onnx_dir, exist_ok=True)
    urllib.request.urlretrieve(_YOLOX_URL.format(model_name), onnx_path)
    if _sha256(onnx_path) != expected:
        os.remove(onnx_path)
        raise RuntimeError(f"sha256 mismatch for downloaded {model_name}.onnx")
    return onnx_path


def _export_mobilenetv2(onnx_dir: str) -> str:
    import torch
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    onnx_path = os.path.join(onnx_dir, "mobilenetv2.onnx")
    if os.path.exists(onnx_path):
        return onnx_path

    os.makedirs(onnx_dir, exist_ok=True)
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes=None,
        dynamo=False,
    )
    return onnx_path
