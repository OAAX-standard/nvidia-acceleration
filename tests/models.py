"""
Model download and ONNX export utilities for the NVIDIA OAAX test suite.

Requires: ultralytics (YOLO), torchvision (MobileNetV2)
Install:  uv sync --group dev
"""

import os
import shutil


# Input name + shape for each model — used by stage2 benchmarking.
MODEL_INPUT_CONFIGS: dict[str, tuple[str, str]] = {
    "yolov8n":      ("images", "1,3,640,640"),
    "yolov8n_int8": ("images", "1,3,640,640"),
    "yolo11n":      ("images", "1,3,640,640"),
    "yolo11s":      ("images", "1,3,640,640"),
    "yolo11n_320":  ("images", "1,3,320,320"),
    "yolo11s_320":  ("images", "1,3,320,320"),
    "mobilenetv2":  ("input",  "1,3,224,224"),
}

# Well-known local COCO128 cache populated by the intel-acceleration project.
_COCO128_DEFAULT = "/home/ayoub/intel-acceleration/tests/compiled_models/calibration/coco128/images/train2017"


def get_calibration_images(calib_dir: str | None = None) -> list[str]:
    """Return a sorted list of JPEG paths suitable for INT8 calibration.

    Falls back to downloading COCO128 via ultralytics if the preferred cache
    directory doesn't exist or is empty.
    """
    import glob

    source = calib_dir or _COCO128_DEFAULT
    images = sorted(glob.glob(os.path.join(source, "*.jpg")))
    if images:
        return images

    # Download coco128 via ultralytics as fallback
    from ultralytics.utils.downloads import download
    import tempfile

    dl_dir = os.path.join(tempfile.gettempdir(), "coco128_calib")
    download("https://ultralytics.com/assets/coco128.zip", dir=dl_dir)
    images = sorted(glob.glob(os.path.join(dl_dir, "coco128", "images", "train2017", "*.jpg")))
    if not images:
        raise FileNotFoundError(f"No calibration images found (tried {source} and download fallback)")
    return images

_YOLO_EXPORT_CONFIGS: dict[str, dict] = {
    "yolov8n":     {"model": "yolov8n.pt",  "imgsz": 640, "batch": 1},
    "yolo11n":     {"model": "yolo11n.pt",  "imgsz": 640, "batch": 1},
    "yolo11s":     {"model": "yolo11s.pt",  "imgsz": 640, "batch": 1},
    "yolo11n_320": {"model": "yolo11n.pt",  "imgsz": 320, "batch": 1},
    "yolo11s_320": {"model": "yolo11s.pt",  "imgsz": 320, "batch": 1},
}


def download_model(model_name: str, onnx_dir: str) -> str:
    """Export a model to ONNX and return the path to the .onnx file."""
    if model_name in _YOLO_EXPORT_CONFIGS:
        return _export_yolo(model_name, onnx_dir)
    if model_name == "mobilenetv2":
        return _export_mobilenetv2(onnx_dir)
    raise ValueError(f"Unknown model: {model_name!r}")


def _export_yolo(model_name: str, onnx_dir: str) -> str:
    from ultralytics import YOLO

    cfg = _YOLO_EXPORT_CONFIGS[model_name]
    onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
    if os.path.exists(onnx_path):
        return onnx_path

    model = YOLO(cfg["model"])
    exported = model.export(
        format="onnx",
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        dynamic=False,
        opset=12,
    )
    if str(exported) != onnx_path:
        os.makedirs(onnx_dir, exist_ok=True)
        shutil.move(str(exported), onnx_path)
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
