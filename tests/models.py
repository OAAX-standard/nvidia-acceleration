"""
YOLO model download and ONNX export utilities for NVIDIA TRT test suite.

Requires ultralytics: uv sync --group dev
"""

import os
from pathlib import Path


# Model name → (ultralytics class, export kwargs)
_YOLO_EXPORT_CONFIGS: dict[str, dict] = {
    "yolov8n":   {"model": "yolov8n.pt",  "imgsz": 640, "batch": 1},
    "yolo11n":   {"model": "yolo11n.pt",  "imgsz": 640, "batch": 1},
    "yolo11s":   {"model": "yolo11s.pt",  "imgsz": 640, "batch": 1},
    "yolo11n_320": {"model": "yolo11n.pt", "imgsz": 320, "batch": 1},
    "yolo11s_320": {"model": "yolo11s.pt", "imgsz": 320, "batch": 1},
}


def download_model(model_name: str, onnx_dir: str) -> str:
    """
    Export a YOLO model to ONNX and return the path to the .onnx file.
    Skips export if the file already exists.
    """
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
    # ultralytics writes the file next to the .pt; move it to onnx_dir
    if str(exported) != onnx_path:
        os.makedirs(onnx_dir, exist_ok=True)
        import shutil
        shutil.move(str(exported), onnx_path)

    return onnx_path


YOLO_MODELS = list(_YOLO_EXPORT_CONFIGS.keys())
