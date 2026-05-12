"""
YOLO integration tests: ONNX → TRT engine via Docker toolchain.

These tests convert YOLO models to TRT engines and verify the output.
Results are cached in tests/compiled_models/ — subsequent runs are fast.

Requires:
  - Docker with NVIDIA runtime (`--gpus all`)
  - oaax-nvidia-tensorrt-toolchain Docker image built locally
  - ultralytics: pip install ultralytics
"""

from pathlib import Path

import pytest


class TestYoloConversion640:
    """YOLO 640×640 models at FP16 precision."""

    @pytest.mark.parametrize("model", ["yolov8n", "yolo11n", "yolo11s"])
    def test_fp16_trt_exists(self, compiled_yolo_models, model):
        key = (model, "FP16")
        assert key in compiled_yolo_models, f"Model {model} FP16 not converted"
        trt = compiled_yolo_models[key]
        assert trt.exists(), f"model.trt not found: {trt}"
        assert trt.stat().st_size > 1024 * 1024, \
            f"model.trt suspiciously small ({trt.stat().st_size} bytes)"

    @pytest.mark.parametrize("model", ["yolov8n", "yolo11n", "yolo11s"])
    def test_logs_produced(self, compiled_yolo_models, model):
        trt = compiled_yolo_models.get((model, "FP16"))
        if trt is None:
            pytest.skip("FP16 model not converted")
        logs = trt.parent / "logs.json"
        assert logs.exists(), f"logs.json missing alongside model.trt"


class TestYoloConversion320:
    """YOLO 320×320 models at FP16 precision."""

    @pytest.mark.parametrize("model", ["yolo11n_320", "yolo11s_320"])
    def test_fp16_trt_exists(self, compiled_yolo_models_320, model):
        key = (model, "FP16")
        assert key in compiled_yolo_models_320
        trt = compiled_yolo_models_320[key]
        assert trt.exists()
        assert trt.stat().st_size > 1024 * 1024

    @pytest.mark.parametrize("model", ["yolo11n_320", "yolo11s_320"])
    def test_320_smaller_than_640(self, compiled_yolo_models, compiled_yolo_models_320, model):
        """320×320 engine should be smaller than the 640×640 equivalent."""
        base_name = model.replace("_320", "")
        trt_640 = compiled_yolo_models.get((base_name, "FP16"))
        trt_320 = compiled_yolo_models_320.get((model, "FP16"))
        if trt_640 is None or trt_320 is None:
            pytest.skip("Both resolutions required for size comparison")
        assert trt_320.stat().st_size <= trt_640.stat().st_size, \
            "320×320 engine is not smaller than 640×640 engine"
