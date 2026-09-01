"""
Model conversion tests: ONNX → output model via Docker toolchain.
Runs for both TRT (→ model.trt) and ORT (→ model.onnx) backends.
"""

import pytest
from tests.backends import TRT

_ALL_MODELS  = ["yolov8n", "yolo11n", "yolo11s", "mobilenetv2"]
_YOLO_640    = ["yolov8n", "yolo11n", "yolo11s"]
_YOLO_320    = ["yolo11n_320", "yolo11s_320"]


class TestModelConversion:
    """Basic output + logs checks for all models."""

    @pytest.mark.parametrize("model", _ALL_MODELS)
    def test_output_exists(self, backend, compiled_models, model):
        path = compiled_models.get(model)
        assert path and path.exists(), f"{backend.output_filename} not found for {model}"
        assert path.stat().st_size > backend.min_output_bytes, \
            f"{backend.output_filename} suspiciously small ({path.stat().st_size} bytes)"

    @pytest.mark.parametrize("model", _ALL_MODELS)
    def test_logs_produced(self, compiled_models, model):
        path = compiled_models.get(model)
        if path is None:
            pytest.skip(f"{model} not converted")
        assert (path.parent / "logs.json").exists()


class TestYoloConversion640:
    """YOLO-specific checks for 640×640 models."""

    @pytest.mark.parametrize("model", ["yolo11n", "yolo11s"])
    def test_larger_than_yolov8n(self, compiled_models, model):
        """yolo11s should not be drastically smaller than yolov8n."""
        base  = compiled_models.get("yolov8n")
        other = compiled_models.get(model)
        if not base or not other:
            pytest.skip("Both models required")
        if model == "yolo11s":
            assert other.stat().st_size >= base.stat().st_size * 0.5, \
                "yolo11s surprisingly small compared to yolov8n"


class TestYoloInt8:
    """TRT INT8 calibration test for yolov8n 640×640."""

    def test_output_exists(self, compiled_yolov8n_int8):
        path = compiled_yolov8n_int8.get("yolov8n_int8")
        assert path and path.exists(), "model.trt not produced for yolov8n_int8"
        assert path.stat().st_size > TRT.min_output_bytes, \
            f"INT8 engine suspiciously small ({path.stat().st_size} bytes)"

    def test_logs_produced(self, compiled_yolov8n_int8):
        path = compiled_yolov8n_int8.get("yolov8n_int8")
        assert path and (path.parent / "logs.json").exists()

    def test_smaller_than_fp16(self, compiled_models, compiled_yolov8n_int8):
        """INT8 engine should be smaller than the fp16 baseline."""
        fp16 = compiled_models.get("yolov8n")
        int8 = compiled_yolov8n_int8.get("yolov8n_int8")
        if not fp16 or not int8:
            pytest.skip("Both fp16 and int8 engines required")
        assert int8.stat().st_size <= fp16.stat().st_size, \
            f"INT8 ({int8.stat().st_size}) should be ≤ fp16 ({fp16.stat().st_size})"


class TestYoloConversion320:
    """YOLO 320×320 model checks."""

    @pytest.mark.parametrize("model", _YOLO_320)
    def test_output_exists(self, backend, compiled_yolo_models_320, model):
        path = compiled_yolo_models_320.get(model)
        assert path and path.exists()
        assert path.stat().st_size > backend.min_output_bytes

    @pytest.mark.parametrize("base", ["yolo11n", "yolo11s"])
    def test_320_vs_640_size(self, backend, compiled_models, compiled_yolo_models_320, base):
        path_640 = compiled_models.get(base)
        path_320 = compiled_yolo_models_320.get(f"{base}_320")
        if not path_640 or not path_320:
            pytest.skip("Both resolutions required")
        size_640, size_320 = path_640.stat().st_size, path_320.stat().st_size
        if backend.name == "trt":
            assert size_320 <= size_640, "320×320 TRT engine should be ≤ 640×640"
        else:
            ratio = abs(size_320 - size_640) / size_640
            assert ratio < 0.05, f"ORT 320 and 640 ONNX differ by {ratio:.1%} (expected <5%)"
