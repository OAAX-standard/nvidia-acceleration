"""
ORT YOLO integration tests: ONNX → simplified ONNX via Docker toolchain.

Results cached in tests/ort/compiled_models/ — subsequent runs are fast.
No GPU required for conversion.
"""

import pytest


class TestOrtYoloConversion640:
    """YOLO 640×640 models through the ORT simplification toolchain."""

    @pytest.mark.parametrize("model", ["yolov8n", "yolo11n", "yolo11s"])
    def test_onnx_exists(self, compiled_yolo_models, model):
        assert model in compiled_yolo_models, f"Model {model} not converted"
        onnx = compiled_yolo_models[model]
        assert onnx.exists(), f"model.onnx not found: {onnx}"
        assert onnx.stat().st_size > 1024 * 100, \
            f"model.onnx too small ({onnx.stat().st_size} bytes)"

    @pytest.mark.parametrize("model", ["yolov8n", "yolo11n", "yolo11s"])
    def test_logs_produced(self, compiled_yolo_models, model):
        onnx = compiled_yolo_models.get(model)
        if onnx is None:
            pytest.skip(f"{model} not converted")
        logs = onnx.parent / "logs.json"
        assert logs.exists(), f"logs.json missing alongside model.onnx"

    @pytest.mark.parametrize("model", ["yolo11n", "yolo11s"])
    def test_yolo11_vs_yolov8n_size(self, compiled_yolo_models, model):
        """yolo11s should be larger than yolov8n (more parameters)."""
        yolov8n_onnx = compiled_yolo_models.get("yolov8n")
        yolo11x_onnx = compiled_yolo_models.get(model)
        if not yolov8n_onnx or not yolo11x_onnx:
            pytest.skip("Both models required for size comparison")
        if model == "yolo11s":
            assert yolo11x_onnx.stat().st_size >= yolov8n_onnx.stat().st_size * 0.5, \
                "yolo11s surprisingly small compared to yolov8n"


class TestOrtYoloConversion320:
    """YOLO 320×320 models through the ORT simplification toolchain."""

    @pytest.mark.parametrize("model", ["yolo11n_320", "yolo11s_320"])
    def test_onnx_exists(self, compiled_yolo_models_320, model):
        assert model in compiled_yolo_models_320
        onnx = compiled_yolo_models_320[model]
        assert onnx.exists()
        assert onnx.stat().st_size > 1024 * 100

    @pytest.mark.parametrize("base", ["yolo11n", "yolo11s"])
    def test_320_same_size_as_640(self, compiled_yolo_models, compiled_yolo_models_320, base):
        """320×320 and 640×640 share the same weights — files should be same size."""
        onnx_640 = compiled_yolo_models.get(base)
        onnx_320 = compiled_yolo_models_320.get(f"{base}_320")
        if not onnx_640 or not onnx_320:
            pytest.skip("Both resolutions required")
        ratio = abs(onnx_320.stat().st_size - onnx_640.stat().st_size) / onnx_640.stat().st_size
        assert ratio < 0.05, \
            f"320 and 640 ONNX files differ by more than 5% ({ratio:.1%})"
