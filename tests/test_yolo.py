"""
YOLO integration tests: ONNX → output model via Docker toolchain.
Runs for both TRT (→ model.trt) and ORT (→ model.onnx) backends.
"""

import pytest


class TestYoloConversion640:
    @pytest.mark.parametrize("model", ["yolov8n", "yolo11n", "yolo11s"])
    def test_output_exists(self, backend, compiled_yolo_models, model):
        path = compiled_yolo_models.get(model)
        assert path and path.exists(), f"{backend.output_filename} not found for {model}"
        assert path.stat().st_size > backend.min_output_bytes, \
            f"{backend.output_filename} suspiciously small ({path.stat().st_size} bytes)"

    @pytest.mark.parametrize("model", ["yolov8n", "yolo11n", "yolo11s"])
    def test_logs_produced(self, compiled_yolo_models, model):
        path = compiled_yolo_models.get(model)
        if path is None:
            pytest.skip(f"{model} not converted")
        assert (path.parent / "logs.json").exists()

    @pytest.mark.parametrize("model", ["yolo11n", "yolo11s"])
    def test_larger_than_yolov8n(self, compiled_yolo_models, model):
        """yolo11s should not be drastically smaller than yolov8n."""
        base = compiled_yolo_models.get("yolov8n")
        other = compiled_yolo_models.get(model)
        if not base or not other:
            pytest.skip("Both models required")
        if model == "yolo11s":
            assert other.stat().st_size >= base.stat().st_size * 0.5, \
                "yolo11s surprisingly small compared to yolov8n"


class TestYoloConversion320:
    @pytest.mark.parametrize("model", ["yolo11n_320", "yolo11s_320"])
    def test_output_exists(self, backend, compiled_yolo_models_320, model):
        path = compiled_yolo_models_320.get(model)
        assert path and path.exists()
        assert path.stat().st_size > backend.min_output_bytes

    @pytest.mark.parametrize("base", ["yolo11n", "yolo11s"])
    def test_320_vs_640_size(self, backend, compiled_yolo_models, compiled_yolo_models_320, base):
        path_640 = compiled_yolo_models.get(base)
        path_320 = compiled_yolo_models_320.get(f"{base}_320")
        if not path_640 or not path_320:
            pytest.skip("Both resolutions required")
        size_640, size_320 = path_640.stat().st_size, path_320.stat().st_size
        if backend.name == "trt":
            # Smaller input → smaller TRT engine
            assert size_320 <= size_640, "320×320 TRT engine should be ≤ 640×640"
        else:
            # Same weights exported at different input sizes → same ONNX file size
            ratio = abs(size_320 - size_640) / size_640
            assert ratio < 0.05, f"ORT 320 and 640 ONNX differ by {ratio:.1%} (expected <5%)"
