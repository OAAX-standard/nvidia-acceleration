"""
Model conversion tests: ONNX → output model via Docker toolchain.
Runs for both TRT (→ model.trt) and ORT (→ model.onnx) backends.
"""

import pytest
from tests.backends import TRT

_ALL_MODELS = ["yolox_nano", "yolox_tiny", "yolox_s", "mobilenetv2"]


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


class TestYoloxConversion:
    """YOLOX-specific checks."""

    def test_s_not_smaller_than_tiny(self, compiled_models):
        """yolox_s has more parameters than yolox_tiny; its output should not be drastically smaller."""
        tiny = compiled_models.get("yolox_tiny")
        small = compiled_models.get("yolox_s")
        if not tiny or not small:
            pytest.skip("Both models required")
        assert small.stat().st_size >= tiny.stat().st_size * 0.5, \
            "yolox_s surprisingly small compared to yolox_tiny"


class TestYoloxInt8:
    """TRT INT8 calibration test for yolox_nano 416×416."""

    def test_output_exists(self, compiled_yolox_nano_int8):
        path = compiled_yolox_nano_int8.get("yolox_nano_int8")
        assert path and path.exists(), "model.trt not produced for yolox_nano_int8"
        assert path.stat().st_size > TRT.min_output_bytes, \
            f"INT8 engine suspiciously small ({path.stat().st_size} bytes)"

    def test_logs_produced(self, compiled_yolox_nano_int8):
        path = compiled_yolox_nano_int8.get("yolox_nano_int8")
        assert path and (path.parent / "logs.json").exists()

    def test_smaller_than_fp16(self, compiled_models, compiled_yolox_nano_int8):
        """INT8 engine should be smaller than the fp16 baseline."""
        fp16 = compiled_models.get("yolox_nano")
        int8 = compiled_yolox_nano_int8.get("yolox_nano_int8")
        if not fp16 or not int8:
            pytest.skip("Both fp16 and int8 engines required")
        assert int8.stat().st_size <= fp16.stat().st_size, \
            f"INT8 ({int8.stat().st_size}) should be ≤ fp16 ({fp16.stat().st_size})"
