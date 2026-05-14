"""
Conversion smoke tests: identity model → output via Docker toolchain.
Runs for both TRT (→ model.trt) and ORT (→ model.onnx) backends.
"""

import json


class TestConversionOutput:
    def test_output_exists(self, backend, toolchain_output):
        assert (toolchain_output / backend.output_filename).exists()

    def test_output_non_empty(self, backend, toolchain_output):
        f = toolchain_output / backend.output_filename
        assert f.stat().st_size > 50

    def test_logs_json_exists(self, toolchain_output):
        assert (toolchain_output / "logs.json").exists()

    def test_logs_json_valid(self, toolchain_output):
        data = json.loads((toolchain_output / "logs.json").read_text())
        assert isinstance(data, (dict, list))

    def test_logs_contain_success(self, backend, toolchain_output):
        text = (toolchain_output / "logs.json").read_text()
        assert any(m in text for m in backend.success_markers)
