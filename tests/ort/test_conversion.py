"""
ORT conversion smoke tests: model.onnx → simplified model via Docker toolchain.

Uses tests/artifacts/model.onnx (identity model, IR version 7).
No GPU required.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
DOCKER_IMAGE = os.environ.get("NVIDIA_ORT_TOOLCHAIN_IMAGE", "oaax-nvidia-toolchain:1.2.0")


def _docker_image_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        if r.returncode != 0:
            return False
        r = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True, text=True, timeout=5,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


@pytest.fixture(scope="module")
def ort_output_dir(tmp_path_factory):
    """Run the ORT Docker toolchain on the identity model once."""
    if not _docker_image_available():
        pytest.skip(
            f"Docker image '{DOCKER_IMAGE}' not available. "
            "Set NVIDIA_ORT_TOOLCHAIN_IMAGE or build the toolchain first."
        )

    onnx = ARTIFACTS_DIR / "model.onnx"
    if not onnx.exists():
        pytest.skip(f"model.onnx not found at {onnx} — run generate_model.py first")

    out_dir = tmp_path_factory.mktemp("ort_output")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(onnx, tmp_path / "model.onnx")
        docker_out = tmp_path / "output"
        docker_out.mkdir()

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{tmp_path}:/input",
                "-v", f"{docker_out}:/output",
                DOCKER_IMAGE,
                "/input/model.onnx",
                "/output",
            ],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            pytest.fail(
                f"ORT toolchain failed (exit {result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )

        for f in docker_out.iterdir():
            shutil.copy2(f, out_dir / f.name)

    return out_dir


class TestOrtConversionOutput:
    def test_onnx_produced(self, ort_output_dir):
        """At least one .onnx file must be produced."""
        onnx_files = list(ort_output_dir.glob("*.onnx"))
        assert onnx_files, f"No .onnx file found in {ort_output_dir}"

    def test_onnx_non_empty(self, ort_output_dir):
        """Output .onnx must be a non-trivial file."""
        onnx_files = list(ort_output_dir.glob("*.onnx"))
        assert onnx_files
        assert onnx_files[0].stat().st_size > 50, \
            f"Output .onnx too small: {onnx_files[0].stat().st_size} bytes"

    def test_logs_json_exists(self, ort_output_dir):
        """logs.json must be produced."""
        assert (ort_output_dir / "logs.json").exists()

    def test_logs_json_valid(self, ort_output_dir):
        """logs.json must be valid JSON."""
        with open(ort_output_dir / "logs.json") as f:
            data = json.load(f)
        assert isinstance(data, (dict, list))

    def test_logs_contain_success(self, ort_output_dir):
        """logs.json should indicate successful conversion."""
        text = (ort_output_dir / "logs.json").read_text()
        assert "Successful" in text or "simplified" in text.lower(), \
            "logs.json does not mention successful conversion"
