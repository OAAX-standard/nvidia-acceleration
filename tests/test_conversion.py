"""
Conversion smoke tests: bundle.zip → TRT engine via Docker toolchain.

These tests verify the end-to-end conversion pipeline using the small
identity model from tests/artifacts/bundle.zip.

Requires:
  - Docker with NVIDIA runtime (`--gpus all`)
  - oaax-nvidia-tensorrt-toolchain Docker image built locally
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DOCKER_IMAGE = "oaax-nvidia-tensorrt-toolchain:latest"


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
def trt_output_dir(tmp_path_factory):
    """Run the Docker TRT toolchain once and return the output directory."""
    if not _docker_image_available():
        pytest.skip(
            f"Docker image '{DOCKER_IMAGE}' not available. "
            "Build with: bash tensorrt/conversion-toolchain/build-toolchain.sh"
        )

    bundle = ARTIFACTS_DIR / "bundle.zip"
    if not bundle.exists():
        pytest.skip(f"bundle.zip not found at {bundle} — run generate_model.py first")

    out_dir = tmp_path_factory.mktemp("trt_output")

    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{bundle.parent}:/input:ro",
            "-v", f"{out_dir}:/output",
            DOCKER_IMAGE,
            "/input/bundle.zip",
            "/output",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        pytest.fail(
            f"Docker conversion failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )

    return out_dir


class TestConversionOutput:
    def test_trt_engine_exists(self, trt_output_dir):
        """model.trt must be produced."""
        trt = trt_output_dir / "model.trt"
        assert trt.exists(), f"model.trt not found in {trt_output_dir}"

    def test_trt_engine_non_empty(self, trt_output_dir):
        """model.trt must be a non-trivial file (> 1 KB)."""
        trt = trt_output_dir / "model.trt"
        assert trt.stat().st_size > 1024, f"model.trt too small: {trt.stat().st_size} bytes"

    def test_logs_json_exists(self, trt_output_dir):
        """logs.json must be produced."""
        logs = trt_output_dir / "logs.json"
        assert logs.exists(), f"logs.json not found in {trt_output_dir}"

    def test_logs_json_valid(self, trt_output_dir):
        """logs.json must be valid JSON."""
        logs = trt_output_dir / "logs.json"
        with open(logs) as f:
            data = json.load(f)
        assert isinstance(data, (dict, list)), "logs.json must be a JSON object or array"

    def test_logs_contain_success(self, trt_output_dir):
        """logs.json should indicate successful conversion."""
        logs = trt_output_dir / "logs.json"
        text = logs.read_text()
        assert "Successful" in text or "model.trt" in text, \
            "logs.json does not mention successful conversion"
