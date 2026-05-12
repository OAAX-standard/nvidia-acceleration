"""
Shared session fixtures for the NVIDIA OAAX TensorRT test suite.

compiled_yolo_models converts YOLO models once per session via the
Docker TRT toolchain image, caching results to tests/compiled_models/.
Stage 1 populates this cache; Stage 2 reads from it without re-converting.

NOTE: The TRT toolchain requires a real NVIDIA GPU at conversion time.
      Docker must be run with --gpus all.
"""

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest

COMPILED_DIR = Path(__file__).parent / "compiled_models"
DOCKER_IMAGE = "oaax-nvidia-tensorrt-toolchain:1.2.0"

# INT8 requires calibration images — only FP16 is tested here
_CONFIGS: dict[str, dict] = {
    "FP16": {"precision": "fp16", "workspace_gb": 2},
}

YOLO_MODELS = ["yolov8n", "yolo11n", "yolo11s"]
YOLO_MODELS_320 = ["yolo11n_320", "yolo11s_320"]


def _docker_available_with_gpu() -> bool:
    """Return True only if Docker is running AND --gpus all is accepted."""
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


def _convert_with_docker(
    model_name: str,
    variant: str,
    onnx_path: Path,
    out_dir: Path,
    config: dict,
) -> None:
    """Convert one model+variant using the Docker TRT toolchain image."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle = tmp_path / "bundle.zip"
        docker_out = tmp_path / "output"
        docker_out.mkdir()

        with zipfile.ZipFile(bundle, "w") as z:
            z.write(onnx_path, arcname="model.onnx")
            z.writestr("config.json", json.dumps(config))

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--gpus", "all",
                "-v", f"{tmp_path}:/input",
                "-v", f"{docker_out}:/output",
                DOCKER_IMAGE,
                f"/input/bundle.zip",
                "/output",
            ],
            capture_output=True,
            text=True,
            # TRT compilation can take several minutes for large models
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Docker TRT conversion failed for {model_name} {variant} "
                f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
            )

        trt_file = docker_out / "model.trt"
        if not trt_file.exists():
            raise RuntimeError(f"No model.trt produced for {model_name} {variant}")

        out_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(trt_file, out_dir / "model.trt")
        logs_file = docker_out / "logs.json"
        if logs_file.exists():
            shutil.copy2(logs_file, out_dir / "logs.json")


@pytest.fixture(scope="session")
def compiled_yolo_models() -> dict:
    """
    Convert YOLO models (FP16) once via the Docker TRT toolchain.
    Caches results to tests/compiled_models/<model>/<variant>/model.trt.
    Returns {(model_name, variant): Path-to-trt}.
    """
    if not _docker_available_with_gpu():
        pytest.skip(
            f"Docker image '{DOCKER_IMAGE}' not available. "
            "Build with: bash tensorrt/conversion-toolchain/build-toolchain.sh"
        )

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pytest.skip("ultralytics not installed — run: pip install ultralytics")

    onnx_dir = COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    from tests.models import download_model

    result = {}
    for model_name in YOLO_MODELS:
        onnx = Path(download_model(model_name, str(onnx_dir)))
        for variant, config in _CONFIGS.items():
            trt = COMPILED_DIR / model_name / variant / "model.trt"
            if trt.exists():
                result[(model_name, variant)] = trt
                continue
            _convert_with_docker(model_name, variant, onnx, trt.parent, config)
            result[(model_name, variant)] = trt

    return result


@pytest.fixture(scope="session")
def compiled_yolo_models_320() -> dict:
    """
    Convert yolo11n/yolo11s at 320×320 (FP16) via the Docker TRT toolchain.
    Returns {(model_name, variant): Path-to-trt}.
    """
    if not _docker_available_with_gpu():
        pytest.skip(f"Docker image '{DOCKER_IMAGE}' not available.")

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pytest.skip("ultralytics not installed — run: pip install ultralytics")

    onnx_dir = COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    from tests.models import download_model

    result = {}
    for model_name in YOLO_MODELS_320:
        onnx = Path(download_model(model_name, str(onnx_dir)))
        for variant, config in _CONFIGS.items():
            trt = COMPILED_DIR / model_name / variant / "model.trt"
            if trt.exists():
                result[(model_name, variant)] = trt
                continue
            _convert_with_docker(model_name, variant, onnx, trt.parent, config)
            result[(model_name, variant)] = trt

    return result
