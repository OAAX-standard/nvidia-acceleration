"""
Shared session fixtures for the NVIDIA OAAX OnnxRuntime test suite.

compiled_yolo_models runs YOLO models through the ORT Docker toolchain
(oaax-nvidia-toolchain), caching simplified ONNX files to
tests/ort/compiled_models/.  Stage 1 populates this cache; Stage 2 reads it.

No GPU is required for conversion — the toolchain only simplifies ONNX graphs.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ORT_COMPILED_DIR = Path(__file__).parent / "compiled_models"
DOCKER_IMAGE = os.environ.get("NVIDIA_ORT_TOOLCHAIN_IMAGE", "oaax-nvidia-toolchain:1.2.0")

YOLO_MODELS = ["yolov8n", "yolo11n", "yolo11s"]
YOLO_MODELS_320 = ["yolo11n_320", "yolo11s_320"]


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


def _downgrade_ir(onnx_path: Path, out_path: Path) -> None:
    """Downgrade ONNX model IR version to 7 (max supported by bundled ORT)."""
    import onnx
    model = onnx.load(str(onnx_path))
    if model.ir_version > 7:
        model.ir_version = 7
    onnx.save(model, str(out_path))


def _convert_with_docker(model_name: str, onnx_path: Path, out_dir: Path) -> Path:
    """
    Convert an ONNX model through the ORT Docker toolchain.
    Returns the path to the output simplified ONNX file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        onnx_in = tmp_path / onnx_path.name

        # Downgrade IR version so the bundled ORT can load it (max IR 12)
        _downgrade_ir(onnx_path, onnx_in)

        docker_out = tmp_path / "output"
        docker_out.mkdir()

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{tmp_path}:/input",
                "-v", f"{docker_out}:/output",
                DOCKER_IMAGE,
                f"/input/{onnx_in.name}",
                "/output",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ORT toolchain failed for {model_name} "
                f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
            )

        # Output is named <stem>-simplified.onnx
        output_files = list(docker_out.glob("*-simplified.onnx"))
        if not output_files:
            # Fallback: original name if simplification was skipped
            output_files = list(docker_out.glob("*.onnx"))
        if not output_files:
            raise RuntimeError(f"No .onnx output produced for {model_name}")

        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "model.onnx"
        shutil.copy2(output_files[0], dest)
        logs_src = docker_out / "logs.json"
        if logs_src.exists():
            shutil.copy2(logs_src, out_dir / "logs.json")

        return dest


@pytest.fixture(scope="session")
def compiled_yolo_models() -> dict:
    """
    Convert YOLO 640×640 models via the ORT Docker toolchain.
    Returns {model_name: Path-to-onnx}.
    """
    if not _docker_image_available():
        pytest.skip(
            f"Docker image '{DOCKER_IMAGE}' not available. "
            "Set NVIDIA_ORT_TOOLCHAIN_IMAGE or build with: "
            "bash onnxruntime/conversion-toolchain/build-toolchain.sh"
        )

    try:
        import onnx  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")

    from tests.models import download_model

    onnx_dir = ORT_COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for model_name in YOLO_MODELS:
        onnx = Path(download_model(model_name, str(onnx_dir)))
        out_dir = ORT_COMPILED_DIR / model_name
        dest = out_dir / "model.onnx"
        if dest.exists():
            result[model_name] = dest
            continue
        result[model_name] = _convert_with_docker(model_name, onnx, out_dir)

    return result


@pytest.fixture(scope="session")
def compiled_yolo_models_320() -> dict:
    """
    Convert YOLO 320×320 models via the ORT Docker toolchain.
    Returns {model_name: Path-to-onnx}.
    """
    if not _docker_image_available():
        pytest.skip(f"Docker image '{DOCKER_IMAGE}' not available.")

    try:
        import onnx  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")

    from tests.models import download_model

    onnx_dir = ORT_COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for model_name in YOLO_MODELS_320:
        onnx = Path(download_model(model_name, str(onnx_dir)))
        out_dir = ORT_COMPILED_DIR / model_name
        dest = out_dir / "model.onnx"
        if dest.exists():
            result[model_name] = dest
            continue
        result[model_name] = _convert_with_docker(model_name, onnx, out_dir)

    return result
