"""
Backend configurations for OAAX toolchain conversion tests.

Each Backend encapsulates how to invoke its Docker toolchain and what output
to expect. Tests parameterize over [TRT, ORT] to exercise both backends.
"""

from __future__ import annotations

import json
import os
import os.path
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

TESTS_DIR = Path(__file__).parent
ARTIFACTS_DIR = TESTS_DIR / "artifacts"


@dataclass
class Backend:
    name: str
    docker_image_env: str
    default_image: str
    gpu_required: bool
    output_filename: str    # "model.trt" or "model.onnx"
    min_output_bytes: int   # minimum size for a non-trivial output file
    conversion_config: dict # extra config passed to TRT toolchain; empty for ORT
    docker_timeout: int     # seconds
    success_markers: list   # strings expected in logs.json on success

    @property
    def cache_dir(self) -> Path:
        return TESTS_DIR / "compiled_models" / self.name

    def docker_image(self) -> str:
        return os.environ.get(self.docker_image_env, self.default_image)

    def is_available(self) -> bool:
        """Return True if Docker is running, the image exists, and GPU is accessible if required."""
        try:
            if subprocess.run(["docker", "info"], capture_output=True, timeout=5).returncode != 0:
                return False
            if self.gpu_required:
                r = subprocess.run(
                    ["docker", "run", "--rm", "--gpus", "all",
                     "nvidia/cuda:12.0.0-base-ubuntu22.04", "true"],
                    capture_output=True, timeout=15,
                )
                if r.returncode != 0:
                    return False
            r = subprocess.run(
                ["docker", "images", "-q", self.docker_image()],
                capture_output=True, text=True, timeout=5,
            )
            return bool(r.stdout.strip())
        except Exception:
            return False

    def convert(self, model_name: str, onnx_path: Path, out_dir: Path) -> Path:
        """Run the Docker toolchain on onnx_path and return the output model file path."""
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docker_out = tmp_path / "output"
            docker_out.mkdir()

            input_arg = self._prepare_input(onnx_path, tmp_path)

            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    *(["--gpus", "all"] if self.gpu_required else []),
                    "-v", f"{tmp_path}:/input",
                    "-v", f"{docker_out}:/output",
                    self.docker_image(),
                    input_arg,
                    "/output",
                ],
                capture_output=True, text=True, timeout=self.docker_timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"{self.name} toolchain failed for '{model_name}' "
                    f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
                )

            output_path = self._find_output(docker_out, model_name)
            dest = out_dir / self.output_filename
            shutil.copy2(output_path, dest)
            if (docker_out / "logs.json").exists():
                shutil.copy2(docker_out / "logs.json", out_dir / "logs.json")
            return dest

    def _prepare_input(self, onnx_path: Path, tmp_dir: Path) -> str:
        raise NotImplementedError

    def _find_output(self, docker_out: Path, model_name: str) -> Path:
        raise NotImplementedError


class _TrtBackend(Backend):
    def _prepare_input(self, onnx_path: Path, tmp_dir: Path) -> str:
        bundle = tmp_dir / "bundle.zip"
        with zipfile.ZipFile(bundle, "w") as z:
            z.write(onnx_path, arcname="model.onnx")
            z.writestr("config.json", json.dumps(self.conversion_config))
        return "/input/bundle.zip"

    def _find_output(self, docker_out: Path, model_name: str) -> Path:
        trt = docker_out / "model.trt"
        if not trt.exists():
            raise RuntimeError(f"No model.trt produced for '{model_name}'")
        return trt


class _TrtInt8Backend(_TrtBackend):
    """TRT backend variant that bundles calibration images for INT8 quantization."""

    def __init__(self, calib_n: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.calib_n = calib_n

    def _prepare_input(self, onnx_path: Path, tmp_dir: Path) -> str:
        from tests.models import get_calibration_images

        images = get_calibration_images()[: self.calib_n]
        bundle = tmp_dir / "bundle.zip"
        with zipfile.ZipFile(bundle, "w") as z:
            z.write(onnx_path, arcname="model.onnx")
            z.writestr("config.json", json.dumps(self.conversion_config))
            for img in images:
                z.write(img, arcname=f"calib/{os.path.basename(img)}")
        return "/input/bundle.zip"


class _OrtBackend(Backend):
    def _prepare_input(self, onnx_path: Path, tmp_dir: Path) -> str:
        import onnx as _onnx
        model = _onnx.load(str(onnx_path))
        if model.ir_version > 7:
            model.ir_version = 7
        dest = tmp_dir / onnx_path.name
        _onnx.save(model, str(dest))
        return f"/input/{onnx_path.name}"

    def _find_output(self, docker_out: Path, model_name: str) -> Path:
        hits = list(docker_out.glob("*-simplified.onnx")) or list(docker_out.glob("*.onnx"))
        if not hits:
            raise RuntimeError(f"No .onnx output produced for '{model_name}'")
        return hits[0]


TRT = _TrtBackend(
    name="trt",
    docker_image_env="NVIDIA_TRT_TOOLCHAIN_IMAGE",
    default_image="oaax-nvidia-tensorrt-toolchain:1.4.0",
    gpu_required=True,
    output_filename="model.trt",
    min_output_bytes=1024 * 1024,
    conversion_config={"precision": "fp16", "workspace_gb": 2},
    docker_timeout=600,
    success_markers=["Successful", "model.trt"],
)

ORT = _OrtBackend(
    name="ort",
    docker_image_env="NVIDIA_ORT_TOOLCHAIN_IMAGE",
    default_image="oaax-nvidia-toolchain:1.3.0",
    gpu_required=False,
    output_filename="model.onnx",
    min_output_bytes=1024 * 100,
    conversion_config={},
    docker_timeout=120,
    success_markers=["Simplified ONNX model successfully"],
)

TRT_INT8 = _TrtInt8Backend(
    calib_n=50,
    name="trt_int8",
    docker_image_env="NVIDIA_TRT_TOOLCHAIN_IMAGE",
    default_image="oaax-nvidia-tensorrt-toolchain:1.4.0",
    gpu_required=True,
    output_filename="model.trt",
    min_output_bytes=1024 * 1024,
    conversion_config={"precision": "int8", "workspace_gb": 2, "calibration_data": "calib/"},
    docker_timeout=900,
    success_markers=["Successful", "model.trt"],
)

BACKENDS: dict[str, Backend] = {"trt": TRT, "ort": ORT, "trt_int8": TRT_INT8}
