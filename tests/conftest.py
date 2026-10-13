"""
Shared fixtures for the NVIDIA OAAX test suite.

The `backend` fixture is parameterized over [TRT, ORT] so every test runs for
both backends. Fixtures that depend on `backend` inherit the parameterization.

Skip conditions:
  TRT — requires Docker with --gpus all + oaax-nvidia-tensorrt-toolchain image
  ORT — requires Docker + oaax-nvidia-toolchain image (no GPU needed)
"""

from pathlib import Path

import pytest

from tests.backends import TRT, ORT, TRT_INT8, ARTIFACTS_DIR

MODELS = ["yolox_nano", "yolox_tiny", "yolox_s", "mobilenetv2"]


def _require_backend(backend):
    if not backend.is_available():
        pytest.skip(
            f"Backend '{backend.name}' not available — "
            f"set {backend.docker_image_env} or build the toolchain image."
        )


# ── Core backend fixture ────────────────────────────────────────────────────────

@pytest.fixture(scope="session", params=[TRT, ORT], ids=["trt", "ort"])
def backend(request):
    return request.param


# ── Conversion smoke fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def toolchain_output(backend, tmp_path_factory):
    """Run the toolchain on the identity model. Returns the output directory."""
    _require_backend(backend)

    onnx = ARTIFACTS_DIR / "model.onnx"
    if not onnx.exists():
        pytest.skip(f"model.onnx not found at {onnx} — run generate_model.py first")

    out_dir = tmp_path_factory.mktemp(f"{backend.name}_output")
    backend.convert("identity", onnx, out_dir)
    return out_dir


# ── Model fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def compiled_models(backend):
    """Convert all models via the toolchain. Returns {model_name: Path}."""
    _require_backend(backend)

    from tests.models import download_model

    onnx_dir = backend.cache_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for model_name in MODELS:
        onnx = Path(download_model(model_name, str(onnx_dir)))
        dest = backend.cache_dir / model_name / backend.output_filename
        if not dest.exists():
            backend.convert(model_name, onnx, dest.parent)
        result[model_name] = dest
    return result


@pytest.fixture(scope="session")
def compiled_yolox_nano_int8():
    """Convert yolox_nano to a TRT INT8 engine. Returns {model_name: Path}."""
    _require_backend(TRT_INT8)

    from tests.models import download_model

    onnx_dir = TRT_INT8.cache_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    onnx = Path(download_model("yolox_nano", str(onnx_dir)))
    dest = TRT_INT8.cache_dir / "yolox_nano_int8" / TRT_INT8.output_filename
    if not dest.exists():
        TRT_INT8.convert("yolox_nano_int8", onnx, dest.parent)
    return {"yolox_nano_int8": dest}
