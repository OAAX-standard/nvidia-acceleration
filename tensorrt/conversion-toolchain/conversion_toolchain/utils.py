"""
Conversion toolchain utilities.

This module implements the two engine-build paths:

  fp32        →  run_trtexec()
  fp16 TRT 10 →  run_trtexec() with --fp16
  fp16 TRT 11 →  build_fp16_engine()  (pre-convert ONNX to FP16, then trtexec)
  int8 TRT 10 →  build_int8_engine()  (calibration-based, TRT Python API)
  int8 TRT 11 →  build_int8_engine_trt11()  (ModelOpt Q/DQ quantization, then trtexec)

TRT 11 removed BuilderFlag.FP16, BuilderFlag.INT8, and the calibration API.
Both int8 and fp16 now require offline model transformation before calling trtexec.

Input zip layout
----------------
  bundle.zip
  ├── model.onnx          # ONNX model (required)
  ├── config.json         # build configuration (required, see validate_config)
  └── calib/              # calibration images (required for int8 only)
      ├── img_001.jpg
      └── ...

config.json schema
------------------
  {
      "precision":           "fp16",    // "fp32" | "fp16" | "int8"
      "workspace_gb":        4,         // TRT builder workspace limit in GB
      "optimization_level":  5,         // optional; trtexec --builderOptimizationLevel (1-5, default 3)
      "avg_timing":          16,        // optional; trtexec --avgTiming (default 8)
      // --- int8 only ---
      "calibration_data": "calib/",     // directory inside the zip (images)
      "preprocessing": {                // optional; ImageNet defaults if omitted
          "mean": [0.485, 0.456, 0.406],
          "std":  [0.229, 0.224, 0.225]
      }
  }
"""

import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

# TRT major versions supported by the conversion toolchain.
# Engines built with a version outside this set may not load on known runtime builds.
_SUPPORTED_TRT_MAJOR_VERSIONS = (10, 11)


def _parse_trtexec_banner(output: str):
    """Extract (major, minor, patch) from a trtexec banner string.

    trtexec prints a version code on its first output line:
      TRT  8.6.0 → [TensorRT v8601]   (4-digit: major minor patch build)
      TRT 10.16.0 → [TensorRT v101600] (6-digit: major(2) minor(2) patch(2))
    """
    m = re.search(r'\[TensorRT v(\d+)\]', output)
    if not m:
        return None
    n = int(m.group(1))
    if n >= 100000:          # TRT 10+
        return (n // 10000, (n % 10000) // 100, n % 100)
    else:                    # TRT 8 / 9
        return (n // 1000, (n % 1000) // 100, (n % 100) // 10)


def _trtexec_version():
    """Return the trtexec TRT version as (major, minor, patch), or None."""
    path = shutil.which('trtexec')
    if not path:
        return None
    result = subprocess.run([path], capture_output=True, text=True)
    return _parse_trtexec_banner(result.stdout + result.stderr)


def _tensorrt_python_version():
    """Return the tensorrt Python package version as (major, minor, patch), or None."""
    if importlib.util.find_spec('tensorrt') is None:
        return None
    try:
        import tensorrt as trt  # noqa: PLC0415
        return tuple(int(x) for x in trt.__version__.split('.')[:3])
    except Exception:
        return None


def _version_warning(name: str, detected):
    """Return a warning string if the detected TRT major version is not supported, else None."""
    if detected is None:
        return (f'{name}: version could not be determined — '
                f'supported major versions: {_SUPPORTED_TRT_MAJOR_VERSIONS}')
    major = detected[0]
    if major not in _SUPPORTED_TRT_MAJOR_VERSIONS:
        return (f'{name}: version {detected[0]}.{detected[1]}.{detected[2]} '
                f'(major {major}) is not in the supported set {_SUPPORTED_TRT_MAJOR_VERSIONS} — '
                f'the engine may not load on known runtime builds')
    return None


def check_nvidia_dependencies(precision: str) -> int | None:
    """Verify NVIDIA dependencies are present and version-compatible.

    Missing dependencies are raised as a RuntimeError (hard stop).
    Version mismatches are printed as warnings to stderr (soft — conversion
    may still succeed but the engine might fail to load at runtime).

    Checks are precision-aware:
      - fp16 on TRT 11+: requires onnxconverter-common (--fp16 flag was removed)
      - int8 on TRT 10: requires tensorrt Python package + pycuda (calibration API)
      - int8 on TRT 11+: requires nvidia-modelopt[onnx] (calibration API removed; Q/DQ nodes instead)

    Args:
        precision: One of "fp32", "fp16", "int8".

    Returns:
        Detected TRT major version, or None if trtexec is missing/unreadable.

    Raises:
        RuntimeError: listing every missing or version-incompatible dependency.
    """
    missing  = []   # hard errors: cannot proceed
    warnings = []   # soft: proceed but flag to the user

    # nvidia-smi: always required — GPU detection and CUDA version.
    if not shutil.which('nvidia-smi'):
        missing.append(
            'nvidia-smi  →  install the NVIDIA driver '
            '(apt-get install nvidia-driver-<version>)')

    # trtexec: always required (fp32/fp16 conversion path).
    trt_version = None
    if not shutil.which('trtexec'):
        missing.append(
            'trtexec  →  install TensorRT and add its bin/ directory to PATH '
            '(typically /usr/src/tensorrt/bin or the TRT package bin/)')
    else:
        trt_version = _trtexec_version()
        w = _version_warning('trtexec', trt_version)
        if w:
            warnings.append(w)

    trt_major = trt_version[0] if trt_version else None

    # fp16 on TRT 11+: --fp16 flag was removed; model is pre-converted to FP16 via onnxconverter-common.
    if precision == 'fp16' and trt_major is not None and trt_major >= 11:
        if importlib.util.find_spec('onnxconverter_common') is None:
            missing.append(
                'onnxconverter-common (required for fp16 with TRT 11+)  →  '
                'pip install onnxconverter-common')

    if precision == 'int8':
        if trt_major is not None and trt_major >= 11:
            # TRT 11: calibration API removed; use ModelOpt to embed Q/DQ nodes offline.
            if importlib.util.find_spec('modelopt') is None:
                missing.append(
                    'nvidia-modelopt[onnx] (required for int8 with TRT 11+)  →  '
                    'pip install "nvidia-modelopt[onnx]" --extra-index-url https://pypi.nvidia.com')
        else:
            # TRT 10: calibration-based INT8 via TRT Python API.
            if importlib.util.find_spec('tensorrt') is None:
                missing.append(
                    'tensorrt  →  pip install tensorrt '
                    '--extra-index-url https://pypi.nvidia.com')
            else:
                w = _version_warning('tensorrt Python package', _tensorrt_python_version())
                if w:
                    warnings.append(w)

            if importlib.util.find_spec('pycuda') is None:
                missing.append('pycuda  →  pip install pycuda')

    if missing:
        lines = '\n'.join(f'  - {m}' for m in missing)
        if warnings:
            lines += '\n' + '\n'.join(f'  [!] {w}' for w in warnings)
        raise RuntimeError(
            f'Dependency issues for {precision} conversion:\n{lines}')

    for w in warnings:
        print(f'[WARNING] {w}', file=sys.stderr)

    return trt_major


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def unzip_input(zip_path: str, extract_dir: str):
    """Extract the input bundle and return the ONNX path and config dict.

    Args:
        zip_path:    Path to the input .zip archive.
        extract_dir: Directory into which the archive is extracted.

    Returns:
        (onnx_path, config) where config is the parsed config.json dict.

    Raises:
        FileNotFoundError: if model.onnx or config.json are absent.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    onnx_path   = os.path.join(extract_dir, 'model.onnx')
    config_path = os.path.join(extract_dir, 'config.json')

    if not os.path.exists(onnx_path):
        raise FileNotFoundError("model.onnx not found in zip archive")
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.json not found in zip archive")

    with open(config_path) as f:
        config = json.load(f)

    return onnx_path, config


# TensorRT 10 supports sm_75 and above.  Architectures below this threshold
# are rejected early so the user gets a clear message instead of a cryptic
# TRT build failure.
_TRT10_MIN_SM = 75


def detect_gpu_architecture() -> str:
    """Detect the compute capability of the first GPU via nvidia-smi.

    Returns:
        GPU architecture string in 'sm_XY' format (e.g. 'sm_86').

    Raises:
        RuntimeError: if nvidia-smi is not found or fails.
        ValueError:   if the detected compute capability is below the TRT 10 minimum.
    """
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed (exit {result.returncode}): {result.stderr.strip()}")

    # compute_cap is e.g. "8.6" — convert to "sm_86"
    cap = result.stdout.strip().splitlines()[0].strip()
    sm_str = 'sm_' + cap.replace('.', '')
    sm_int = int(cap.replace('.', ''))

    if sm_int < _TRT10_MIN_SM:
        raise ValueError(
            f"Detected GPU compute capability {cap} (sm_{sm_int}) is not supported "
            f"by TensorRT 10 (minimum: sm_{_TRT10_MIN_SM} / Turing).")

    return sm_str


def validate_config(config: dict):
    """Validate the contents of config.json.

    Raises:
        ValueError: on any missing or invalid field.
    """
    required = ['precision', 'workspace_gb']
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    # --- precision ---
    valid_precisions = ['fp32', 'fp16', 'int8']
    if config['precision'] not in valid_precisions:
        raise ValueError(
            f"precision must be one of {valid_precisions}, got '{config['precision']}'")

    # --- workspace_gb ---
    if not isinstance(config['workspace_gb'], (int, float)) or config['workspace_gb'] <= 0:
        raise ValueError(
            f"workspace_gb must be a positive number, got '{config['workspace_gb']}'")

    # --- optimization_level (optional) ---
    if 'optimization_level' in config:
        v = config['optimization_level']
        if not isinstance(v, int) or not (1 <= v <= 5):
            raise ValueError(f"optimization_level must be an integer 1-5, got '{v}'")

    # --- avg_timing (optional) ---
    if 'avg_timing' in config:
        v = config['avg_timing']
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"avg_timing must be a positive integer, got '{v}'")

    # --- int8-specific ---
    if config['precision'] == 'int8' and not config.get('calibration_data'):
        raise ValueError(
            "'calibration_data' is required in config.json when precision is 'int8'")


# ---------------------------------------------------------------------------
# fp32 / fp16 path: trtexec
# ---------------------------------------------------------------------------

def run_trtexec(onnx_path: str, output_trt_path: str, config: dict, logs):
    """Build a TensorRT engine (fp32 or fp16) by invoking trtexec.

    trtexec handles ONNX parsing, layer fusion, and kernel selection.
    The --skipInference flag builds the engine without running a warmup
    pass, which is all we need at conversion time.

    Args:
        onnx_path:       Path to the source ONNX model.
        output_trt_path: Destination path for the serialised .trt engine.
        config:          Validated config dict (see validate_config).
        logs:            Logs instance for structured output.

    Raises:
        RuntimeError: if trtexec exits with a non-zero return code.
    """
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={output_trt_path}',
        f'--memPoolSize=workspace:{config["workspace_gb"]}g',
        '--skipInference',
    ]

    # --fp16 was removed in TRT 11; fp16 on TRT 11+ uses build_fp16_engine() instead.
    trt_ver = _trtexec_version()
    if config['precision'] == 'fp16' and trt_ver and trt_ver[0] < 11:
        cmd.append('--fp16')

    if 'optimization_level' in config:
        cmd.append(f'--builderOptimizationLevel={config["optimization_level"]}')

    if 'avg_timing' in config:
        cmd.append(f'--avgTiming={config["avg_timing"]}')

    logs.add_message('Running trtexec', {'command': ' '.join(cmd)})

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Trim large output to avoid bloating logs.json
    stdout = result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout
    stderr = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
    logs.add_data(stdout=stdout, stderr=stderr, return_code=result.returncode)

    if result.returncode != 0:
        raise RuntimeError(
            f"trtexec failed (exit {result.returncode}). "
            f"stderr: {result.stderr[-500:]}")


# ---------------------------------------------------------------------------
# fp16 path for TRT 11+: Python API (--fp16 trtexec flag was removed in TRT 11)
# ---------------------------------------------------------------------------

def build_fp16_engine(onnx_path: str, output_trt_path: str, config: dict, logs):
    """Build an FP16 TensorRT engine for TRT 11+.

    TRT 11 removed BuilderFlag.FP16 and uses strongly-typed networks by default,
    so precision is driven by the ONNX model's own tensor types. This function
    converts the (FP32) ONNX model to FP16 first, then delegates to trtexec.

    Args:
        onnx_path:       Path to the source ONNX model.
        output_trt_path: Destination path for the serialised .trt engine.
        config:          Validated config dict (see validate_config).
        logs:            Logs instance for structured output.

    Raises:
        RuntimeError: if ONNX conversion or trtexec fails.
    """
    import tempfile
    import onnx
    from onnxconverter_common import float16

    logs.add_message('Converting ONNX model to FP16 (TRT 11+ strongly-typed mode)')

    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        fp16_onnx_path = f.name
    onnx.save(model_fp16, fp16_onnx_path)

    # Build with fp32 config — the model is already FP16, no precision flag needed.
    fp32_config = {**config, 'precision': 'fp32'}
    run_trtexec(fp16_onnx_path, output_trt_path, fp32_config, logs)

    os.unlink(fp16_onnx_path)


# ---------------------------------------------------------------------------
# int8 path: TRT Python API + image calibration
# ---------------------------------------------------------------------------

def get_input_shape(onnx_path: str):
    """Return the first data input shape from an ONNX model as (N, C, H, W).

    Older ONNX opsets (< 9) include weight initializers in graph.input
    alongside the actual data inputs.  This function filters them out by
    name so only the true network input is returned.  Dynamic dimensions
    are replaced with 1.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        A 4-tuple (N, C, H, W) of ints.

    Raises:
        ValueError: if no data input is found, or if the input is not 4-D.
    """
    import onnx
    model = onnx.load(onnx_path)
    initializer_names = {init.name for init in model.graph.initializer}
    data_inputs = [inp for inp in model.graph.input
                   if inp.name not in initializer_names]
    if not data_inputs:
        raise ValueError("No data inputs found in ONNX model")
    dims = data_inputs[0].type.tensor_type.shape.dim
    shape = [d.dim_value if d.dim_value > 0 else 1 for d in dims]
    if len(shape) != 4:
        raise ValueError(f"Expected 4D input (N, C, H, W), got shape {shape}")
    return tuple(shape)


def _collect_images(calib_dir: str):
    """Return a sorted list of image paths found in calib_dir.

    Supported extensions: jpg, jpeg, png, bmp, webp (case-insensitive).

    Raises:
        FileNotFoundError: if the directory contains no matching images.
    """
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(calib_dir, ext)))
        paths.extend(glob.glob(os.path.join(calib_dir, ext.upper())))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"No images found in calibration directory: {calib_dir}")
    return paths


def _preprocess_image(image_path: str, H: int, W: int, preprocessing: dict):
    """Load and preprocess a single image for TRT INT8 calibration.

    Processing steps:
      1. Open and convert to RGB.
      2. Resize to (W, H) using bilinear interpolation.
      3. Scale pixels to [0, 1].
      4. Normalise with per-channel mean and std.
      5. Transpose HWC → CHW and add a batch dimension → (1, C, H, W).

    Args:
        image_path:    Path to the image file.
        H, W:          Target height and width (from the model's input shape).
        preprocessing: Dict with optional 'mean' and 'std' keys (lists of 3
                       floats, one per RGB channel).  Defaults to ImageNet
                       values if the keys are absent.

    Returns:
        A contiguous float32 numpy array of shape (1, C, H, W).
    """
    import numpy as np
    from PIL import Image

    mean = preprocessing.get('mean', [0.485, 0.456, 0.406])
    std  = preprocessing.get('std',  [0.229, 0.224, 0.225])

    img = Image.open(image_path).convert('RGB')
    img = img.resize((W, H), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0           # HWC in [0, 1]
    img = (img - np.array(mean, dtype=np.float32)) \
             /  np.array(std,  dtype=np.float32)             # HWC normalised
    img = img.transpose(2, 0, 1)                             # CHW
    img = np.expand_dims(img, axis=0)                        # 1CHW
    return np.ascontiguousarray(img, dtype=np.float32)


def build_int8_engine(onnx_path: str, output_trt_path: str,
                      calib_dir: str, config: dict, logs):
    """Build an INT8 TensorRT engine using image-based post-training calibration.

    Unlike the fp32/fp16 path (which uses trtexec), this function drives the
    TRT builder entirely through the Python API.  This is necessary because
    TRT's calibration interface expects a Python object that yields batches of
    GPU memory pointers — something trtexec cannot accept directly from image
    files.

    Calibration algorithm: IInt8EntropyCalibrator2 (minimises KL-divergence
    between the FP32 and INT8 activation distributions — the recommended
    default for CNNs).

    The calibration cache is written alongside the engine as
    <output_trt_path>.calib.  On subsequent builds of the same model, TRT
    reads the cache instead of re-running calibration, which saves significant
    time.

    Args:
        onnx_path:       Path to the source ONNX model.
        output_trt_path: Destination path for the serialised .trt engine.
        calib_dir:       Directory containing calibration images (JPG/PNG/…).
        config:          Validated config dict (see validate_config).
        logs:            Logs instance for structured output.

    Raises:
        FileNotFoundError: if no images are found in calib_dir.
        RuntimeError:      if ONNX parsing or the TRT engine build fails.
    """
    import pycuda.autoinit  # noqa: F401 — creates the CUDA context for pycuda
    import pycuda.driver as cuda
    import tensorrt as trt

    # _ImageCalibrator must be defined inside this function so that
    # trt.IInt8EntropyCalibrator2 is already resolved when Python evaluates
    # the class statement (trt is imported just above).
    class _ImageCalibrator(trt.IInt8EntropyCalibrator2):
        """Feeds preprocessed images to the TRT INT8 calibrator one at a time.

        Each call to get_batch() uploads a single image to a persistent GPU
        buffer and returns its device pointer.  TRT reads from that buffer
        during the forward pass it runs to collect activation statistics.
        """

        def __init__(self, image_paths, input_shape, preprocessing, cache_file):
            super().__init__()
            _, C, H, W = input_shape
            self.C, self.H, self.W = C, H, W
            self.image_paths   = image_paths
            self.preprocessing = preprocessing
            self.cache_file    = cache_file
            self.index         = 0
            # Allocate a persistent GPU buffer sized for one float32 image
            self.device_buf = cuda.mem_alloc(C * H * W * 4)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            """Upload the next image to the GPU buffer and return its pointer.

            Returns None when all images have been consumed, signalling to TRT
            that calibration is complete.
            """
            if self.index >= len(self.image_paths):
                return None
            img = _preprocess_image(
                self.image_paths[self.index], self.H, self.W, self.preprocessing)
            cuda.memcpy_htod(self.device_buf, img)
            self.index += 1
            return [int(self.device_buf)]

        def read_calibration_cache(self):
            """Return the cached calibration data if it already exists."""
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            """Persist the calibration data so future builds can skip this step."""
            with open(self.cache_file, 'wb') as f:
                f.write(cache)

    preprocessing = config.get('preprocessing', {})
    workspace_gb  = config['workspace_gb']

    image_paths = _collect_images(calib_dir)
    input_shape = get_input_shape(onnx_path)  # (N, C, H, W)

    logs.add_message('Starting INT8 calibration', {
        'num_images':   len(image_paths),
        'input_shape':  list(input_shape),
        'preprocessing': preprocessing or 'ImageNet defaults',
    })

    # Build the TRT network from the ONNX model
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(trt_logger)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser     = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parsing failed: {errors}")

    # Configure the builder: INT8 mode + calibrator
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024 ** 3))
    build_config.set_flag(trt.BuilderFlag.INT8)

    cache_file = output_trt_path + '.calib'
    build_config.int8_calibrator = _ImageCalibrator(
        image_paths, input_shape, preprocessing, cache_file)

    logs.add_message('Building INT8 engine (this may take a few minutes)')

    engine_bytes = builder.build_serialized_network(network, build_config)
    if engine_bytes is None:
        raise RuntimeError("INT8 engine build failed — check TRT logs above")

    with open(output_trt_path, 'wb') as f:
        f.write(engine_bytes)

    logs.add_message('INT8 engine built', {
        'calibration_cache':      cache_file,
        'num_calibration_images': len(image_paths),
    })


# ---------------------------------------------------------------------------
# int8 path for TRT 11+: ModelOpt Q/DQ quantization
# ---------------------------------------------------------------------------

def build_int8_engine_trt11(onnx_path: str, output_trt_path: str,
                             calib_dir: str, config: dict, logs):
    """Build an INT8 TensorRT engine for TRT 11+ using ModelOpt ONNX quantization.

    TRT 11 removed the calibration-based INT8 API (BuilderFlag.INT8,
    IBuilderConfig.int8_calibrator).  The new approach is to embed Q/DQ nodes
    directly into the ONNX model offline, which TRT 11 detects automatically.
    ModelOpt handles the quantization; the quantized ONNX is then passed to
    trtexec with no special precision flags.

    Args:
        onnx_path:       Path to the source ONNX model.
        output_trt_path: Destination path for the serialised .trt engine.
        calib_dir:       Directory containing calibration images (JPG/PNG/…).
        config:          Validated config dict (see validate_config).
        logs:            Logs instance for structured output.

    Raises:
        FileNotFoundError: if no images are found in calib_dir.
        RuntimeError:      if quantization or the TRT engine build fails.
    """
    import tempfile
    import numpy as np
    import onnx
    from modelopt.onnx.quantization import quantize as modelopt_quantize

    preprocessing = config.get('preprocessing', {})
    input_shape = get_input_shape(onnx_path)
    _, C, H, W = input_shape

    # Resolve the input tensor name from the ONNX graph.
    model = onnx.load(onnx_path)
    initializer_names = {init.name for init in model.graph.initializer}
    data_inputs = [inp for inp in model.graph.input
                   if inp.name not in initializer_names]
    input_name = data_inputs[0].name

    image_paths = _collect_images(calib_dir)
    logs.add_message('Collecting calibration data for ModelOpt quantization', {
        'num_images':    len(image_paths),
        'input_shape':   list(input_shape),
        'preprocessing': preprocessing or 'ImageNet defaults',
    })

    # Stack all images into (N, C, H, W) — ModelOpt accepts a dict of numpy arrays.
    calib_array = np.stack([
        _preprocess_image(p, H, W, preprocessing).squeeze(0)  # (C, H, W)
        for p in image_paths
    ])

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        quantized_onnx_path = f.name

    logs.add_message('Embedding INT8 Q/DQ nodes via ModelOpt (this may take a few minutes)')

    modelopt_quantize(
        onnx_path,
        quantize_mode='int8',
        calibration_data={input_name: calib_array},
        output_path=quantized_onnx_path,
    )

    logs.add_message('Q/DQ quantization complete, building TRT engine')

    # TRT 11 detects Q/DQ nodes automatically — no --int8 flag needed.
    fp32_config = {**config, 'precision': 'fp32'}
    run_trtexec(quantized_onnx_path, output_trt_path, fp32_config, logs)

    os.unlink(quantized_onnx_path)

    logs.add_message('INT8 engine built', {'num_calibration_images': len(image_paths)})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def md5_hash(file_path: str) -> str:
    """Return the MD5 hex digest of a file (used for integrity logging)."""
    h = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def build_mime_type(gpu_arch: str, cpu_arch: str) -> str:
    """Build the MIME type string for a TensorRT engine.

    Args:
        gpu_arch: GPU architecture in 'sm_XY' format (from detect_gpu_architecture).
        cpu_arch: CPU architecture string, e.g. 'x86_64' or 'aarch64'.

    Returns:
        e.g. 'application/vnd.oaax.tensorrt; arch=x86_64; sm=86; trt=10'
    """
    sm = gpu_arch.replace('sm_', '')
    trt_ver = _trtexec_version()
    trt_major = trt_ver[0] if trt_ver else _RUNTIME_TRT_VERSION[0]
    return f'application/x-tensorrt; arch={cpu_arch}; sm={sm}; trt={trt_major}'
