# Runtime Library (ONNX Runtime)

C++ shared library (`libRuntimeLibrary.so` / `RuntimeLibrary.dll`) that implements the OAAX v2 C API on top of ONNX Runtime with the CUDA Execution Provider. It loads the simplified `.onnx` model produced by the [conversion toolchain](../conversion-toolchain/README.md) and runs inference on an NVIDIA GPU.

## API overview

Same OAAX v2 C API as the TensorRT backend (see [`include/interface.h`](include/interface.h) and the [TensorRT runtime README](../../tensorrt/runtime-library/README.md#api-overview)): `runtime_init`, `runtime_load_models`, `runtime_enqueue_input`, `runtime_retrieve_output`, `runtime_cleanup`, `runtime_get_error`, `runtime_get_version`, `runtime_get_name`, `runtime_get_info`. Only these symbols are exported (`src/runtime_exports.version` on Linux, `src/runtime_exports.def` on Windows).

### Config keys (`runtime_init`)

| Key | Values | Description |
|---|---|---|
| `log_level` | `0`–`4` | Log verbosity (0 = silent, 4 = trace) |
| `log_file` | path | Log file. If it cannot be created, the runtime falls back to console-only logging. On Windows the default is a per-process file in the temp directory |
| `log_stdout` | `true` | Also log to stdout |
| `perf_mode` | `power`, `eco` | `power`: all CPU cores and all GPU memory. `eco`: half the cores and half the GPU memory |
| `num_threads` | `1`–`512` | ORT intra-op threads (overrides `perf_mode`) |
| `inter_op_threads` | `0`–`512` | ORT inter-op threads (overrides `perf_mode`) |
| `gpu_mem_limit` | bytes | CUDA EP memory limit (overrides `perf_mode`) |

Unknown keys are ignored with a warning. Invalid numeric values make `runtime_init` return an error; no C++ exception crosses the C API.

## Build the runtime

**Requirements:** CMake ≥ 3.14. No GPU needed to build.

| Target | Build host | Toolchain |
|---|---|---|
| Linux `X86_64`, `AARCH64` (Jetson/embedded), `AARCH64_SBSA` (DGX Spark / Grace) | Linux x86_64, cross-compiled | GCC toolchains extracted under `/opt` (exact names in `CMakeLists.txt`; the download URLs are in the *Install cross-compilation toolchains* step of `.github/workflows/build-onnxruntime-runtime.yml`) |
| `WINDOWS` x86_64 | Windows | MSVC (use an **x64 Native Tools Command Prompt for VS**) |

ONNX Runtime itself is pre-built and vendored in `deps/onnxruntime/<PLATFORM>_CUDA_<N>/`; cuDNN is downloaded at build time by `../../scripts/download-cudnn.sh` and bundled into the package.

```bash
# Linux: all platform/CUDA combinations
bash build-runtimes.sh

# Linux: one combination — PLATFORM CUDA_VERSION
bash build-runtimes.sh X86_64 12
```

```bat
REM Windows: CUDA 11, 12 and 13
.\build-runtimes.bat
```

| Platform | CUDA versions |
|---|---|
| `X86_64` | 11, 12, 13 |
| `AARCH64` | 11, 12, 13 |
| `AARCH64_SBSA` | 13 |
| `WINDOWS` | 11, 12, 13 |

Output (in `artifacts/`):
- Linux: `runtime-library-<PLATFORM>-cuda_<N>.tar.gz`, containing `libRuntimeLibrary.so`, the ORT libraries and the bundled cuDNN. `fix-provider-rpath.sh` rewrites the RUNPATH of `libonnxruntime_providers_cuda.so` to `$ORIGIN` so cuDNN resolves from the same folder.
- Windows: `artifacts/Windows/cuda_<N>/` with `RuntimeLibrary.dll`, ORT, cuDNN and the MSVC runtime DLLs.

## Deploy the runtime

| Requirement | Notes |
|---|---|
| NVIDIA driver | Recent enough for the CUDA major version of the package |
| CUDA Toolkit libraries | **Not bundled** — must be installed on the target (Windows: on `PATH`) |
| cuDNN | Bundled in the package |

Copy the package contents as-is to one folder and load the library from there. GPU detection uses the CUDA *driver* API (`libcuda.so.1` / `nvcuda.dll`), loaded dynamically, so the library loads without a GPU and reports it in the log.

**Windows hosts:** load the DLL with `LoadLibraryEx(..., LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`; plain `LoadLibrary` does not search the DLL's own folder for its dependencies.

## Test

- C++ tests in [`../../tests/runtime/`](../../tests/runtime/) (`simple_test`, `inference_test`, `robustness_test`; build with `build-tests.sh`).
- Windows load/dependency diagnostic (`windows_load_test.c`): see [`../../tests/README.md`](../../tests/README.md#testing-the-ort-runtime-on-windows).
