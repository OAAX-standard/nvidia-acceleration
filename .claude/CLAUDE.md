# nvidia-acceleration — OAAX Runtime for NVIDIA GPUs

## What This Repo Is

NVIDIA GPU implementation of the [OAAX standard](https://github.com/OAAX-standard/OAAX) with two backends, each in its own top-level folder:

- **`tensorrt/`** — Native TensorRT inference (no ONNX Runtime); compiles ONNX to `.trt` engine on the deployment machine.
- **`onnxruntime/`** — ONNX Runtime + CUDA Execution Provider; uses pre-built ORT libraries.

Each backend has two deliverables:
1. **`conversion-toolchain/`** — Docker image that prepares the model for inference.
2. **`runtime-library/`** — C++ shared library (`libRuntimeLibrary.so`) implementing the OAAX C API.

---

## Repository Structure

```
nvidia-acceleration/
├── tensorrt/
│   ├── conversion-toolchain/        # Dockerfile (nvcr.io tensorrt base), build-toolchain.sh, build-wheel.sh,
│   │                                # conversion_toolchain/{main,utils,logger}.py, scripts/convert.sh
│   └── runtime-library/             # CMakeLists.txt, build-runtimes.sh (matrix PLATFORM:CUDA:TRT),
│                                    # src/runtime_{core,utils}.cpp, include/, test/test_runtime.c
├── onnxruntime/
│   ├── conversion-toolchain/        # python:3.11-slim + onnxsim; build-toolchain.sh, build-wheel.sh
│   └── runtime-library/             # CMakeLists.txt, build-runtimes.{sh,bat}, fix-provider-rpath.sh,
│                                    # src/ (runtime_exports.{def,version} limit exported symbols),
│                                    # deps/ (pre-built ORT per platform/CUDA, cpuinfo, re2, windows-runtime DLLs, submodules)
├── scripts/                         # setup-env.sh (toolchains, TRT, CUDA), download-cuda.sh, download-cudnn.sh,
│                                    # unpack-trt.sh, build-in-docker.sh
├── tests/                           # stage1 (conversion), stage2 (benchmark), runtime/ C++ tests (incl. Windows)
├── .github/workflows/               # build-{runtime,toolchain}.yml (TRT), build-onnxruntime-{runtime,toolchain}.yml (ORT),
│                                    # delete-temporary-artifacts.yml (cleans a PR's S3 keys on close, never a release-looking name),
│                                    # ../actions/publish (upload step shared by the build jobs)
└── VERSION                          # semver, read by build scripts and test image tags
```

---

## How the Runtime Works

### Inference Pipeline

```
runtime_enqueue_input() ─► input_queue ─► inference slots ─► output_queue ─► runtime_retrieve_output()
                          (ConcurrentQueue)  (TRT enqueueV3,    (ConcurrentQueue)
                                              one CUDA stream per slot)
```

- `runtime_enqueue_input(model_id, Tensors*)` enqueues non-blocking; `max_concurrent` config key sets the number of concurrent slots per model
- Each slot copies inputs to GPU, runs `context->enqueueV3(stream)`, copies outputs to host
- `runtime_retrieve_output(&model_id, &out, timeout_ms)` polls the output queue (1ms sleep) and returns `RUNTIME_STATUS_NO_OUTPUT_AVAILABLE` on timeout
- Multiple models per runtime instance (`runtime_load_models`); default CUDA device 0

### Execution Engine

Direct TensorRT deserialization — no ONNX Runtime:
```cpp
nvinfer1::IRuntime* trt_runtime = nvinfer1::createInferRuntime(logger);
engine = trt_runtime->deserializeCudaEngine(engine_data, size);
context = engine->createExecutionContext();
for (int i = 0; i < engine->getNbIOTensors(); i++)
    cudaMalloc(&gpu_bufs[i], tensor_byte_size(engine, i));
```

### Memory Model

- Input `tensors_struct*` is owned and freed by the runtime after inference
- Output `tensors_struct*` is `malloc`'d by the runtime; caller must free it
- All data in `tensors_struct.data[]` is host memory (`cudaMemcpy` to/from GPU internally)

---

## Conversion Toolchain

- Runs on the **deployment machine** — TensorRT targets the GPU that is physically present
- Input: zip archive with `model.onnx` + `config.json` (`gpu_architecture`, `precision`, `workspace_gb`)
- Processing: `trtexec --onnx=model.onnx --saveEngine=model.trt [--fp16|--int8] --skipInference`
- Output: `model.trt` + `logs.json`
- For int8: calibration images in `calib/` + `preprocessing` config (mean/std)
- Engine is **not** portable across GPU architectures or CPU platforms (x86_64 ≠ aarch64)

---

## Build System

### Platforms

| PLATFORM | Target | Toolchain |
|---|---|---|
| `X86_64` | Linux x86_64 (`x86-64`) | `.deps/toolchains/x86-64-v2--glibc--stable-2022.08-1` |
| `AARCH64_JETSON` | Jetson / JetPack (`armv8-a`) → `aarch64-jetson` | `.deps/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu` |
| `AARCH64_SBSA` | DGX Spark / Grace (`armv8-a`) → `aarch64-sbsa` | `.deps/toolchains/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu` |

### External Dependencies (not in repo)

- `TRT_DIR`: defaults to `.deps/tensorrt/trt<V>/cuda-<N>/<platform>/` (set by `setup-env.sh`)
- `CUDA_DIR`: defaults to `.deps/cuda/<N>/<platform>/` (set by `setup-env.sh`)

### Build Output

`tensorrt/runtime-library/build/NVIDIA_TENSORRT/<arch>/Ubuntu/<ubuntu_version>/library-trt<V>-cuda_<N>.tar.gz` — archive of `bin/` containing `libRuntimeLibrary.so` + bundled TRT + CUDA libs (TRT 10 and 11 variants)

Where `<arch>` is `x86_64`, `aarch64-jetson`, or `aarch64-sbsa`.

### CI Artifacts (S3)

- Runtime: `s3://oaax/runtimes/<version>/NVIDIA_TRT/...` (TRT) and `.../NVIDIA_ORT/...` (ORT), same `<arch>/Ubuntu/<ubuntu_version>/` layout as the build output
- Toolchain: `s3://oaax/conversion-toolchain/<version>/NVIDIA/oaax-nvidia-{tensorrt,onnxruntime}-toolchain.tar` (+ the TRT conversion wheel)
- Uploads go through `scripts/publish-artifacts.sh <release|branch|dry> <name> <set>`, called by `.github/actions/publish` at the end of each build job:
  - `v*` tag (must equal `VERSION`, commit on `main`) = release: `<VERSION>/` is **write once** (refuses if any key exists) and `latest/` is copied from it
  - pull request with secrets = `<branch>/`, overwritten each run, never `latest/`; names that look like a release (`latest`, leading digit, `v<digit>`) are refused
  - `main` push, fork PR = dry run (files checked, keys printed, nothing uploaded)
- The Windows ORT runtime is built on a GitHub-hosted runner and handed to a Linux job (`publish-runtime-windows-x86_64`) as an Actions artifact: `s3cmd` on the Windows runner did not finish the ~3 GB upload within the 60 min job timeout.
- Each set lists its expected files (the build matrix is repeated in the script): a missing or unlisted archive fails the job. Adding a build variant means adding it to the script.
- The four build workflows run independently on a tag. If one fails midway through a release upload, its keys are partly in the bucket and the retry refuses: delete that set's partial `<VERSION>/` keys (maintainer, with a backup of the sha256 list) and re-run the job.
- Release: bump `VERSION` and `CHANGELOG.md` on `main`, then push the tag `vX.Y.Z` on that commit. Update oaax-manager's `index.json` afterwards.

---

## Known Limitations

1. **Conversion runs on the deployment machine** — TensorRT requires the target GPU to be physically present to compile kernels; no cross-compilation of engines.
2. **Engine not portable** — a `.trt` engine is tied to the GPU architecture and CPU platform it was built on; must reconvert for each target.
3. **Fixed batch size** — dynamic shapes not supported in v1; batch size is fixed at 1.
4. **Single GPU only** — always the default CUDA device (0).
5. **Polling latency** — slots and `runtime_retrieve_output` poll with a 1ms sleep.
6. **Error reporting is a single string** — `runtime_get_error()` returns only the last error message; no structured error state.
7. **Output always on CPU** — output is always `cudaMemcpy`'d to host; no GPU pointer exposure.
