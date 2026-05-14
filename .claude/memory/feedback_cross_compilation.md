---
name: Deployment and packaging lessons
description: Hard-won lessons on cross-compiling the runtime, deploying to devices, and packaging the conversion toolchain
type: feedback
---

## Cross-compilation

Use toolchains at `.deps/toolchains/` (set up by `scripts/setup-env.sh`):
- X86_64: `x86-64-v2--glibc--stable-2022.08-1`, prefix `x86_64-linux-`
- AARCH64_JETSON: `gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu`, prefix `aarch64-none-linux-gnu-`
- AARCH64_SBSA: `arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu`, prefix `aarch64-none-linux-gnu-`

Use `-Wl,--allow-shlib-undefined` for AARCH64/AARCH64_SBSA — Jetson's `libnvinfer.so` depends on `libcudla.so` which is only present on-device.

**Why:** Avoids linker errors when cross-compiling against empty stub .so files for TRT/CUDA aarch64 libs that aren't available on the x86 build machine.

## x86-64 march selection

Always check the target CPU before picking a march variant (`grep -o 'avx[^ ]*\|sse[^ ]*' /proc/cpuinfo | sort -u`). Use `x86-64-v3` for modern x86 desktops/servers with AVX2; fall back to baseline only for unknown/old hardware.

**Why:** We initially deployed baseline `x86-64` to device-4 (i7-12700K, AVX2 capable). Correct choice is `x86-64-v3`. Performance difference is negligible when GPU dominates, but shipping the right binary matters for correctness.

## Deployment to aarch64 devices

- **rsync symlinks**: always use `--copy-links` when syncing directories that contain relative symlinks to paths outside the sync root.
- **glibc mismatch**: binaries from Ubuntu 24.04 (glibc 2.38) won't run on Ubuntu 22.04 (glibc 2.35) — build or install natively.
- **Device-to-device copy**: devices cannot SSH to each other by hostname — relay through the dev machine.

## Docker GPU access on SBSA servers

`nvidia-container-toolkit` is not in standard Ubuntu 24.04 arm64 apt repos (as of 2026-03). DGX Spark B200 had no Docker GPU access.

**Fix:** Bare trtexec approach — ship slim TRT toolkit (trtexec binary + SM-specific `libnvinfer_builder_resource_smXXX.so` only, ~1.3 GB vs 5 GB+), run with `LD_LIBRARY_PATH`.

## TRT version compatibility (Docker vs wheel)

Docker conversion pins TRT (guaranteed match with runtime). The Python wheel uses system TRT — mismatch means engine fails to deserialize.

**How to apply:** When bumping runtime TRT, update `_RUNTIME_TRT_VERSION` in `conversion-toolchain/conversion_toolchain/utils.py` and rebuild the wheel. The dependency checker warns automatically if system TRT differs.

## OAAX memory ownership

After `send_input()`, the runtime owns and frees the input `tensors_struct*`. Never call `deep_free_tensors_struct()` on it afterwards — causes double-free at shutdown.
