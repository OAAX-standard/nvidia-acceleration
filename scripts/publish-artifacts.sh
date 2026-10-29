#!/usr/bin/env bash
# Upload one set of CI build outputs to the OAAX bucket. Run from the repository root.
#
#   scripts/publish-artifacts.sh release <version> <set>   # on a v* tag
#   scripts/publish-artifacts.sh branch  <branch>  <set>   # on a pull request, for testing
#   scripts/publish-artifacts.sh dry     <name>    <set>   # only check the files and print the keys
#
# <set> is what one CI job builds: ort-runtime, ort-runtime-windows, ort-toolchain, trt-runtime, trt-toolchain
# or trt-wheel. S3_ACCESS_KEY, S3_SECRET_KEY and S3_ENDPOINT_URL must be set for release and branch.
#
# release: keys under <version>/ are written once (if any exists nothing is uploaded: a published version is
#          immutable, oaax-manager pins its sha256); latest/ is copied from them afterwards.
# branch:  keys under <branch>/ are overwritten on every run, latest/ is never touched, and a name that could be a
#          release (latest, or starting with a digit or v<digit>) is refused so a branch can never replace one.
# Every expected file must exist and no other archive may sit next to them, so a build variant that silently
# disappeared (or a new one that is not listed here) fails the job instead of shipping a partial release.
set -euo pipefail

usage="usage: publish-artifacts.sh release|branch|dry <name> ort-runtime|ort-runtime-windows|ort-toolchain|trt-runtime|trt-toolchain|trt-wheel"
mode=${1:?$usage}
name=${2:?$usage}
set_name=${3:?$usage}

case $mode in
  release | dry) ;;
  branch)
    if [[ $name =~ ^(latest|v?[0-9]) ]]; then
      echo "branch name '$name' looks like a release, rename the branch to upload its artifacts" >&2
      exit 1
    fi
    ;;
  *) echo "unknown mode: $mode" >&2; exit 1 ;;
esac

# local file -> key, with @ standing for <version>/latest/<branch>
declare -A files=()
scan_dir="" scan_glob="" scan_skip="-"

ort_dir=onnxruntime/runtime-library/artifacts
trt_dir=tensorrt/runtime-library/build/NVIDIA_TENSORRT
case $set_name in
  ort-runtime)
    scan_dir=$ort_dir scan_glob='runtime-library-*-cuda_*.tar.gz' scan_skip='*Windows*'  # built by another job
    for cuda in 11 12 13; do
      files["$ort_dir/runtime-library-X86_64-cuda_$cuda.tar.gz"]="runtimes/@/NVIDIA_ORT/x86_64/Ubuntu/library-cuda_$cuda.tar.gz"
      files["$ort_dir/runtime-library-AARCH64-cuda_$cuda.tar.gz"]="runtimes/@/NVIDIA_ORT/aarch64-jetson/Ubuntu/library-cuda_$cuda.tar.gz"
    done
    files["$ort_dir/runtime-library-AARCH64_SBSA-cuda_13.tar.gz"]="runtimes/@/NVIDIA_ORT/aarch64-sbsa/Ubuntu/library-cuda_13.tar.gz"
    ;;
  ort-runtime-windows)
    scan_dir=$ort_dir scan_glob='runtime-library-*-Windows-cuda_*.tar.gz'
    for cuda in 11 12 13; do
      files["$ort_dir/runtime-library-X86_64-Windows-cuda_$cuda.tar.gz"]="runtimes/@/NVIDIA_ORT/x86_64/Windows/library-cuda_$cuda.tar.gz"
    done
    ;;
  ort-toolchain)
    files["onnxruntime/conversion-toolchain/artifacts/oaax-nvidia-toolchain.tar"]="conversion-toolchain/@/NVIDIA/oaax-nvidia-onnxruntime-toolchain.tar"
    ;;
  trt-runtime)
    # keep in sync with MATRIX in tensorrt/runtime-library/build-runtimes.sh: <arch>:<ubuntu>:<trt>:<cuda>
    scan_dir=$trt_dir scan_glob='library-trt*-cuda_*.tar.gz'
    for v in x86_64:22.04:10:12 x86_64:22.04:10:13 aarch64-jetson:22.04:10:12 aarch64-sbsa:24.04:10:12 \
      aarch64-sbsa:24.04:10:13 x86_64:22.04:11:12 x86_64:22.04:11:13 aarch64-sbsa:24.04:11:13; do
      IFS=: read -r arch ubuntu trt cuda <<<"$v"
      rel="$arch/Ubuntu/$ubuntu/library-trt$trt-cuda_$cuda.tar.gz"
      files["$trt_dir/$rel"]="runtimes/@/NVIDIA_TRT/$rel"
    done
    ;;
  trt-toolchain)
    files["tensorrt/conversion-toolchain/artifacts/oaax-nvidia-tensorrt-toolchain.tar"]="conversion-toolchain/@/NVIDIA/oaax-nvidia-tensorrt-toolchain.tar"
    ;;
  trt-wheel)
    shopt -s nullglob
    wheels=(tensorrt/conversion-toolchain/artifacts/*.whl)
    [ ${#wheels[@]} -eq 1 ] || { echo "expected exactly one wheel, found ${#wheels[@]}" >&2; exit 1; }
    files["${wheels[0]}"]="conversion-toolchain/@/NVIDIA/$(basename "${wheels[0]}")"
    ;;
  *) echo "unknown set: $set_name" >&2; exit 1 ;;
esac

for f in "${!files[@]}"; do
  [ -s "$f" ] || { echo "missing or empty artifact: $f" >&2; exit 1; }
done
if [ -n "$scan_dir" ]; then
  while IFS= read -r f; do
    [[ -v "files[$f]" ]] || { echo "unexpected artifact $f: add it to scripts/publish-artifacts.sh" >&2; exit 1; }
  done < <(find "$scan_dir" -name "$scan_glob" ! -name "$scan_skip" -type f | sort)
fi

if [ "$mode" = dry ]; then
  for f in "${!files[@]}"; do
    echo "would upload $f -> s3://oaax/${files[$f]//@/$name}"
  done | sort
  exit 0
fi

s3() {
  s3cmd --access_key="$S3_ACCESS_KEY" --secret_key="$S3_SECRET_KEY" \
    --host="$S3_ENDPOINT_URL" --host-bucket="%(bucket)s.$S3_ENDPOINT_URL" --bucket-location=us-east-1 --ssl "$@"
}

# Transient failures happen on the large uploads; a failed put leaves no object behind, so a retry is safe.
s3_retry() {
  local attempt
  for attempt in 1 2 3; do
    s3 "$@" && return 0
    echo "s3cmd $1 failed (attempt $attempt/3), retrying..." >&2
    sleep $((10 * attempt))
  done
  echo "s3cmd $* failed after 3 attempts" >&2
  return 1
}

if [ "$mode" = release ]; then
  # s3cmd info: 0 = exists, 12 = not found, anything else = a real error (credentials, network)
  for f in "${!files[@]}"; do
    url="s3://oaax/${files[$f]//@/$name}"
    rc=0
    s3 info "$url" >/dev/null 2>&1 || rc=$?
    case $rc in
      0) echo "$url already exists: versions are immutable, bump VERSION for a new release" >&2; exit 1 ;;
      12) ;;
      *) echo "could not check $url (s3cmd exit $rc)" >&2; exit 1 ;;
    esac
  done
fi

for f in "${!files[@]}"; do
  s3_retry put "$f" "s3://oaax/${files[$f]//@/$name}"
done
if [ "$mode" = release ]; then
  # server-side copy, after every versioned key is in place, so latest/ never points to a partial release
  for f in "${!files[@]}"; do
    s3_retry cp "s3://oaax/${files[$f]//@/$name}" "s3://oaax/${files[$f]//@/latest}"
  done
fi
