#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 path/to/model_int8_uint8.tflite" >&2
  exit 2
fi

input=$1
if [[ ! -f "$input" ]]; then
  echo "Model not found: $input" >&2
  exit 1
fi
if [[ "$input" != *.tflite ]]; then
  echo "Input must be a .tflite file: $input" >&2
  exit 1
fi

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required but was not found." >&2
  exit 1
}

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not ready. Start Docker Desktop (or the Docker daemon) and retry." >&2
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
input_dir=$(cd "$(dirname "$input")" && pwd)
input_name=$(basename "$input")
default_image=edgetpu_compiler:16.0.384591198
image=${AICAM_EDGETPU_IMAGE:-$default_image}
platform=${AICAM_DOCKER_PLATFORM:-linux/amd64}
# Keep console output separate from the compiler's own *_edgetpu.log report.
log_path="$input_dir/${input_name%.tflite}_edgetpu.compile.log"

if ! docker image inspect "$image" >/dev/null 2>&1; then
  if [[ "$image" != "$default_image" ]]; then
    echo "Compiler image not found: $image. Load it before retrying." >&2
    exit 1
  fi
  archive="$script_dir/edgetpu-compiler-16.0.384591198-linux-amd64.tar"
  if [[ ! -f "$archive" ]]; then
    echo "Compiler archive not found. From the repository root, run:" >&2
    echo "  dvc pull aicam_laptop/tools/edge_tpu_compiler/$(basename "$archive").dvc" >&2
    exit 1
  fi
  # The Lenovo archive contains edgetpu_compiler:latest; retain a versioned tag.
  docker image load -i "$archive"
  docker image tag edgetpu_compiler:latest "$default_image"
fi

docker run --rm --pull=never --platform "$platform" \
  -v "$input_dir:/work" \
  "$image" \
  edgetpu_compiler -s -o /work "/work/$input_name" 2>&1 | tee "$log_path"
