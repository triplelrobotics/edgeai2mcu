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

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
input_dir=$(cd "$(dirname "$input")" && pwd)
input_name=$(basename "$input")
image=${AICAM_EDGETPU_IMAGE:-aicam-edgetpu-compiler}
platform=${AICAM_DOCKER_PLATFORM:-linux/amd64}
log_path="$input_dir/${input_name%.tflite}_edgetpu.log"

docker build --platform "$platform" -t "$image" "$script_dir"
docker run --rm --platform "$platform" \
  -v "$input_dir:/work" \
  "$image" \
  edgetpu_compiler -s -o /work "/work/$input_name" 2>&1 | tee "$log_path"
