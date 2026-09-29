#!/bin/sh

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  printf 'Usage: %s USER@HOST [REPORT_PATH]\n' "$0" >&2
  exit 2
fi

device="$1"
timestamp="$(date '+%Y%m%d_%H%M%S')"
report_path="${2:-${TMPDIR:-/tmp}/h618_app_diff_${timestamp}.txt}"
remote_app="${AICAM_REMOTE_APP:-/workspace/aicam_coral/app}"
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
local_app="$script_dir/app"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/compare-h618-app.XXXXXX")"
local_manifest="$tmp_dir/mac.sha256"
device_manifest="$tmp_dir/device.sha256"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT HUP INT TERM

if [ ! -d "$local_app" ]; then
  printf 'Local app directory not found: %s\n' "$local_app" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$report_path")"

(
  cd "$local_app"
  find . -type f \
    ! -path '*/__pycache__/*' \
    ! -name '*.pyc' \
    ! -name '.DS_Store' \
    -print |
    LC_ALL=C sort |
    while IFS= read -r path; do
      hash="$(shasum -a 256 "$path" | awk '{print $1}')"
      printf '%s  %s\n' "$hash" "$path"
    done
) > "$local_manifest"

ssh "$device" "
  set -eu
  cd '$remote_app'
  find . -type f \\
    ! -path '*/__pycache__/*' \\
    ! -name '*.pyc' \\
    ! -name '.DS_Store' \\
    -print |
    LC_ALL=C sort |
    while IFS= read -r path; do
      hash=\$(sha256sum \"\$path\" | awk '{print \$1}')
      printf '%s  %s\\n' \"\$hash\" \"\$path\"
    done
" > "$device_manifest"

local_count="$(wc -l < "$local_manifest" | tr -d ' ')"
device_count="$(wc -l < "$device_manifest" | tr -d ' ')"

{
  printf 'Mac app:    %s\n' "$local_app"
  printf 'Device app: %s:%s\n' "$device" "$remote_app"
  printf 'Mac files:  %s\n' "$local_count"
  printf 'Device files: %s\n\n' "$device_count"

  if diff -u "$device_manifest" "$local_manifest"; then
    printf 'No file-content differences found.\n'
  else
    status="$?"
    if [ "$status" -ne 1 ]; then
      exit "$status"
    fi
  fi
} > "$report_path"

cat "$report_path"
printf '\nReport written to %s\n' "$report_path"
