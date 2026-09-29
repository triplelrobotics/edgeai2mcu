#!/bin/sh

set -eu

aicam_root="${AICAM_ROOT:-/workspace/aicam_coral}"

mkdir -p \
  "$aicam_root/app" \
  "$aicam_root/var/captures/line_follow" \
  "$aicam_root/var/cache/coral/classify" \
  "$aicam_root/var/cache/coral/detect" \
  "$aicam_root/var/cache/coral/segment" \
  "$aicam_root/var/log" \
  "$aicam_root/var/run"

printf 'H618 directory layout initialized under %s\n' "$aicam_root"
