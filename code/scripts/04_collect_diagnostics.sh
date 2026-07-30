#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_device

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT="${BUNDLE_ROOT}/results/diagnostics_${RUN_ID}.txt"

{
  echo "=== adb devices ==="
  adb_run devices -l
  echo "=== device properties ==="
  adb_run shell 'getprop ro.product.model; getprop ro.soc.model; getprop ro.product.cpu.abi; getprop ro.build.version.release; getprop ro.build.version.sdk'
  echo "=== kernel ==="
  adb_run shell uname -a
  echo "=== free space ==="
  adb_run shell df -h /data/local/tmp
  echo "=== deployed files ==="
  adb_run shell "ls -l $(shell_quote "${DEVICE_DIR}")"
  echo "=== output files ==="
  adb_run shell "ls -l $(shell_quote "${DEVICE_DIR}/outputs") 2>/dev/null || true"
} 2>&1 | tee "${OUT}"

echo "Diagnostics saved in: ${OUT}"
