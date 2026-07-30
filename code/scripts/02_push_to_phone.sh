#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_device
for file in "${required_bundle_files[@]}"; do
  require_host_file "${file}"
done

echo "Creating ${DEVICE_DIR}/outputs"
adb_run shell "mkdir -p $(shell_quote "${DEVICE_DIR}/outputs")"

for file in "${required_bundle_files[@]}"; do
  echo "Pushing $(basename "${file}")"
  adb_run push "${file}" "${DEVICE_DIR}/"
done

adb_run shell "chmod 755 $(shell_quote "${DEVICE_DIR}/qnn_llama_runner")"

echo "Files on phone:"
adb_run shell "ls -lh $(shell_quote "${DEVICE_DIR}")"

echo "No T-MAN op package was included or pushed."
