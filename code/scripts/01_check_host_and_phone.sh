#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

echo "Bundle: ${BUNDLE_ROOT}"
echo "Phone directory: ${DEVICE_DIR}"

for file in "${required_bundle_files[@]}"; do
  require_host_file "${file}"
done
echo "All required bundle files are present."

require_adb
adb_run version
adb_run devices -l
require_device

echo "Phone identity:"
adb_run shell 'printf "model="; getprop ro.product.model; printf "soc="; getprop ro.soc.model; printf "abi="; getprop ro.product.cpu.abi'

echo "Free space under /data/local/tmp:"
adb_run shell 'df -h /data/local/tmp'
