#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_file "${BUNDLE_ROOT}/hybrid_llama_qnn.pte"
require_file "${BUNDLE_ROOT}/hybrid_llama_qnn_quant_attrs.txt"
require_file "${DATASET_PATH}"
require_file "${TOKENIZER_PATH}"
require_file "${PYTHON_BIN}"
require_device

"${PYTHON_BIN}" -c 'import tokenizers' >/dev/null

echo "Host artifacts:"
sha256sum \
  "${BUNDLE_ROOT}/hybrid_llama_qnn.pte" \
  "${BUNDLE_ROOT}/hybrid_llama_qnn_quant_attrs.txt" \
  "${TOKENIZER_PATH}"

echo
echo "Phone:"
adb_run shell \
  'printf "model="; getprop ro.product.model; printf " soc="; getprop ro.soc.model; printf " abi="; getprop ro.product.cpu.abi; printf " android="; getprop ro.build.version.release'

echo
echo "Existing runtime files under ${DEVICE_DIR}:"
for filename in "${required_phone_files[@]}"; do
  if ! adb_run shell "test -f '${DEVICE_DIR}/${filename}'"; then
    echo "Missing phone runtime file: ${DEVICE_DIR}/${filename}" >&2
    exit 1
  fi
  adb_run shell "ls -lh '${DEVICE_DIR}/${filename}'"
done

echo
echo "Phone free space:"
adb_run shell df -h /data/local/tmp

echo
echo "Dataset samples:"
wc -l "${DATASET_PATH}"

echo "Environment check passed."
