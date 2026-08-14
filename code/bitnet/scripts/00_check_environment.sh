#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_file "${BUNDLE_ROOT}/hybrid_llama_qnn.pte"
require_file "${BUNDLE_ROOT}/hybrid_llama_qnn_quant_attrs.txt"
require_file "${DATASET_PATH}"
require_file "${TOKENIZER_PATH}"
require_file "${RUNTIME_BUNDLE_ROOT}/runtime/qnn_llama_runner"
require_file "${RUNTIME_BUNDLE_ROOT}/runtime/libqnn_executorch_backend.so"
require_file "${PYTHON_BIN}"
require_device

"${PYTHON_BIN}" -c 'import tokenizers' >/dev/null

echo "Host artifacts:"
sha256sum \
  "${BUNDLE_ROOT}/hybrid_llama_qnn.pte" \
  "${BUNDLE_ROOT}/hybrid_llama_qnn_quant_attrs.txt" \
  "${TOKENIZER_PATH}" \
  "${RUNTIME_BUNDLE_ROOT}/runtime/qnn_llama_runner" \
  "${RUNTIME_BUNDLE_ROOT}/runtime/libqnn_executorch_backend.so"

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
echo "Checking deployed BitNet runtime hashes:"
for filename in qnn_llama_runner libqnn_executorch_backend.so; do
  host_path="${RUNTIME_BUNDLE_ROOT}/runtime/${filename}"
  host_hash="$(sha256sum "${host_path}" | awk '{print $1}')"
  phone_hash="$(adb_run shell "sha256sum '${DEVICE_DIR}/${filename}'" | awk '{print $1}')"
  if [[ "${host_hash}" != "${phone_hash}" ]]; then
    echo "Runtime SHA-256 mismatch: ${filename}" >&2
    echo "host=${host_hash}" >&2
    echo "phone=${phone_hash}" >&2
    exit 1
  fi
  echo "${filename}: ${host_hash}"
done

echo
echo "Phone free space:"
adb_run shell df -h /data/local/tmp

echo
echo "Dataset samples:"
wc -l "${DATASET_PATH}"

echo "Environment check passed."
