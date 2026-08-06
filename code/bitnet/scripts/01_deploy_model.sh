#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_device
require_file "${BUNDLE_ROOT}/hybrid_llama_qnn.pte"
require_file "${BUNDLE_ROOT}/hybrid_llama_qnn_quant_attrs.txt"

local_model="${BUNDLE_ROOT}/hybrid_llama_qnn.pte"
remote_model="${DEVICE_DIR}/${REMOTE_MODEL_NAME}"
remote_partial="${remote_model}.partial"
local_hash="$(sha256sum "${local_model}" | awk '{print $1}')"

adb_run shell "mkdir -p '${DEVICE_DIR}'"

remote_hash=""
if adb_run shell "test -f '${remote_model}'"; then
  remote_hash="$(adb_run shell "sha256sum '${remote_model}'" | awk '{print $1}')"
fi

if [[ "${remote_hash}" == "${local_hash}" ]]; then
  echo "Model already deployed with matching SHA-256: ${remote_model}"
else
  echo "Pushing Hybrid-1024 PTE to temporary path..."
  adb_run push "${local_model}" "${remote_partial}"
  pushed_hash="$(adb_run shell "sha256sum '${remote_partial}'" | awk '{print $1}')"
  if [[ "${pushed_hash}" != "${local_hash}" ]]; then
    echo "SHA-256 mismatch after adb push." >&2
    echo "host=${local_hash}" >&2
    echo "phone=${pushed_hash}" >&2
    exit 1
  fi
  adb_run shell "mv '${remote_partial}' '${remote_model}'"
  echo "Model deployed: ${remote_model}"
fi

adb_run push \
  "${BUNDLE_ROOT}/hybrid_llama_qnn_quant_attrs.txt" \
  "${DEVICE_DIR}/${REMOTE_QUANT_ATTRS_NAME}"

echo
adb_run shell "ls -lh '${remote_model}' '${DEVICE_DIR}/${REMOTE_QUANT_ATTRS_NAME}'"
echo "SHA-256: ${local_hash}"
