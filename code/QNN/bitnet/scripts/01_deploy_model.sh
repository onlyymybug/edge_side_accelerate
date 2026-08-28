#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_device
require_file "${BUNDLE_ROOT}/hybrid_llama_qnn.pte"
require_file "${BUNDLE_ROOT}/hybrid_llama_qnn_quant_attrs.txt"
require_file "${QNN_RUNNER_LOCAL_PATH}"
require_file "${RUNTIME_BUNDLE_ROOT}/runtime/libqnn_executorch_backend.so"

local_model="${BUNDLE_ROOT}/hybrid_llama_qnn.pte"
remote_model="${DEVICE_DIR}/${REMOTE_MODEL_NAME}"
remote_partial="${remote_model}.partial"
local_hash="$(sha256sum "${local_model}" | awk '{print $1}')"

adb_run shell "mkdir -p '${DEVICE_DIR}'"

remote_hash=""
if adb_run shell "test -f '${remote_model}'"; then
  remote_hash="$(phone_sha256 "${remote_model}" || true)"
fi

if [[ "${remote_hash}" == "${local_hash}" ]]; then
  echo "Model already deployed with matching SHA-256: ${remote_model}"
else
  echo "Pushing Hybrid-1024 PTE to temporary path..."
  adb_run push "${local_model}" "${remote_partial}"
  pushed_hash="$(phone_sha256 "${remote_partial}")"
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

deploy_runtime_file() {
  local host_path="$1"
  local filename="$2"
  local mode="$3"
  local phone_path="${DEVICE_DIR}/${filename}"
  local partial_path="${phone_path}.partial"
  local host_runtime_hash phone_runtime_hash pushed_runtime_hash

  host_runtime_hash="$(sha256sum "${host_path}" | awk '{print $1}')"
  phone_runtime_hash="$(phone_sha256 "${phone_path}" 2>/dev/null || true)"
  if [[ "${phone_runtime_hash}" == "${host_runtime_hash}" ]]; then
    echo "Runtime already matches: ${phone_path}"
    return
  fi

  if adb_run shell 'pidof qnn_llama_runner qnn_llama_runner_raw_prompt qnn_llm_runner_new >/dev/null 2>&1'; then
    echo "Refusing to replace BitNet runtime while a runner process is active." >&2
    exit 1
  fi

  echo "Pushing ${filename} to temporary path..."
  adb_run push "${host_path}" "${partial_path}"
  pushed_runtime_hash="$(phone_sha256 "${partial_path}")"
  if [[ "${pushed_runtime_hash}" != "${host_runtime_hash}" ]]; then
    echo "Runtime SHA-256 mismatch after adb push: ${filename}" >&2
    echo "host=${host_runtime_hash}" >&2
    echo "phone=${pushed_runtime_hash}" >&2
    exit 1
  fi
  adb_run shell "chmod '${mode}' '${partial_path}' && mv '${partial_path}' '${phone_path}'"
  echo "Runtime deployed: ${phone_path} (${host_runtime_hash})"
}

deploy_runtime_file "${QNN_RUNNER_LOCAL_PATH}" "${QNN_RUNNER_NAME}" 755
deploy_runtime_file "${RUNTIME_BUNDLE_ROOT}/runtime/libqnn_executorch_backend.so" libqnn_executorch_backend.so 644

echo
adb_run shell "ls -lh '${remote_model}' '${DEVICE_DIR}/${REMOTE_QUANT_ATTRS_NAME}' '${QNN_RUNNER_REMOTE_PATH}' '${DEVICE_DIR}/libqnn_executorch_backend.so'"
echo "SHA-256: ${local_hash}"
