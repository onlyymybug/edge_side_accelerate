#!/usr/bin/env bash
set -euo pipefail

LLM_NPU_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${LLM_NPU_CODE_ROOT}/config.env"
set +a

PAYLOAD_DIR="${LLM_NPU_LOCAL_BUNDLE}/payload"
if [[ ! -d "${PAYLOAD_DIR}" ]]; then
  echo "llm.npu payload is missing: ${PAYLOAD_DIR}" >&2
  exit 1
fi

ADB_ARGS=(-P "${ADB_SERVER_PORT}" -s "${ANDROID_SERIAL}")
"${ADB_BIN}" "${ADB_ARGS[@]}" get-state >/dev/null
"${ADB_BIN}" "${ADB_ARGS[@]}" shell "mkdir -p '${LLM_NPU_PHONE_DIR}'"

for local_file in "${PAYLOAD_DIR}"/*; do
  name="$(basename "${local_file}")"
  if [[ "${name}" == "${LLM_NPU_CONTEXT_FILE}" ]]; then
    local_size="$(wc -c < "${local_file}")"
    remote_size="$("${ADB_BIN}" "${ADB_ARGS[@]}" shell \
      "stat -c %s '${LLM_NPU_PHONE_DIR}/${name}' 2>/dev/null || true" | tr -d '\r')"
    if [[ "${local_size}" == "${remote_size}" ]]; then
      echo "Context already present (${local_size} bytes); skipping ${name}"
      continue
    fi
  fi
  "${ADB_BIN}" "${ADB_ARGS[@]}" push "${local_file}" "${LLM_NPU_PHONE_DIR}/"
done

"${ADB_BIN}" "${ADB_ARGS[@]}" shell \
  "chmod 755 '${LLM_NPU_PHONE_DIR}/mllm-qwen2-aot-runner' '${LLM_NPU_PHONE_DIR}/run_on_phone.sh'"

local_context_size="$(wc -c < "${PAYLOAD_DIR}/${LLM_NPU_CONTEXT_FILE}")"
remote_context_size="$("${ADB_BIN}" "${ADB_ARGS[@]}" shell \
  "stat -c %s '${LLM_NPU_PHONE_DIR}/${LLM_NPU_CONTEXT_FILE}'" | tr -d '\r')"
if [[ "${local_context_size}" != "${remote_context_size}" ]]; then
  echo "Context size mismatch: local=${local_context_size} remote=${remote_context_size}" >&2
  exit 1
fi

echo "llm.npu bundle deployed to ${DEVICE_LABEL} (${ANDROID_SERIAL}) at ${LLM_NPU_PHONE_DIR}"
