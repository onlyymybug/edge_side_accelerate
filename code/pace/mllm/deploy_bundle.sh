#!/usr/bin/env bash
set -euo pipefail

MLLM_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${MLLM_CODE_ROOT}/config.env"
set +a

BUNDLE_DIR="${MLLM_LOCAL_BUNDLE}"
required_local=(
  "${MLLM_RUNNER_FILE}"
  "${MLLM_CONTEXT_FILE}"
  "${MLLM_REMOTE_TOKENIZER_FILE}"
  "${MLLM_MODEL_CONFIG_FILE}"
  "lib/libMllmQNNBackend.so"
  "lib/libMllmRT.so"
  "lib/libQnnHtp.so"
  "dsp/libQnnHtpV79Skel.so"
  "dsp/libQnnLLaMAPackage.so"
  "SHA256SUMS"
)
for relative in "${required_local[@]}"; do
  if [[ ! -f "${BUNDLE_DIR}/${relative}" ]]; then
    echo "mllm bundle file is missing: ${BUNDLE_DIR}/${relative}" >&2
    exit 1
  fi
done

(
  cd "${BUNDLE_DIR}"
  sha256sum -c SHA256SUMS
)

ADB_ARGS=(-P "${ADB_SERVER_PORT}" -s "${ANDROID_SERIAL}")
"${ADB_BIN}" "${ADB_ARGS[@]}" get-state >/dev/null
"${ADB_BIN}" "${ADB_ARGS[@]}" shell \
  "mkdir -p '${MLLM_PHONE_DIR}/bin' '${MLLM_PHONE_DIR}/lib' '${MLLM_PHONE_DIR}/dsp' '${MLLM_PHONE_DIR}/model'"

for directory in bin lib dsp; do
  "${ADB_BIN}" "${ADB_ARGS[@]}" push \
    "${BUNDLE_DIR}/${directory}/." "${MLLM_PHONE_DIR}/${directory}/"
done

for name in config.json context-info.json tokenizer.json; do
  "${ADB_BIN}" "${ADB_ARGS[@]}" push \
    "${BUNDLE_DIR}/model/${name}" "${MLLM_PHONE_DIR}/model/${name}"
done
for name in README.md SHA256SUMS run_local.sh; do
  if [[ -f "${BUNDLE_DIR}/${name}" ]]; then
    "${ADB_BIN}" "${ADB_ARGS[@]}" push \
      "${BUNDLE_DIR}/${name}" "${MLLM_PHONE_DIR}/${name}"
  fi
done

local_context="${BUNDLE_DIR}/${MLLM_CONTEXT_FILE}"
remote_context="${MLLM_PHONE_DIR}/${MLLM_CONTEXT_FILE}"
local_size="$(wc -c < "${local_context}")"
local_sha="$(sha256sum "${local_context}" | awk '{print $1}')"
remote_size="$("${ADB_BIN}" "${ADB_ARGS[@]}" shell \
  "stat -c %s '${remote_context}' 2>/dev/null || true" | tr -d '\r')"
remote_sha=""
if [[ "${local_size}" == "${remote_size}" ]]; then
  remote_sha="$("${ADB_BIN}" "${ADB_ARGS[@]}" shell \
    "sha256sum '${remote_context}' 2>/dev/null | awk '{print \$1}'" | tr -d '\r')"
fi
if [[ "${local_sha}" == "${remote_sha}" ]]; then
  echo "Context already verified (${local_sha}); skipping upload"
else
  "${ADB_BIN}" "${ADB_ARGS[@]}" push "${local_context}" "${remote_context}"
fi

"${ADB_BIN}" "${ADB_ARGS[@]}" shell \
  "chmod 0755 '${MLLM_PHONE_DIR}/${MLLM_RUNNER_FILE}' && cd '${MLLM_PHONE_DIR}' && sha256sum -c SHA256SUMS"

echo "mllm bundle deployed to ${DEVICE_LABEL} (${ANDROID_SERIAL}) at ${MLLM_PHONE_DIR}"
