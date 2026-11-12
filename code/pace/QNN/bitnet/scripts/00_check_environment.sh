#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_file "${BUNDLE_ROOT}/hybrid_llama_qnn.pte"
require_file "${BUNDLE_ROOT}/hybrid_llama_qnn_quant_attrs.txt"
require_file "${TOKENIZER_PATH}"
require_file "${QNN_RUNNER_LOCAL_PATH}"
require_file "${RUNTIME_BUNDLE_ROOT}/runtime/libqnn_executorch_backend.so"
require_file "${PYTHON_BIN}"
require_device

"${PYTHON_BIN}" -c 'import tokenizers' >/dev/null

echo "Host artifacts:"
sha256sum \
  "${BUNDLE_ROOT}/hybrid_llama_qnn.pte" \
  "${BUNDLE_ROOT}/hybrid_llama_qnn_quant_attrs.txt" \
  "${TOKENIZER_PATH}" \
  "${QNN_RUNNER_LOCAL_PATH}" \
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
for filename in "${QNN_RUNNER_NAME}" libqnn_executorch_backend.so; do
  if [[ "${filename}" == "${QNN_RUNNER_NAME}" ]]; then
    host_path="${QNN_RUNNER_LOCAL_PATH}"
  else
    host_path="${RUNTIME_BUNDLE_ROOT}/runtime/${filename}"
  fi
  host_hash="$(sha256sum "${host_path}" | awk '{print $1}')"
  phone_hash="$(phone_sha256 "${DEVICE_DIR}/${filename}")"
  if [[ "${host_hash}" != "${phone_hash}" ]]; then
    echo "Runtime SHA-256 mismatch: ${filename}" >&2
    echo "host=${host_hash}" >&2
    echo "phone=${phone_hash}" >&2
    exit 1
  fi
  echo "${filename}: ${host_hash}"
done

echo
echo "Checking raw-prompt runner flags:"
runner_help="$(adb_run shell "cd '${DEVICE_DIR}' && export LD_LIBRARY_PATH=\$PWD && export LD_PRELOAD=\$PWD/libc++_shared.so && './${QNN_RUNNER_NAME}' --help 2>&1" || true)"
grep -q -- '-raw_prompt' <<<"${runner_help}" || {
  echo "Deployed runner does not expose --raw_prompt" >&2
  exit 1
}
grep -q -- '-htp_performance_mode' <<<"${runner_help}" || {
  echo "Deployed runner does not expose --htp_performance_mode" >&2
  exit 1
}
echo "${QNN_RUNNER_NAME}: --raw_prompt and --htp_performance_mode available"

echo
echo "Phone free space:"
adb_run shell df -h /data/local/tmp

echo "Environment check passed."
