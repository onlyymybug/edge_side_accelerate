#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=../config.env
source "${BUNDLE_ROOT}/config.env"

ADB_CMD=("${ADB_BIN}")
if [[ -n "${ANDROID_SERIAL}" ]]; then
  ADB_CMD+=("-s" "${ANDROID_SERIAL}")
fi

adb_run() {
  "${ADB_CMD[@]}" "$@"
}

require_host_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_adb() {
  if ! command -v "${ADB_BIN}" >/dev/null 2>&1; then
    echo "adb was not found. Set ADB_BIN in config.env or add adb to PATH." >&2
    exit 1
  fi
}

require_device() {
  require_adb
  local state
  state="$(adb_run get-state 2>/dev/null || true)"
  if [[ "${state}" != "device" ]]; then
    echo "No authorized Android device selected. Current adb state: ${state:-none}" >&2
    echo "Run: ${ADB_BIN} devices -l" >&2
    exit 1
  fi
}

shell_quote() {
  local value="$1"
  value=${value//\'/\'\\\'\'}
  printf "'%s'" "${value}"
}

required_bundle_files=(
  "${BUNDLE_ROOT}/model/kv_llama_qnn.pte"
  "${BUNDLE_ROOT}/model/tokenizer.json"
  "${BUNDLE_ROOT}/runtime/qnn_llama_runner"
  "${BUNDLE_ROOT}/runtime/libqnn_executorch_backend.so"
  "${BUNDLE_ROOT}/runtime/libc++_shared.so"
  "${BUNDLE_ROOT}/qnn/libQnnHtp.so"
  "${BUNDLE_ROOT}/qnn/libQnnHtpV79Stub.so"
  "${BUNDLE_ROOT}/qnn/libQnnHtpV79Skel.so"
  "${BUNDLE_ROOT}/qnn/libQnnHtpPrepare.so"
  "${BUNDLE_ROOT}/qnn/libQnnSystem.so"
  "${BUNDLE_ROOT}/qnn/libQnnModelDlc.so"
)
