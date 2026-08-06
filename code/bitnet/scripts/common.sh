#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BITNET_CODE_ROOT="$(CDPATH='' cd -- "${SCRIPT_DIR}/.." && pwd)"

set -a
# shellcheck source=../config.env
source "${BITNET_CODE_ROOT}/config.env"
set +a

ADB_CMD=("${ADB_BIN}")
if [[ -n "${ANDROID_SERIAL}" ]]; then
  ADB_CMD+=("-s" "${ANDROID_SERIAL}")
fi

adb_run() {
  "${ADB_CMD[@]}" "$@"
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_device() {
  require_command "${ADB_BIN}"
  local state
  state="$(adb_run get-state 2>/dev/null | tr -d '\r' || true)"
  if [[ "${state}" != "device" ]]; then
    echo "No authorized Android device selected: ${ANDROID_SERIAL}" >&2
    echo "Current state: ${state:-none}" >&2
    adb_run devices -l || true
    exit 1
  fi
}

required_phone_files=(
  "qnn_llama_runner"
  "libqnn_executorch_backend.so"
  "libc++_shared.so"
  "tokenizer.json"
  "libQnnHtp.so"
  "libQnnHtpV79Stub.so"
  "libQnnHtpV79Skel.so"
  "libQnnHtpPrepare.so"
  "libQnnSystem.so"
  "libQnnModelDlc.so"
)
