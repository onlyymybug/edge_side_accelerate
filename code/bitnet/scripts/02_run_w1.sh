#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_device
require_file "${DATASET_PATH}"
require_file "${TOKENIZER_PATH}"
require_file "${PYTHON_BIN}"

if ! adb_run shell "test -f '${DEVICE_DIR}/${REMOTE_MODEL_NAME}'"; then
  echo "Hybrid model is not deployed: ${DEVICE_DIR}/${REMOTE_MODEL_NAME}" >&2
  echo "Run scripts/01_deploy_model.sh first." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${BITNET_CODE_ROOT}/scripts/run_w1.py" "$@"
