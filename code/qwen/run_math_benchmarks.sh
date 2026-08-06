#!/usr/bin/env bash

set -euo pipefail

QWEN_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck source=config.env
source "${QWEN_CODE_ROOT}/config.env"
set +a

exec "${PYTHON_BIN}" "${QWEN_CODE_ROOT}/run_pace_datasets.py" \
  --tasks GSM8K MATH500 "$@"
