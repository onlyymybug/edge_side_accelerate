#!/usr/bin/env bash

set -euo pipefail

QWEN_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck source=config.env
source "${QWEN_CODE_ROOT}/config.env"
set +a

export PYTHONUNBUFFERED=1
export HAMMERBENCH_DATASET_ROOT="${HAMMERBENCH_DATASET_ROOT:-/home/lyyyy/HammerBench/full_dataset/data}"

exec "${PYTHON_BIN}" -u "${QWEN_CODE_ROOT}/run_hammerbench.py" "$@"
