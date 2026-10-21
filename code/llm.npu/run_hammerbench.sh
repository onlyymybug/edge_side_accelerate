#!/usr/bin/env bash
set -euo pipefail
LLM_NPU_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${LLM_NPU_CODE_ROOT}/config.env"
set +a

SPLIT="${HAMMERBENCH_SPLIT:-single-turn}"
case "${SPLIT}" in
  single-turn) TASK_DIR=hammerbench-single ;;
  multi-turn) TASK_DIR=hammerbench-multi ;;
  *) echo "HAMMERBENCH_SPLIT must be single-turn or multi-turn" >&2; exit 2 ;;
esac
OUTPUT_ROOT="${LLM_NPU_RESULTS_ROOT}/${DEVICE_LABEL}/${TASK_DIR}/${RESULT_MODEL_DIR_NAME}"

exec "${PYTHON_BIN}" "${LLM_NPU_CODE_ROOT}/run_hammerbench.py" \
  --dataset-root "${HAMMERBENCH_DATASET_ROOT}" \
  --language "${HAMMERBENCH_LANGUAGE:-en}" \
  --split "${SPLIT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${HAMMERBENCH_RUN_ID:-${LLM_NPU_CONFIGURATION_NAME}}" \
  --continue-on-error \
  "$@"
