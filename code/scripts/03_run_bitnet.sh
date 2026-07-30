#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_device

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOCAL_RESULT_DIR="${BUNDLE_ROOT}/results/${RUN_ID}"
mkdir -p "${LOCAL_RESULT_DIR}"

remote_command="
cd $(shell_quote "${DEVICE_DIR}") || exit 1
export LD_LIBRARY_PATH=\"\$PWD\"
export LD_PRELOAD=\"\$PWD/libc++_shared.so\"
export ADSP_LIBRARY_PATH=\"\$PWD\${ADSP_LIBRARY_PATH:+;\$ADSP_LIBRARY_PATH}\"
unset QNN_OP_PACKAGE_PATHS
mkdir -p outputs
./qnn_llama_runner \
  --model_path kv_llama_qnn.pte \
  --tokenizer_path tokenizer.json \
  --prompt $(shell_quote "${PROMPT}") \
  --system_prompt $(shell_quote "${SYSTEM_PROMPT}") \
  --seq_len ${SEQ_LEN} \
  --eval_mode ${EVAL_MODE} \
  --kv_updater ${KV_UPDATER} \
  --temperature ${TEMPERATURE} \
  --logits_scale ${LOGITS_SCALE} \
  --logits_offset ${LOGITS_OFFSET} \
  --num_iters ${NUM_ITERS} \
  --output_path outputs/output.txt \
  --performance_output_path outputs/inference_speed.txt
"

echo "Running BitNet. Host log: ${LOCAL_RESULT_DIR}/run.log"
set -o pipefail
adb_run shell "${remote_command}" 2>&1 | tee "${LOCAL_RESULT_DIR}/run.log"

adb_run pull "${DEVICE_DIR}/outputs/output.txt" "${LOCAL_RESULT_DIR}/output.txt"
adb_run pull "${DEVICE_DIR}/outputs/inference_speed.txt" "${LOCAL_RESULT_DIR}/inference_speed.txt"

echo
echo "Generated text:"
cat "${LOCAL_RESULT_DIR}/output.txt"
echo
echo "Tokens per second:"
cat "${LOCAL_RESULT_DIR}/inference_speed.txt"
echo
echo "Results saved in: ${LOCAL_RESULT_DIR}"
