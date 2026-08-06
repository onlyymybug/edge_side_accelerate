#!/usr/bin/env bash

set -euo pipefail

BITNET_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck source=config.env
source "${BITNET_CODE_ROOT}/config.env"
set +a

tag="${1:-comparison_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi

"${BITNET_CODE_ROOT}/scripts/02_run_w1.sh" \
  --run-id "${tag}_w1" --continue-on-error "$@"
"${BITNET_CODE_ROOT}/scripts/03_run_w2.sh" \
  --run-id "${tag}_w2" \
  --prompt-token-calibration-csv "${W2_PROMPT_TOKEN_CALIBRATION_CSV}" \
  --continue-on-error "$@"
"${BITNET_CODE_ROOT}/scripts/04_run_w3.sh" \
  --run-id "${tag}_w3" \
  --prompt-token-calibration-csv "${W3_PROMPT_TOKEN_CALIBRATION_CSV}" \
  --continue-on-error "$@"
