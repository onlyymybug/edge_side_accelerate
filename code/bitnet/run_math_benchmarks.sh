#!/usr/bin/env bash

set -euo pipefail

BITNET_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck source=config.env
source "${BITNET_CODE_ROOT}/config.env"
set +a

tag="${1:-bitnet}"
if [[ $# -gt 0 ]]; then
  shift
fi

"${BITNET_CODE_ROOT}/scripts/05_run_gsm8k.sh" \
  --run-id "${tag}_gsm8k" --continue-on-error "$@"
"${BITNET_CODE_ROOT}/scripts/06_run_math500.sh" \
  --run-id "${tag}_math500" --continue-on-error "$@"
