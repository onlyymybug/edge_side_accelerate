#!/usr/bin/env bash

set -euo pipefail

BITNET_CODE_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck source=config.env
source "${BITNET_CODE_ROOT}/config.env"
set +a

"${BITNET_CODE_ROOT}/scripts/05_run_gsm8k.sh" \
  --continue-on-error "$@"
"${BITNET_CODE_ROOT}/scripts/06_run_math500.sh" \
  --continue-on-error "$@"
