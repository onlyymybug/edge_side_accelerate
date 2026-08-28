#!/usr/bin/env bash
# Run all requested OnePlus Llama math combinations in sequence.
set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for budget in 256 512 no_budget; do
  echo "===== OnePlus Llama GSM8K: ${budget} ====="
  GSM8K_BUDGET="${budget}" "${SCRIPT_DIR}/oneplus_llama_gsm8k.sh" "$@"
done

for budget in 256 512 no_budget; do
  echo "===== OnePlus Llama MATH500: ${budget} ====="
  MATH500_BUDGET="${budget}" "${SCRIPT_DIR}/oneplus_llama_math500.sh" "$@"
done
