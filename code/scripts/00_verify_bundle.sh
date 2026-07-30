#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${BUNDLE_ROOT}"

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "sha256sum is required to verify this bundle." >&2
  exit 1
fi

sha256sum --check SHA256SUMS
echo "Bundle verification passed."
