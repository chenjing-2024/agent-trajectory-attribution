#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SCRIPT="$SKILL_ROOT/scripts/common/prepare_annotation_input.py"
require_file "$SCRIPT"
case "${INPUT_FORMAT:-normalized}" in
  raw)
    [[ -n "${RAW_INPUT:-}" ]] || { echo "ERROR: INPUT_FORMAT=raw requires RAW_INPUT" >&2; exit 2; }
    run_step "Normalize raw input" python3 "$SCRIPT" \
      --input "$RAW_INPUT" --input_format raw --output_root "$NORMALIZED_ROOT"
    ;;
  normalized)
    run_step "Validate normalized input (preserve IDs)" python3 "$SCRIPT" \
      --input "${1:-$NORMALIZED_CASES}" --input_format normalized --check_only
    ;;
  *) echo "ERROR: INPUT_FORMAT must be raw or normalized" >&2; exit 2 ;;
esac
