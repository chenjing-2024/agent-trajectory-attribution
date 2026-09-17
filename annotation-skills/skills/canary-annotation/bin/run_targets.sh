#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_api_key
INPUT_DIR="${1:-$NORMALIZED_CASES}"
require_dir "$INPUT_DIR"
require_file "$PIPELINE_DIR/annotate_primary_targets.py"

run_step \
  "Annotate targets" \
  python \
  "$PIPELINE_DIR/annotate_primary_targets.py" \
  --input_dir "$INPUT_DIR" \
  --output_root "$TARGET_ROOT" \
  --model "$MODEL_ID" \
  --api_key_env "$API_KEY_ENV_NAME" \
  --base_url "$OPENAI_BASE_URL" \
  --max_tokens "$MAX_TOKENS" \
  --timeout "$TIMEOUT" \
  --retries "$RETRIES" \
  --sleep_seconds "$SLEEP_SECONDS"

