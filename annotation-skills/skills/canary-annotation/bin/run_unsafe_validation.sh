#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_api_key
require_dir "$UNSAFE_MERGED_ROOT/cases"
require_dir "$NORMALIZED_CASES"

run_step \
  "Deterministic unsafe-action validation" \
  python \
  "$PIPELINE_DIR/validate_unsafe_action_annotations_deterministic_v3.py" \
  --annotations_dir "$UNSAFE_MERGED_ROOT/cases" \
  --normalized_dir "$NORMALIZED_CASES" \
  --output_root "$UNSAFE_DETERMINISTIC_ROOT" \
  --force \
  --verbose

run_step \
  "Semantic unsafe-action validation" \
  python \
  "$PIPELINE_DIR/validate_unsafe_action_annotations_semantic_v3_root_cause.py" \
  --annotations_dir "$UNSAFE_MERGED_ROOT/cases" \
  --normalized_dir "$NORMALIZED_CASES" \
  --output_root "$UNSAFE_SEMANTIC_ROOT" \
  --provider openai \
  --model "$MODEL_ID" \
  --base_url "$OPENAI_BASE_URL" \
  --api_key_env "$API_KEY_ENV_NAME" \
  --max_tokens "$MAX_TOKENS" \
  --verbose
