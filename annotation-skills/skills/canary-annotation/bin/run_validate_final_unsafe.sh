#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_api_key
require_dir "$UNSAFE_FINAL_ROOT/cases"
require_dir "$NORMALIZED_CASES"

run_step \
  "Validate final unsafe annotations deterministically" \
  python \
  "$PIPELINE_DIR/validate_unsafe_action_annotations_deterministic_v3.py" \
  --annotations_dir "$UNSAFE_FINAL_ROOT/cases" \
  --normalized_dir "$NORMALIZED_CASES" \
  --output_root "$UNSAFE_FINAL_DETERMINISTIC_ROOT" \
  --force \
  --verbose

run_step \
  "Validate final unsafe annotations semantically" \
  python \
  "$PIPELINE_DIR/validate_unsafe_action_annotations_semantic_v3_root_cause.py" \
  --annotations_dir "$UNSAFE_FINAL_ROOT/cases" \
  --normalized_dir "$NORMALIZED_CASES" \
  --output_root "$UNSAFE_FINAL_SEMANTIC_ROOT" \
  --provider openai \
  --model "$MODEL_ID" \
  --base_url "$OPENAI_BASE_URL" \
  --api_key_env "$API_KEY_ENV_NAME" \
  --max_tokens "$MAX_TOKENS" \
  --max_retries "$RETRIES" \
  --request_timeout "$TIMEOUT" \
  --force \
  --verbose

run_step "Assess final unsafe quality" \
  python3 "$BIN_DIR/check_final_quality.py" \
  --kind unsafe \
  --deterministic "$UNSAFE_FINAL_DETERMINISTIC_ROOT/deterministic_validation_report.json" \
  --semantic "$UNSAFE_FINAL_SEMANTIC_ROOT/semantic_validation_report.json" \
  --output "$UNSAFE_FINAL_DETERMINISTIC_ROOT/final_quality.json"
