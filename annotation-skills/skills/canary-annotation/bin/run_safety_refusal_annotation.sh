#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_api_key
require_dir "$NORMALIZED_CASES"
require_dir "$TARGET_ROOT/by_target_type/safety_refusal"
require_file "$PIPELINE_DIR/annotate_primary_attribution_fixed_target.py"
require_file "$PIPELINE_DIR/merge_safety_refusal_annotations.py"
require_file "$PIPELINE_DIR/validate_safety_refusal_annotations_deterministic.py"

run_step \
  "Annotate safety-refusal primary root causes" \
  python \
  "$PIPELINE_DIR/annotate_primary_attribution_fixed_target.py" \
  --input_dir "$NORMALIZED_CASES" \
  --target_dir "$TARGET_ROOT/by_target_type/safety_refusal" \
  --output_root "$PRIMARY_REFUSAL_ROOT" \
  --model "$MODEL_ID" \
  --api_key_env "$API_KEY_ENV_NAME" \
  --base_url "$OPENAI_BASE_URL" \
  --max_tokens "$PRIMARY_MAX_TOKENS" \
  --timeout "$TIMEOUT" \
  --retries "$RETRIES" \
  --sleep_seconds "$SLEEP_SECONDS"

require_dir "$PRIMARY_REFUSAL_ROOT/by_target_type/safety_refusal"

run_step \
  "Merge safety-refusal target and primary annotations" \
  python \
  "$PIPELINE_DIR/merge_safety_refusal_annotations.py" \
  --target_dir "$TARGET_ROOT/by_target_type/safety_refusal" \
  --primary_dir "$PRIMARY_REFUSAL_ROOT/by_target_type/safety_refusal" \
  --output_dir "$SAFETY_REFUSAL_MERGED_ROOT/cases" \
  --force

run_step \
  "Validate safety-refusal annotations" \
  python \
  "$PIPELINE_DIR/validate_safety_refusal_annotations_deterministic.py" \
  --annotations_dir "$SAFETY_REFUSAL_MERGED_ROOT/cases" \
  --normalized_dir "$NORMALIZED_CASES" \
  --output_root "$SAFETY_REFUSAL_VALIDATION_ROOT" \
  --force \
  --verbose

run_step "Assess final refusal quality" \
  python3 "$BIN_DIR/check_final_quality.py" \
  --kind refusal \
  --deterministic "$SAFETY_REFUSAL_VALIDATION_ROOT/deterministic_validation_report.json" \
  --output "$SAFETY_REFUSAL_VALIDATION_ROOT/final_quality.json"
