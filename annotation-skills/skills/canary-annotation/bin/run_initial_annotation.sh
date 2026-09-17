#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_api_key

if [[ $# -gt 0 && "${INPUT_FORMAT:-normalized}" == raw ]]; then
  echo "ERROR: use RAW_INPUT for raw input, not a positional input directory" >&2
  exit 2
fi

bash "$BIN_DIR/run_normalize.sh" "${1:-$NORMALIZED_CASES}"
INPUT_DIR="${1:-$NORMALIZED_CASES}"

require_dir "$INPUT_DIR"

require_file "$PIPELINE_DIR/annotate_primary_targets.py"
require_file "$PIPELINE_DIR/annotate_primary_attribution_fixed_target.py"
require_file "$PIPELINE_DIR/annotation_attackchain.py"
require_file "$PIPELINE_DIR/annotation_executionchain.py"
require_file "$PIPELINE_DIR/merge_full_annotations_per_case.py"

bash "$BIN_DIR/run_targets.sh" "$INPUT_DIR"
if [[ "${ANNOTATION_DRY_RUN:-0}" != 1 ]] && python3 -c 'from pathlib import Path; import sys; sys.exit(any(Path(sys.argv[1]).glob("*.json")))' "$TARGET_ROOT/by_target_type/unsafe_action"; then
  mkdir -p "$UNSAFE_MERGED_ROOT/cases"
  exit 0
fi

require_dir "$TARGET_ROOT/by_target_type/unsafe_action"

run_step \
  "Stage 2/5: Annotate unsafe-action primary root cause" \
  python \
  "$PIPELINE_DIR/annotate_primary_attribution_fixed_target.py" \
  --input_dir "$INPUT_DIR" \
  --target_dir "$TARGET_ROOT/by_target_type/unsafe_action" \
  --output_root "$PRIMARY_UNSAFE_ROOT" \
  --model "$MODEL_ID" \
  --api_key_env "$API_KEY_ENV_NAME" \
  --base_url "$OPENAI_BASE_URL" \
  --max_tokens "$PRIMARY_MAX_TOKENS" \
  --timeout "$TIMEOUT" \
  --retries "$RETRIES" \
  --sleep_seconds "$SLEEP_SECONDS"

require_dir "$PRIMARY_UNSAFE_ROOT/by_target_type/unsafe_action"

run_step \
  "Stage 3/5: Annotate attack chains" \
  python \
  "$PIPELINE_DIR/annotation_attackchain.py" \
  --input_dir "$INPUT_DIR" \
  --target_dir "$TARGET_ROOT/by_target_type/unsafe_action" \
  --primary_dir "$PRIMARY_UNSAFE_ROOT/by_target_type/unsafe_action" \
  --output_root "$ATTACK_CHAIN_ROOT" \
  --model "$MODEL_ID" \
  --api_key_env "$API_KEY_ENV_NAME" \
  --base_url "$OPENAI_BASE_URL" \
  --sleep_seconds "$SLEEP_SECONDS"

run_step \
  "Stage 4/5: Annotate execution chains" \
  python \
  "$PIPELINE_DIR/annotation_executionchain.py" \
  --input_root "$INPUT_DIR" \
  --primary_root "$PRIMARY_UNSAFE_ROOT" \
  --attack_chain_root "$ATTACK_CHAIN_ROOT" \
  --output_root "$EXECUTION_CHAIN_ROOT" \
  --model "$MODEL_ID" \
  --api_key_env "$API_KEY_ENV_NAME" \
  --base_url "$OPENAI_BASE_URL" \
  --max_retries "$RETRIES" \
  --sleep_seconds "$SLEEP_SECONDS"

require_dir "$ATTACK_CHAIN_ROOT/cases"
require_dir "$EXECUTION_CHAIN_ROOT/cases"

run_step \
  "Stage 5/5: Merge unsafe-action annotations" \
  python \
  "$PIPELINE_DIR/merge_full_annotations_per_case.py" \
  --target_dir "$TARGET_ROOT/by_target_type/unsafe_action" \
  --primary_dir "$PRIMARY_UNSAFE_ROOT/by_target_type/unsafe_action" \
  --attack_chain_dir "$ATTACK_CHAIN_ROOT/cases" \
  --execution_chain_dir "$EXECUTION_CHAIN_ROOT/cases" \
  --output_dir "$UNSAFE_MERGED_ROOT" \
  --complete_only

echo
if [[ "${ANNOTATION_DRY_RUN:-0}" == 1 ]]; then
  echo "Initial unsafe-action execution plan recorded; no annotations produced."
else
  echo "Initial unsafe-action annotation finished."
fi
echo "Output:"
echo "  $UNSAFE_MERGED_ROOT/cases"
