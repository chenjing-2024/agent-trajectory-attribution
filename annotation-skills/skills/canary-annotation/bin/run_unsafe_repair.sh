#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_api_key
require_file "$UNSAFE_SEMANTIC_ROOT/all_cases.jsonl"

run_step "Build joint deterministic and semantic repair queue" \
  python3 "$PIPELINE_DIR/build_joint_repair_queue.py" \
  --targets "$TARGET_ROOT/by_target_type/unsafe_action" \
  --normalized "$NORMALIZED_CASES" --annotations "$UNSAFE_MERGED_ROOT/cases" \
  --deterministic "$UNSAFE_DETERMINISTIC_ROOT/all_cases.jsonl" \
  --semantic "$UNSAFE_SEMANTIC_ROOT/all_cases.jsonl" --output "$UNSAFE_REPAIR_ROOT"

run_step \
  "Prepare unsafe-action repair cases" \
  python \
  "$PIPELINE_DIR/prepare_unsafe_action_repair_cases.py" \
  --semantic_all_cases_jsonl "$UNSAFE_REPAIR_ROOT/joint_cases.jsonl" \
  --annotations_dir "$UNSAFE_MERGED_ROOT/cases" \
  --normalized_dir "$NORMALIZED_CASES" \
  --output_root "$UNSAFE_REPAIR_ROOT" \
  --force

if [[ "${ANNOTATION_DRY_RUN:-0}" != 1 ]] && python3 -c 'import json,sys; sys.exit(bool(json.load(open(sys.argv[1]))["executable"]))' "$UNSAFE_REPAIR_ROOT/queue_summary.json"; then
  mkdir -p "$UNSAFE_REANNOTATION_ROOT/merged/cases"
  echo "No executable repair cases. Final audit will check blocked cases."
  exit 0
fi

run_step \
  "Reannotate unsafe-action repair cases" \
  "$PIPELINE_DIR/rerun_full_annotation_pipeline.sh" \
  --input_dir "$UNSAFE_REPAIR_ROOT/normalized_cases" \
  --output_root "$UNSAFE_REANNOTATION_ROOT" \
  --model "$MODEL_ID" \
  --base_url "$OPENAI_BASE_URL" \
  --api_key_env "$API_KEY_ENV_NAME" \
  --max_tokens "$MAX_TOKENS" \
  --timeout "$TIMEOUT" \
  --retries "$RETRIES" \
  --sleep_seconds "$SLEEP_SECONDS" \
  --force
