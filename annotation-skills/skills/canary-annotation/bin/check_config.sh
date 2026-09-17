#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

required_variables=(
  MODEL_ID OPENAI_BASE_URL NORMALIZED_CASES TARGET_ROOT
  PRIMARY_UNSAFE_ROOT PRIMARY_REFUSAL_ROOT ATTACK_CHAIN_ROOT
  EXECUTION_CHAIN_ROOT UNSAFE_MERGED_ROOT UNSAFE_SEMANTIC_ROOT
  UNSAFE_REPAIR_ROOT UNSAFE_REANNOTATION_ROOT UNSAFE_FINAL_ROOT
  UNSAFE_FINAL_DETERMINISTIC_ROOT UNSAFE_FINAL_SEMANTIC_ROOT
  SAFETY_REFUSAL_MERGED_ROOT SAFETY_REFUSAL_VALIDATION_ROOT
  SAFETY_REFUSAL_FINAL_ROOT SAFETY_REFUSAL_FINAL_VALIDATION_ROOT
)

for name in "${required_variables[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: Required config variable is empty: $name" >&2
    exit 2
  fi
done

require_api_key
case "${INPUT_FORMAT:-normalized}" in
  normalized) require_dir "$NORMALIZED_CASES" ;;
  raw)
    [[ -n "${RAW_INPUT:-}" && -e "$RAW_INPUT" ]] || {
      echo "ERROR: INPUT_FORMAT=raw requires an existing RAW_INPUT" >&2
      exit 2
    }
    ;;
  *) echo "ERROR: INPUT_FORMAT must be raw or normalized" >&2; exit 2 ;;
esac

required_scripts=(
  common/prepare_annotation_input.py
  common/normalize_components.py
  annotate_primary_targets.py
  annotate_primary_attribution_fixed_target.py
  annotation_attackchain.py
  annotation_executionchain.py
  merge_full_annotations_per_case.py
  merge_safety_refusal_annotations.py
  validate_unsafe_action_annotations_deterministic_v3.py
  validate_unsafe_action_annotations_semantic_v3_root_cause.py
  validate_safety_refusal_annotations_deterministic.py
  prepare_unsafe_action_repair_cases.py
  rerun_full_annotation_pipeline.sh
  merge_previous_pass_and_repaired.py
  merge_reclassified_safety_refusal.py
)

for name in "${required_scripts[@]}"; do
  require_file "$PIPELINE_DIR/$name"
done

echo "Configuration is usable."
echo "Config:           $CONFIG_FILE"
echo "Normalized cases: $NORMALIZED_CASES"
echo "Model:            $MODEL_ID"
echo "API key env:      $API_KEY_ENV_NAME"
