#!/usr/bin/env bash
set -Eeuo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'HELP'
Usage:
  annotation_skill.sh <command> [--dry_run --plan /path/to/new-plan.json]

Commands:
  initial
      Run initial unsafe-action annotation:
      target → primary → attack chain → execution chain → merge

  validate-unsafe
      Run deterministic and semantic unsafe-action validation.

  repair-unsafe
      Prepare repair cases and rerun their full annotations.

  finalize-unsafe
      Merge previous semantic-pass cases with repaired unsafe cases.

  finalize-refusal
      Add repair cases reclassified as safety_refusal to the
      existing safety-refusal collection.

  safety-refusal-all
      Annotate targets, then refusal primary causes, merge, and validate.

  safety-refusal
      Annotate safety-refusal primary causes, merge, and validate.

  validate-final-unsafe
      Validate the repaired and finalized unsafe-action collection.

  validate-final-refusal
      Validate the finalized safety-refusal collection.

  unsafe-all
      Run:
      initial → validate-unsafe → repair-unsafe → finalize-unsafe
      → validate-final-unsafe

  finalize-all
      Run:
      finalize-unsafe → validate-final-unsafe → finalize-refusal
      → validate-final-refusal

  check-config
      Validate paths, required scripts, model settings, and API-key env.

  show-config
      Print the active configuration file.

Environment:
  ANNOTATION_CONFIG
      Optional path to another annotation.env file.
HELP
}

COMMAND="${1:-}"
[[ $# -eq 0 ]] || shift
export ANNOTATION_DRY_RUN=0
export ANNOTATION_PLAN=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run|--dry-run) export ANNOTATION_DRY_RUN=1; shift ;;
    --plan)
      [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: --plan requires a path" >&2; exit 2; }
      export ANNOTATION_PLAN="$2"; shift 2 ;;
    --*) echo "ERROR: Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
if [[ ${#POSITIONAL[@]} -gt 0 && "$COMMAND" != initial ]]; then
  echo "ERROR: Unexpected arguments for $COMMAND" >&2
  exit 2
fi
if [[ "$ANNOTATION_DRY_RUN" == 1 ]]; then
  case "$COMMAND" in
    initial|validate-unsafe|repair-unsafe|finalize-unsafe|finalize-refusal|unsafe-all|safety-refusal|safety-refusal-all|validate-final-unsafe|validate-final-refusal|finalize-all) ;;
    *) echo "ERROR: dry run requires a workflow command" >&2; exit 2 ;;
  esac
  [[ -n "$ANNOTATION_PLAN" ]] || { echo "ERROR: --dry_run requires --plan PATH" >&2; exit 2; }
  source "$BIN_DIR/common.sh"
  INPUT_CHECK="$NORMALIZED_CASES"
  if [[ "${INPUT_FORMAT:-normalized}" == raw && ( "$COMMAND" == unsafe-all || "$COMMAND" == safety-refusal-all || "$COMMAND" == initial ) ]]; then
    INPUT_CHECK="${RAW_INPUT:-}"
  fi
  if [[ "$COMMAND" == initial && ${#POSITIONAL[@]} -gt 0 ]]; then INPUT_CHECK="${POSITIONAL[0]}"; fi
  [[ -e "$INPUT_CHECK" ]] || { echo "ERROR: Input directory not found: $INPUT_CHECK" >&2; exit 2; }
  python3 "$BIN_DIR/record_plan.py" init "$ANNOTATION_PLAN" "$COMMAND" "$INPUT_CHECK" "$MODEL_ID" "$CONFIG_FILE"
  trap 'code=$?; python3 "$BIN_DIR/record_plan.py" finish "$ANNOTATION_PLAN" "$code"; exit "$code"' EXIT
elif [[ -n "$ANNOTATION_PLAN" ]]; then
  echo "ERROR: --plan requires --dry_run" >&2
  exit 2
fi

# Full workflows and final validation expose the same per-case result contract.
if [[ "$ANNOTATION_DRY_RUN" != 1 ]]; then
  case "$COMMAND" in
    unsafe-all|safety-refusal-all|safety-refusal|validate-final-unsafe|validate-final-refusal)
      source "$BIN_DIR/common.sh"
      mkdir -p "$RESULTS_ROOT/skill_runs"
      export UNIFIED_OUTPUT_ROOT="$(mktemp -d "$RESULTS_ROOT/skill_runs/${COMMAND}.XXXXXX")"
      python3 "$BIN_DIR/workflow_output.py" begin "$UNIFIED_OUTPUT_ROOT" "$COMMAND"
      if [[ "$COMMAND" == unsafe-all || "$COMMAND" == safety-refusal-all ]]; then
        if [[ "${INPUT_FORMAT:-normalized}" == raw ]]; then
          export UNIFIED_RAW_INPUT="$UNIFIED_OUTPUT_ROOT/raw/cases"
          export UNIFIED_NORMALIZED_ROOT="$NORMALIZED_ROOT"
        else
          export UNIFIED_NORMALIZED_ROOT="$UNIFIED_OUTPUT_ROOT/normalized"
        fi
      fi
      trap 'code=$?; trap - EXIT; python3 "$BIN_DIR/workflow_output.py" finish "$UNIFIED_OUTPUT_ROOT" "$COMMAND" "$code"; exit $?' EXIT
      if python3 -c 'import json,sys; sys.exit(bool(json.load(open(sys.argv[1]))["cases"]))' "$UNIFIED_OUTPUT_ROOT/selected_cases.json"; then
        exit 0
      fi
      ;;
  esac
fi

# Keep the dispatcher alive so its EXIT trap finalizes the plan.
dispatch() { bash "$@"; }

case "$COMMAND" in
  initial)
    if [[ ${#POSITIONAL[@]} -gt 1 ]]; then echo "ERROR: initial accepts one input directory" >&2; exit 2; fi
    if [[ ${#POSITIONAL[@]} -eq 1 ]]; then
      dispatch "$BIN_DIR/run_initial_annotation.sh" "${POSITIONAL[0]}"
    else
      dispatch "$BIN_DIR/run_initial_annotation.sh"
    fi
    ;;

  validate-unsafe)
    dispatch "$BIN_DIR/run_unsafe_validation.sh"
    ;;

  repair-unsafe)
    dispatch "$BIN_DIR/run_unsafe_repair.sh"
    ;;

  finalize-unsafe)
    dispatch "$BIN_DIR/run_finalize_unsafe.sh"
    ;;

  finalize-refusal)
    dispatch "$BIN_DIR/run_finalize_safety_refusal.sh"
    ;;

  unsafe-all)
    "$BIN_DIR/run_initial_annotation.sh"
    if [[ "$ANNOTATION_DRY_RUN" != 1 ]] && python3 -c 'from pathlib import Path; import sys; sys.exit(any(Path(sys.argv[1]).glob("*.json")))' "$TARGET_ROOT/by_target_type/unsafe_action"; then
      exit 0
    fi
    "$BIN_DIR/run_unsafe_validation.sh"
    "$BIN_DIR/run_unsafe_repair.sh"
    "$BIN_DIR/run_finalize_unsafe.sh"
    "$BIN_DIR/run_validate_final_unsafe.sh"
    ;;

  safety-refusal-all)
    dispatch "$BIN_DIR/run_normalize.sh"
    dispatch "$BIN_DIR/run_targets.sh"
    if [[ "$ANNOTATION_DRY_RUN" != 1 ]] && python3 -c 'from pathlib import Path; import sys; sys.exit(any(Path(sys.argv[1]).glob("*.json")))' "$TARGET_ROOT/by_target_type/safety_refusal"; then
      exit 0
    fi
    dispatch "$BIN_DIR/run_safety_refusal_annotation.sh"
    ;;

  safety-refusal)
    dispatch "$BIN_DIR/run_safety_refusal_annotation.sh"
    ;;

  validate-final-unsafe)
    dispatch "$BIN_DIR/run_validate_final_unsafe.sh"
    ;;

  validate-final-refusal)
    dispatch "$BIN_DIR/run_validate_final_safety_refusal.sh"
    ;;

  finalize-all)
    "$BIN_DIR/run_finalize_unsafe.sh"
    "$BIN_DIR/run_validate_final_unsafe.sh"
    "$BIN_DIR/run_finalize_safety_refusal.sh"
    "$BIN_DIR/run_validate_final_safety_refusal.sh"
    ;;

  show-config)
    source "$BIN_DIR/common.sh"
    cat "$CONFIG_FILE"
    ;;

  check-config)
    dispatch "$BIN_DIR/check_config.sh"
    ;;

  help|-h|--help|"")
    usage
    ;;

  *)
    echo "ERROR: Unknown command: $COMMAND" >&2
    usage
    exit 2
    ;;
esac
