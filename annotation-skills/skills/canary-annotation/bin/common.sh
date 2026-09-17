#!/usr/bin/env bash
set -Eeuo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$BIN_DIR/.." && pwd)"

CONFIG_FILE="${ANNOTATION_CONFIG:-$SKILL_ROOT/config/annotation.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE" >&2
  echo "Copy config/annotation.example.env to your run directory and set ANNOTATION_CONFIG." >&2
  exit 2
fi

set -a
source "$CONFIG_FILE"
set +a
if [[ -n "${UNIFIED_NORMALIZED_ROOT:-}" ]]; then
  export NORMALIZED_ROOT="$UNIFIED_NORMALIZED_ROOT"
  export NORMALIZED_CASES="$NORMALIZED_ROOT/cases"
fi
if [[ -n "${UNIFIED_RAW_INPUT:-}" ]]; then export RAW_INPUT="$UNIFIED_RAW_INPUT"; fi

API_KEY_ENV_NAME="${OPENAI_API_KEY_ENV:-OPENAI_API_KEY}"

require_api_key() {
  [[ "${ANNOTATION_DRY_RUN:-0}" != 1 ]] || return 0
  if [[ -z "${!API_KEY_ENV_NAME:-}" ]]; then
    echo "ERROR: API key variable not set: $API_KEY_ENV_NAME" >&2
    exit 2
  fi
}

require_file() {
  local path="$1"
  if [[ "${ANNOTATION_DRY_RUN:-0}" == 1 && "$path" != *.py && "$path" != *.sh ]]; then
    python3 "$BIN_DIR/record_plan.py" dependency "$ANNOTATION_PLAN" "$path" file
    return 0
  fi
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Required file not found: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ "${ANNOTATION_DRY_RUN:-0}" == 1 ]]; then
    python3 "$BIN_DIR/record_plan.py" dependency "$ANNOTATION_PLAN" "$path" directory
    return 0
  fi
  if [[ ! -d "$path" ]]; then
    echo "ERROR: Required directory not found: $path" >&2
    exit 2
  fi
}

run_step() {
  local title="$1"
  shift

  if [[ "${ANNOTATION_DRY_RUN:-0}" == 1 ]]; then
    python3 "$BIN_DIR/record_plan.py" stage "$ANNOTATION_PLAN" "$title" "$@"
    return 0
  fi

  echo
  echo "================================================================================"
  echo "$title"
  echo "================================================================================"
  printf 'Running:'
  printf ' %q' "$@"
  echo
  echo "================================================================================"

  if [[ -n "${UNIFIED_OUTPUT_ROOT:-}" ]]; then
    python3 "$BIN_DIR/workflow_output.py" stage "$UNIFIED_OUTPUT_ROOT" "$title" start
  fi
  "$@"
  if [[ -n "${UNIFIED_OUTPUT_ROOT:-}" ]]; then
    python3 "$BIN_DIR/workflow_output.py" stage "$UNIFIED_OUTPUT_ROOT" "$title" end
  fi
}
