#!/usr/bin/env bash
set -Eeuo pipefail


# =============================================================================
# Defaults
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR_DEFAULT="$SCRIPT_DIR"

INPUT_DIR=""
OUTPUT_ROOT=""

MODEL="${MODEL_ID:-gpt-5.6-terra}"
BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
API_KEY_ENV="OPENAI_API_KEY"

PIPELINE_DIR="$PIPELINE_DIR_DEFAULT"

MAX_TOKENS="4096"
TIMEOUT="300"
RETRIES="3"
SLEEP_SECONDS="3"

FORCE=0
START_STAGE="target"
STOP_STAGE="merge"


# =============================================================================
# Help
# =============================================================================

usage() {
  cat <<'EOF'
Usage:

  rerun_full_annotation_pipeline.sh \
    --input_dir <normalized_cases_dir> \
    --output_root <output_root> \
    [options]

Required:
  --input_dir PATH
      Directory containing normalized trajectory JSON files.

  --output_root PATH
      Root directory for target, primary, attack-chain,
      execution-chain, and merged outputs.

Optional:
  --model MODEL
      Default: $MODEL_ID or gpt-5.6-terra

  --base_url URL
      Default: $OPENAI_BASE_URL or https://api.openai.com/v1

  --api_key_env NAME
      Default: OPENAI_API_KEY

  --pipeline_dir PATH
      Directory containing the annotation Python scripts.

  --max_tokens N
      Default: 4096

  --timeout N
      Default: 300

  --retries N
      Default: 3

  --sleep_seconds N
      Default: 3

  --force
      Re-run and overwrite existing outputs when supported.

  --start_stage STAGE
      Start from:
        target
        primary
        attack
        execution
        merge

  --stop_stage STAGE
      Stop after:
        target
        primary
        attack
        execution
        merge

Examples:

  # Run the complete pipeline.
  rerun_full_annotation_pipeline.sh \
    --input_dir /path/to/normalized_cases \
    --output_root /path/to/reannotation

  # Resume from attack-chain annotation.
  rerun_full_annotation_pipeline.sh \
    --input_dir /path/to/normalized_cases \
    --output_root /path/to/reannotation \
    --start_stage attack

Environment overrides for script paths:
  TARGET_SCRIPT
  PRIMARY_SCRIPT
  ATTACK_SCRIPT
  EXECUTION_SCRIPT
EOF
}


# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base_url)
      BASE_URL="$2"
      shift 2
      ;;
    --api_key_env)
      API_KEY_ENV="$2"
      shift 2
      ;;
    --pipeline_dir)
      PIPELINE_DIR="$2"
      shift 2
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --retries)
      RETRIES="$2"
      shift 2
      ;;
    --sleep_seconds)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --start_stage)
      START_STAGE="$2"
      shift 2
      ;;
    --stop_stage)
      STOP_STAGE="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done


# =============================================================================
# Validation
# =============================================================================

if [[ -z "$INPUT_DIR" ]]; then
  echo "ERROR: --input_dir is required." >&2
  exit 2
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "ERROR: --output_root is required." >&2
  exit 2
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: Input directory not found: $INPUT_DIR" >&2
  exit 2
fi

if [[ ! -d "$PIPELINE_DIR" ]]; then
  echo "ERROR: Pipeline directory not found: $PIPELINE_DIR" >&2
  exit 2
fi

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "ERROR: API key environment variable is not set: $API_KEY_ENV" >&2
  exit 2
fi

VALID_STAGES=(
  target
  primary
  attack
  execution
  merge
)

stage_index() {
  local wanted="$1"
  local i

  for i in "${!VALID_STAGES[@]}"; do
    if [[ "${VALID_STAGES[$i]}" == "$wanted" ]]; then
      echo "$i"
      return 0
    fi
  done

  return 1
}

START_INDEX="$(stage_index "$START_STAGE")" || {
  echo "ERROR: Invalid --start_stage: $START_STAGE" >&2
  exit 2
}

STOP_INDEX="$(stage_index "$STOP_STAGE")" || {
  echo "ERROR: Invalid --stop_stage: $STOP_STAGE" >&2
  exit 2
}

if (( START_INDEX > STOP_INDEX )); then
  echo "ERROR: start stage occurs after stop stage." >&2
  exit 2
fi


# =============================================================================
# Resolve scripts
# =============================================================================

find_first_script() {
  local exact_path="$1"
  shift

  if [[ -f "$exact_path" ]]; then
    printf '%s\n' "$exact_path"
    return 0
  fi

  local pattern
  local found

  for pattern in "$@"; do
    found="$(
      find "$PIPELINE_DIR" \
        -maxdepth 1 \
        -type f \
        -name "$pattern" \
        | sort \
        | head -1
    )"

    if [[ -n "$found" ]]; then
      printf '%s\n' "$found"
      return 0
    fi
  done

  return 1
}

TARGET_SCRIPT="${TARGET_SCRIPT:-$PIPELINE_DIR/annotate_primary_targets.py}"
PRIMARY_SCRIPT="${PRIMARY_SCRIPT:-$PIPELINE_DIR/annotate_primary_attribution_fixed_target.py}"
ATTACK_SCRIPT="${ATTACK_SCRIPT:-$PIPELINE_DIR/annotation_attackchain.py}"

if [[ -z "${EXECUTION_SCRIPT:-}" ]]; then
  EXECUTION_SCRIPT="$(
    find_first_script \
      "$PIPELINE_DIR/annotation_executionchain.py" \
      '*execution*chain*.py' \
      '*executionchain*.py'
  )" || true
fi


# =============================================================================
# Output paths
# =============================================================================

TARGET_ROOT="$OUTPUT_ROOT/target"
PRIMARY_ROOT="$OUTPUT_ROOT/primary"
ATTACK_ROOT="$OUTPUT_ROOT/attack_chain"
EXECUTION_ROOT="$OUTPUT_ROOT/execution_chain"
MERGED_ROOT="$OUTPUT_ROOT/merged"
MERGED_CASES="$MERGED_ROOT/cases"

TARGET_CASES="$TARGET_ROOT/cases"
PRIMARY_CASES="$PRIMARY_ROOT/cases"
ATTACK_CASES="$ATTACK_ROOT/cases"
EXECUTION_CASES="$EXECUTION_ROOT/cases"

mkdir -p \
  "$OUTPUT_ROOT" \
  "$TARGET_ROOT" \
  "$PRIMARY_ROOT" \
  "$ATTACK_ROOT" \
  "$EXECUTION_ROOT" \
  "$MERGED_CASES"


# =============================================================================
# CLI compatibility helpers
# =============================================================================

declare -A HELP_CACHE

script_help() {
  local script="$1"

  if [[ -z "${HELP_CACHE[$script]+x}" ]]; then
    HELP_CACHE["$script"]="$(python "$script" --help 2>&1 || true)"
  fi

  printf '%s\n' "${HELP_CACHE[$script]}"
}

supports_flag() {
  local script="$1"
  local flag="$2"

  script_help "$script" | grep -q -- "$flag"
}

append_option_if_supported() {
  local script="$1"
  local array_name="$2"
  local flag="$3"
  local value="$4"

  if supports_flag "$script" "$flag"; then
    local -n command_ref="$array_name"
    command_ref+=("$flag" "$value")
  fi
}

append_switch_if_supported() {
  local script="$1"
  local array_name="$2"
  local flag="$3"

  if supports_flag "$script" "$flag"; then
    local -n command_ref="$array_name"
    command_ref+=("$flag")
  fi
}

run_command() {
  echo
  echo "================================================================================"
  printf 'Running:'
  printf ' %q' "$@"
  echo
  echo "================================================================================"

  "$@"
}

ensure_script() {
  local stage="$1"
  local script="$2"

  if [[ -z "$script" || ! -f "$script" ]]; then
    echo "ERROR: $stage script not found: $script" >&2
    echo "Set the corresponding environment variable explicitly." >&2
    exit 2
  fi
}

should_run_stage() {
  local stage="$1"
  local index

  index="$(stage_index "$stage")"

  (( index >= START_INDEX && index <= STOP_INDEX ))
}


# =============================================================================
# Shared model arguments
# =============================================================================

append_model_args() {
  local script="$1"
  local array_name="$2"

  append_option_if_supported \
    "$script" "$array_name" \
    "--model" "$MODEL"

  append_option_if_supported \
    "$script" "$array_name" \
    "--base_url" "$BASE_URL"

  append_option_if_supported \
    "$script" "$array_name" \
    "--api_key_env" "$API_KEY_ENV"

  append_option_if_supported \
    "$script" "$array_name" \
    "--max_tokens" "$MAX_TOKENS"

  append_option_if_supported \
    "$script" "$array_name" \
    "--timeout" "$TIMEOUT"

  append_option_if_supported \
    "$script" "$array_name" \
    "--retries" "$RETRIES"

  append_option_if_supported \
    "$script" "$array_name" \
    "--sleep_seconds" "$SLEEP_SECONDS"

  if (( FORCE == 1 )); then
    append_switch_if_supported \
      "$script" "$array_name" \
      "--force"
  fi
}


# =============================================================================
# Stage 1: target
# =============================================================================

if should_run_stage target; then
  ensure_script "target" "$TARGET_SCRIPT"

  TARGET_CMD=(
    python
    "$TARGET_SCRIPT"
  )

  append_option_if_supported \
    "$TARGET_SCRIPT" TARGET_CMD \
    "--input_dir" "$INPUT_DIR"

  append_option_if_supported \
    "$TARGET_SCRIPT" TARGET_CMD \
    "--data_dir" "$INPUT_DIR"

  append_option_if_supported \
    "$TARGET_SCRIPT" TARGET_CMD \
    "--output_root" "$TARGET_ROOT"

  append_option_if_supported \
    "$TARGET_SCRIPT" TARGET_CMD \
    "--output_dir" "$TARGET_ROOT"

  append_model_args \
    "$TARGET_SCRIPT" TARGET_CMD

  run_command "${TARGET_CMD[@]}"
fi


# =============================================================================
# Stage 2: primary root attribution
# =============================================================================

if should_run_stage primary; then
  ensure_script "primary" "$PRIMARY_SCRIPT"

  if [[ ! -d "$TARGET_CASES" ]]; then
    echo "ERROR: Target cases directory not found: $TARGET_CASES" >&2
    exit 2
  fi

  PRIMARY_CMD=(
    python
    "$PRIMARY_SCRIPT"
  )

  append_option_if_supported \
    "$PRIMARY_SCRIPT" PRIMARY_CMD \
    "--input_dir" "$INPUT_DIR"

  append_option_if_supported \
    "$PRIMARY_SCRIPT" PRIMARY_CMD \
    "--target_dir" "$TARGET_CASES"

  append_option_if_supported \
    "$PRIMARY_SCRIPT" PRIMARY_CMD \
    "--output_root" "$PRIMARY_ROOT"

  append_option_if_supported \
    "$PRIMARY_SCRIPT" PRIMARY_CMD \
    "--output_dir" "$PRIMARY_ROOT"

  append_model_args \
    "$PRIMARY_SCRIPT" PRIMARY_CMD

  run_command "${PRIMARY_CMD[@]}"
fi


# =============================================================================
# Stage 3: attack chain
# =============================================================================

if should_run_stage attack; then
  ensure_script "attack-chain" "$ATTACK_SCRIPT"

  if [[ ! -d "$TARGET_CASES" ]]; then
    echo "ERROR: Target cases directory not found: $TARGET_CASES" >&2
    exit 2
  fi

  if [[ ! -d "$PRIMARY_CASES" ]]; then
    echo "ERROR: Primary cases directory not found: $PRIMARY_CASES" >&2
    exit 2
  fi

  ATTACK_CMD=(
    python
    "$ATTACK_SCRIPT"
  )

  append_option_if_supported \
    "$ATTACK_SCRIPT" ATTACK_CMD \
    "--input_dir" "$INPUT_DIR"

  append_option_if_supported \
    "$ATTACK_SCRIPT" ATTACK_CMD \
    "--target_dir" "$TARGET_CASES"

  append_option_if_supported \
    "$ATTACK_SCRIPT" ATTACK_CMD \
    "--primary_dir" "$PRIMARY_CASES"

  append_option_if_supported \
    "$ATTACK_SCRIPT" ATTACK_CMD \
    "--output_root" "$ATTACK_ROOT"

  append_option_if_supported \
    "$ATTACK_SCRIPT" ATTACK_CMD \
    "--output_dir" "$ATTACK_ROOT"

  append_model_args \
    "$ATTACK_SCRIPT" ATTACK_CMD

  run_command "${ATTACK_CMD[@]}"
fi


# =============================================================================
# Stage 4: execution chain
# =============================================================================

if should_run_stage execution; then
  ensure_script "execution-chain" "$EXECUTION_SCRIPT"

  if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input root not found: $INPUT_DIR" >&2
    exit 2
  fi

  if [[ ! -d "$PRIMARY_ROOT" ]]; then
    echo "ERROR: Primary root not found: $PRIMARY_ROOT" >&2
    exit 2
  fi

  if [[ ! -d "$ATTACK_ROOT" ]]; then
    echo "ERROR: Attack-chain root not found: $ATTACK_ROOT" >&2
    exit 2
  fi

  EXECUTION_CMD=(
    python
    "$EXECUTION_SCRIPT"
    --input_root
    "$INPUT_DIR"
    --primary_root
    "$PRIMARY_ROOT"
    --attack_chain_root
    "$ATTACK_ROOT"
    --output_root
    "$EXECUTION_ROOT"
  )

  append_model_args \
    "$EXECUTION_SCRIPT" EXECUTION_CMD

  run_command "${EXECUTION_CMD[@]}"
fi


# =============================================================================
# Stage 5: merge
# =============================================================================

if should_run_stage merge; then
  export TARGET_CASES
  export PRIMARY_CASES
  export ATTACK_CASES
  export EXECUTION_CASES
  export MERGED_CASES
  export MERGED_ROOT

  python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


target_dir = Path(os.environ["TARGET_CASES"])
primary_dir = Path(os.environ["PRIMARY_CASES"])
attack_dir = Path(os.environ["ATTACK_CASES"])
execution_dir = Path(os.environ["EXECUTION_CASES"])
output_dir = Path(os.environ["MERGED_CASES"])
output_root = Path(os.environ["MERGED_ROOT"])

output_dir.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def key_for(path: Path) -> str:
    name = path.name

    for suffix in (
        ".execution.json",
        ".execution_chain.json",
        ".attack.json",
        ".attack_chain.json",
        ".primary.json",
        ".target.json",
        ".annotation.json",
        ".json",
    ):
        if name.endswith(suffix):
            return name[:-len(suffix)]

    return path.stem


def index_dir(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}

    output: dict[str, Path] = {}

    for path in sorted(directory.glob("*.json")):
        output[key_for(path)] = path

    return output


target_files = index_dir(target_dir)
primary_files = index_dir(primary_dir)
attack_files = index_dir(attack_dir)
execution_files = index_dir(execution_dir)

all_keys = sorted(
    set(target_files)
    | set(primary_files)
    | set(attack_files)
    | set(execution_files)
)

merged = 0
failures: list[dict[str, Any]] = []


for key in all_keys:
    missing = []

    if key not in target_files:
        missing.append("target")
    if key not in primary_files:
        missing.append("primary")
    if key not in attack_files:
        missing.append("attack_chain")
    if key not in execution_files:
        missing.append("execution_chain")

    if missing:
        failures.append({
            "case_key": key,
            "error": "missing_annotation_files",
            "missing": missing,
        })
        continue

    try:
        target = load_json(target_files[key])
        primary = load_json(primary_files[key])
        attack = load_json(attack_files[key])
        execution = load_json(execution_files[key])

        target_action = (
            target.get("target_action")
            or target.get("target_unsafe_action")
        )

        primary_component = primary.get(
            "primary_attribution_component"
        )

        attack_chain = attack.get("attack_chain")
        execution_chain = execution.get("execution_chain")

        if not isinstance(target_action, dict):
            raise ValueError("Missing target action object.")

        if not isinstance(primary_component, dict):
            raise ValueError(
                "Missing primary_attribution_component object."
            )

        if not isinstance(attack_chain, list):
            raise ValueError("Missing attack_chain list.")

        if not isinstance(execution_chain, list):
            raise ValueError("Missing execution_chain list.")

        trajectory_ids = [
            target.get("trajectory_id"),
            primary.get("trajectory_id"),
            attack.get("trajectory_id"),
            execution.get("trajectory_id"),
        ]

        non_null_ids = {
            str(value)
            for value in trajectory_ids
            if value is not None
        }

        if len(non_null_ids) > 1:
            raise ValueError(
                f"trajectory_id mismatch: {sorted(non_null_ids)}"
            )

        merged_obj = {
            "trajectory_id": next(
                (
                    value
                    for value in trajectory_ids
                    if value is not None
                ),
                None,
            ),
            "target_type": (
                target.get("target_type")
                or primary.get("target_type")
            ),
            "attack_success": target.get(
                "attack_success",
                primary.get("attack_success"),
            ),
            "target_action": target_action,
            "primary_attribution_component": primary_component,
            "attack_chain": attack_chain,
            "execution_chain": execution_chain,
            "confidence": {
                "target": target.get("confidence"),
                "primary": primary.get("confidence"),
                "attack_chain": attack.get("confidence"),
                "execution_chain": execution.get("confidence"),
            },
            "needs_review": {
                "target": target.get("needs_review", False),
                "primary": primary.get("needs_review", False),
                "attack_chain": attack.get("needs_review", False),
                "execution_chain": execution.get(
                    "needs_review",
                    False,
                ),
            },
            "_metadata": {
                "target_annotation_path": str(
                    target_files[key]
                ),
                "primary_annotation_path": str(
                    primary_files[key]
                ),
                "attack_chain_annotation_path": str(
                    attack_files[key]
                ),
                "execution_chain_annotation_path": str(
                    execution_files[key]
                ),
                "target_metadata": target.get(
                    "_metadata",
                    {},
                ),
                "primary_metadata": primary.get(
                    "_metadata",
                    {},
                ),
                "attack_chain_metadata": attack.get(
                    "_metadata",
                    {},
                ),
                "execution_chain_metadata": execution.get(
                    "_metadata",
                    {},
                ),
            },
        }

        write_json(
            output_dir / f"{key}.annotation.json",
            merged_obj,
        )

        merged += 1

    except Exception as exc:
        failures.append({
            "case_key": key,
            "error": repr(exc),
            "target_path": str(target_files.get(key)),
            "primary_path": str(primary_files.get(key)),
            "attack_chain_path": str(attack_files.get(key)),
            "execution_chain_path": str(
                execution_files.get(key)
            ),
        })


report = {
    "num_target_files": len(target_files),
    "num_primary_files": len(primary_files),
    "num_attack_chain_files": len(attack_files),
    "num_execution_chain_files": len(execution_files),
    "num_merged": merged,
    "num_failures": len(failures),
    "failures": failures,
    "directories": {
        "target": str(target_dir),
        "primary": str(primary_dir),
        "attack_chain": str(attack_dir),
        "execution_chain": str(execution_dir),
        "merged": str(output_dir),
    },
}

write_json(
    output_root / "merge_report.json",
    report,
)

print("=" * 80)
print("Annotation merge finished")
print("=" * 80)
print(f"Target files:          {len(target_files)}")
print(f"Primary files:         {len(primary_files)}")
print(f"Attack-chain files:    {len(attack_files)}")
print(f"Execution-chain files: {len(execution_files)}")
print(f"Merged:                {merged}")
print(f"Failures:              {len(failures)}")
print(f"Output:                {output_dir}")
print("=" * 80)

if failures:
    raise SystemExit(1)
PY
fi


# =============================================================================
# Final summary
# =============================================================================

echo
echo "================================================================================"
echo "Reannotation pipeline finished"
echo "================================================================================"
echo "Input:           $INPUT_DIR"
echo "Output root:     $OUTPUT_ROOT"
echo "Target:          $TARGET_ROOT"
echo "Primary:         $PRIMARY_ROOT"
echo "Attack chain:    $ATTACK_ROOT"
echo "Execution chain: $EXECUTION_ROOT"
echo "Merged cases:    $MERGED_CASES"
echo "================================================================================"
