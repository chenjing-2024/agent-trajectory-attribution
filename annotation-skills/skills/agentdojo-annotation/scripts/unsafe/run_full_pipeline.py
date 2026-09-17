#!/usr/bin/env python3
"""Run the complete unsafe annotation, review, repair, and finalization flow."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def append_option(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend([name, str(value)])


def redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, value in enumerate(redacted[:-1]):
        if value == "--base_url":
            redacted[index + 1] = "<redacted-base-url>"
    return redacted


def run_stage(
    *,
    name: str,
    command: list[str],
    dry_run: bool,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    record = {
        "name": name,
        "command": redact_command(command),
        "started_at": now(),
        "status": "planned" if dry_run else "running",
    }
    manifest["stages"].append(record)
    write_json(manifest_path, manifest)
    print(f"\n[{name}]")
    print("+", shlex.join(command))

    if dry_run:
        record["finished_at"] = now()
        write_json(manifest_path, manifest)
        return

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        record["status"] = "failed"
        record["returncode"] = exc.returncode
        record["finished_at"] = now()
        manifest["status"] = "failed"
        manifest["failed_stage"] = name
        write_json(manifest_path, manifest)
        raise

    record["status"] = "ok"
    record["returncode"] = 0
    record["finished_at"] = now()
    write_json(manifest_path, manifest)


def require_file(path: Path, stage: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{stage} did not produce required file: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run complete AgentDojo unsafe annotation: initial annotation, "
            "deterministic and semantic validation, repair, merge, final "
            "revalidation, and final summary."
        )
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--suite_name", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--repair_model",
        default=None,
        help="Repair model; defaults to --model.",
    )
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include_non_security_true", action="store_true")
    parser.add_argument("--use_max_completion_tokens", action="store_true")
    parser.add_argument("--semantic_max_tokens", type=int, default=8192)
    parser.add_argument("--repair_max_completion_tokens", type=int, default=16384)
    parser.add_argument(
        "--repair_reasoning_effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="medium",
    )
    parser.add_argument("--request_timeout", type=float, default=300.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python_bin", default=sys.executable)
    return parser


def pipeline_main() -> None:
    args = build_parser().parse_args()
    requested_limit = args.limit
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")
    if output_root.exists():
        raise FileExistsError(
            "output_root already exists; complete runs require a new directory: "
            f"{output_root}"
        )

    output_root.mkdir(parents=True)
    if not args.dry_run:
        from unified_output import select_directory
        select_directory(raw_root, output_root / "00_selection/input", args.limit, security=None if args.include_non_security_true else True)
        raw_root = output_root / "00_selection/input"
        from unified_output import read
        if not read(output_root / "00_selection/selected_cases.json")["cases"]:
            return
        args.limit = None  # The cohort was limited once, before normalization.

    initial_root = output_root / "initial"
    components_root = initial_root / "components"
    initial_annotations = initial_root / "annotations"
    initial_deterministic = initial_root / "validation"
    initial_semantic = output_root / "validation" / "initial_semantic"
    repaired_annotations = output_root / "repaired_annotations"
    final_annotations = output_root / "final_annotations"
    final_deterministic = output_root / "validation" / "final_deterministic"
    final_semantic = output_root / "validation" / "final_semantic"
    final_summary = output_root / "final_summary"
    manifest_path = output_root / "full_run_manifest.json"

    manifest: dict[str, Any] = {
        "schema_version": "agentdojo_unsafe_full_pipeline_v1",
        "created_at": now(),
        "status": "dry_run" if args.dry_run else "running",
        "run_name": args.run_name,
        "settings": {
            "suite_name": args.suite_name,
            "model": args.model,
            "repair_model": args.repair_model or args.model,
            "uses_custom_base_url": bool(args.base_url),
            "api_key_env": args.api_key_env,
            "limit": requested_limit,
            "use_max_completion_tokens": args.use_max_completion_tokens,
        },
        "paths": {
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "components_root": str(components_root),
            "initial_annotations": str(initial_annotations),
            "initial_deterministic": str(initial_deterministic),
            "initial_semantic": str(initial_semantic),
            "repaired_annotations": str(repaired_annotations),
            "final_annotations": str(final_annotations),
            "final_deterministic": str(final_deterministic),
            "final_semantic": str(final_semantic),
            "final_summary": str(final_summary),
        },
        "stages": [],
    }
    write_json(manifest_path, manifest)

    initial_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "run_initial_pipeline.py"),
        "--raw_root",
        str(raw_root),
        "--output_root",
        str(initial_root),
        "--model",
        args.model,
        "--base_url",
        args.base_url,
        "--api_key_env",
        args.api_key_env,
    ]
    append_option(initial_cmd, "--suite_name", args.suite_name)
    append_option(initial_cmd, "--run_name", args.run_name)
    append_option(initial_cmd, "--limit", args.limit)
    if args.force:
        initial_cmd.append("--force")
    if args.include_non_security_true:
        initial_cmd.append("--include_non_security_true")
    if args.dry_run:
        initial_cmd.append("--dry_run")

    initial_semantic_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "validate_semantic.py"),
        "--input_cases_jsonl",
        str(initial_deterministic / "semantic_input_cases.jsonl"),
        "--normalized_root",
        str(components_root),
        "--output_root",
        str(initial_semantic),
        "--model",
        args.model,
        "--base_url",
        args.base_url,
        "--api_key_env",
        args.api_key_env,
        "--max_tokens",
        str(args.semantic_max_tokens),
        "--request_timeout",
        str(args.request_timeout),
        "--max_retries",
        str(args.max_retries),
        "--force",
    ]
    if args.use_max_completion_tokens:
        initial_semantic_cmd.append("--use_max_completion_tokens")
    append_option(initial_semantic_cmd, "--limit", args.limit)

    repair_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "repair.py"),
        "--normalized_root",
        str(components_root),
        "--old_annotation_root",
        str(initial_annotations),
        "--semantic_review_jsonl",
        str(initial_semantic / "review_cases.jsonl"),
        "--deterministic_error_jsonl",
        str(initial_deterministic / "repair_cases.jsonl"),
        "--output_root",
        str(repaired_annotations),
        "--model",
        args.repair_model or args.model,
        "--base_url",
        args.base_url,
        "--api_key_env",
        args.api_key_env,
        "--reasoning_effort",
        args.repair_reasoning_effort,
        "--max_completion_tokens",
        str(args.repair_max_completion_tokens),
        "--request_timeout",
        str(args.request_timeout),
        "--max_retries",
        str(args.max_retries),
        "--force",
    ]
    merge_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "merge.py"),
        "--components_root",
        str(components_root),
        "--original_root",
        str(initial_annotations),
        "--repaired_root",
        str(repaired_annotations),
        "--output_root",
        str(final_annotations),
    ]

    final_deterministic_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "validate_deterministic.py"),
        "--components_root",
        str(components_root),
        "--annotations_root",
        str(final_annotations),
        "--output_dir",
        str(final_deterministic),
    ]
    append_option(final_deterministic_cmd, "--limit", args.limit)

    final_semantic_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "validate_semantic.py"),
        "--input_cases_jsonl",
        str(final_deterministic / "semantic_input_cases.jsonl"),
        "--normalized_root",
        str(components_root),
        "--output_root",
        str(final_semantic),
        "--model",
        args.model,
        "--base_url",
        args.base_url,
        "--api_key_env",
        args.api_key_env,
        "--max_tokens",
        str(args.semantic_max_tokens),
        "--request_timeout",
        str(args.request_timeout),
        "--max_retries",
        str(args.max_retries),
        "--force",
    ]
    if args.use_max_completion_tokens:
        final_semantic_cmd.append("--use_max_completion_tokens")
    append_option(final_semantic_cmd, "--limit", args.limit)

    final_summary_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "build_summary.py"),
        "--components_root",
        str(components_root),
        "--annotations_root",
        str(final_annotations),
        "--output_dir",
        str(final_summary),
    ]

    stages = [
        ("initial_annotation", initial_cmd),
        ("initial_semantic_validation", initial_semantic_cmd),
        ("repair", repair_cmd),
        ("merge_final_annotations", merge_cmd),
        ("final_deterministic_validation", final_deterministic_cmd),
        ("final_semantic_validation", final_semantic_cmd),
        ("final_summary", final_summary_cmd),
    ]

    try:
        for name, command in stages:
            run_stage(
                name=name,
                command=command,
                dry_run=args.dry_run,
                manifest=manifest,
                manifest_path=manifest_path,
            )
            if args.dry_run:
                continue
            if name == "initial_annotation":
                require_file(
                    initial_deterministic / "semantic_input_cases.jsonl", name
                )
                require_file(initial_deterministic / "repair_cases.jsonl", name)
            elif name == "initial_semantic_validation":
                require_file(initial_semantic / "review_cases.jsonl", name)
            elif name == "repair":
                require_file(repaired_annotations / "_repair_manifest.json", name)
            elif name == "merge_final_annotations":
                require_file(final_annotations / "merge_report.json", name)
            elif name == "final_deterministic_validation":
                require_file(
                    final_deterministic / "validation_report.json", name
                )
            elif name == "final_semantic_validation":
                require_file(
                    final_semantic / "semantic_validation_report.json", name
                )
            elif name == "final_summary":
                require_file(final_summary / "case_summary.jsonl", name)
    except Exception:
        if manifest.get("status") != "failed":
            manifest["status"] = "failed"
            manifest["finished_at"] = now()
            write_json(manifest_path, manifest)
        raise

    if args.dry_run:
        manifest["status"] = "dry_run_complete"
        manifest["finished_at"] = now()
        write_json(manifest_path, manifest)
        print(f"\nDry run complete. Manifest: {manifest_path}")
        return

    deterministic_report = load_json(
        final_deterministic / "validation_report.json"
    )
    semantic_report = load_json(
        final_semantic / "semantic_validation_report.json"
    )
    deterministic_summary = deterministic_report.get("summary", {})
    semantic_summary = semantic_report.get("summary", {})

    unresolved = {
        "deterministic_hard_errors": int(
            deterministic_summary.get("num_hard_error", 0) or 0
        ),
        "semantic_uncertain": int(
            semantic_summary.get("num_uncertain", 0) or 0
        ),
        "semantic_fail": int(semantic_summary.get("num_fail", 0) or 0),
        "semantic_validator_error": int(
            semantic_summary.get("num_validator_error", 0) or 0
        ),
    }
    manifest["final_quality"] = {
        "deterministic_summary": deterministic_summary,
        "semantic_summary": semantic_summary,
        "unresolved": unresolved,
    }
    manifest["status"] = (
        "ok"
        if not any(unresolved.values())
        else "completed_with_review"
    )
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)

    print("\nFull unsafe annotation pipeline complete.")
    print(f"Status:            {manifest['status']}")
    print(f"Final annotations: {final_annotations}")
    print(f"Final summary:     {final_summary}")
    print(f"Manifest:          {manifest_path}")


def main():
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from pipeline_output import execute
    args = build_parser().parse_args()
    return execute(pipeline_main, args, "unsafe")


if __name__ == "__main__":
    raise SystemExit(main())
