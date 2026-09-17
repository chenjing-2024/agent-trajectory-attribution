#!/usr/bin/env python3
"""Run complete Agent3Sigma unsafe annotation and repair."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_case_coverage import audit


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "common"


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


def redacted(command: list[str]) -> list[str]:
    result = list(command)
    for index, value in enumerate(result[:-1]):
        if value == "--base_url":
            result[index + 1] = "<redacted-base-url>"
    return result


def add(command: list[str], option: str, value: Any) -> None:
    if value is not None:
        command.extend([option, str(value)])


def run_stage(
    name: str,
    command: list[str],
    *,
    dry_run: bool,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    record = {
        "name": name,
        "command": redacted(command),
        "started_at": now(),
        "status": "planned" if dry_run else "running",
    }
    manifest["stages"].append(record)
    write_json(manifest_path, manifest)
    print(f"\n[{name}]\n+ {shlex.join(command)}")
    if dry_run:
        record["finished_at"] = now()
        write_json(manifest_path, manifest)
        return
    try:
        if name == "audit_case_coverage":
            summary_path = manifest_path.parent / "coverage_summary.json"
            summary_path.unlink(missing_ok=True)
            result = subprocess.run(command, check=False)
            coverage = load_json(summary_path)
            if coverage.get("schema_version") != "unsafe_case_coverage_v1" or coverage.get("returncode") != result.returncode:
                raise ValueError("Invalid coverage audit result")
            record.update(status=coverage["status"], returncode=result.returncode, finished_at=now())
            write_json(manifest_path, manifest)
            return
        subprocess.run(command, check=True)
    except Exception as exc:
        record.update(
            status="failed",
            returncode=getattr(exc, "returncode", 2),
            finished_at=now(),
        )
        manifest.update(status="failed", failed_stage=name, finished_at=now())
        write_json(manifest_path, manifest)
        raise
    record.update(status="ok", returncode=0, finished_at=now())
    write_json(manifest_path, manifest)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Prepare Agent3Sigma cases, annotate unsafe causal structure, "
            "validate, repair, finalize, and revalidate."
        )
    )
    value.add_argument(
        "--input",
        required=True,
        help="Agent3Sigma detailed.json or prepared case directory.",
    )
    value.add_argument("--input_format", choices=("raw", "normalized"), default="raw",
                       help="normalized reuses existing components after validation; IDs are preserved.")
    value.add_argument("--output_root", required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--base_url", default=None)
    value.add_argument("--api_key_env", default="OPENAI_API_KEY")
    value.add_argument(
        "--provider",
        choices=("openai", "compatible"),
        default="compatible",
    )
    value.add_argument("--limit", type=int, default=None)
    value.add_argument("--temperature", type=float, default=0.0)
    value.add_argument("--max_tokens", type=int, default=2200)
    # Some stage CLIs intentionally accept only whole-second timeouts.
    # Keep the orchestrator type equally strict so it never forwards "600.0".
    value.add_argument("--timeout", type=int, default=600)
    value.add_argument("--retries", type=int, default=3)
    value.add_argument("--sleep_seconds", type=float, default=2.0)
    value.add_argument("--force", action="store_true")
    value.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing failed output_root from its failed stage. "
            "The recorded input and model must match."
        ),
    )
    value.add_argument("--dry_run", action="store_true")
    value.add_argument("--python_bin", default=sys.executable)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input does not exist: {input_path}")
    if not (input_path.is_dir() or input_path.is_file()):
        raise ValueError(f"input must be a file or directory: {input_path}")
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            "output_root already exists; choose a new directory or pass "
            f"--resume for a failed run: {output_root}"
        )
    if args.resume and not output_root.is_dir():
        raise FileNotFoundError(
            f"--resume requires an existing output_root: {output_root}"
        )
    if args.resume and args.dry_run:
        raise ValueError("--resume and --dry_run cannot be used together")
    if args.provider == "compatible" and not args.base_url:
        raise ValueError("--base_url is required for provider=compatible")

    output_root.mkdir(parents=True, exist_ok=args.resume)
    selection_root = output_root / "00_selection"
    selection = selection_root / "selected_cases.json"
    queue_root = output_root / "08_repair_queue"
    normalized = output_root / "01_normalized"
    target = output_root / "02_target"
    primary = output_root / "03_primary"
    attack = output_root / "04_attack_chain"
    execution = output_root / "05_execution_chain"
    combined = output_root / "06_combined"
    deterministic = output_root / "07_deterministic_validation"
    semantic = output_root / "08_semantic_validation"
    repairs = output_root / "09_repairs"
    final_annotations = repairs / "final_annotations" / "cases"
    completed_annotations = repairs / "completed_annotations" / "cases"
    final_deterministic = output_root / "10_final_deterministic"
    final_semantic = output_root / "11_final_semantic"
    manifest_path = output_root / "run_manifest.json"

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"--resume requires run_manifest.json: {manifest_path}"
            )
        manifest = load_json(manifest_path)
        if manifest.get("status") != "failed":
            raise ValueError(
                "--resume only accepts a run whose manifest status is 'failed'"
            )
        recorded_input = manifest.get("paths", {}).get("input")
        if manifest.get("settings", {}).get("input_format", "raw") != args.input_format:
            raise ValueError("Resume input_format must match the frozen selected set")
        recorded_model = manifest.get("settings", {}).get("model")
        if manifest.get("settings", {}).get("limit") != args.limit:
            raise ValueError("Resume --limit must match the frozen selected set")
        if recorded_input != str(input_path):
            raise ValueError(
                f"resume input mismatch: recorded={recorded_input!r}, "
                f"requested={str(input_path)!r}"
            )
        if recorded_model != args.model:
            raise ValueError(
                f"resume model mismatch: recorded={recorded_model!r}, "
                f"requested={args.model!r}"
            )
        if manifest.get("failed_stage") == "select_run_cases":
            raise ValueError("Input selection failed; fix the input and use a new output_root")
        manifest.setdefault("resume_history", []).append(
            {
                "resumed_at": now(),
                "from_stage": manifest.get("failed_stage"),
            }
        )
        manifest["status"] = "running"
    else:
        manifest = {
            "schema_version": "agent3sigma_unsafe_annotation_pipeline_v1",
            "created_at": now(),
            "status": "dry_run" if args.dry_run else "running",
            "settings": {
                "input_format": args.input_format,
                "model": args.model,
                "provider": args.provider,
                "api_key_env": args.api_key_env,
                "limit": args.limit,
                "uses_custom_base_url": bool(args.base_url),
            },
            "paths": {
                "input": str(input_path),
                "selected_cases": str(selection),
                "repair_queue": str(queue_root / "repair_tasks.jsonl"),
                "case_outcomes": str(output_root / "case_outcomes.jsonl"),
                "coverage_summary": str(output_root / "coverage_summary.json"),
                "output_root": str(output_root),
                "normalized_cases": str(normalized / "cases"),
                "initial_annotations": str(combined / "cases"),
                "final_annotations": str(final_annotations),
                "final_deterministic": str(final_deterministic),
                "final_semantic": str(final_semantic),
            },
            "stages": [],
        }
    write_json(manifest_path, manifest)

    stages: list[tuple[str, list[str]]] = []
    select_command = [args.python_bin, str(SCRIPT_DIR / "select_run_cases.py"),
                      "--input", str(input_path), "--output_root", str(selection_root)]
    select_command.extend(["--input_format", args.input_format])
    add(select_command, "--limit", args.limit)
    stages.append(("select_run_cases", select_command))
    cases_input = selection_root / "cases"

    normalize = [
        args.python_bin,
        str(COMMON_DIR / "normalize_components.py"),
        "--input",
        str(cases_input),
        "--output",
        str(normalized / "normalized_trajectories.jsonl"),
        "--summary",
        str(normalized / "summary.json"),
        "--cases_output_dir",
        str(normalized / "cases"),
    ]
    if args.input_format == "normalized":
        normalize = [args.python_bin, str(COMMON_DIR / "prepare_annotation_input.py"),
                     "--input", str(cases_input), "--input_format", "normalized",
                     "--output_root", str(normalized)]
        stages.append(("validate_reuse_components", normalize))
        print("已选择标准化输入：校验后复用组件，跳过标准化和组件拆分。")
    else:
        stages.append(("normalize_components", normalize))

    def model_command(script: str, output: Path) -> list[str]:
        command = [
            args.python_bin,
            str(SCRIPT_DIR / script),
            "--output_root",
            str(output),
            "--model",
            args.model,
            "--api_key_env",
            args.api_key_env,
            "--temperature",
            str(args.temperature),
            "--max_tokens",
            str(args.max_tokens),
            "--timeout",
            str(args.timeout),
        ]
        add(command, "--base_url", args.base_url)
        if args.force:
            command.append("--force")
        return command

    target_cmd = model_command("annotate_target.py", target)
    target_cmd.extend(["--input_dir", str(normalized / "cases")])
    target_cmd.extend(["--retries", str(args.retries)])
    stages.append(("annotate_target", target_cmd))

    primary_cmd = model_command("annotate_primary.py", primary)
    primary_cmd.extend(
        [
            "--input_dir",
            str(normalized / "cases"),
            "--target_dir",
            str(target / "cases"),
            "--retries",
            str(args.retries),
            "--sleep_seconds",
            str(args.sleep_seconds),
        ]
    )
    stages.append(("annotate_primary", primary_cmd))

    attack_cmd = model_command("annotate_attack_chain.py", attack)
    attack_cmd.extend(
        [
            "--input_dir",
            str(normalized / "cases"),
            "--target_dir",
            str(target / "cases"),
            "--primary_dir",
            str(primary / "cases"),
            "--retries",
            str(args.retries),
            "--sleep_seconds",
            str(args.sleep_seconds),
        ]
    )
    stages.append(("annotate_attack_chain", attack_cmd))

    execution_cmd = model_command("annotate_execution_chain.py", execution)
    execution_cmd.extend(
        [
            "--input_root",
            str(normalized),
            "--target_root",
            str(target),
            "--primary_root",
            str(primary),
            "--attack_chain_root",
            str(attack),
            "--max_retries",
            str(args.retries),
            "--sleep_seconds",
            str(args.sleep_seconds),
        ]
    )
    stages.append(("annotate_execution_chain", execution_cmd))

    merge = [
        args.python_bin,
        str(SCRIPT_DIR / "merge_annotations.py"),
        "--run_root",
        str(output_root),
        "--target_dir",
        str(target / "cases"),
        "--primary_dir",
        str(primary / "cases"),
        "--attack_chain_dir",
        str(attack / "cases"),
        "--execution_chain_dir",
        str(execution / "cases"),
        "--output_root",
        str(combined),
    ]
    stages.append(("merge_annotations", merge))

    def deterministic_command(annotations: Path, destination: Path) -> list[str]:
        command = [
            args.python_bin,
            str(SCRIPT_DIR / "validate_deterministic.py"),
            "--annotations_dir",
            str(annotations),
            "--trajectories_dir",
            str(normalized / "cases"),
            "--output_root",
            str(destination),
        ]
        return command

    stages.append(
        (
            "initial_deterministic_validation",
            deterministic_command(combined / "cases", deterministic),
        )
    )

    def semantic_command(annotations: Path, destination: Path) -> list[str]:
        command = [
            args.python_bin,
            str(SCRIPT_DIR / "validate_semantic.py"),
            "--annotations_dir",
            str(annotations),
            "--trajectories_dir",
            str(normalized / "cases"),
            "--output_root",
            str(destination),
            "--model",
            args.model,
            "--provider",
            args.provider,
            "--api_key_env",
            args.api_key_env,
            "--temperature",
            str(args.temperature),
            "--max_tokens",
            str(args.max_tokens),
            "--max_retries",
            str(args.retries),
            "--request_timeout",
            str(args.timeout),
            "--force",
        ]
        add(command, "--base_url", args.base_url)
        return command

    stages.append(
        (
            "initial_semantic_validation",
            semantic_command(combined / "cases", semantic),
        )
    )

    stages.append(("build_repair_queue", [args.python_bin, str(SCRIPT_DIR / "build_repair_queue.py"),
                   "--selection", str(selection), "--normalized", str(normalized / "cases"),
                   "--annotations", str(combined / "cases"), "--deterministic", str(deterministic),
                   "--semantic", str(semantic), "--output_root", str(queue_root)]))

    repair = [
        args.python_bin,
        str(SCRIPT_DIR / "run_repairs.py"),
        "--validation_root",
        str(queue_root),
        "--trajectory_root",
        str(normalized / "cases"),
        "--output_root",
        str(repairs),
        "--category",
        "all",
        "--model",
        args.model,
        "--api_key_env",
        args.api_key_env,
        "--temperature",
        str(args.temperature),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--sleep_seconds",
        str(args.sleep_seconds),
        "--clean",
        "--force",
    ]
    add(repair, "--base_url", args.base_url)
    stages.append(("repair_annotations", repair))

    finalize = [
        args.python_bin,
        str(SCRIPT_DIR / "finalize_annotations.py"),
        "--run_root",
        str(repairs),
        "--clean",
        "--allow_empty",
    ]
    stages.append(("finalize_annotations", finalize))
    stages.append(
        (
            "final_deterministic_validation",
            deterministic_command(completed_annotations, final_deterministic),
        )
    )
    stages.append(
        (
            "final_semantic_validation",
            semantic_command(completed_annotations, final_semantic),
        )
    )

    stages.append(("audit_case_coverage", [args.python_bin, str(SCRIPT_DIR / "audit_case_coverage.py"), "--run_root", str(output_root)]))

    start_index = 0
    if args.resume:
        failed_stage = manifest.get("failed_stage")
        stage_names = [name for name, _ in stages]
        if failed_stage not in stage_names:
            raise ValueError(
                f"Cannot resume unknown failed stage: {failed_stage!r}"
            )
        start_index = stage_names.index(failed_stage)
        manifest["failed_stage"] = None
        write_json(manifest_path, manifest)
        print(
            f"Resuming at stage {stage_names[start_index]!r}; "
            f"skipping {start_index} completed stage(s)."
        )

    try:
        for name, command in stages[start_index:]:
            run_stage(
                name,
                command,
                dry_run=args.dry_run,
                manifest=manifest,
                manifest_path=manifest_path,
            )
            if not args.dry_run and name == "select_run_cases":
                manifest["input_case_count"] = load_json(selection)["selected_count"]
                if manifest["input_case_count"] == 0:
                    manifest["coverage"] = audit(output_root)
                    manifest.update(status="no_eligible_cases", returncode=0, finished_at=now())
                    write_json(manifest_path, manifest)
                    return 0
                write_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(status="failed", failed_stage=manifest.get("failed_stage") or name,
                        finished_at=now(), error=str(exc), returncode=2)
        if not args.dry_run:
            try:
                manifest["coverage"] = audit(output_root, pipeline_error={"stage": name, "error": str(exc)})
            except Exception as audit_error:
                write_json(output_root / "coverage_summary.json", {"status": "failed", "returncode": 2,
                           "error": str(audit_error), "pipeline_error": str(exc)})
        write_json(manifest_path, manifest)
        return 2

    if args.dry_run:
        status, code = "dry_run_complete", 0
    else:
        coverage = load_json(output_root / "coverage_summary.json")
        status, code = coverage["status"], coverage["returncode"]
        manifest["coverage"] = coverage
        manifest["final_quality"] = coverage.get("final_quality", {})
        if status == "failed":
            manifest["failed_stage"] = "audit_case_coverage"

    manifest.update(status=status, returncode=code, finished_at=now())
    write_json(manifest_path, manifest)
    print(f"\nPipeline status: {manifest['status']}")
    print(f"Final annotations: {final_annotations}")
    print(f"Manifest: {manifest_path}")
    return code


if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from unified_output import finish, jsonl
    result = main()
    options = parser().parse_args()
    root = Path(options.output_root).resolve()
    manifest = load_json(root / "run_manifest.json")
    coverage = manifest.get("coverage", {})
    issues = list(coverage.get("problems", []))
    if manifest.get("status") == "failed":
        issues.append(manifest.get("error", "Pipeline or coverage audit failed"))
    result = finish(root, "Agent3Sigma", "unsafe", jsonl(root / "case_outcomes.jsonl"),
                    problems=issues, dry_run=options.dry_run, manifest=manifest)
    coverage_path = root / "coverage_summary.json"
    if coverage_path.exists():
        legacy = load_json(coverage_path)
        quality = load_json(root / "quality_summary.json")
        legacy.update(status=quality["status"], returncode=result,
                      quality_summary=str(root / "quality_summary.json"))
        write_json(coverage_path, legacy)
    raise SystemExit(result)
