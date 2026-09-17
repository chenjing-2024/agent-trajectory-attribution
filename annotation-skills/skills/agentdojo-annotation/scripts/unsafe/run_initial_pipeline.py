#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# Basic helpers
# ============================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def redact_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for index, value in enumerate(redacted[:-1]):
        if value == "--base_url":
            redacted[index + 1] = "<redacted-base-url>"
    return redacted


def count_json_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for p in root.rglob("*.json")
        if not p.name.endswith(".error.json")
    )


def count_error_json_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*.error.json"))


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def resolve_path(path: str | None, project_root: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    return p


# ============================================================
# Subprocess runner
# ============================================================

def run_stage(
    *,
    stage_name: str,
    cmd: list[str],
    cwd: Path,
    dry_run: bool,
    manifest: dict[str, Any],
) -> None:
    cmd_text = shell_join(cmd)
    manifest_cmd = redact_command(cmd)

    print()
    print("=" * 100)
    print(f"Stage: {stage_name}")
    print("=" * 100)
    print(cmd_text)

    record: dict[str, Any] = {
        "stage": stage_name,
        "command": manifest_cmd,
        "command_text": shell_join(manifest_cmd),
        "cwd": str(cwd),
        "start_time": now_iso(),
        "dry_run": dry_run,
    }

    if dry_run:
        record["status"] = "dry_run_skipped"
        record["returncode"] = None
        record["end_time"] = now_iso()
        record["elapsed_seconds"] = 0
        manifest["stages"].append(record)
        return

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    elapsed = time.time() - t0

    record["returncode"] = proc.returncode
    record["end_time"] = now_iso()
    record["elapsed_seconds"] = round(elapsed, 3)

    if proc.returncode == 0:
        record["status"] = "ok"
        manifest["stages"].append(record)
        return

    record["status"] = "failed"
    manifest["stages"].append(record)

    raise RuntimeError(
        f"Stage failed: {stage_name}\n"
        f"Return code: {proc.returncode}\n"
        f"Command: {cmd_text}"
    )


# ============================================================
# Commands
# ============================================================

def make_build_components_cmd(
    *,
    python_bin: str,
    script: Path,
    raw_root: Path,
    components_dir: Path,
    include_non_security_true: bool,
) -> list[str]:
    cmd = [
        python_bin,
        str(script),
        "--src_root",
        str(raw_root),
        "--out_root",
        str(components_dir),
    ]

    if include_non_security_true:
        cmd.append("--include_non_security_true")

    return cmd


def make_annotate_cmd(
    *,
    python_bin: str,
    script: Path,
    components_dir: Path,
    annotations_dir: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    limit: int | None,
    force: bool,
) -> list[str]:
    cmd = [
        python_bin,
        str(script),
        "--input_root",
        str(components_dir),
        "--output_root",
        str(annotations_dir),
        "--model",
        model,
        "--api_key_env",
        api_key_env,
    ]

    if base_url:
        cmd.extend(["--base_url", base_url])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if force:
        cmd.append("--force")

    return cmd


def make_validate_cmd(
    *,
    python_bin: str,
    script: Path,
    components_dir: Path,
    annotations_dir: Path,
    validation_dir: Path,
    limit: int | None,
) -> list[str]:
    cmd = [
        python_bin,
        str(script),
        "--components_root",
        str(components_dir),
        "--annotations_root",
        str(annotations_dir),
        "--output_dir",
        str(validation_dir),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return cmd


def make_summary_cmd(
    *,
    python_bin: str,
    script: Path,
    components_dir: Path,
    annotations_dir: Path,
    summary_dir: Path,
) -> list[str]:
    return [
        python_bin,
        str(script),
        "--components_root",
        str(components_dir),
        "--annotations_root",
        str(annotations_dir),
        "--output_dir",
        str(summary_dir),
    ]


# ============================================================
# Manifest
# ============================================================

def init_manifest(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    safe_args = vars(args).copy()
    if safe_args.get("base_url"):
        safe_args["base_url"] = "<redacted-base-url>"
    return {
        "skill_name": "agentdojo-annotation",
        "skill_version": "v1",
        "description": (
            "Initial unsafe annotation pipeline for AgentDojo trajectories. "
            "Outputs components, initial C1 annotations, deterministic "
            "validation, and an initial summary."
        ),
        "run_name": args.run_name,
        "suite_name": args.suite_name,
        "model": args.model,
        "project_root": str(project_root),
        "start_time": now_iso(),
        "end_time": None,
        "status": "running",
        "args": safe_args,
        "paths": {},
        "stages": [],
        "counts": {},
        "validation_summary": None,
    }


def finalize_manifest(
    *,
    manifest: dict[str, Any],
    status: str,
    output_root: Path,
    raw_root: Path,
    components_dir: Path,
    annotations_dir: Path,
    validation_dir: Path,
    summary_dir: Path,
) -> None:
    manifest["status"] = status
    manifest["end_time"] = now_iso()

    manifest["paths"] = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "components_dir": str(components_dir),
        "annotations_dir": str(annotations_dir),
        "validation_dir": str(validation_dir),
        "summary_dir": str(summary_dir),
        "validation_report": str(validation_dir / "validation_report.json"),
        "validation_errors": str(validation_dir / "validation_errors.jsonl"),
        "case_summary_jsonl": str(summary_dir / "case_summary.jsonl"),
        "case_summary_csv": str(summary_dir / "case_summary.csv"),
        "run_manifest": str(output_root / "run_manifest.json"),
    }

    manifest["counts"] = {
        "components_json_files": count_json_files(components_dir),
        "annotations_json_files": count_json_files(annotations_dir),
        "annotation_error_json_files": count_error_json_files(annotations_dir),
        "case_summary_rows": count_jsonl_lines(summary_dir / "case_summary.jsonl"),
    }

    report_path = validation_dir / "validation_report.json"
    if report_path.exists():
        try:
            report = load_json(report_path)
            manifest["validation_summary"] = report.get("summary", report)
        except Exception as e:
            manifest["validation_summary"] = {
                "error": f"Could not load validation report: {type(e).__name__}: {e}"
            }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run initial AgentDojo unsafe annotation: raw trajectories -> "
            "components -> initial C1 annotations -> deterministic validation "
            "-> initial case summary."
        )
    )

    parser.add_argument(
        "--raw_root",
        required=True,
        type=str,
        help="Input raw AgentDojo trajectory directory.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        type=str,
        help="Output root for this skill run.",
    )
    parser.add_argument(
        "--suite_name",
        default=None,
        type=str,
        help="Optional suite name, e.g. banking/workspace/travel/slack.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Optional run name stored in run_manifest.json.",
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Annotation model name.",
    )
    parser.add_argument(
        "--api_key_env",
        default="OPENAI_API_KEY",
        type=str,
        help="Environment variable containing API key.",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        type=str,
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Optional limit for annotation stage. Useful for testing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to annotate_trajectory_c1.py.",
    )

    parser.add_argument(
        "--include_non_security_true",
        action="store_true",
        help=(
            "If set, componentize all raw trajectories. "
            "By default build_action_result_components.py keeps only security=true trajectories."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and write manifest without executing.",
    )

    parser.add_argument(
        "--project_root",
        default=str(Path(__file__).resolve().parents[2]),
        type=str,
        help="Skill root. Defaults to the agentdojo-annotation directory.",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        type=str,
        help="Python executable used for subprocess stages.",
    )

    parser.add_argument(
        "--build_components_script",
        default="scripts/unsafe/build_components.py",
        type=str,
        help="Path to build_action_result_components.py relative to project_root.",
    )
    parser.add_argument(
        "--annotate_script",
        default="scripts/unsafe/annotate.py",
        type=str,
        help="Path to annotate_trajectory_c1.py relative to project_root.",
    )
    parser.add_argument(
        "--validate_script",
        default="scripts/unsafe/validate_deterministic.py",
        type=str,
        help="Path to validate_c1_annotations.py relative to project_root.",
    )
    parser.add_argument(
        "--build_summary_script",
        default="scripts/unsafe/build_summary.py",
        type=str,
        help="Path to build_case_summary.py relative to project_root.",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    ensure_exists(project_root, "project_root")

    raw_root = resolve_path(args.raw_root, project_root)
    output_root = resolve_path(args.output_root, project_root)

    assert raw_root is not None
    assert output_root is not None

    ensure_exists(raw_root, "raw_root")
    output_root.mkdir(parents=True, exist_ok=True)

    build_components_script = resolve_path(args.build_components_script, project_root)
    annotate_script = resolve_path(args.annotate_script, project_root)
    validate_script = resolve_path(args.validate_script, project_root)
    build_summary_script = resolve_path(args.build_summary_script, project_root)

    assert build_components_script is not None
    assert annotate_script is not None
    assert validate_script is not None
    assert build_summary_script is not None

    ensure_exists(build_components_script, "build_components_script")
    ensure_exists(annotate_script, "annotate_script")
    ensure_exists(validate_script, "validate_script")
    ensure_exists(build_summary_script, "build_summary_script")

    components_dir = output_root / "components"
    annotations_dir = output_root / "annotations"
    validation_dir = output_root / "validation"
    summary_dir = output_root / "summary"

    manifest = init_manifest(args, project_root)
    manifest_path = output_root / "run_manifest.json"

    manifest["paths"] = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "components_dir": str(components_dir),
        "annotations_dir": str(annotations_dir),
        "validation_dir": str(validation_dir),
        "summary_dir": str(summary_dir),
    }

    write_json(manifest_path, manifest)

    try:
        # Stage 1: raw -> components
        run_stage(
            stage_name="build_action_result_components",
            cmd=make_build_components_cmd(
                python_bin=args.python_bin,
                script=build_components_script,
                raw_root=raw_root,
                components_dir=components_dir,
                include_non_security_true=args.include_non_security_true,
            ),
            cwd=project_root,
            dry_run=args.dry_run,
            manifest=manifest,
        )
        write_json(manifest_path, manifest)

        # Stage 2: components -> C1 annotations
        run_stage(
            stage_name="annotate_trajectory_c1",
            cmd=make_annotate_cmd(
                python_bin=args.python_bin,
                script=annotate_script,
                components_dir=components_dir,
                annotations_dir=annotations_dir,
                model=args.model,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                limit=args.limit,
                force=args.force,
            ),
            cwd=project_root,
            dry_run=args.dry_run,
            manifest=manifest,
        )
        write_json(manifest_path, manifest)

        # Stage 3: validate
        run_stage(
            stage_name="validate_c1_annotations",
            cmd=make_validate_cmd(
                python_bin=args.python_bin,
                script=validate_script,
                components_dir=components_dir,
                annotations_dir=annotations_dir,
                validation_dir=validation_dir,
                limit=args.limit,
            ),
            cwd=project_root,
            dry_run=args.dry_run,
            manifest=manifest,
        )
        write_json(manifest_path, manifest)

        # Stage 4: build summary
        run_stage(
            stage_name="build_case_summary",
            cmd=make_summary_cmd(
                python_bin=args.python_bin,
                script=build_summary_script,
                components_dir=components_dir,
                annotations_dir=annotations_dir,
                summary_dir=summary_dir,
            ),
            cwd=project_root,
            dry_run=args.dry_run,
            manifest=manifest,
        )

        finalize_manifest(
            manifest=manifest,
            status="dry_run_complete" if args.dry_run else "ok",
            output_root=output_root,
            raw_root=raw_root,
            components_dir=components_dir,
            annotations_dir=annotations_dir,
            validation_dir=validation_dir,
            summary_dir=summary_dir,
        )
        write_json(manifest_path, manifest)

        print()
        print("=" * 100)
        print("AgentDojo atom annotation skill complete.")
        print("=" * 100)
        print(f"Status:              {manifest['status']}")
        print(f"Raw root:            {raw_root}")
        print(f"Output root:         {output_root}")
        print(f"Components:          {components_dir}")
        print(f"Annotations:         {annotations_dir}")
        print(f"Validation:          {validation_dir}")
        print(f"Summary:             {summary_dir}")
        print(f"Manifest:            {manifest_path}")
        print()
        print("Counts:")
        for k, v in manifest["counts"].items():
            print(f"  {k}: {v}")

        if manifest.get("validation_summary"):
            print()
            print("Validation summary:")
            for k, v in manifest["validation_summary"].items():
                print(f"  {k}: {v}")

    except Exception as e:
        finalize_manifest(
            manifest=manifest,
            status="failed",
            output_root=output_root,
            raw_root=raw_root,
            components_dir=components_dir,
            annotations_dir=annotations_dir,
            validation_dir=validation_dir,
            summary_dir=summary_dir,
        )
        manifest["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }
        write_json(manifest_path, manifest)

        print()
        print("=" * 100)
        print("AgentDojo atom annotation skill failed.")
        print("=" * 100)
        print(f"Error: {type(e).__name__}: {e}")
        print(f"Manifest: {manifest_path}")
        raise


if __name__ == "__main__":
    main()
