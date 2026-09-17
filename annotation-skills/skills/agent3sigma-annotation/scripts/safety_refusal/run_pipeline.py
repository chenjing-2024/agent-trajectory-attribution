#!/usr/bin/env python3
"""Run Agent3Sigma safety-refusal target and root-cause annotation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quality import summarize


SCRIPT_DIR = Path(__file__).resolve().parent


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def redact(command: list[str]) -> list[str]:
    value = list(command)
    for index, item in enumerate(value[:-1]):
        if item == "--base_url":
            value[index + 1] = "<redacted-base-url>"
    return value


def run(
    name: str,
    command: list[str],
    *,
    dry_run: bool,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    record = {
        "name": name,
        "command": redact(command),
        "status": "planned" if dry_run else "running",
        "started_at": now(),
    }
    manifest["stages"].append(record)
    save(manifest_path, manifest)
    print(f"\n[{name}]\n+ {shlex.join(command)}")
    if dry_run:
        record["finished_at"] = now()
        save(manifest_path, manifest)
        return
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        record.update(
            status="failed",
            returncode=exc.returncode,
            finished_at=now(),
        )
        manifest.update(status="failed", failed_stage=name, finished_at=now())
        save(manifest_path, manifest)
        raise
    record.update(status="ok", returncode=0, finished_at=now())
    save(manifest_path, manifest)


def pipeline_main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect optional review cases, annotate safety-refusal targets, "
            "annotate primary root causes, and validate them."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data_dir", help="Already normalized cases; IDs are preserved.")
    source.add_argument("--raw_input", help="Raw item/turns case, prepared directory, or detailed.json.")
    source.add_argument("--reviews_file")
    parser.add_argument(
        "--cases_dir",
        help="Normalized case directory required with --reviews_file.",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()
    requested_limit = args.limit

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output_root already exists; choose a new directory: {output_root}"
        )

    if args.reviews_file:
        reviews_file = Path(args.reviews_file).resolve()
        if not reviews_file.is_file():
            raise FileNotFoundError(f"reviews_file not found: {reviews_file}")
        if not args.cases_dir:
            parser.error("--cases_dir is required with --reviews_file")
        cases_dir = Path(args.cases_dir).resolve()
        if not cases_dir.is_dir():
            raise FileNotFoundError(f"cases_dir not found: {cases_dir}")
        data_dir = output_root / "00_selected_cases"
    elif args.raw_input:
        raw_input = Path(args.raw_input).resolve()
        if not raw_input.exists():
            raise FileNotFoundError(raw_input)
        data_dir = output_root / "00_normalized" / "cases"
    else:
        data_dir = Path(args.data_dir).resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

    output_root.mkdir(parents=True)
    if not args.dry_run:
        sys.path.insert(0, str(SCRIPT_DIR.parent))
        from unified_output import select_ids, read
        requested = None
        if args.reviews_file:
            entries = read(reviews_file)
            requested = [r.get("trajectory_id") if isinstance(r, dict) else None for r in entries]
        selection_input = raw_input if args.raw_input else cases_dir if args.reviews_file else data_dir
        selected = select_ids(selection_input, output_root / "00_selection/input", raw=bool(args.raw_input),
                              limit=args.limit, requested=requested)
        args.limit = None
        if args.raw_input:
            raw_input = output_root / "00_selection/input"
        else:
            data_dir = output_root / "00_selection/input"
        # Review requests are now selected by content ID, with missing IDs retained.
        args.reviews_file = None
    target = output_root / "01_target"
    primary = output_root / "02_primary"
    validation = output_root / "03_validation"
    manifest_path = output_root / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "agent3sigma_safety_refusal_annotation_pipeline_v1",
        "created_at": now(),
        "status": "dry_run" if args.dry_run else "running",
        "settings": {
            "model": args.model,
            "api_key_env": args.api_key_env,
            "limit": requested_limit,
            "uses_custom_base_url": bool(args.base_url),
        },
        "paths": {
            "data_dir": str(data_dir),
            "target_annotations": str(
                target / "safety_refusal_annotations.json"
            ),
            "primary_cases": str(primary / "cases"),
            "validation": str(validation),
        },
        "stages": [],
    }
    save(manifest_path, manifest)
    if not args.dry_run and not selected:
        manifest.update(status="no_eligible_cases", returncode=0)
        save(manifest_path, manifest)
        return 0

    stages: list[tuple[str, list[str]]] = []
    if args.reviews_file:
        stages.append(
            (
                "collect_review_cases",
                [
                    args.python_bin,
                    str(SCRIPT_DIR / "collect_review_cases.py"),
                    "--reviews_file",
                    str(reviews_file),
                    "--cases_dir",
                    str(cases_dir),
                    "--output_dir",
                    str(data_dir),
                ],
            )
        )

    prepare_script = SCRIPT_DIR.parent / "common" / "prepare_annotation_input.py"
    if args.raw_input:
        prepare_command = [args.python_bin, str(prepare_script), "--input", str(raw_input),
                           "--input_format", "raw", "--output_root", str(data_dir.parent)]
        stages.append(("normalize_raw_input", prepare_command))
    else:
        stages.append(("validate_normalized_input", [args.python_bin, str(prepare_script),
                       "--input", str(data_dir), "--input_format", "normalized", "--check_only"]))

    target_command = [
        args.python_bin,
        str(SCRIPT_DIR / "annotate_target.py"),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(target),
        "--model_id",
        args.model,
        "--api_key_env",
        args.api_key_env,
        "--temperature",
        str(args.temperature),
        "--max_retries",
        str(args.retries),
    ]
    if args.base_url:
        target_command.extend(["--base_url", args.base_url])
    if args.limit is not None:
        target_command.extend(["--limit", str(args.limit)])
    stages.append(("annotate_refusal_target", target_command))

    primary_command = [
        args.python_bin,
        str(SCRIPT_DIR / "annotate_primary.py"),
        "--data_dir",
        str(data_dir),
        "--target_annotations",
        str(target / "safety_refusal_annotations.json"),
        "--output_dir",
        str(primary),
        "--model_id",
        args.model,
        "--api_key_env",
        args.api_key_env,
        "--temperature",
        str(args.temperature),
        "--max_tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--max_retries",
        str(args.retries),
    ]
    if args.base_url:
        primary_command.extend(["--base_url", args.base_url])
    if args.limit is not None:
        primary_command.extend(["--limit", str(args.limit)])
    stages.append(("annotate_primary_root_cause", primary_command))

    stages.append(
        (
            "validate_primary_root_causes",
            [
                args.python_bin,
                str(SCRIPT_DIR / "validate_primary.py"),
                "--cases_dir",
                str(primary / "cases"),
                "--output_dir",
                str(validation),
                "--report_only",
            ],
        )
    )

    no_refusal_targets = False
    try:
        for name, command in stages:
            if no_refusal_targets and name in {"annotate_primary_root_cause", "validate_primary_root_causes"}:
                manifest["stages"].append({"name": name, "status": "skipped_no_refusal_targets"})
                continue
            run(name, command, dry_run=args.dry_run, manifest=manifest, manifest_path=manifest_path)
            if not args.dry_run and name == "annotate_refusal_target":
                target_summary = load(target / "summary.json")
                no_refusal_targets = target_summary["label_counts"]["safety_refusal"] == 0

        if args.dry_run:
            status, code = "dry_run_complete", 0
        else:
            target_summary = load(target / "summary.json")
            primary_summary = None if no_refusal_targets else load(primary / "summary.json")
            validation_summary = None if no_refusal_targets else load(validation / "summary.json")
            status, code, quality = summarize(target_summary, primary_summary, validation_summary)
            manifest["final_quality"] = quality
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), finished_at=now(), returncode=2)
        save(manifest_path, manifest)
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 2

    manifest.update(status=status, returncode=code, finished_at=now())
    save(manifest_path, manifest)
    print(f"\nPipeline status: {manifest['status']}")
    print(f"Primary annotations: {primary / 'cases'}")
    print(f"Manifest: {manifest_path}")
    return code


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        return pipeline_main()
    from output_report import report
    from unified_output import finish, read
    # Parse only output options here; the workflow parser owns all other arguments.
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output_root", required=True)
    output_parser.add_argument("--dry_run", action="store_true")
    options, _ = output_parser.parse_known_args()
    root = Path(options.output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Choose a new output root: {root}")
    try:
        pipeline_main()
        return report(root, dry_run=options.dry_run)
    except Exception as exc:
        selection = root / "00_selection/selected_cases.json"
        selected = read(selection)["cases"] if selection.exists() else []
        return finish(root, "Agent3Sigma", "safety_refusal",
                      [{**r, "outcome": "failed", "reason": "pipeline_or_evidence_error"} for r in selected],
                      problems=[str(exc)])


if __name__ == "__main__":
    raise SystemExit(main())
