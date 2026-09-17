#!/usr/bin/env python3
"""Run the deterministic AgentDojo task-alignment annotation pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(command: list[str], dry_run: bool) -> None:
    print("+", shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize task-alignment trajectories, filter by component count, "
            "and build deterministic task-alignment annotations."
        )
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--min_components", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write_skipped", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--task_constraints", help="Reviewed exact-argument specifications keyed by relative case path.")
    parser.add_argument("--python_bin", default=sys.executable)
    return parser.parse_args()


def pipeline_main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")

    normalized = output_root / "normalized"
    filtered = output_root / "filtered"
    annotations = output_root / "annotations"
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        from unified_output import select_directory
        select_directory(raw_root, output_root / "00_selection/input", security=False)
        raw_root = output_root / "00_selection/input"
        from unified_output import read
        if not read(output_root / "00_selection/selected_cases.json")["cases"]:
            return


    normalize_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "normalize.py"),
        "--src_root",
        str(raw_root),
        "--out_root",
        str(normalized),
    ]
    filter_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "filter_cases.py"),
        "--src_root",
        str(normalized),
        "--out_root",
        str(filtered),
        "--min_components",
        str(args.min_components),
    ]
    annotate_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "annotate_deterministic.py"),
        "--src_root",
        str(filtered),
        "--out_root",
        str(annotations),
    ]
    if args.task_constraints:
        constraints = Path(args.task_constraints).resolve()
        if not constraints.is_file():
            raise FileNotFoundError(constraints)
        annotate_cmd.extend(["--task_constraints", str(constraints)])

    if args.overwrite:
        filter_cmd.append("--overwrite")
        annotate_cmd.append("--overwrite")
    if True:  # Persist every explicit exclusion for the final case ledger.
        annotate_cmd.append("--write_skipped")

    for command in (normalize_cmd, filter_cmd, annotate_cmd):
        run(command, args.dry_run)

    quality = None
    status = "dry_run_complete"
    if not args.dry_run:
        quality = json.loads((annotations / "_annotation_summary.json").read_text())
        if quality.get("errors", 0):
            status = "completed_with_review"
        elif not quality.get("annotated_trajectories", 0):
            status = "no_eligible_cases"
        elif quality.get("skipped_trajectories", 0):
            status = "completed_with_skips"
        else:
            status = "ok"
    manifest = {
        "schema_version": "agentdojo_task_alignment_pipeline_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "annotation_summary": quality,
        "raw_root": str(raw_root),
        "normalized_root": str(normalized),
        "filtered_root": str(filtered),
        "annotations_root": str(annotations),
        "min_components": args.min_components,
        "stages": [
            {"name": name, "command": command,
             "status": "planned" if args.dry_run else "ok"}
            for name, command in (
                ("normalize", normalize_cmd),
                ("filter_cases", filter_cmd),
                ("annotate_deterministic", annotate_cmd),
            )
        ],
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from pipeline_output import execute
    args = parse_args()
    return execute(pipeline_main, args, "task_alignment")


if __name__ == "__main__":
    raise SystemExit(main())
