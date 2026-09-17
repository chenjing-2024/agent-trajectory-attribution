#!/usr/bin/env python3
"""Merge a complete annotation tree with a sparse repaired annotation tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def annotation_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.json")
        if "manifest" not in path.name and not path.name.endswith(".error.json")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the original annotation tree and replace matching files with "
            "repaired annotations. Inputs are never modified."
        )
    )
    parser.add_argument("--original_root", required=True)
    parser.add_argument("--repaired_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--components_root", help="Allow newly repaired annotations only for known component files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_root = Path(args.original_root).resolve()
    repaired_root = Path(args.repaired_root).resolve()
    output_root = Path(args.output_root).resolve()

    for label, path in (
        ("original_root", original_root),
        ("repaired_root", repaired_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    if output_root.exists():
        raise FileExistsError(
            f"output_root already exists; choose a new path: {output_root}"
        )

    original = annotation_files(original_root)
    repaired = annotation_files(repaired_root)
    original_rel = {path.relative_to(original_root) for path in original}
    repaired_rel = {path.relative_to(repaired_root) for path in repaired}
    component_root = Path(args.components_root).resolve() if args.components_root else None
    unmatched = sorted(rel for rel in repaired_rel - original_rel
                       if component_root is None or not (component_root / rel).is_file())
    if unmatched:
        preview = "\n".join(f"  - {path}" for path in unmatched[:20])
        raise ValueError(
            "Repaired files without matching originals:\n"
            f"{preview}"
            + ("\n  ..." if len(unmatched) > 20 else "")
        )

    shutil.copytree(original_root, output_root)
    manifest_path = output_root / "merge_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for relative_path in sorted(repaired_rel):
            repaired_path = repaired_root / relative_path
            output_path = output_root / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repaired_path, output_path)
            manifest.write(
                json.dumps(
                    {
                        "relative_path": str(relative_path),
                        "action": "replaced_with_repaired" if relative_path in original_rel else "added_from_repair",
                        "original_file": str(original_root / relative_path) if relative_path in original_rel else None,
                        "repaired_file": str(repaired_path),
                        "final_file": str(output_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    final_count = len(annotation_files(output_root))
    if final_count != len(original_rel | repaired_rel):
        raise RuntimeError(
            f"Final annotation count differs from expected union: {len(original_rel | repaired_rel)} -> {final_count}"
        )

    report = {
        "schema_version": "agentdojo_annotation_merge_v1",
        "original_root": str(original_root),
        "repaired_root": str(repaired_root),
        "output_root": str(output_root),
        "num_original_annotations": len(original),
        "num_repaired_annotations": len(repaired),
        "num_final_annotations": final_count,
        "manifest": str(manifest_path),
    }
    report_path = output_root / "merge_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
