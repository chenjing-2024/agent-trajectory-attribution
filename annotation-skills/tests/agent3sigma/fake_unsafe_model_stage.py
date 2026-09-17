"""Substitute model outputs only; all local pipeline stages execute normally."""
import json
import os
from pathlib import Path
import subprocess
import sys

script, *args = sys.argv[1:]
name = Path(script).name
mode = os.environ.get("UNSAFE_TEST_MODE", "pass")

def option(key):
    return Path(args[args.index(key) + 1])

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

if name.startswith("annotate_"):
    if mode == "interrupted":
        raise SystemExit(9)
    source = option("--input_dir") if "--input_dir" in args else option("--input_root") / "cases"
    stage = name.removeprefix("annotate_").removesuffix(".py")
    for path in source.glob("*.json"):
        tid = json.loads(path.read_text())["trajectory_id"]
        value = {"trajectory_id": tid, "confidence": 0.9, "needs_review": False}
        if stage == "target":
            value.update(target_status="no_target", attack_success=False, target_component_id=None, target_unsafe_action=None)
        elif stage == "primary":
            value.update(primary_component_id=None, primary_attribution_component=None)
        else:
            value[stage] = []
        write(option("--output_root") / "cases" / (path.stem + "." + stage + ".json"), value)
elif name == "validate_semantic.py":
    out = option("--output_root")
    final = out.name == "11_final_semantic"
    status = "uncertain" if final and mode == "review" else "pass"
    records = [{"trajectory_id": json.loads(p.read_text())["trajectory_id"],
                "semantic_status": status, "recommended_action": "keep_annotation"}
               for p in option("--annotations_dir").glob("*.json")]
    out.mkdir(parents=True, exist_ok=True)
    (out / "all_cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    if not (final and mode == "missing_report"):
        write(out / "semantic_validation_report.json", {"summary": {
            "num_cases_seen": len(records), **{"num_" + s: len(records) if status == s else 0
                for s in ("pass", "uncertain", "fail", "validator_error")}}})
else:
    raise SystemExit(subprocess.run([sys.executable, script, *args]).returncode)
