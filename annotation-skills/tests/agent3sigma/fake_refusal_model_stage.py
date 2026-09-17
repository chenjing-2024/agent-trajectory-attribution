"""Offline model-stage fixture; execute real input and validation scripts."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

script, *args = sys.argv[1:]
def option(name):
    return Path(args[args.index(name) + 1])
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

fixture = json.loads(Path(os.environ["REFUSAL_TEST_FIXTURE"]).read_text())
name = Path(script).name
if name not in {"annotate_target.py", "annotate_primary.py"}:
    raise SystemExit(subprocess.run([sys.executable, script, *args]).returncode)
out = option("--output_dir")
target = {"trajectory_id": "example", "target_component_id": "C2",
          "target_summary": "Synthetic refusal", "label": "safety_refusal",
          "needs_review": fixture.get("target_review", False),
          "review_reason": "Target requires review" if fixture.get("target_review") else None}
if name == "annotate_target.py":
    refused = 0 if fixture.get("no_target") else 1
    if not refused:
        target.update(label="no_target", target_component_id=None)
    write(out / "cases/example.json", target)
    write(out / "safety_refusal_annotations.json", [target] if refused else [])
    write(out / "summary.json", {"saved_annotations": 1, "failed_this_run": fixture.get("target_failed", 0),
                                  "needs_review": int(target["needs_review"]),
                                  "label_counts": {"safety_refusal": refused, "no_target": 1-refused}})
else:
    # Import the production record builder without an SDK or any possible API call.
    fake = types.ModuleType("openai")
    def forbidden(*args, **kwargs):
        raise AssertionError("No model clients may be constructed in this test")
    fake.OpenAI = forbidden
    sys.modules["openai"] = fake
    namespace = runpy.run_path(script)
    primary = {"component_id": "C99" if fixture.get("invalid") else "C1", "role": "user",
               "summary": "Synthetic cause", "assistant_generated_cause": False, "confidence": 0.5,
               "needs_review": fixture.get("primary_review", False),
               "review_reason": "Primary requires review" if fixture.get("primary_review") else None}
    record = namespace["build_output_record"]("example", next(option("--data_dir").glob("*.json")),
                                             option("--target_annotations"), target, primary, "fixture-model")
    write(out / "cases/example.json", record)
    write(out / "summary.json", {"selected_safety_refusal_targets": 1, "processed": 1, "failed": 0,
                                  "needs_review": int(record["needs_review"])})
