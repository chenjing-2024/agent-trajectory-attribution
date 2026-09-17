#!/usr/bin/env python3
"""Record shell workflow plans without executing annotation commands."""

import json
from pathlib import Path
import shlex
import sys


def main():
    action, filename, *args = sys.argv[1:]
    path = Path(filename)
    if action == "init":
        workflow, input_root, model, config = args
        plan = {
            "schema_version": "canary_dry_run_v1",
            "status": "planning",
            "workflow": workflow,
            "input_root": input_root,
            "model": model,
            "config": config,
            "stages": [],
            "dependencies": [],
            "limitations": "Commands are planned, not executed. Intermediate data and model quality are not validated. Repair reannotation is recorded as one shell stage.",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(plan, stream, indent=2)
            stream.write("\n")
        return

    plan = json.loads(path.read_text())
    if action == "stage":
        title, *command = args
        for arg in command:
            if arg.endswith((".py", ".sh")) and not Path(arg).is_file():
                raise FileNotFoundError(f"Missing stage script: {arg}")
        for index, arg in enumerate(command[:-1]):
            if arg == "--base_url":
                command[index + 1] = "<redacted-base-url>"
        plan["stages"].append({"name": title, "command": command, "status": "planned"})
        print(f"[planned] {title}\n+ {shlex.join(command)}")
    elif action == "dependency":
        value, kind = args
        source = Path(value)
        exists = source.is_dir() if kind == "directory" else source.is_file()
        plan["dependencies"].append({"path": value, "kind": kind,
                                     "status": "exists" if exists else "deferred"})
    elif action == "finish":
        code = int(args[0])
        plan["status"] = "dry_run_complete" if code == 0 else "failed"
        plan["returncode"] = code
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from unified_output import finish
        finish(path.with_suffix(".outputs"), "Canary", plan["workflow"], [],
               dry_run=code == 0, problems=[] if code == 0 else ["Dry-run planning failed"], manifest=plan)
    else:
        raise ValueError(f"Unknown action: {action}")
    path.write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
