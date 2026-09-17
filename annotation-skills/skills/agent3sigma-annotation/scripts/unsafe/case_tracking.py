"""Shared file contracts for unsafe input selection, repair routing, and audit."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def rows(path, required=False):
    path = Path(path)
    if not path.exists() and not required:
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def index_cases(directory):
    index, problems = {}, []
    for path in sorted(Path(directory).rglob("*.json")):
        try:
            value = read(path)
            tid = value.get("trajectory_id")
            if not isinstance(tid, str) or not tid:
                raise ValueError("Missing trajectory_id")
            index.setdefault(tid, []).append((path, value))
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            problems.append({"path": str(path), "error": str(exc)})
    for tid, matches in index.items():
        if len(matches) != 1:
            problems.append({"trajectory_id": tid, "error": "duplicate_id", "paths": [str(p) for p, _ in matches]})
    return {tid: matches[0] for tid, matches in index.items() if len(matches) == 1}, problems
