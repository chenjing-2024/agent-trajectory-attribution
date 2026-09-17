"""Independent Skill output contract v1; keep the three bundled copies identical."""
import json
from collections import Counter
from pathlib import Path

OUTCOMES = ("finalized", "excluded", "needs_review", "failed")
LABELS = {"ok": "完成", "no_eligible_cases": "无符合条件的案例", "completed_with_review": "需要复核",
          "failed": "存在失败", "dry_run_complete": "演练完成，未生成标注"}

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def jsonl(path, required=False):
    path = Path(path)
    if not path.exists() and not required:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def index(records, key, problems):
    result = {}
    duplicates = set()
    for row in records:
        identity = row.get(key)
        if not isinstance(identity, str) or not identity:
            problems.append("Missing identity in evidence: " + key)
        elif identity in result:
            duplicates.add(identity)
            problems.append("Duplicate evidence: " + identity)
        else:
            result[identity] = row
    for identity in duplicates:
        result.pop(identity, None)
    return result

def scan(directory, problems, pattern="*.json"):
    records = []
    for path in sorted(Path(directory).rglob(pattern)):
        try:
            row = read(path)
            if not isinstance(row, dict):
                raise ValueError("Expected object")
            records.append({**row, "_artifact": str(path)})
        except Exception as exc:
            problems.append(f"Invalid artifact {path}: {exc}")
    return records

def finish(root, skill, workflow, cases, *, problems=None, dry_run=False, manifest=None):
    root = Path(root)
    problems = list(problems or [])
    cases = list(cases)
    keys = set()
    for case in cases:
        key = case.get("case_key")
        if not key or key in keys:
            problems.append("Missing or duplicate selected case key")
        keys.add(key)
        if case.get("outcome") not in OUTCOMES:
            case.update(outcome="failed", reason="unaccounted_case")
    counts = {name: 0 for name in OUTCOMES}
    counts.update(Counter(case["outcome"] for case in cases))
    if dry_run:
        status, code = "dry_run_complete", 0
        cases, counts = [], {name: 0 for name in OUTCOMES}
    elif problems or counts["failed"]:
        status, code = "failed", 2
    elif counts["needs_review"]:
        status, code = "completed_with_review", 1
    elif counts["finalized"]:
        status, code = "ok", 0
    else:
        status, code = "no_eligible_cases", 0
    summary = {"schema_version": "annotation_quality_v1", "skill": skill, "workflow": workflow,
        "status": status, "status_zh": LABELS[status], "returncode": code,
        "selected": None if dry_run else len(cases), "counts": counts, "problems": problems,
        "case_outcomes": str(root / "case_outcomes.jsonl"), "dry_run": dry_run}
    root.mkdir(parents=True, exist_ok=True)
    (root / "case_outcomes.jsonl").write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    write(root / "quality_summary.json", summary)
    if manifest is None:
        path = root / "run_manifest.json"
        manifest = read(path) if path.exists() else {}
    manifest = dict(manifest)
    if manifest.get("schema_version") not in {None, "annotation_run_v1"}:
        manifest.setdefault("source_schema_version", manifest["schema_version"])
    manifest.update(schema_version="annotation_run_v1", skill=skill, workflow=workflow,
                    status=status, returncode=code, case_outcomes=str(root / "case_outcomes.jsonl"),
                    quality_summary=str(root / "quality_summary.json"))
    write(root / "run_manifest.json", manifest)
    print(f"\nSkill：{skill}\n工作流：{workflow}")
    if not dry_run:
        print(f"本轮 {len(cases)} 条：完成 {counts['finalized']} 条，明确排除 {counts['excluded']} 条，待复核 {counts['needs_review']} 条，失败 {counts['failed']} 条。")
    print(f"运行结果：{LABELS[status]}。")
    if problems:
        print("运行问题：" + str(problems[0]))
    print(f"结果目录：{root}\n逐案例结果及原因：{root / 'case_outcomes.jsonl'}\n完整运行报告：{root / 'run_manifest.json'}")
    return code

def select_directory(source, destination, limit=None, security=None):
    """Snapshot screened inputs once, preserving relative paths for annotations."""
    source, destination = Path(source), Path(destination)
    candidates = []
    for path in sorted(source.rglob("*.json")):
        error = None
        try:
            value = read(path)
            if not isinstance(value, dict):
                raise ValueError("Expected input object")
            if security is not None and value.get("security") is not security:
                continue
        except Exception as exc:
            value, error = None, str(exc)
        candidates.append((path, value, error))
    eligible_count = len(candidates)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        candidates = candidates[:limit]
    selected = []
    destination.mkdir(parents=True, exist_ok=False)
    for path, value, error in candidates:
        rel = path.relative_to(source)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        selected.append({"case_key": rel.as_posix(), "source": str(path), "snapshot": str(target), "input_error": error})
    write(destination.parent / "selected_cases.json", {"schema_version": "annotation_selection_v1", "source": str(source), "limit": limit,
          "security": security, "eligible_count": eligible_count, "selected_count": len(selected), "cases": selected})
    return selected

def select_ids(source, destination, *, raw=False, limit=None, requested=None):
    """Snapshot raw or normalized input by stable ID, recording invalid rows too."""
    source, destination = Path(source), Path(destination)
    files = [source] if source.is_file() else sorted(source.rglob('*.json'))
    if raw and source.is_dir():
        trajectories = [p for p in files if p.name == 'trajectory.json']
        files = trajectories or files
    candidates = []
    for path in files:
        try:
            value = read(path)
            values = value['results'] if raw and isinstance(value, dict) and isinstance(value.get('results'), list) else [value]
            for i, case in enumerate(values):
                origin = str(path) + (f'#results[{i}]' if len(values) != 1 else '')
                try:
                    tid = (case.get('item') or {}).get('id') if raw else case.get('trajectory_id')
                    candidates.append((origin, tid, case, None))
                except Exception as exc:
                    candidates.append((origin, None, case, str(exc)))
        except Exception as exc:
            candidates.append((str(path), None, None, str(exc)))
    if requested is not None:
        by_id = {}
        for row in candidates:
            if isinstance(row[1], str):
                by_id.setdefault(row[1], []).append(row)
        candidates = [row for tid in requested for row in
                      (by_id.get(tid) or [(str(source), tid, None, 'Requested input not found')]
                       if isinstance(tid, str) else [(str(source), None, None, 'Invalid requested trajectory_id')])]
    eligible_count = len(candidates)
    if limit is not None:
        if limit <= 0:
            raise ValueError('--limit must be positive')
        candidates = candidates[:limit]
    records = []
    seen = Counter(tid for _, tid, _, _ in candidates if isinstance(tid, str))
    destination.mkdir(parents=True, exist_ok=False)
    for i, (origin, tid, value, error) in enumerate(candidates):
        if not isinstance(tid, str) or not tid.strip():
            error, tid = error or 'Missing valid trajectory_id', None
        elif seen[tid] > 1:
            error = 'Duplicate trajectory_id'
        snapshot = destination / f'{i:06d}.json'
        if value is not None:
            write(snapshot, value)
        records.append({'case_key': f'case-{i:06d}', 'trajectory_id': tid, 'source': origin,
                        'snapshot': str(snapshot), 'input_error': error})
    write(destination.parent / 'selected_cases.json', {'schema_version': 'annotation_selection_v1', 'source': str(source), 'limit': limit,
          'input_format': 'raw' if raw else 'normalized', 'eligible_count': eligible_count, 'selected_count': len(records), 'cases': records})
    return records
