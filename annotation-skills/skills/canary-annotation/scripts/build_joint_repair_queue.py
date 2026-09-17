#!/usr/bin/env python3
"""Union deterministic and semantic findings; keep hard errors out of pass input."""
import argparse
import json
from pathlib import Path
import shutil
from unified_output import read, write, scan, index, jsonl


def build(targets, normalized, annotations, deterministic, semantic, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    problems = []
    target = index(scan(targets, problems), 'trajectory_id', problems)
    norm = index(scan(normalized, problems), 'trajectory_id', problems)
    ann = index(scan(annotations, problems), 'trajectory_id', problems)
    det = index(jsonl(deterministic, required=True), 'trajectory_id', problems)
    sem = index(jsonl(semantic, required=True), 'trajectory_id', problems)
    if (set(ann) | set(det) | set(sem)) - set(target):
        problems.append('Unexpected unsafe annotation or validation IDs')
    if problems:
        write(output / 'queue_errors.json', problems)
        raise ValueError('; '.join(problems))
    for folder in ('accepted_cases', 'normalized_cases'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    tasks, executable = [], []
    for tid, t in target.items():
        d, s, a, n = det.get(tid, {}), sem.get(tid, {}), ann.get(tid), norm.get(tid)
        hard = d.get('hard_errors', [])
        task = {**s, 'trajectory_id': tid, 'case_id': Path(n['_artifact']).stem if n else tid,
                'deterministic': d, 'semantic': s, 'hard_errors': hard}
        if not n or d.get('status') == 'validator_error' or any(str(e).startswith(('normalized_', 'trajectory_id_mismatch')) for e in hard):
            task.update(recommended_action='blocked', disposition='failed', reason='missing_or_invalid_normalized_input')
        elif s.get('semantic_status') == 'validator_error' and not hard:
            task.update(recommended_action='blocked', disposition='failed', reason='semantic_validator_error')
        elif a and d.get('status') in {'pass', 'warning'} and not hard and s.get('semantic_status') == 'pass' and s.get('recommended_action') == 'keep_annotation' and not a.get('needs_review'):
            task.update(recommended_action='keep_annotation', reason='both_validators_passed')
            shutil.copy2(a['_artifact'], output / 'accepted_cases' / Path(a['_artifact']).name)
        else:
            task.update(recommended_action='rerun_all_annotations', reason='deterministic_or_semantic_repair',
                semantic_status=s.get('semantic_status', 'uncertain'),
                files={'annotation': a['_artifact'] if a else None, 'normalized': n['_artifact']})
            executable.append(task)
        tasks.append(task)
    for name, records in [('repair_tasks.jsonl', tasks), ('joint_cases.jsonl', executable)]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    write(output / 'queue_summary.json', {'selected': len(target), 'executable': len(executable),
         'blocked': sum(t['recommended_action'] == 'blocked' for t in tasks)})
    return tasks


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('targets','normalized','annotations','deterministic','semantic','output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    build(a.targets,a.normalized,a.annotations,a.deterministic,a.semantic,a.output)
