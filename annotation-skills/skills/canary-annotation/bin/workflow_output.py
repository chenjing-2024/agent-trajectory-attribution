#!/usr/bin/env python3
"""Canary dispatcher bridge to the common per-case output contract."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from unified_output import read, write, jsonl, scan, index, select_ids, finish


def envpath(name):
    return Path(os.environ[name])


def begin(root, command):
    full = command in {'unsafe-all', 'safety-refusal-all'}
    unsafe = command in {'unsafe-all', 'validate-final-unsafe'}
    if full:
        raw = os.environ.get('INPUT_FORMAT', 'normalized') == 'raw'
        source = envpath('RAW_INPUT' if raw else 'NORMALIZED_CASES')
        destination = root / ('raw' if raw else 'normalized') / 'cases'
        scope = 'full_input_cohort'
    else:
        raw = False
        source = envpath('TARGET_ROOT') / 'by_target_type' / ('unsafe_action' if unsafe else 'safety_refusal')
        scope = 'selected_target_cohort'
        if not source.is_dir():
            source = envpath('UNSAFE_FINAL_ROOT' if unsafe else 'SAFETY_REFUSAL_FINAL_ROOT') / 'cases'
            scope = 'provided_annotations_only'
        destination = root / 'selection/cases'
    manifest = {'schema_version': 'annotation_run_v1', 'skill': 'Canary', 'workflow': 'unsafe' if unsafe else 'safety_refusal',
                'command': command, 'scope': scope, 'source': str(source), 'status': 'running', 'stages': [],
                'paths': {key: value for key, value in os.environ.items() if key in {'NORMALIZED_ROOT','NORMALIZED_CASES','TARGET_ROOT','PRIMARY_UNSAFE_ROOT','PRIMARY_REFUSAL_ROOT','UNSAFE_MERGED_ROOT','UNSAFE_REPAIR_ROOT','UNSAFE_REANNOTATION_ROOT','UNSAFE_FINAL_ROOT','UNSAFE_FINAL_DETERMINISTIC_ROOT','UNSAFE_FINAL_SEMANTIC_ROOT','SAFETY_REFUSAL_MERGED_ROOT','SAFETY_REFUSAL_VALIDATION_ROOT','SAFETY_REFUSAL_FINAL_ROOT','SAFETY_REFUSAL_FINAL_VALIDATION_ROOT'}}}
    if full and not raw:
        manifest['paths'].update(NORMALIZED_ROOT=str(root / 'normalized'), NORMALIZED_CASES=str(root / 'normalized/cases'))
    write(root / 'run_manifest.json', manifest)
    if not source.exists():
        write(root / 'selected_cases.json', {'cases': []})
        raise FileNotFoundError(source)
    records = select_ids(source, destination, raw=raw)
    write(root / 'selected_cases.json', {'schema_version': 'annotation_selection_v1', 'cases': records})
    # A full run must not reuse old model artifacts; standalone validation may.
    if full:
        roots = ['TARGET_ROOT', 'UNSAFE_MERGED_ROOT', 'UNSAFE_REPAIR_ROOT', 'UNSAFE_REANNOTATION_ROOT', 'UNSAFE_FINAL_ROOT'] if unsafe else ['TARGET_ROOT', 'PRIMARY_REFUSAL_ROOT', 'SAFETY_REFUSAL_MERGED_ROOT']
        if raw:
            roots.append('NORMALIZED_ROOT')
        for name in roots:
            if envpath(name).exists():
                raise FileExistsError(f'Full runs require new output paths: {name}={envpath(name)}')


def final_report(root, command, code):
    manifest = read(root / 'run_manifest.json')
    problems = []
    stages = manifest.get('stages', [])
    quality_stop = code == 1 and stages and stages[-1]['name'].startswith('Assess final')
    if code and not quality_stop:
        problems.append(f'Pipeline interrupted (exit {code})' + (': ' + stages[-1]['name'] if stages else ''))
    for stage in stages:
        if stage['status'] == 'running':
            stage.update(status='needs_review' if quality_stop else 'failed', returncode=code)
    full = command in {'unsafe-all', 'safety-refusal-all'}
    unsafe = manifest['workflow'] == 'unsafe'
    selected = read(root / 'selected_cases.json')['cases']
    targets = index(scan(envpath('TARGET_ROOT') / 'cases', problems), 'trajectory_id', problems) if full else {}
    repaired_targets = index(scan(envpath('UNSAFE_REANNOTATION_ROOT') / 'target/cases', problems), 'trajectory_id', problems) if unsafe else {}
    annotation_root = envpath('UNSAFE_FINAL_ROOT') if unsafe else envpath('SAFETY_REFUSAL_MERGED_ROOT' if command in {'safety-refusal','safety-refusal-all'} else 'SAFETY_REFUSAL_FINAL_ROOT')
    annotations = index(scan(annotation_root / 'cases', problems), 'trajectory_id', problems)
    detroot = envpath('UNSAFE_FINAL_DETERMINISTIC_ROOT') if unsafe else envpath('SAFETY_REFUSAL_VALIDATION_ROOT' if command in {'safety-refusal','safety-refusal-all'} else 'SAFETY_REFUSAL_FINAL_VALIDATION_ROOT')
    det = index(jsonl(detroot / 'all_cases.jsonl'), 'trajectory_id', problems)
    sem = index(jsonl(envpath('UNSAFE_FINAL_SEMANTIC_ROOT') / 'all_cases.jsonl'), 'trajectory_id', problems) if unsafe else {}
    tasks = index(jsonl(envpath('UNSAFE_REPAIR_ROOT') / 'repair_tasks.jsonl'), 'trajectory_id', problems) if unsafe else {}
    expected = {r['trajectory_id'] for r in selected}
    for data in (annotations, det, sem, targets, repaired_targets):
        if set(data) - expected:
            problems.append('Unexpected case IDs in run artifacts')
    quality_path = detroot / 'final_quality.json'
    if annotations or quality_path.exists():
        try:
            quality = read(quality_path)
            if quality.get('schema_version') != 'canary_final_quality_v1' or quality.get('status') not in {'ok','no_eligible_cases','completed_with_review','failed'}:
                raise ValueError('Invalid final quality schema or status')
            if quality['status'] == 'failed':
                problems.append(quality.get('error', 'Final validator failed'))
            for name, records in [('deterministic', det)] + ([('semantic', sem)] if unsafe else []):
                summary = quality['summaries'][name]
                if summary['num_cases'] != len(records):
                    raise ValueError('Report total disagrees with case evidence')
                status_key = 'semantic_status' if name == 'semantic' else 'status'
                for status, count in summary['status_counts'].items():
                    if count != sum(r.get(status_key) == status for r in records.values()):
                        raise ValueError('Report status counts disagree with case evidence')
        except Exception as exc:
            problems.append(f'Missing or invalid final quality: {exc}')
    cases = []
    for row in selected:
        tid = row['trajectory_id']
        t = repaired_targets.get(tid) or targets.get(tid)
        a, d, s = annotations.get(tid), det.get(tid), sem.get(tid)
        case = {**row, 'outcome': 'failed', 'reason': row.get('input_error') or 'unaccounted_case',
                'evidence': {'target': t, 'deterministic': d, 'semantic': s, 'repair_task': tasks.get(tid)}}
        if not row.get('input_error'):
            if t and t.get('target_type') in {'unsafe_action','safety_refusal','no_target'} and t['target_type'] != ('unsafe_action' if unsafe else 'safety_refusal'):
                if a:
                    case['reason'] = 'final_annotation_conflicts_with_target_reclassification'
                else:
                    case.update(outcome='needs_review' if t.get('needs_review') else 'excluded', reason='target_classification_' + t['target_type'])
            elif not a:
                case['reason'] = tasks.get(tid, {}).get('reason', 'missing_final_annotation_or_classification')
            elif not d or (unsafe and not s):
                case['reason'] = 'missing_final_validation'
            elif d.get('status') == 'validator_error' or (s or {}).get('semantic_status') == 'validator_error':
                case['reason'] = 'validator_error'
            elif d.get('status') not in {'pass','warning'} or (unsafe and s.get('semantic_status') != 'pass') or a.get('needs_review') or (t or {}).get('needs_review'):
                case.update(outcome='needs_review', reason='annotation_or_validation_requires_review')
            else:
                case.update(outcome='finalized', reason='final_validations_passed')
            if a:
                case['annotation_path'] = a['_artifact']
        cases.append(case)
    return finish(root, 'Canary', manifest['workflow'], cases, problems=problems, manifest=manifest)


if __name__ == '__main__':
    mode, directory, command, *extra = sys.argv[1:]
    root = Path(directory)
    try:
        if mode == 'begin':
            begin(root, command)
        elif mode == 'stage':
            manifest = read(root / 'run_manifest.json')
            if extra[0] == 'start':
                manifest['stages'].append({'name': command, 'status': 'running'})
            else:
                manifest['stages'][-1].update(status='ok', returncode=0)
            write(root / 'run_manifest.json', manifest)
        elif mode == 'finish':
            raise SystemExit(final_report(root, command, int(extra[0])))
    except Exception as exc:
        selected_path = root / 'selected_cases.json'
        selected = read(selected_path)['cases'] if selected_path.exists() else []
        raise SystemExit(finish(root, 'Canary', 'unsafe' if 'unsafe' in command else 'safety_refusal',
            [{**r, 'outcome': 'failed', 'reason': 'pipeline_or_evidence_error'} for r in selected], problems=[str(exc)]))
