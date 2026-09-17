"""Per-case evidence adapters for the two independent AgentDojo workflows."""
from pathlib import Path
from unified_output import read, write, jsonl, index, scan, finish


def report(root, workflow, error=None, dry_run=False):
    root = Path(root)
    issues = [str(error)] if error else []
    manifest_path = root / ('full_run_manifest.json' if workflow == 'unsafe' else 'run_manifest.json')
    manifest = read(manifest_path) if manifest_path.exists() else {}
    if dry_run:
        return finish(root, 'AgentDojo', workflow, [], dry_run=not error, problems=issues, manifest=manifest)
    selected_path = root / '00_selection/selected_cases.json'
    selected = read(selected_path)['cases'] if selected_path.exists() else []
    if not selected_path.exists():
        issues.append('Missing selected-case manifest')
    outcomes = []
    stage_errors = {}
    for path in (root / 'normalized_outcomes.jsonl', root / 'filtered_outcomes.jsonl', root / 'initial/components_outcomes.jsonl', root / 'annotations/_failed_annotations.jsonl'):
        for row in jsonl(path):
            if row.get('status') == 'failed' or path.name == '_failed_annotations.jsonl':
                stage_errors[row['source_path']] = row['reason']

    if workflow == 'unsafe':
        d = index(scan(root / 'validation/final_deterministic/case_validation', issues), 'relative_path', issues)
        s = index(jsonl(root / 'validation/final_semantic/all_cases.jsonl'), 'relative_path', issues)
        expected = {r['case_key'] for r in selected}
        if (set(d) | set(s)) - expected:
            issues.append('Unexpected final validation case IDs')
        if selected and not error:
            for path, records, key, total, statuses in [
                (root / 'validation/final_deterministic/validation_report.json', d, 'status', 'num_component_files', ('clean', 'ok_with_warnings', 'hard_error')),
                (root / 'validation/final_semantic/semantic_validation_report.json', s, 'semantic_status', 'num_cases_seen', ('pass', 'uncertain', 'fail', 'validator_error'))]:
                try:
                    counts = read(path)['summary']
                    if any(row.get(key) not in statuses for row in records.values()):
                        raise ValueError('Unknown validation status in case evidence')
                    if type(counts[total]) is not int or counts[total] != len(records):
                        raise ValueError('Report total differs from per-case evidence')
                    for status in statuses:
                        n = counts['num_' + status]
                        if type(n) is not int or n != sum(r.get(key) == status for r in records.values()):
                            raise ValueError('Report status counts differ from evidence')
                except Exception as exc:
                    issues.append(f'{path}: {exc}')
    else:
        skipped = {r['source_path']: r for r in jsonl(root / 'annotations/_skipped_annotations.jsonl')}
    for row in selected:
        key = row['case_key']
        case = {**row, 'outcome': 'failed', 'reason': 'unaccounted_case'}
        try:
            if row.get('input_error'):
                case['reason'] = row['input_error']
            elif workflow == 'unsafe':
                annotation = root / 'final_annotations' / key
                norm = root / 'initial/components' / key
                det, sem = d.get(key), s.get(key)
                case['evidence'] = {'deterministic': det, 'semantic': sem, 'normalized': str(norm)}
                if not norm.is_file():
                    case['reason'] = 'normalization_failed_or_missing'
                elif not annotation.is_file():
                    case['reason'] = 'missing_final_annotation'
                else:
                    ann = read(annotation)
                    ann = ann.get("annotation", ann)
                    case['annotation_path'] = str(annotation)
                    if not det:
                        case['reason'] = 'missing_final_deterministic_validation'
                    elif any(str(h).startswith(('component_invalid_json', 'component_missing_ids')) for h in det.get('hard_errors', [])):
                        case['reason'] = 'invalid_normalized_input'
                    elif det['status'] == 'hard_error':
                        case.update(outcome='needs_review', reason='unresolved_deterministic_errors')
                    elif not sem or sem.get('semantic_status') == 'validator_error':
                        case['reason'] = 'missing_or_failed_semantic_validation'
                    elif sem.get('semantic_status') != 'pass' or sem.get('needs_review') or ann.get('needs_review'):
                        case.update(outcome='needs_review', reason='final_annotation_or_validation_requires_review')
                    elif ann.get('attack_success') is False:
                        case.update(outcome='excluded', reason='validated_no_unsafe_action')
                    else:
                        case.update(outcome='finalized', reason='final_validations_passed')
            else:
                norm = root / 'normalized' / key
                filtered = root / 'filtered' / key
                ann = root / 'annotations' / key
                if not norm.is_file():
                    case['reason'] = 'normalization_failed_or_missing'
                elif not filtered.is_file():
                    value = read(norm)
                    n = value.get('num_components', len(value.get('trajectory', [])))
                    if n < manifest.get('min_components', 4):
                        case.update(outcome='excluded', reason='below_min_components')
                    else:
                        case['reason'] = 'filter_failed_or_missing'
                elif ann.is_file():
                    value = read(ann)
                    if value.get('needs_review'):
                        case.update(outcome='needs_review', reason='annotation_requires_review')
                    elif not value.get('target') or not value.get('attribution'):
                        case['reason'] = 'invalid_annotation_structure'
                    else:
                        case.update(outcome='finalized', reason='deterministic_task_constraints_satisfied')
                    case['annotation_path'] = str(ann)
                elif str(filtered) in skipped:
                    reason = skipped[str(filtered)]['reason']
                    outcome = 'needs_review' if reason in {'unverified_task_constraints', 'unverified_task_preconditions'} else 'excluded'
                    case.update(outcome=outcome, reason=reason)
                else:
                    case['reason'] = 'annotation_failed_or_missing'
        except Exception as exc:
            case.update(outcome='failed', reason=str(exc))
        if case['outcome'] == 'failed':
            for source in (row.get('snapshot'), str(root / 'normalized' / key), str(root / 'filtered' / key)):
                if source in stage_errors:
                    case['reason'] = stage_errors[source]
        outcomes.append(case)
    code = finish(root, 'AgentDojo', workflow, outcomes, problems=issues, manifest=manifest)
    if workflow == 'unsafe':
        write(manifest_path, read(root / 'run_manifest.json'))
    return code


def execute(callback, args, workflow):
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(f'Choose a new output root: {root}')
    error = None
    try:
        callback()
    except Exception as exc:
        error = exc
    try:
        return report(root, workflow, error, args.dry_run)
    except Exception as exc:
        # Preserve one failed disposition per known input even if evidence is corrupt.
        path = root / '00_selection/selected_cases.json'
        selected = read(path)['cases'] if path.exists() else []
        return finish(root, 'AgentDojo', workflow,
            [{**r, 'outcome': 'failed', 'reason': 'invalid_output_evidence'} for r in selected], problems=[str(exc), str(error)])
