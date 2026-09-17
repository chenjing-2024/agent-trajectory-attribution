"""Refusal output audit against the selected inputs, including no-target cases."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_output import read, scan, index, finish


def report(root, dry_run=False):
    root = Path(root)
    manifest = read(root / 'run_manifest.json')
    problems = [manifest.get('error', 'Pipeline failed')] if manifest.get('status') == 'failed' else []
    if dry_run:
        return finish(root, 'Agent3Sigma', 'safety_refusal', [], dry_run=True, manifest=manifest)
    selected = read(root / '00_selection/selected_cases.json')['cases']
    targets = index(scan(root / '01_target/cases', problems), 'trajectory_id', problems)
    primary = index(scan(root / '02_primary/cases', problems), 'trajectory_id', problems)
    checks = index(scan(root / '03_validation/pass', problems) + scan(root / '03_validation/review', problems), 'trajectory_id', problems)
    expected = {r['trajectory_id'] for r in selected}
    for records in (targets, primary, checks):
        if set(records) - expected:
            problems.append('Unexpected trajectory ID in output')
    cases = []
    for row in selected:
        tid = row['trajectory_id']
        t, p, v = targets.get(tid), primary.get(tid), checks.get(tid)
        case = {**row, 'outcome': 'failed', 'reason': row.get('input_error') or 'missing_target_annotation',
                'evidence': {'target': t, 'primary': p, 'validation': v}}
        if not row.get('input_error') and t:
            if t.get('label') == 'no_target':
                case.update(outcome='needs_review' if t.get('needs_review') else 'excluded',
                            reason=t.get('review_reason') or 'no_refusal_target')
            elif t.get('label') != 'safety_refusal':
                case['reason'] = 'invalid_target_label'
            elif not p or not v:
                case['reason'] = 'missing_primary_or_validation'
            elif t.get('needs_review') or p.get('needs_review') or v.get('status') == 'review':
                case.update(outcome='needs_review', reason='target_primary_or_validation_requires_review')
            elif v.get('status') == 'pass':
                case.update(outcome='finalized', reason='refusal_validation_passed')
            else:
                case['reason'] = 'invalid_validation_status'
            case['annotation_path'] = (p or t).get('_artifact')
        cases.append(case)
    return finish(root, 'Agent3Sigma', 'safety_refusal', cases, problems=problems, manifest=manifest)
