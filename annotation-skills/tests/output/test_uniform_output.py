import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / 'skills'
sys.path.insert(0, str(SKILLS / 'agentdojo-annotation/scripts'))
from unified_output import finish, read, write, select_ids


class UniformOutput(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_independent_bundles_share_contract(self):
        copies = [p.read_bytes() for p in SKILLS.glob('*/scripts/unified_output.py')]
        self.assertEqual(len(copies), 3)
        self.assertEqual(len(set(copies)), 1)

    def test_counts_status_and_unknown_failure(self):
        scenarios = [(['finalized'], 'ok', 0), (['excluded'], 'no_eligible_cases', 0),
                     (['finalized','needs_review'], 'completed_with_review', 1),
                     (['failed','needs_review'], 'failed', 2), (['unknown'], 'failed', 2), ([], 'no_eligible_cases', 0)]
        for i, (states, status, expected) in enumerate(scenarios):
            with self.subTest(states=states), contextlib.redirect_stdout(io.StringIO()) as stdout:
                out = self.root / str(i)
                code = finish(out, 'Synthetic', 'test', [{'case_key': str(n), 'outcome': s, 'reason': 'fixture'} for n,s in enumerate(states)])
                summary = read(out / 'quality_summary.json')
                self.assertEqual((code, summary['status']), (expected, status))
                self.assertEqual(summary['selected'], sum(summary['counts'].values()))
                self.assertEqual(read(out / 'run_manifest.json')['returncode'], code)
                self.assertIn('运行结果：', stdout.getvalue())

    def test_dry_run_does_not_count_annotations(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = finish(self.root, 'Synthetic', 'test', [], dry_run=True)
        summary = read(self.root / 'quality_summary.json')
        self.assertEqual(code, 0)
        self.assertIsNone(summary['selected'])
        self.assertIn('未生成标注', output.getvalue())
        self.assertNotIn('本轮 0 条', output.getvalue())

    def test_raw_malformed_record_does_not_drop_following_case(self):
        source = self.root / 'raw.json'
        write(source, {'results': [{'item': {'id':'a'}}, 3, {'item': {'id':'b'}}]})
        rows = select_ids(source, self.root / 'selection/cases', raw=True)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]['trajectory_id'], 'b')
        self.assertTrue(rows[1]['input_error'])

    def test_task_alignment_mixed_input_errors_exclusions_and_success(self):
        source = self.root / 'raw'
        shutil.copytree(ROOT / 'examples/task-alignment/raw', source)
        good = read(source / 'workspace/example.json')
        write(source / 'short.json', {**good, 'messages': good['messages'][:1]})
        (source / 'broken.json').write_text('{broken')
        write(source / 'missing_messages.json', {'security': False})
        # Screened out before the cohort, rather than incorrectly counted as a failure.
        write(source / 'unsafe.json', {'security': True})
        out = self.root / 'output'
        r = subprocess.run([sys.executable, str(SKILLS / 'agentdojo-annotation/scripts/task_alignment/run_pipeline.py'),
            '--raw_root', str(source), '--output_root', str(out)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        summary = read(out / 'quality_summary.json')
        self.assertEqual(summary['counts'], {'finalized':1,'excluded':1,'needs_review':0,'failed':2})
        self.assertEqual(summary['selected'], 4)
        self.assertIn('本轮 4 条：完成 1 条，明确排除 1 条，待复核 0 条，失败 2 条。', r.stdout)
        self.assertEqual(read(out / 'run_manifest.json')['status'], 'failed')

    def test_task_alignment_all_excluded_and_empty(self):
        for name, value in [('excluded', {'security':False,'messages':[{'role':'user','content':'hi'}]}), ('empty', {'security':True})]:
            source = self.root / name
            write(source / 'a.json', value)
            out = self.root / (name + '_output')
            r = subprocess.run([sys.executable, str(SKILLS / 'agentdojo-annotation/scripts/task_alignment/run_pipeline.py'),
                '--raw_root', str(source), '--output_root', str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(read(out / 'quality_summary.json')['status'], 'no_eligible_cases')

    def test_dojo_final_hard_errors_missing_reports_and_input_failures(self):
        from pipeline_output import report
        for scenario, expected in [('pass',0),('hard',1),('missing_report',2),('missing_annotation',2),('missing_normalized',2)]:
            out = self.root / scenario
            write(out / '00_selection/selected_cases.json', {'cases':[{'case_key':'a.json','source':'synthetic','input_error':None}]})
            if scenario != 'missing_normalized':
                write(out / 'initial/components/a.json', {'trajectory':[]})
            if scenario != 'missing_annotation':
                write(out / 'final_annotations/a.json', {'annotation':{'attack_success':True}})
            d = 'hard_error' if scenario == 'hard' else 'clean'
            write(out / 'validation/final_deterministic/case_validation/a.json', {'relative_path':'a.json','status':d})
            write(out / 'validation/final_deterministic/validation_report.json', {'summary':{'num_component_files':1, 'num_clean':int(d=='clean'),'num_ok_with_warnings':0,'num_hard_error':int(d=='hard_error')}})
            sem = out / 'validation/final_semantic';sem.mkdir(parents=True)
            (sem / 'all_cases.jsonl').write_text(json.dumps({'relative_path':'a.json','semantic_status':'pass'})+'\n' if d=='clean' else '')
            if scenario != 'missing_report':
                write(sem / 'semantic_validation_report.json', {'summary':{'num_cases_seen':int(d=='clean'),'num_pass':int(d=='clean'),'num_uncertain':0,'num_fail':0,'num_validator_error':0}})
            with contextlib.redirect_stdout(io.StringIO()):
                code = report(out,'unsafe')
            self.assertEqual(code,expected,read(out / 'quality_summary.json'))


if __name__ == '__main__':
    unittest.main()
