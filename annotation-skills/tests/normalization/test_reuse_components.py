import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DOJO = ROOT / 'skills/agentdojo-annotation/scripts'
SIGMA = ROOT / 'skills/agent3sigma-annotation/scripts'


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True)


def read(path):
    return json.loads(path.read_text())


class ReuseComponents(unittest.TestCase):
    def test_dojo_both_formats_roundtrip_without_changing_ids_or_content(self):
        for workflow, script in [('unsafe', 'unsafe/build_components.py'), ('task_alignment', 'task_alignment/normalize.py')]:
            with self.subTest(workflow=workflow), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'raw'; source.mkdir()
                raw = read(ROOT / 'examples/task-alignment/raw/workspace/example.json')
                raw['security'] = workflow == 'unsafe'
                (source / 'a.json').write_text(json.dumps(raw))
                for src, dest in [(source, root/'first'), (root/'first', root/'second')]:
                    result = run(DOJO/script, '--src_root', src, '--out_root', dest)
                    self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(read(root/'first/a.json'), read(root/'second/a.json'))
                ledger = read(root/'second_outcomes.jsonl')
                self.assertEqual(ledger['status'], 'reused')
                self.assertIn('[REUSE]', result.stdout)

    def test_task_full_run_reuses_components_and_rejects_corrupt_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first=root/'first'
            result=run(DOJO/'task_alignment/run_pipeline.py', '--raw_root', ROOT/'examples/task-alignment/raw', '--output_root', first)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            prepared=read(first/'normalized/workspace/example.json')
            second=root/'second'
            result=run(DOJO/'task_alignment/run_pipeline.py', '--raw_root', first/'normalized', '--output_root', second)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(read(second/'normalized/workspace/example.json'),prepared)
            self.assertEqual(read(second/'quality_summary.json')['counts']['finalized'],1)
            bad=root/'bad';bad.mkdir()
            prepared['components_metadata'][0]['component_id']=99
            (bad/'broken.json').write_text(json.dumps(prepared))
            out=root/'failed'
            result=run(DOJO/'task_alignment/run_pipeline.py','--raw_root',bad,'--output_root',out)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            self.assertEqual(read(out/'quality_summary.json')['counts']['failed'],1)
            self.assertFalse((out/'normalized/broken.json').exists())

    def test_dojo_incompatible_workflow_is_not_reinterpreted_as_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'raw';raw.mkdir()
            value=read(ROOT/'examples/task-alignment/raw/workspace/example.json')
            value['security']=True
            (raw/'a.json').write_text(json.dumps(value))
            run(DOJO/'unsafe/build_components.py','--src_root',raw,'--out_root',root/'unsafe')
            value=read(root/'unsafe/a.json');value['security']=False
            value['messages']=[{'role':'user','content':'must not normalize this fallback'}]
            (root/'unsafe/a.json').write_text(json.dumps(value))
            run(DOJO/'task_alignment/normalize.py','--src_root',root/'unsafe','--out_root',root/'rejected')
            self.assertFalse((root/'rejected/a.json').exists())
            self.assertEqual(read(root/'rejected_outcomes.jsonl')['status'],'failed')

    def test_sigma_normalized_selection_limit_validation_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'prepared';src.mkdir()
            case={'trajectory_id':'example','trajectory':[{'role':'user','content':'keep exact text','component_id':'C7'}]}
            (src/'a.json').write_text(json.dumps(case))
            (src/'b.json').write_text(json.dumps(case)) # duplicate beyond this run's limit
            selection=root/'selection'
            result=run(SIGMA/'unsafe/select_run_cases.py','--input',src,'--output_root',selection,'--input_format','normalized','--limit',1)
            self.assertEqual(result.returncode,0,result.stderr)
            result=run(SIGMA/'common/prepare_annotation_input.py','--input',selection/'cases','--input_format','normalized','--output_root',root/'normalized')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(read(root/'normalized/cases/example.json'),case)
            result=run(SIGMA/'unsafe/select_run_cases.py','--input',src,'--output_root',root/'duplicate','--input_format','normalized')
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(len(read(root/'duplicate/selected_cases.json')['cases']),2)
            case['trajectory'][0]['component_id']='invalid'
            (src/'a.json').write_text(json.dumps(case))
            result=run(SIGMA/'unsafe/select_run_cases.py','--input',src,'--output_root',root/'invalid','--input_format','normalized','--limit',1)
            self.assertNotEqual(result.returncode,0)
            self.assertIsNotNone(read(root/'invalid/selected_cases.json')['cases'][0]['input_error'])

    def test_sigma_normalized_full_run_uses_real_local_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'input';src.mkdir()
            prepare=run(SIGMA/'common/prepare_annotation_input.py','--input',ROOT/'examples/raw-turns.json','--input_format','raw','--output_root',root/'prepared')
            self.assertEqual(prepare.returncode,0,prepare.stderr)
            fixture=ROOT/'tests/agent3sigma/fake_unsafe_model_stage.py'
            shim=root/'python';shim.write_text('#!'+sys.executable+'\n'+fixture.read_text());shim.chmod(0o755)
            out=root/'out'
            result=run(SIGMA/'unsafe/run_full_pipeline.py','--input',root/'prepared/cases','--input_format','normalized','--output_root',out,'--model','fixture','--base_url','https://example.invalid','--python_bin',shim)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            originals={read(p)['trajectory_id']:read(p) for p in (root/'prepared/cases').glob('*.json')}
            actual={read(p)['trajectory_id']:read(p) for p in (out/'01_normalized/cases').glob('*.json')}
            self.assertEqual(actual,originals)
            self.assertEqual(read(out/'quality_summary.json')['selected'],len(originals))

    def test_sigma_full_dry_plan_replaces_normalization_with_validated_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'input';src.mkdir()
            result=run(SIGMA/'unsafe/run_full_pipeline.py','--input',src,'--input_format','normalized','--output_root',root/'out','--model','unused','--base_url','https://example.invalid','--dry_run')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            stages=read(root/'out/run_manifest.json')['stages']
            names=[s['name'] for s in stages]
            self.assertIn('validate_reuse_components',names)
            self.assertNotIn('normalize_components',names)
            self.assertFalse((root/'out/01_normalized').exists())


if __name__=='__main__':
    unittest.main()
