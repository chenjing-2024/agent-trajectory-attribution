import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / 'skills/canary-annotation'
sys.path.insert(0, str(SKILL / 'scripts'))
from unified_output import read, write
from build_joint_repair_queue import build


class CanaryWorkflowOutput(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(self,mode,command='safety-refusal-all',raw=False):
        project = self.root / (mode+command)
        cases = project / 'results/canary/cases'
        for tid in ('a','b'):
            write(cases / (tid+'.json'),{'trajectory_id':tid,'trajectory':[
                {'component_id':'C1','role':'user','content':'request'},
                {'component_id':'C2','role':'assistant','content':'I refuse.'}]})
        raw_input = project / 'raw.json'
        if raw:
            import shutil
            shutil.rmtree(project / 'results/canary')
            write(raw_input, {'results':[{'item':{'id':tid},'turns':[{'user':'request','agent':'I refuse.'}]} for tid in ('a','b')]})
        config = project / 'config.env' 
        config.write_text((SKILL / 'config/annotation.example.env').read_text())
        shim = project / 'bin/python';shim.parent.mkdir()
        shim.write_text('#!'+sys.executable+'\n'+Path(__file__).with_name('fake_canary_model.py').read_text());shim.chmod(0o755)
        env = dict(os.environ,PROJECT_ROOT=str(project),ANNOTATION_CONFIG=str(config),MODEL_ID='fixture',
                   OPENAI_API_KEY='synthetic-fixture',CANARY_OUTPUT_TEST_MODE=mode,PATH=str(shim.parent)+os.pathsep+os.environ['PATH'])
        if raw:
            env.update(INPUT_FORMAT='raw', RAW_INPUT=str(raw_input))
        result = subprocess.run(['bash',str(SKILL / 'bin/annotation_skill.sh'),command],env=env,capture_output=True,text=True)
        summary_path = next(project.glob('results/skill_runs/*/quality_summary.json'))
        return result,read(summary_path),summary_path.parent

    def test_full_refusal_workflow_status_matrix(self):
        for mode,code,counts in [('pass',0,{'finalized':2}),('review',1,{'needs_review':2}),
                                 ('excluded',0,{'excluded':2}),('missing',2,{'finalized':1,'failed':1}),
                                 ('interrupted',2,{'failed':2})]:
            with self.subTest(mode=mode):
                result,summary,root = self.run_fixture(mode)
                self.assertEqual(result.returncode,code,result.stdout+result.stderr)
                self.assertEqual({k:v for k,v in summary['counts'].items() if v},counts,summary)
                self.assertEqual(summary['selected'],2)
                self.assertIn('本轮 2 条：',result.stdout)
                self.assertEqual(read(root / 'run_manifest.json')['returncode'],code)

    def test_full_unsafe_nonunsafe_classifications_explicitly_excluded(self):
        result,summary,_ = self.run_fixture('excluded','unsafe-all')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(summary['counts']['excluded'],2)
        self.assertEqual(summary['status'],'no_eligible_cases')

    def test_raw_full_run_keeps_normalization_available_for_later_workflows(self):
        result,summary,root = self.run_fixture('pass',raw=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(summary['counts']['finalized'],2)
        manifest=read(root/'run_manifest.json')
        self.assertTrue(Path(manifest['paths']['NORMALIZED_CASES']).is_dir())
        self.assertEqual(len(list(Path(manifest['paths']['NORMALIZED_CASES']).glob('*.json'))),2)

    def test_joint_queue_hard_error_overrides_semantic_keep(self):
        p = self.root
        for folder in ('targets','normalized','annotations'):
            write(p / folder / 'a.json',{'trajectory_id':'a'})
        (p / 'd.jsonl').write_text(json.dumps({'trajectory_id':'a','status':'fail','hard_errors':['missing_target_component_id']})+'\n')
        (p / 's.jsonl').write_text(json.dumps({'trajectory_id':'a','semantic_status':'pass','recommended_action':'keep_annotation'})+'\n')
        tasks = build(p/'targets',p/'normalized',p/'annotations',p/'d.jsonl',p/'s.jsonl',p/'output')
        self.assertEqual(tasks[0]['recommended_action'],'rerun_all_annotations')
        self.assertEqual(list((p/'output/accepted_cases').glob('*.json')),[])
        r=subprocess.run([sys.executable,str(SKILL/'scripts/prepare_unsafe_action_repair_cases.py'),
            '--semantic_all_cases_jsonl',str(p/'output/joint_cases.jsonl'),'--annotations_dir',str(p/'annotations'),
            '--normalized_dir',str(p/'normalized'),'--output_root',str(p/'package')],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(len(list((p/'package/normalized_cases').glob('*.json'))),1)
        task=json.loads((p/'package/repair_manifest.jsonl').read_text())
        self.assertEqual(task['hard_errors'],['missing_target_component_id'])

    def test_refusal_validator_preserves_review_in_original_source(self):
        m=runpy.run_path(str(SKILL/'scripts/merge_safety_refusal_annotations.py'))
        write(self.root/'target.json',{'trajectory_id':'a','target_type':'safety_refusal','target_action':{'component_id':'C2'},'needs_review':True,'review_reason':'target'})
        write(self.root/'primary.json',{'trajectory_id':'a','primary_attribution_component':{'component_id':'C1'},'needs_review':False})
        result=m['merge_one'](target_path=self.root/'target.json',primary_path=self.root/'primary.json')
        self.assertTrue(result['needs_review'])
        self.assertEqual(result['review_reason'],'target')
        self.assertTrue(result['_metadata']['review_sources']['target']['needs_review'])


    def test_reclassified_repair_has_explicit_exclusion_record(self):
        (self.root/'passed').mkdir()
        write(self.root/'repaired/a.annotation.json',{'trajectory_id':'a','target_type':'safety_refusal','needs_review':False,
            'target_action':{'component_id':'C2'},'primary_attribution_component':{'component_id':'C1'},'attack_chain':[],'execution_chain':[]})
        result=subprocess.run([sys.executable,str(SKILL/'scripts/merge_previous_pass_and_repaired.py'),
            '--pass_dir',str(self.root/'passed'),'--repaired_dir',str(self.root/'repaired'),
            '--output_dir',str(self.root/'final'),'--required_target_type','unsafe_action'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(list((self.root/'final/cases').glob('*.json')),[])
        excluded=json.loads((self.root/'final/excluded_reclassified.jsonl').read_text())
        self.assertEqual(excluded['trajectory_id'],'a')
        self.assertEqual(excluded['reason'],'reclassified_safety_refusal')


if __name__ == '__main__':
    unittest.main()
