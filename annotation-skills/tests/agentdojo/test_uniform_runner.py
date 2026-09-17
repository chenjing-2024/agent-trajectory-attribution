import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'skills/agentdojo-annotation/scripts/unsafe/run_full_pipeline.py'

class UniformRunner(unittest.TestCase):
    def test_full_runner_exit_summary_and_frozen_limit(self):
        for mode,expected,status in [('pass',0,'no_eligible_cases'),('review',1,'completed_with_review'),('missing_report',2,'failed')]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);raw=root/'raw';raw.mkdir()
                for filename,security in [('00_skip',False),('01_selected',True),('02_beyond_limit',True)]:
                    (raw/(filename+'.json')).write_text(json.dumps({'security':security,'messages':[]}))
                shim=root/'python';shim.write_text('#!'+sys.executable+'\n'+Path(__file__).with_name('fake_unsafe_output_stage.py').read_text());shim.chmod(0o755)
                out=root/'output'
                result=subprocess.run([sys.executable,str(SCRIPT),'--raw_root',str(raw),'--output_root',str(out),
                    '--limit','1','--model','fixture','--base_url','https://example.invalid','--python_bin',str(shim)],
                    env={**os.environ,'DOJO_OUTPUT_MODE':mode},capture_output=True,text=True)
                self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
                summary=json.loads((out/'quality_summary.json').read_text())
                self.assertEqual(summary['status'],status,summary)
                self.assertEqual(summary['selected'],1)
                selection=json.loads((out/'00_selection/selected_cases.json').read_text())
                self.assertEqual(selection['cases'][0]['case_key'],'01_selected.json')
                manifest=json.loads((out/'run_manifest.json').read_text())
                self.assertTrue(all('--limit' not in stage['command'] for stage in manifest['stages']))
                self.assertEqual(manifest['returncode'],expected)

if __name__=='__main__':
    unittest.main()
