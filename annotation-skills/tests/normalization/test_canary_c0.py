import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]

class CanaryC0(unittest.TestCase):
    def prepare(self,case,root,name):
        src=root/(name+'.json');src.write_text(json.dumps(case))
        dest=root/name
        p=subprocess.run([sys.executable,str(ROOT/'skills/canary-annotation/scripts/common/prepare_annotation_input.py'),'--input',str(src),'--input_format','normalized','--output_root',str(dest)],capture_output=True,text=True)
        return p,dest

    def test_preserve_c0_and_metadata(self):
        case={'trajectory_id':'example','trajectory':[{'role':'user','content':'request','component_id':'C0'},{'role':'assistant','content':'response','component_id':'C1'}],'components_metadata':[{'component_id':'C0'},{'component_id':'C1'}]}
        with tempfile.TemporaryDirectory() as temp:
            p,out=self.prepare(case,Path(temp),'valid')
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads((out/'cases/example.json').read_text()),case)

    def test_invalid_duplicate_and_mismatched_ids_remain_rejected(self):
        for ids,metadata in [(['C0','C0'],None),(['C-1'],None),(['C00'],None),(['C0'],[{'component_id':'C1'}])]:
            with self.subTest(ids=ids),tempfile.TemporaryDirectory() as temp:
                case={'trajectory_id':'example','trajectory':[{'role':'user','content':'request','component_id':cid} for cid in ids]}
                if metadata is not None:case['components_metadata']=metadata
                p,out=self.prepare(case,Path(temp),'bad')
                self.assertNotEqual(p.returncode,0)
                self.assertFalse((out/'cases').exists())

if __name__=='__main__':unittest.main()
