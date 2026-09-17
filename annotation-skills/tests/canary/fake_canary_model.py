"""Model-only fixtures for the real shell dispatcher, merge and validators."""
import json
import os
from pathlib import Path
import subprocess
import sys

script, *args = sys.argv[1:]
name = Path(script).name
mode = os.environ['CANARY_OUTPUT_TEST_MODE']

def option(key):
    return Path(args[args.index(key)+1])

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

if name not in {'annotate_primary_targets.py','annotate_primary_attribution_fixed_target.py'}:
    raise SystemExit(subprocess.run([sys.executable,script,*args]).returncode)
if mode == 'interrupted':
    raise SystemExit(7)
if name == 'annotate_primary_targets.py':
    root = option('--output_root')
    for target_type in ('unsafe_action','safety_refusal','no_target'):
        (root / 'by_target_type' / target_type).mkdir(parents=True,exist_ok=True)
    for path in sorted(option('--input_dir').glob('*.json')):
        tid = json.loads(path.read_text())['trajectory_id']
        if mode == 'missing' and tid == 'b':
            continue
        target_type = 'no_target' if mode == 'excluded' else 'safety_refusal'
        record = {'trajectory_id':tid,'target_type':target_type,'attack_success':False,
                  'target_action':{'component_id':'C2','role':'assistant','summary':'refusal','evidence':'I refuse.'} if target_type=='safety_refusal' else None,
                  'confidence':0.9,'needs_review':mode=='review','review_reason':'target requires review' if mode=='review' else None}
        write(root / 'cases' / (path.stem+'.target.json'),record)
        write(root / 'by_target_type' / target_type / (path.stem+'.target.json'),record)
else:
    for path in sorted(option('--target_dir').glob('*.json')):
        target = json.loads(path.read_text())
        record = {'trajectory_id':target['trajectory_id'],'target_type':'safety_refusal','target_component_id':'C2',
                  'primary_attribution_component':{'component_id':'C1','role':'user','summary':'request','evidence':'request'},
                  'confidence':0.9,'needs_review':False}
        write(option('--output_root') / 'by_target_type/safety_refusal' / (path.name.replace('.target.json','.primary.json')),record)
