"""Artifact fixtures for the full runner; final deterministic validation is real."""
import json
import os
from pathlib import Path
import subprocess
import sys

script, *args = sys.argv[1:]
name = Path(script).name
mode = os.environ['DOJO_OUTPUT_MODE']

def option(flag):
    return Path(args[args.index(flag)+1])

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))

if name == 'run_initial_pipeline.py':
    root = option('--output_root')
    for source in sorted(option('--raw_root').rglob('*.json')):
        rel = source.relative_to(option('--raw_root'))
        write(root/'components'/rel,{'component_scheme':'action_result_pair_schemeA','trajectory':[{'role':'user','content':'request'},{'role':'assistant','content':'safe result'}],
                                     'components_metadata':[{'component_id':'C1','role':'user'},{'component_id':'C2','role':'assistant'}]})
        write(root/'annotations'/rel,{'annotation':{'attack_success':False,'annotations':[]}})
    (root/'validation').mkdir()
    for filename in ('repair_cases.jsonl','semantic_input_cases.jsonl'):
        (root/'validation'/filename).write_text('')
elif name == 'repair.py':
    write(option('--output_root')/'_repair_manifest.json',{})
elif name == 'validate_semantic.py':
    root = option('--output_root');root.mkdir(parents=True,exist_ok=True)
    records=[]
    if root.name == 'final_semantic':
        for line in option('--input_cases_jsonl').read_text().splitlines():
            row=json.loads(line)
            records.append({'relative_path':row['relative_path'],'semantic_status':'uncertain' if mode=='review' else 'pass'})
    for filename in ('all_cases.jsonl','review_cases.jsonl'):
        (root/filename).write_text(''.join(json.dumps(r)+'\n' for r in records))
    if mode != 'missing_report' or root.name != 'final_semantic':
        write(root/'semantic_validation_report.json',{'summary':{'num_cases_seen':len(records),
            'num_pass':sum(r['semantic_status']=='pass' for r in records),'num_uncertain':sum(r['semantic_status']=='uncertain' for r in records),
            'num_fail':0,'num_validator_error':0}})
elif name == 'build_summary.py':
    root=option('--output_dir');root.mkdir(parents=True,exist_ok=True);(root/'case_summary.jsonl').write_text('')
else:
    raise SystemExit(subprocess.run([sys.executable,script,*args]).returncode)
