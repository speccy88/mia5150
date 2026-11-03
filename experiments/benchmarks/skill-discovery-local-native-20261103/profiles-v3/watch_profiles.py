import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
previous=None
while True:
    state=json.loads((ROOT/'evidence/profile-status.json').read_text())
    finished=[(x['model'],x['profile'],x['state']) for x in state['stages'] if x['state'] in ('completed','blocked','completed_no_generation')]
    if previous!=finished:
        subprocess.run([sys.executable,str(ROOT/'review_profiles.py')],cwd=ROOT,check=True)
        previous=finished
    last=state['stages'][-1]
    name=last['model']+'-'+last['profile']
    rows=[]
    path=ROOT/'results'/name/'details.csv'
    if path.exists(): rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    if last['state']=='benchmarking':
        sample=subprocess.run(['pwsh','-NoProfile','-File',str(ROOT/'sample_wddm_memory.ps1'),'-ProofPath',str(ROOT/'runtime'/name/'gpu-proof.json')],cwd=ROOT,capture_output=True,text=True)
        if not sample.returncode:
            with (ROOT/'evidence/wddm-telemetry.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(json.loads(sample.stdout))+'\n')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'suite_state':state['state'],'model':last['model'],'profile':last['profile'],'profile_state':last['state'],'saved_rows':len(rows),'measured':sum(bool(x['input_tokens']) for x in rows),'gpu':gpu.stdout.strip(),'resolved_stages':len(finished)}
    (ROOT/'evidence/live-status.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    print(json.dumps(snapshot),flush=True)
    if state['state']=='completed':break
    time.sleep(45)
