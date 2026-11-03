"""Read-only monitoring plus regenerated reports; never starts/stops model processes."""
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def tail(path,n=2):
    return path.read_text(encoding='utf-8',errors='replace').splitlines()[-n:] if path.exists() else []

previous=None
while True:
    state=json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text(encoding='utf-8-sig'))
    completed=[x['alias'] for x in state['models'] if x['state'] in ['completed','blocked']]
    if completed!=previous:
        subprocess.run([sys.executable,str(ROOT/'write_native_v2_report.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'offline_reasoning_audit.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        previous=completed
    last=state['models'][-1]
    csv_path=ROOT/'results/local-native-v2'/last['alias']/'details.csv'
    rows=list(csv.DictReader(csv_path.open(encoding='utf-8-sig',newline=''))) if csv_path.exists() else []
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'suite_state':state['state'],
        'active_or_last':last['alias'],'model_state':last['state'],'saved_rows':len(rows),
        'measured_responses':sum(bool(r['input_tokens']) for r in rows),'finished_models':completed,
        'gpu':gpu.stdout.strip(),'download':tail(ROOT/'evidence/downloads-console-8mb.log'),
        'nanbeige':tail(ROOT/'evidence/nanbeige-download-8mb.log'),
        'conversion':tail(ROOT/'evidence/conversion-resume-console.log')}
    (ROOT/'evidence/live-status.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    print(json.dumps(snapshot),flush=True)
    if state['state']=='completed':
        break
    time.sleep(45)
