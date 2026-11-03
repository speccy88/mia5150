"""Stop only owned Qwen3-8B work after a durable row; preserve every acquired result."""
import datetime
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from run_local_suite_v2 import stopped_owned

ROOT=Path(__file__).resolve().parent
alias='qwen3-8b-q4km'
state=json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text(encoding='utf-8-sig'))
if state['models'][-1]['alias']!=alias:
    raise SystemExit('Expected Qwen3-8B to be active; no process changed')
proof=json.loads((ROOT/'runtime'/(alias+'-native-v2')/'gpu-proof.json').read_text(encoding='utf-8-sig'))
dbpath=ROOT/'results/local-native-v2'/alias/'state.sqlite3'
db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
def count():
    return db.execute('SELECT COUNT(*) FROM trials WHERE row_json IS NOT NULL').fetchone()[0]
before=count()
script=ROOT/'stop_vram_owned.ps1'
script.write_text("""param([ValidateSet('controller','bench','monitor')][string]$Kind)
$all=Get-CimInstance Win32_Process
if ($Kind -eq 'controller') { $ids=@(18512); $pattern='run_local_suite_v2.py' }
elseif ($Kind -eq 'bench') { $ids=@(14616,9336); $pattern='bench_native_v2.py' }
else { $ids=@(); $pattern='watch_local_progress.py' }
$selected=@($all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match [regex]::Escape($pattern) -and ($_.CommandLine.Contains('task-5\\local-benchmark') -or $_.ParentProcessId -in $ids) })
if ($Kind -eq 'bench') { $selected=@($selected | Where-Object { $_.CommandLine.Contains('qwen3-8b-q4km') }) }
foreach ($p in $selected) { Stop-Process -Id $p.ProcessId -ErrorAction SilentlyContinue }
$selected | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress
""",encoding='utf-8')
actions=[]
for kind in ['monitor','controller']:
    p=subprocess.run(['pwsh','-NoProfile','-File',str(script),kind],capture_output=True,text=True,check=True)
    actions.append({'kind':kind,'processes':p.stdout.strip()})
print('Controller paused. Waiting for next durable row.',flush=True)
while count()==before:
    time.sleep(.02)
after=count()
p=subprocess.run(['pwsh','-NoProfile','-File',str(script),'bench'],capture_output=True,text=True,check=True)
actions.append({'kind':'bench','processes':p.stdout.strip()})
time.sleep(.2)
acquired=count()
pending=[x[0] for x in db.execute('SELECT key FROM trials WHERE row_json IS NULL')]
db.close()
stopped_owned(proof)
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':alias,
    'reason':'CUDA_STATIC_ALLOCATIONS_EXCEED_PHYSICAL_VRAM','cuda_buffers_mib':9097.34,'physical_gpu_mib':8192,
    'rows_before_pause':before,'rows_at_next_commit':after,'acquired_rows_preserved':acquired,
    'pending_intents_not_replayed':pending,'actions':actions,
    'interpretation':'Diagnostic generations under an overcommitted CUDA configuration; not accepted as VRAM-resident profile evidence. No acquired inference is replayed.'}
(ROOT/'evidence/qwen3-8b-vram-pause.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='actions'}),flush=True)
