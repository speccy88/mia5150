"""Sequential resource-aware execution; never replay attempted outcomes."""
import argparse,json,pathlib,subprocess,sys,time
from ledger import canonical_attempts
HERE=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def wait_headroom(deadline):
    while time.monotonic()<deadline:
        try:
            sample=read(HERE.parents[3]/'host-memory.json')
            if time.time()-sample['utc']<15 and sample['available_bytes']>=4.5*1024**3:return True
        except (ValueError,OSError):pass
        time.sleep(10)
    return False
def main():
    a=argparse.ArgumentParser();a.add_argument('--runtime-config',required=True);a.add_argument('--models',nargs='+',default=['qwen3_06_q8','qwen35_4b_q4']);args=a.parse_args()
    for model in args.models:
        while True:
            path=HERE/f'results/{model}-cases.jsonl'
            records=[]
            if path.exists():
                with path.open() as f:records=[json.loads(x) for x in f if x.strip()]
                pending=sum(r['status'].startswith('not_run') for r in canonical_attempts(records).values())
                if not pending:print(model,'all cases attempted or structurally excluded',flush=True);break
                receipt=read(HERE/f'results/{model}-execution.json');cap=read(HERE/'protocol.json')['runtime_cap_seconds_per_model'][model]
                if receipt.get('cumulative_elapsed_seconds',receipt['elapsed_seconds'])>=cap:print(model,'frozen cumulative runtime cap reached',flush=True);break
            print(model,'waiting for4.5GiB host headroom; guard reserve remains2GiB',flush=True)
            if not wait_headroom(time.monotonic()+900):raise SystemExit('RESOURCE_PENDING: headroom did not recover within15minutes; no model launched')
            completed=subprocess.run([sys.executable,str(HERE/'run.py'),'select','--model',model,'--runtime-config',args.runtime_config])
            if completed.returncode:raise SystemExit(completed.returncode)
    print('Sequential model stages ended; inspect execution receipts and pending counts',flush=True)
if __name__=='__main__':main()
