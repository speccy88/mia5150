"""Fail-closed supervisor; no credentials or network. New run directory per attempt."""
import argparse, collections, hashlib, json, math, os, pathlib, shutil, signal, subprocess, sys, time

def sample(host_path):
    # Host telemetry contains only timestamp, memory and CPU; never application logs.
    with open(host_path, 'rb') as f:
        f.seek(max(0, os.fstat(f.fileno()).st_size-8192))
        lines=f.read().decode().splitlines()
    host=json.loads(lines[-1])
    now=time.time()
    if not 0 <= now-host['utc'] < 8: raise RuntimeError('stale host telemetry')
    mem={s.split(':')[0]:int(s.split()[1])*1024 for s in pathlib.Path('/proc/meminfo').read_text().splitlines()}
    swap={s.split()[0]:int(s.split()[1]) for s in pathlib.Path('/proc/vmstat').read_text().splitlines() if s.startswith(('pswpin ', 'pswpout '))}
    gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).strip().split(',')
    row={'utc':now,'host':host,'guest_available_bytes':mem['MemAvailable'],'swap':swap,'gpu_used_mib':int(gpu[0]),'gpu_temp_c':int(gpu[1]),'gpu_utilization_pct':int(gpu[2])}
    if min(host['available_bytes'],mem['MemAvailable']) < 2*1024**3: raise RuntimeError('less than 2 GiB host/guest reserve: '+json.dumps(row))
    if row['gpu_used_mib']>7168 or row['gpu_temp_c']>80: raise RuntimeError('GPU memory/temperature guard: '+json.dumps(row))
    cpu=host['cpu_pct']
    if isinstance(cpu,bool) or not isinstance(cpu,(float,int)) or not math.isfinite(cpu) or not 0<=cpu<=100: raise RuntimeError('invalid host CPU')
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--run-dir',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    out=pathlib.Path(a.run_dir);out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir();hashes={}
    for path in pathlib.Path(__file__).resolve().parent.glob('*.py'):
        shutil.copy2(path,source/path.name);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'source-sha256.json').write_text(json.dumps(hashes,indent=2))
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    (out/'command.json').write_text(json.dumps(cmd,indent=2))
    child=None; code=1; history=collections.deque(maxlen=5)
    try:
        before=sample(a.host)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',SKILLRET_RESOURCE_GUARD='1')
        with (out/'worker.log').open('w') as log,(out/'telemetry.jsonl').open('w') as tel:
            child=subprocess.Popen(['nice','-n','10']+cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while True:
                row=sample(a.host);tel.write(json.dumps(row)+'\n');tel.flush()
                if row['swap']!=before['swap']: raise RuntimeError('active swap; preserving checkpoints')
                history.append(row['host']['cpu_pct'])
                if len(history)==5 and all(v>80 for v in history): raise RuntimeError('host CPU above 80% for five samples')
                code=child.poll()
                if code is not None: break
                time.sleep(1)
    except Exception as exc:
        (out/'guard-stop.json').write_text(json.dumps({'error':str(exc),'utc':time.time()},indent=2));print(str(exc),flush=True);code=1
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        (out/'exit.json').write_text(json.dumps({'returncode':code,'utc':time.time()},indent=2))
    return code

if __name__=='__main__':sys.exit(main())
