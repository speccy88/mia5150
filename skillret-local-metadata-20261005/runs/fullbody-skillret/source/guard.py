import json, os, pathlib, subprocess, sys, time, signal,hashlib,shutil
ROOT=pathlib.Path(os.environ.get('MIA5150_ROOT',pathlib.Path(__file__).resolve().parent))
HOST=pathlib.Path(os.environ.get('MIA_HOST_TELEMETRY','/mnt/c/Users/user/Documents/Codex/2026-10-05/task-2/host.jsonl'))
def sample():
 h=json.loads(HOST.read_text().splitlines()[-1]); m={s.split(':')[0]:int(s.split()[1])*1024 for s in pathlib.Path('/proc/meminfo').read_text().splitlines()}
 swap={s.split()[0]:int(s.split()[1]) for s in pathlib.Path('/proc/vmstat').read_text().splitlines() if s.startswith(('pswpin ','pswpout '))}
 g=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
 r={'time':time.time(),'host':h,'guest_available':m['MemAvailable'],'swap':swap,'gpu_free_mib':int(g[0]),'gpu_temp':int(g[1]),'gpu_utilization':int(g[2])}
 assert r['time']-h['utc']<8, 'stale host monitor'
 assert h['available_bytes']>=2*1024**3 and m['MemAvailable']>=2*1024**3,f'RAM reserve: host_bytes={h["available_bytes"]}, guest_bytes={m["MemAvailable"]}'
 assert int(g[0])>=1024 and int(g[1])<=80,f'GPU reserve/temperature: free_mib={g[0]}, celsius={g[1]}'
 return r
if __name__=='__main__':
 out=ROOT/'runs'/sys.argv[1];out.mkdir(parents=True,exist_ok=False); cmd=sys.argv[2:]
 source=out/'source';source.mkdir();hashes={}
 for path in ROOT.glob('*.py'):
  shutil.copy2(path,source/path.name);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 (out/'source-hashes.json').write_text(json.dumps(hashes,indent=2))
 (out/'command.json').write_text(json.dumps(cmd));p=None
 try:
  prev=sample();env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',UV_CONCURRENT_DOWNLOADS='1')
  with (out/'worker.log').open('w') as log,(out/'telemetry.jsonl').open('w') as tel:
   p=subprocess.Popen(['nice','-n','10']+cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while True:
    r=sample()
    try:
     fields=pathlib.Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split();r['worker_cpu_seconds']=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK');r['worker_rss_bytes']=int(fields[21])*os.sysconf('SC_PAGE_SIZE');r['worker_threads']=int(fields[17])
    except (FileNotFoundError,ProcessLookupError):pass
    r['guest_cpu_ticks']=list(map(int,pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:9]));tel.write(json.dumps(r)+'\n');tel.flush();assert r['swap']==prev['swap'],'active swap';prev=r
    if p.poll() is not None:break
    time.sleep(1)
  (out/'exit.json').write_text(json.dumps({'returncode':p.returncode}));sys.exit(p.returncode)
 except Exception as e:
  (out/'guard-stop.json').write_text(json.dumps({'error':repr(e)}));raise
 finally:
  if p and p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=20)
