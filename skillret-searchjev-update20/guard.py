import os,sys,time,json,pathlib,subprocess,signal,hashlib
ROOT=pathlib.Path(__file__).resolve().parent
HOST=ROOT/'host.jsonl'
def tail_json(p):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-4096)); lines=f.read().splitlines()
 return json.loads(lines[-1])
def snapshot():
 h=tail_json(HOST); now=time.time()
 mem={x.split(':')[0]:int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines()}
 sw={x.split()[0]:int(x.split()[1]) for x in pathlib.Path('/proc/vmstat').read_text().splitlines() if x.startswith(('pswpin ','pswpout '))}
 g=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).strip().split(',')
 r=dict(utc=now,host=h,guest_available_bytes=mem['MemAvailable'],swap_used_bytes=mem['SwapTotal']-mem['SwapFree'],swap=sw,gpu_used_mib=int(g[0]),gpu_temp_c=int(g[1]),gpu_utilization=int(g[2]))
 assert 0<=now-h['utc']<8,'stale host telemetry'
 assert h['available_bytes']>=2*1024**3 and mem['MemAvailable']>=2*1024**3,'host or guest available RAM below 2 GiB'
 assert int(g[0])<=7168,'VRAM above 7168 MiB'
 assert int(g[1])<=80,'GPU above 80C'
 return r
def main():
 run=ROOT/'runs'/sys.argv[1];run.mkdir(parents=True,exist_ok=False)
 cmd=sys.argv[2:]; (run/'command.json').write_text(json.dumps(cmd,indent=2))
 (run/'source-hashes.json').write_text(json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in ROOT.glob('*.py')},indent=2))
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_XET='1',DO_NOT_TRACK='1',WANDB_DISABLED='true',USE_HUB_KERNELS='NO',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PILOT_RUN=str(run),PYTHONPATH='/home/user/code/embeddinggemma2-pilot/overlay')
 p=None; high=0;good=0;prev=None;stamp=None
 try:
  with (run/'telemetry.jsonl').open('w') as tel,(run/'worker.log').open('w') as log:
   for iteration in range(100000):
    r=snapshot()
    if prev is not None:assert r['swap']==prev['swap'],'active swap'
    if r['host']['utc']!=stamp:
     high=high+1 if r['host']['cpu_percent']>80 else 0
     good=good+1 if r['host']['cpu_percent']<=80 else 0
     stamp=r['host']['utc']
    assert high<5,'host CPU above 80 percent sustained five samples'
    r['worker_pid']=p.pid if p else None
    tel.write(json.dumps(r)+'\n');tel.flush();prev=r
    if p is None and good>=5:p=subprocess.Popen(['nice','-n','10']+cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    if p and p.poll() is not None:
     (run/'exit.json').write_text(json.dumps(dict(returncode=p.returncode,utc=time.time())))
     return p.returncode
    time.sleep(1)
 except Exception as e:
  (run/'guard-stop.json').write_text(json.dumps(dict(error=repr(e),utc=time.time(),sample=locals().get('r'),previous=prev),indent=2))
  raise
 finally:
  if p and p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
if __name__=='__main__':sys.exit(main())
