import os,sys,pathlib,subprocess,threading,time,json,signal
TOP=pathlib.Path.home()/'code/embeddinggemma2-pilot'
role=sys.argv[1];root=TOP/role;root.mkdir(parents=True,exist_ok=True)
os.environ['BENCH_OUTPUT']=str(root);os.environ['HOST_TELEMETRY']='/mnt/c/Users/user/Documents/Codex/2026-10-05/task/eg2-host.jsonl'
import pilot_common as c
c.state['stage']='baseline';child=None
gpuout=open(root/'gpu.csv','w');gpu=subprocess.Popen(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv','-lms','1000'],stdout=gpuout)
t=threading.Thread(target=c.monitor,daemon=True);t.start()
try:
 for _ in range(8):time.sleep(1);c.check()
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',UV_CONCURRENT_DOWNLOADS='1',PILOT_RESULT=str(root))
 cmd=['nice','-n','10']+sys.argv[2:];c.save(root/'command.json',cmd)
 log=open(root/'worker.log','w');c.state['stage']='work';child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);c.state['pid']=child.pid
 while child.poll() is None:
  c.check()
  try:c.state['stage']=json.loads((root/'stage.json').read_text())['stage']
  except (OSError,ValueError):pass
  time.sleep(.2)
 c.check();c.save(root/'exit.json',{'returncode':child.returncode});assert child.returncode==0, 'Worker failed; see worker.log'
finally:
 if child:
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  child.wait(timeout=15)
 c.state['pid']=None;c.stop.set();t.join(timeout=3);gpu.terminate();gpu.wait(timeout=10);gpuout.close()




