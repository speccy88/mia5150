"""Pinned cached llama.cpp, no credentials or external serving."""
from harness import *
import subprocess,os,time,urllib.request,contextlib,resource
def post(path,data):
 req=urllib.request.Request('http://127.0.0.1:18089'+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as f:return json.load(f)
@contextlib.contextmanager
def server(model,tag,context=2048):
 base=pathlib.Path('/home/user/code/frognano-benchmark');env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(str(base/x) for x in ['runtime/deps/usr/lib/x86_64-linux-gnu','runtime/llama-b11425','runtime/cudart-llama-b11425-bin-ubuntu-cuda-12.8-x64']);cmd=json.loads((base/'results/server-command.json').read_text())
 for flag,value in [('-m',str(model)),('-t','1'),('-c',str(context)),('-b','128'),('-ub','128')]:cmd[cmd.index(flag)+1]=value
 cmd+=['-tb','1','--poll','0','--poll-batch','0','--prio','-1','--load-mode','dio','--cache-ram','0','--ctx-checkpoints','0'];save(R/f'evidence/{tag}-server.json',cmd);p=None
 child_before=resource.getrusage(resource.RUSAGE_CHILDREN);server_started=time.monotonic()
 try:
  with (R/f'runs/{tag}-server.log').open('w') as log:
   p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   for _ in range(240):
    if p.poll() is not None:raise RuntimeError('llama server exited')
    try:
     with urllib.request.urlopen('http://127.0.0.1:18089/health',timeout=1) as f:
      if json.load(f).get('status')=='ok':break
    except OSError:pass
    time.sleep(.5)
   else:raise TimeoutError('server startup')
   yield
 finally:
  if p and p.poll() is None:p.terminate();p.wait(timeout=20)
  child_after=resource.getrusage(resource.RUSAGE_CHILDREN)
  save(R/f'evidence/{tag}-server-resources.json',{'server_lifetime_seconds':time.monotonic()-server_started,'child_user_cpu_seconds':child_after.ru_utime-child_before.ru_utime,'child_system_cpu_seconds':child_after.ru_stime-child_before.ru_stime,'child_peak_rss_bytes':child_after.ru_maxrss*1024,'measurement':'Linux child resource accounting after server wait; separate from guard Python-worker RSS/CPU. Includes server startup and all physical completions in this process; not cache-adjusted end-to-end serving latency.'})
