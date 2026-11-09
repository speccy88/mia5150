"""Supervise only this experiment; preserve resource-pressure stops."""
from common import *
import sys,subprocess,time,signal,psutil,shutil,shlex

def main():
 name=sys.argv[1];cmd=sys.argv[2:];deadline=float(os.environ.get('CHALLENGER_MAX_SECONDS','86400'));folder=CACHE/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
 operator_signal=None
 def request_stop(signum,frame):
  nonlocal operator_signal
  operator_signal=signum
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 if (folder/'exit.json').exists():raise RuntimeError('Use a new attempt name; preserve old receipts')
 def gpu():
  return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
 def owners():
  output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
  return {int(x.strip()) for x in output.splitlines() if x.strip().isdigit()}
 idle=gpu()
 if int(idle[0])>=500 or owners():
  save(folder/'gated.json',{'status':'not_started','reason':'GPU occupied','device_used_mib':int(idle[0])})
  (folder/'resume-command.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(['env',f'CHALLENGER_MAX_SECONDS={deadline}',sys.executable,str(Path(__file__).resolve()),name+'-resume',*cmd])+'\n')
  raise SystemExit('GPU occupied; no model execution started. Gated resume command saved in the external run directory.')
 assert psutil.virtual_memory().available>2*1024**3
 # Do not publish machine-specific command paths.
 argv=[Path(cmd[0]).name]+[Path(x).name if x.startswith('/') or i==0 and Path(x).suffix=='.py' else x for i,x in enumerate(cmd[1:])]
 save(folder/'command.json',{'argv':argv,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'time_limit_seconds':deadline})
 source=folder/'source';source.mkdir(exist_ok=True);hashes={}
 for src in (STUDY/'src').glob('*.py'):
  shutil.copyfile(src,source/src.name);hashes[src.name]=sha(src)
 save(folder/'source-sha256.json',hashes)
 execution=[]
 for arg in cmd:
  candidate=Path(arg)
  execution.append(str(source/candidate.name) if candidate.parent.resolve()==(STUDY/'src').resolve() and candidate.suffix=='.py' else arg)
 child_env=dict(os.environ,SKILLRET_STUDY_ROOT=str(STUDY),SKILLRET_EXECUTION_SOURCE=str(source.resolve()))
 if operator_signal is not None:
  save(folder/'gated.json',{'status':'not_started','reason':'operator_stop_before_launch','signal':operator_signal})
  raise SystemExit(75)
 t=time.monotonic();base=psutil.swap_memory();p=subprocess.Popen(execution,start_new_session=True,env=child_env);pressure=0;reason=None;owned={p.pid}
 while p.poll() is None:
  v=psutil.virtual_memory();sw=psutil.swap_memory();mem,util,temp=map(int,gpu());low=v.available<1024**3 or mem>7680 or temp>=80
  pressure=pressure+1 if low else 0
  disk_free=shutil.disk_usage(CACHE).free
  append(folder/'telemetry.jsonl',{'seconds':time.monotonic()-t,'gpu_mib':mem,'gpu_util':util,'gpu_temp_c':temp,'host_available':v.available,'host_used':v.used,'swap_in_delta':sw.sin-base.sin,'swap_out_delta':sw.sout-base.sout,'disk_free_bytes':disk_free})
  try:owned.update(x.pid for x in psutil.Process(p.pid).children(recursive=True))
  except psutil.NoSuchProcess:pass
  external_gpu=bool(owners()-owned)
  disk_low=disk_free<3*1024**3
  if operator_signal is not None or external_gpu or pressure>=3 or sw.sout-base.sout>128*1024**2 or time.monotonic()-t>deadline or disk_low:
   reason='operator_signal' if operator_signal is not None else 'external_gpu_workload' if external_gpu else 'disk_reserve' if disk_low else 'time_budget' if time.monotonic()-t>deadline else 'sustained_resource_pressure'
   try:os.killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   break
  time.sleep(3)
 try:code=p.wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
 save(folder/'exit.json',{'exit_code':code,'guard_reason':reason,'wall_seconds':time.monotonic()-t});sys.exit(code if code>=0 else 1)
if __name__=='__main__':main()
