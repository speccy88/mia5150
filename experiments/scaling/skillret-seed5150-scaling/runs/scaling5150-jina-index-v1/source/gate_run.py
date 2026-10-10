"""Wait for startup headroom; retain stricter continuous guard during work."""
from guard import sample,ROOT
import time,sys,json,subprocess
tag=sys.argv[1];samples=[];stable=0;prev=None;deadline=time.monotonic()+180
while time.monotonic()<deadline:
 try:
  r=sample();samples.append(r);ready=r['host']['available_bytes']>=4.5*1024**3 and r['guest_available']>=4*1024**3 and r['gpu_free_mib']>=7680 and r['gpu_utilization']<=5 and (prev is None or prev['swap']==r['swap']);stable=stable+1 if ready else 0;prev=r
 except Exception as e:samples.append({'time':time.time(),'not_ready':repr(e)});stable=0
 if stable>=3:break
 time.sleep(2)
(ROOT/'evidence'/f'gate-{tag}.json').write_text(json.dumps({'startup_host_gib':4.5,'startup_guest_gib':4,'continuous_limits_unchanged':True,'ready':stable>=3,'samples':samples},indent=2))
if stable<3:print('Insufficient startup headroom after180seconds; no work launched.',flush=True);sys.exit(75)
sys.exit(subprocess.run([sys.executable,'guard.py',tag]+sys.argv[2:],cwd=ROOT).returncode)
