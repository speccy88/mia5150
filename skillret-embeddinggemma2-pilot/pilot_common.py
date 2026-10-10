import csv,json,os,pathlib,re,subprocess,time,threading,urllib.request
BASE=pathlib.Path.home()/'code/frognano-benchmark';ROOT=pathlib.Path(os.environ.get('BENCH_OUTPUT',str(pathlib.Path.home()/'code/lowcpu-benchmark')));ROOT.mkdir(exist_ok=True)
HOST=pathlib.Path(os.environ.get('HOST_TELEMETRY','/mnt/c/Users/user/Documents/Codex/2026-10-05/task/lowcpu-host.jsonl'))
env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(str(BASE/x) for x in ['runtime/deps/usr/lib/x86_64-linux-gnu','runtime/llama-b11425','runtime/cudart-llama-b11425-bin-ubuntu-cuda-12.8-x64'])
state={'stage':'baseline','pid':None};stop=threading.Event();abort=[];lock=threading.Lock();hz=os.sysconf('SC_CLK_TCK');nproc=os.cpu_count()
def save(p,x):p.write_text(json.dumps(x,indent=2))
def mem():return {x.split(':')[0]:int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.split(':')[0] in ['MemAvailable','MemTotal','SwapFree','SwapTotal']}
def swap():return {x.split()[0]:int(x.split()[1]) for x in pathlib.Path('/proc/vmstat').read_text().splitlines() if x.startswith(('pswpin ','pswpout '))}
def cpu():return list(map(int,pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
def proc(pid):
 try:
  f=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return {'seconds':(int(f[11])+int(f[12]))/hz,'rss_bytes':int(f[21])*os.sysconf('SC_PAGE_SIZE'),'threads':int(f[17])}
 except (FileNotFoundError,ProcessLookupError):return None
def collect_monitor():
 prev=cpu();prevtime=time.monotonic();prevproc=None;prevpid=None;prevswap=swap();high=0;last_host_stamp=None
 with open(ROOT/'telemetry.jsonl','w') as f:
  while not stop.wait(1):
   now=time.monotonic();wall=time.time();cur=cpu();delta=sum(cur)-sum(prev);wcpu=100*(1-((cur[3]+cur[4])-(prev[3]+prev[4]))/delta) if delta else 0
   pid=state['pid'];ps=proc(pid) if pid else None;pcpu=(ps['seconds']-prevproc['seconds'])/(now-prevtime)*100 if ps and prevproc and pid==prevpid else None
   host=json.loads(HOST.read_text().splitlines()[-1]);m=mem();s=swap();ds={k:s[k]-prevswap[k] for k in s}
   row={'utc':wall,'stage':state['stage'],'pid':pid,'host':host,'wsl_cpu_percent':wcpu,'process_cpu_core_percent':pcpu,'process':ps,'wsl_memory':m,'swap_delta_pages':ds}
   try:
    lines=(ROOT/'gpu.csv').read_text().splitlines();vals=next(csv.reader([lines[-1]]));row['gpu']={'memory_mib':float(vals[1].split()[0]),'utilization_percent':float(vals[2].split()[0]),'temperature_c':float(vals[3].split()[0]),'power_w':float(vals[4].split()[0])}
   except (ValueError,IndexError):pass
   f.write(json.dumps(row)+'\n');f.flush()
   if host['utc']!=last_host_stamp:high=high+1 if host['cpu_percent']>80 else 0;last_host_stamp=host['utc']
   reasons=[]
   if high>=5:reasons.append('host CPU >80% for five successive samples')
   if host['available_bytes']<2*1024**3:reasons.append('host available RAM <2 GiB')
   if m['MemAvailable']<2*1024**3:reasons.append('WSL available RAM <2 GiB')
   if any(ds.values()):reasons.append('active WSL swapping')
   if row.get('gpu',{}).get('temperature_c',0)>80:reasons.append('GPU temperature >80 C')
   if row.get('gpu',{}).get('memory_mib',0)>7168:reasons.append('GPU free-memory reserve below 1 GiB')
   if wall-host['utc']>8:reasons.append('host telemetry stale')
   if reasons:
    abort.extend(reasons);save(ROOT/'guard-stop.json',row|{'reasons':reasons})
    if pid:
     try:os.kill(pid,15)
     except ProcessLookupError:pass
    return
   prev=cur;prevtime=now;prevproc=ps;prevpid=pid;prevswap=s
def monitor():
 try:collect_monitor()
 except Exception as e:
  abort.append('monitor failure: '+repr(e));save(ROOT/'guard-monitor-error.json',{'error':repr(e)})
  if state['pid']:
   try:os.kill(state['pid'],15)
   except ProcessLookupError:pass
def check():
 if abort:raise RuntimeError('Guard stopped workload: '+str(abort))
def api(path,data=None,stream=False):
 check();req=urllib.request.Request('http://127.0.0.1:18089'+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=120) as f:
  if not stream:return json.load(f)
  chunks=[];first=None
  for line in f:
   check()
   if not line.startswith(b'data: '):continue
   payload=line[6:].strip()
   if payload==b'[DONE]':break
   item=json.loads(payload);elapsed=time.monotonic()-start;chunks.append({'elapsed_seconds':elapsed,'data':item})
   if first is None and item.get('content'):first=elapsed
  return {'ttft_seconds':first,'wall_seconds':time.monotonic()-start,'chunks':chunks,'timings':next((x['data']['timings'] for x in reversed(chunks) if 'timings' in x['data']),None)}
def test(d,filename):
 cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--clearenv','--setenv','PATH','/usr/bin','--setenv','HOME','/tmp','--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--ro-bind',str(d),'/work','--chdir','/work','/usr/bin/prlimit','--as=536870912','--cpu=5','--nproc=32','--fsize=1048576','--nofile=64','--','/usr/bin/python3','-B',filename]
 try:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=12);return {'command':cmd.copy(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'passed':p.returncode==0 and p.stdout.strip()=='PASS'}
 except subprocess.TimeoutExpired:return {'command':cmd.copy(),'timeout':True,'passed':False}
