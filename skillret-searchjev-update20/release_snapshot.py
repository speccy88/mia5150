import pathlib,subprocess,json,time,os
root=pathlib.Path(__file__).resolve().parent
names=['guard.py','train_pilot.py','pilot20.py','pilot20_adapter_reload.py','pilot20_dynamic.py','validate_checkpoint.py','verify_final_adapter.py','download_models.py']
active=[]
for entry in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=entry.joinpath('cmdline').read_bytes().split(b'\0')
  decoded=[x.decode(errors='replace') for x in args if x]
  if any(pathlib.Path(x).name in names for x in decoded):active.append(dict(pid=int(entry.name),args=decoded))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
m={s.split(':')[0]:int(s.split()[1])*1024 for s in pathlib.Path('/proc/meminfo').read_text().splitlines()}
s={s.split()[0]:int(s.split()[1]) for s in pathlib.Path('/proc/vmstat').read_text().splitlines() if s.startswith(('pswpin ','pswpout '))}
g=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
r=dict(utc=time.time(),owned_workload_processes=active,gpu=g,guest_available_bytes=m['MemAvailable'],guest_total_bytes=m['MemTotal'],guest_swap_used_bytes=m['SwapTotal']-m['SwapFree'],guest_swap_counters=s,guest_cpu_count=os.cpu_count())
assert not active
(root/'release-guest.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
