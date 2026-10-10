from harness import *
import numpy as np
summary={}
for folder in (R/'runs').iterdir():
 if not folder.is_dir():continue
 p=folder/'telemetry.jsonl'
 if not p.exists():continue
 rows=[json.loads(s) for s in p.read_text().splitlines()];status='running-or-interrupted'
 if (folder/'exit.json').exists():status=json.loads((folder/'exit.json').read_text())
 if (folder/'guard-stop.json').exists():status=json.loads((folder/'guard-stop.json').read_text())
 if not rows:summary[folder.name]={'status':status};continue
 summary[folder.name]={'status':status,'elapsed_seconds':rows[-1]['time']-rows[0]['time'],'min_host_available_gib':min(r['host']['available_bytes'] for r in rows)/1024**3,'min_guest_available_gib':min(r['guest_available'] for r in rows)/1024**3,'peak_gpu_used_mib':8192-min(r['gpu_free_mib'] for r in rows),'peak_gpu_c':max(r['gpu_temp'] for r in rows),'worker_peak_rss_gib':max(r.get('worker_rss_bytes',0) for r in rows)/1024**3,'worker_cpu_seconds':max(r.get('worker_cpu_seconds',0) for r in rows),'swap_delta':{k:rows[-1]['swap'][k]-rows[0]['swap'][k] for k in rows[0]['swap']},'note':'Whole-device GPU includes display baseline; workerCPU/RSS excludes subprocess servers; guestCPUticks retained in raw telemetry. Startup/warm/index timings remain separate.'}
save(R/'results/resources.json',summary)
