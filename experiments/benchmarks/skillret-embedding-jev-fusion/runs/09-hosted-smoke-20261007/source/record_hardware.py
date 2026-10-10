"""Record nonsecret machine/runtime facts for this local inference experiment."""
import json, os, pathlib, platform, subprocess, time
root=pathlib.Path(__file__).resolve().parent
mem={s.split(':')[0]:int(s.split()[1])*1024 for s in pathlib.Path('/proc/meminfo').read_text().splitlines()}
gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
record={'recorded_utc_epoch':time.time(),'kernel':platform.release(),'architecture':platform.machine(),'guest_logical_cpus':os.cpu_count(),'guest_mem_total_bytes':mem['MemTotal'],'guest_swap_total_bytes':mem['SwapTotal'],'gpu_name':gpu[0].strip(),'nvidia_driver_version':gpu[1].strip(),'gpu_total_mib':int(gpu[2]),'deployment':'Windows host, WSL2 Ubuntu-26.04','requested_limits':{'host_available_bytes_min':2*1024**3,'guest_available_bytes_min':2*1024**3,'vram_mib_max':7168,'gpu_temperature_c_max':80,'host_cpu_percent_max_sustained_5_samples':80,'host_telemetry_age_seconds_max':8,'guest_swap_counter_changes_allowed':0}}
(root/'results/hardware.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
