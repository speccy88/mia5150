"""Sample this pod's GPU and encoder-process resource use during Qwen embedding."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

def query(*args):
    p=subprocess.run(["nvidia-smi",*args],capture_output=True,text=True,timeout=10)
    return p.stdout.strip() if p.returncode==0 else ""

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--seconds",type=int,default=1800);a=p.parse_args()
    out=a.root/"evidence/qwen_resource_samples.jsonl"
    ready=False
    end=time.time()+a.seconds
    while time.time()<end:
        gpu=query("--query-gpu=memory.used,memory.total,utilization.gpu",
                  "--format=csv,noheader,nounits")
        apps=query("--query-compute-apps=pid,used_gpu_memory",
                   "--format=csv,noheader,nounits")
        rows=[]
        for line in apps.splitlines():
            fields=[x.strip() for x in line.split(",")]
            if len(fields)<2 or not fields[0].isdigit(): continue
            pid=int(fields[0]); rss=None
            try:
                for status in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if status.startswith("VmRSS:"):
                        rss=int(status.split()[1]);break
            except OSError: pass
            rows.append({"pid":pid,"gpu_used_mib":fields[1],"cpu_rss_kib":rss})
        with out.open("a") as f:
            f.write(json.dumps({"utc":datetime.now(timezone.utc).isoformat(),
                                "gpu_memory_used_total_util_percent":gpu,"processes":rows})+"\n")
        ready |= bool(rows)
        if ready and not rows: break
        time.sleep(10)

if __name__=="__main__":main()
