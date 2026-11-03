"""One GPU model at a time; no hosted inference, automatic retries, or CPU fallback."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import bench_native_v2 as bench
from inspect_gguf import inspect

ROOT=Path(__file__).resolve().parent
SERVER=ROOT/'tools/llama-b11960-cuda12.4/llama-server.exe'
SUITE=ROOT/'results/local-native-v2'
PYTHON=Path(sys.executable)
PORT=28085

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(path,data):
    bench.atomic_write(path,json.dumps(data,indent=2))

def stopped_owned(record):
    # Windows checks process start time AND executable; a recycled PID is never stopped.
    script=("$p=Get-Process -Id "+str(int(record['pid']))+" -ErrorAction SilentlyContinue; "
            "if ($p) { if ($p.Path -ne $args[0] -or $p.StartTime.ToUniversalTime().ToString('o') -ne $args[1]) { exit 3 }; Stop-Process -Id $p.Id }")
    # Use a temporary script file so all paths are structured arguments, not shell code.
    stop=ROOT/'stop_owned.ps1'
    if not stop.exists():
        stop.write_text("param([int]$PidToStop,[string]$Exe,[string]$Started)\n"
            "$p=Get-Process -Id $PidToStop -ErrorAction SilentlyContinue\n"
            "if ($p) { if ($p.Path -ne $Exe -or $p.StartTime.ToUniversalTime().ToString('o') -ne $Started) { exit 3 }; Stop-Process -Id $p.Id }\n")
    subprocess.run(['pwsh','-NoProfile','-File',str(stop),str(record['pid']),record['executable_path'],record['process_start']],check=True,cwd=ROOT)

def main():
    SUITE.mkdir(parents=True,exist_ok=True)
    state_path=SUITE/'suite-status.json'
    if not state_path.exists():
        raise SystemExit('Explicit continuation requires an existing inspected suite state.')
    jobs=json.loads((ROOT/'local-models.json').read_text())
    state=json.loads(state_path.read_text(encoding='utf-8-sig'))
    state.update(state='running',continued_utc=now(),continuation_reason='Physical VRAM admission after preserved Qwen3-8B diagnostic')
    write(state_path,state)
    skills,queries=bench.load_data(ROOT/'data/test.jsonl',ROOT/'data/test_queries.json')
    for job in jobs:
        alias=job['alias']
        completed_path=SUITE/alias/'status.json'
        if completed_path.exists() and json.loads(completed_path.read_text())['complete']:
            continue
        if (SUITE/alias/'state.sqlite3').exists():
            raise RuntimeError('Unresolved acquired or in-flight trial state; no replay: '+alias)
        entry={'alias':alias,'requested':job['requested'],'started_utc':now(),'state':'preparing'}
        state['models']=[x for x in state['models'] if x['alias']!=alias]+[entry]
        state['state']='running'
        write(state_path,state)
        config={'backend':'llama','model':alias+'-native-v2','base_url':f'http://127.0.0.1:{PORT}',
                'context_window':32768,'quantization':job.get('quantization','not loaded'),
                'gpu_proof':str(ROOT/'runtime'/(alias+'-native-v2')/'gpu-proof.json'),
                'timeout_s':1200,'require_token_usage':True,'source':job,
                'raw_response_dir':str(ROOT/'private-responses/local-native-v2'/alias)}
        config_path=ROOT/'configs/native-v2'/(alias+'.json')
        out=SUITE/alias
        if job.get('blocked'):
            config['startup_error']=job['blocked']
            entry.update(state='blocked',reason=job['blocked'])
            # All 140 are unattempted. The normal estimate gate is retained where applicable.
            write(config_path,config)
            bench.run(config,skills,queries,out)
            entry['finished_utc']=now()
            write(state_path,state)
            continue
        model=Path(job['path'])
        if not model.is_absolute():
            model=ROOT/model
        while not model.exists():
            entry['state']='waiting_for_artifact'
            write(state_path,state)
            time.sleep(10)
        if job.get('sha256') and bench.file_hash(model)!=job['sha256']:
            raise RuntimeError('Artifact identity mismatch: '+alias)
        metadata=inspect(model)
        write(ROOT/'evidence'/(alias+'-gguf.json'),metadata)
        # A foreign CUDA compute workload is a real blocker, never killed.
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if any(line.strip().isdigit() for line in gpu.stdout.splitlines()):
            entry.update(state='blocked',reason='FOREIGN_GPU_WORKLOAD_PRESENT')
            write(state_path,state)
            raise RuntimeError('GPU occupied; no third-party process changed')
        write(config_path,config)
        runtime=ROOT/'runtime'/(alias+'-native-v2')
        previous=set(runtime.glob('launch-*')) if runtime.exists() else set()
        entry['state']='loading'
        write(state_path,state)
        with (ROOT/'evidence'/(alias+'-v2-launcher.log')).open('w',encoding='utf-8') as log:
            started=subprocess.run(['pwsh','-NoProfile','-File',str(ROOT/'Start-LocalServer.ps1'),
                        '-ServerExe',str(SERVER),'-ModelFile',str(model),'-Config',str(config_path)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        launches=set(runtime.glob('launch-*'))-previous
        if len(launches)!=1:
            raise RuntimeError('Could not uniquely identify owned launch: '+alias)
        launch=next(iter(launches))
        owned=json.loads((launch/'owned-process.json').read_text(encoding='utf-8-sig'))
        entry['launch']=str(launch.relative_to(ROOT))
        try:
            if started.returncode:
                text=(launch/'server.stderr.log').read_text(errors='replace')
                admission=launch/'admission-failure.json'
                if admission.exists():
                    verdict=json.loads(admission.read_text(encoding='utf-8-sig'))
                    error=verdict['kind']
                    config['admission_evidence']=verdict
                else:
                    error='HARDWARE_OOM' if 'out of memory' in text.lower() else 'RUNTIME_STARTUP_FAILED'
                config.update(startup_error=error,
                    startup_error_log_sha256=bench.file_hash(launch/'server.stderr.log'))
                entry.update(state='blocked',reason=config['startup_error'])
                write(config_path,config)
                bench.run(config,skills,queries,out)
            else:
                entry['state']='benchmarking'
                write(state_path,state)
                with (ROOT/'evidence'/(alias+'-v2-benchmark.log')).open('w',encoding='utf-8') as log:
                    result=subprocess.run([str(PYTHON),str(ROOT/'bench_native_v2.py'),'run','--config',str(config_path),'--out',str(out)],
                                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    entry.update(state='blocked',reason='BENCHMARK_PROCESS_FAILED')
                    write(state_path,state)
                    raise RuntimeError('Inspect local benchmark failure: '+alias)
                entry['state']='completed'
        finally:
            stopped_owned(owned)
        entry['finished_utc']=now()
        write(state_path,state)
        print(alias+' '+entry['state'],flush=True)
    state.update(state='completed',finished_utc=now())
    write(state_path,state)

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(type(e).__name__+': '+str(e),flush=True)
        raise
