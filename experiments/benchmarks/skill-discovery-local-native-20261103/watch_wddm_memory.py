"""Read-only WDDM telemetry for the verified owned active server; no inference."""
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
while True:
    state = json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text(encoding='utf-8-sig'))
    if state['state'] == 'completed':
        break
    entry = state['models'][-1]
    proof = ROOT/'runtime'/(entry['alias']+'-native-v2')/'gpu-proof.json'
    if entry['state'] == 'benchmarking' and proof.exists():
        result = subprocess.run(['pwsh', '-NoProfile', '-File', str(ROOT/'sample_wddm_memory.ps1'), '-ProofPath', str(proof)],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            sample = json.loads(result.stdout)
            with (ROOT/'evidence/wddm-memory-telemetry.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(sample)+'\n')
            print(json.dumps({'model': entry['alias'], 'sampled_utc': sample['utc']}), flush=True)
        else:
            print(json.dumps({'model': entry['alias'], 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'sampling_exit': result.returncode}), flush=True)
    time.sleep(45)
