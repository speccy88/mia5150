"""Read public publisher metadata only; no authentication or gated access workaround."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
REPOS = [
    'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-0.6B-GGUF',
    'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-1.7B-GGUF',
    'meta-llama/Llama-3.2-3B-Instruct',
    'Nanbeige/Nanbeige4.2-3B', 'Nanbeige/Nanbeige4.2-3B-GGUF',
    'microsoft/Phi-4-mini-instruct',
    'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-4B-GGUF',
    'h2oai/h2o-lightning-4b', 'h2oai/h2o-lightning-4b-gguf',
    'google/gemma-3-4b-it', 'google/gemma-4-E2B-it',
    'google/gemma-4-E2B-it-qat-q4_0-gguf',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'Qwen/Qwen3-8B', 'Qwen/Qwen3-8B-GGUF',
    'Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-9B-GGUF',
    'TokenRhythm/NeoHorse-1-9B', 'TokenRhythm/NeoHorse-1-9B-GGUF',
    'ornith-ai/Ornith-1.5-9B', 'ornith-ai/Ornith-1.5-9B-GGUF',
    'nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB',
    'nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16',
    'nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF',
    'inclusionAI/Ling-3.0-tiny', 'inclusionAI/Ling-3.0-tiny-GGUF',
    'prism-ml/Ternary-Bonsai-2-9B-gguf',
    'LiquidAI/LFM3-Omni-2B', 'LiquidAI/LFM3-Omni-2B-GGUF',
]

def read(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'local-skill-benchmark-inventory/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b''

def inspect(repo):
    target = ROOT/'evidence'/'sources'/repo.replace('/', '--')
    target.mkdir(parents=True, exist_ok=True)
    status, raw = read('https://huggingface.co/api/models/'+repo+'?blobs=true')
    summary = {'repo':repo, 'metadata_status':status}
    if status != 200:
        return summary
    (target/'metadata.json').write_bytes(raw)
    info = json.loads(raw)
    summary.update(revision=info.get('sha'), gated=info.get('gated'),
                   license=info.get('cardData',{}).get('license'),
                   files=[{'name':s['rfilename'], **(s.get('lfs') or {})}
                          for s in info.get('siblings',[]) if s['rfilename'].endswith('.gguf')])
    # A gated model is reported and left untouched, including config files.
    if not info.get('gated'):
        for name in ['README.md','config.json']:
            code, data = read('https://huggingface.co/'+repo+'/resolve/'+info['sha']+'/'+name)
            if code == 200:
                (target/name).write_bytes(data)
    return summary

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(inspect, REPOS))
    out = {'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'sources':rows}
    (ROOT/'evidence'/'source-inventory.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    for r in rows:
        print(json.dumps(r),flush=True)
