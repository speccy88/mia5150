"""Sequential downloads of selected public official GGUFs, never inference."""
import json
from pathlib import Path
import urllib.request
from fetch_official import fetch, ROOT

SELECTED=[
 ('h2oai/h2o-lightning-4b-gguf','h2o-lightning-4b-stock-Q4_K_M.gguf'),
 ('google/gemma-4-E2B-it-qat-q4_0-gguf','gemma-4-E2B_q4_0-it.gguf'),
 ('Qwen/Qwen3-8B-GGUF','Qwen3-8B-Q4_K_M.gguf'),
 ('TokenRhythm/NeoHorse-1-9B-GGUF','NeoHorse-1-9B-Q4_K_M.gguf'),
 ('ornith-ai/Ornith-1.5-9B-GGUF','Ornith-1.5-9B-Q4_K_M.gguf'),
 ('nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB','Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf'),
 ('prism-ml/Ternary-Bonsai-2-9B-gguf','Ternary-Bonsai-2-9B-Q2_0.gguf'),
 ('LiquidAI/LFM3-Omni-2B-GGUF','LFM3-Omni-2B-Q4_K_M.gguf'),
]

if __name__=='__main__':
    records=[]
    for repo,name in SELECTED:
        record={'repo':repo,'file':name,'status':'inflight'}
        records.append(record)
        try:
            fetch(repo,name)
            record['status']='verified'
        except Exception as e:
            record.update(status='failed',error=type(e).__name__)
            print('Download stopped for '+repo+': '+type(e).__name__,flush=True)
        (ROOT/'evidence'/'downloads-status.json').write_text(json.dumps(records,indent=2))
