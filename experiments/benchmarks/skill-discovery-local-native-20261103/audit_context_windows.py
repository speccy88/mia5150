"""Sidecar audit: published/configured/artifact/effective windows; no model calls."""
import csv
import datetime
import json
from pathlib import Path
from inspect_gguf import inspect

ROOT=Path(__file__).resolve().parent
BASES=[
 'Qwen/Qwen3-0.6B','Qwen/Qwen3-1.7B','Qwen/Qwen3.5-4B',
 'inclusionAI/Ling-3.0-tiny','Qwen/Qwen3.5-9B',
 'meta-llama/Llama-3.2-3B-Instruct','google/gemma-3-4b-it',
 'h2oai/h2o-lightning-4b','google/gemma-4-E2B-it','Qwen/Qwen3-8B',
 'TokenRhythm/NeoHorse-1-9B','ornith-ai/Ornith-1.5-9B',
 'nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16','prism-ml/Ternary-Bonsai-2-9B-gguf',
 'LiquidAI/LFM3-Omni-2B','microsoft/Phi-4-mini-instruct',
 'mistralai/Mistral-7B-Instruct-v0.3','Nanbeige/Nanbeige4.2-3B']
PUBLISHED={
 'Qwen3-0.6B':{'native':32768,'note':'Publisher README; released config and GGUF additionally contain 40960.'},
 'Qwen3-1.7B':{'native':32768,'note':'Publisher README; released config and GGUF additionally contain 40960.'},
 'Qwen3.5-4B':{'native':262144,'extension':1010000,'note':'Publisher README distinguishes native from RoPE/YaRN extension.'},
 'Qwen3.5-9B':{'native':262144,'extension':1010000,'note':'Publisher README distinguishes native from RoPE/YaRN extension.'},
 'Ling-3.0-tiny':{'native':131072,'extension':262144,'note':'Released max_position_embeddings/original YaRN length=131072; README recipe extends to 262144.'},
 'Ternary Bonsai 2 9B':{'native':262144,'note':'Publisher model card states inherited Qwen3.5 native context.'}}

def limits(obj,prefix=''):
    out={}
    if isinstance(obj,dict):
        for k,v in obj.items():
            path=prefix+'.'+k if prefix else k
            if k in ['max_position_embeddings','original_max_position_embeddings','max_seq_len','n_positions','model_max_length']:
                out[path]=v
            elif isinstance(v,dict):
                out.update(limits(v,path))
    return out

def main():
    rows=[]
    for job,repo in zip(json.loads((ROOT/'local-models.json').read_text()),BASES):
        src=ROOT/'evidence/sources'/repo.replace('/','--')
        metadata=json.loads((src/'metadata.json').read_text(encoding='utf-8')) if (src/'metadata.json').exists() else {}
        cfg=json.loads((src/'config.json').read_text(encoding='utf-8')) if (src/'config.json').exists() else {}
        artifact={}
        if job.get('path'):
            path=Path(job['path'])
            if not path.is_absolute(): path=ROOT/path
            if path.exists():
                info=inspect(path)['metadata']
                artifact={k:v for k,v in info.items() if k.endswith('.context_length')}
        proof=ROOT/'runtime'/(job['alias']+'-native-v2')/'gpu-proof.json'
        actual=json.loads(proof.read_text(encoding='utf-8-sig'))['context_window'] if proof.exists() else None
        detail=ROOT/'results/local-native-v2'/job['alias']/'details.csv'
        data=list(csv.DictReader(detail.open(encoding='utf-8-sig'))) if detail.exists() else []
        rows.append({'model':job['requested'],'source_repo':repo,'source_revision':metadata.get('sha'),
            'published_statement':PUBLISHED.get(job['requested']), 'released_config_limits':limits(cfg),
            'gguf_context_metadata':artifact,'configured_n_ctx':32768,'observed_effective_n_ctx':actual,
            'configuration_reason':'Uniform conservative local profile inherited from preparation; not a measured per-model VRAM maximum.',
            'estimated_context_rows_recorded':sum(x['error']=='CONTEXT_LIMIT_ESTIMATED' for x in data),
            'context_skip_interpretation':'Estimate gate against configured window; no tokenizer/inference observation at skipped sizes; not evidence of intrinsic model context limit.'})
    audit={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'threshold_tokens':0.9*32768,'estimates':{n:44.5*n+200 for n in [50,100,250,500,1000,2000,5000]},
        'why_sixty':'For every 32768-token configuration, N=1000,2000,5000 exceed the v1 90% estimate threshold. Three sizes times twenty questions gives sixty rows, regardless of larger published native windows.',
        'hardware_limit_not_explored':'No search for maximum feasible n_ctx was performed. A 32768 configuration is not a demonstrated VRAM ceiling.',
        'models':rows}
    (ROOT/'evidence/context-window-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Wrote context-window-audit.json; no inference or CSV changes.')

if __name__=='__main__': main()
