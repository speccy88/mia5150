"""CPU conversion/quantization only, using pinned publisher inputs and llama.cpp code."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from fetch_official import ROOT, sha

ITEMS=[
 ('microsoft/Phi-4-mini-instruct','Phi-4-mini-instruct'),
 ('mistralai/Mistral-7B-Instruct-v0.3','Mistral-7B-Instruct-v0.3'),
 ('Nanbeige/Nanbeige4.2-3B','Nanbeige4.2-3B'),
]

def main():
    converter=next((ROOT/'tools/converter').glob('llama.cpp-*'))
    quantizer_dir=ROOT/'tools/quantize-cpu'
    quantizer_dir.mkdir(exist_ok=True)
    runtime=ROOT/'tools/llama-b11960-cuda12.4'
    for p in runtime.iterdir():
        if p.name in ('llama-quantize.exe','libomp.dll') or (p.suffix=='.dll' and p.name.startswith(('llama','ggml')) and 'cuda' not in p.name):
            target=quantizer_dir/p.name
            if not target.exists():
                shutil.copy2(p,target)
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='-1')
    statuses=[]
    for repo,name in ITEMS:
        source=ROOT/'conversion-inputs'/repo.replace('/','--')
        while not (source/'source.json').exists():
            time.sleep(10)
        final=ROOT/'models'/(name+'-Q4_K_M.gguf')
        if final.exists():
            raise ValueError('Existing converted artifact: inspect rather than replay')
        intermediate=ROOT/'conversion-inputs'/(name+'-BF16.gguf')
        item={'repo':repo,'name':name,'state':'converting'}
        statuses.append(item)
        status_path=ROOT/'evidence/conversion-status.json'
        status_path.write_text(json.dumps(statuses,indent=2))
        conversion_log=ROOT/'evidence'/(name+'-conversion.log')
        if intermediate.exists():
            if not conversion_log.exists() or 'Model successfully exported' not in conversion_log.read_text(encoding='utf-8',errors='replace'):
                raise ValueError('Existing intermediate lacks successful conversion evidence; inspect before continuing')
        else:
            with conversion_log.open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(converter/'convert_hf_to_gguf.py'),str(source),
                                '--outfile',str(intermediate),'--outtype','bf16'],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        item['state']='quantizing_cpu'
        status_path.write_text(json.dumps(statuses,indent=2))
        partial=final.with_suffix('.gguf.part')
        with (ROOT/'evidence'/(name+'-quantization.log')).open('w',encoding='utf-8') as log:
            subprocess.run([str(quantizer_dir/'llama-quantize.exe'),str(intermediate),str(partial),'Q4_K_M','6'],
                           cwd=quantizer_dir,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        partial.rename(final)
        provenance={'official_source':json.loads((source/'source.json').read_text()),
                    'converter':json.loads((ROOT/'evidence/converter-source.json').read_text()),
                    'intermediate_sha256':sha(intermediate),'quantizer_sha256':sha(quantizer_dir/'llama-quantize.exe'),
                    'quantization':'Q4_K_M','sha256':sha(final),'size':final.stat().st_size,
                    'conversion_and_quantization_device':'CPU; no model inference'}
        final.with_suffix('.gguf.source.json').write_text(json.dumps(provenance,indent=2))
        item.update(state='verified',sha256=provenance['sha256'],size=provenance['size'])
        status_path.write_text(json.dumps(statuses,indent=2))
        print('Converted and verified '+name,flush=True)

if __name__=='__main__':
    main()
