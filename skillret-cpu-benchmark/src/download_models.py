import os,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'models/.hf');os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
names=['bge-small','e5-small','embeddinggemma-int8','gliner','verdict','laya','julia-1']
models=json.loads((ROOT/'docs/model_sources.json').read_text())['models']; receipts=[]
for name,m in zip(names,models):
 p=ROOT/'models'/name
 snapshot_download(m['repo'],revision=m['revision'],local_dir=p,allow_patterns=m.get('files'),ignore_patterns=['*.png','*.jpg','*.svg','*.webp'] if name=='julia-1' else None,max_workers=2)
 files=[]
 for f in sorted(p.rglob('*')):
  if f.is_file() and '.cache' not in f.parts:
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
   files.append({'path':str(f.relative_to(p)),'bytes':f.stat().st_size,'sha256':h.hexdigest()})
 receipt={'slug':name,'repo':m['repo'],'revision':m['revision'],'files':files};receipts.append(receipt)
 (ROOT/'docs/download_receipts.json').write_text(json.dumps(receipts,indent=2));print('READY',name,flush=True)
