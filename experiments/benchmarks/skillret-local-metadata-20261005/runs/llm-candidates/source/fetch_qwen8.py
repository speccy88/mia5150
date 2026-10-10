from harness import *
import urllib.request,os
meta=json.loads((R/'evidence/qwen8_rerank_gguf.json').read_text());assert meta['gated'] is False and meta['cardData']['license']=='apache-2.0';name='Qwen3-Reranker-8B.Q4_K_M.gguf';assert any(s['rfilename']==name for s in meta['siblings']);p=R/'models/qwen8_gguf';p.mkdir(exist_ok=True);target=p/name
if not target.exists():
 partial=target.with_suffix('.partial');offset=partial.stat().st_size if partial.exists() else 0;req=urllib.request.Request(f'https://huggingface.co/{meta["id"]}/resolve/{meta["sha"]}/{name}',headers={'Range':f'bytes={offset}-'} if offset else {})
 with urllib.request.urlopen(req,timeout=120) as response,partial.open('ab' if response.status==206 else 'wb') as f:
  n=0
  while block:=response.read(1024*1024):
   f.write(block);n+=1
   if n%16==0:f.flush();os.fdatasync(f.fileno());os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 partial.rename(target)
hasher=hashlib.sha256()
with target.open('rb') as f:
 offset=0
 while block:=f.read(8*1024*1024):
  hasher.update(block);os.posix_fadvise(f.fileno(),offset,len(block),os.POSIX_FADV_DONTNEED);offset+=len(block)
sha=hasher.hexdigest()
save(p/'manifest.json',{'repository':meta['id'],'revision':meta['sha'],'file':name,'sha256':sha,'bytes':target.stat().st_size,'provenance':'Third-party GGUF of Qwen/Qwen3-Reranker-8B; no FP16 parity claim. Weights only, no downloaded code executed.'})
