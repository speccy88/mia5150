import json,pathlib,urllib.request,hashlib,sys,time,os
R=pathlib.Path('/home/user/code/mia5150-local')
for role in sys.argv[1:]:
 meta=json.loads((R/f'evidence/{role}.json').read_text());assert meta.get('gated') is False;assert meta.get('cardData',{}).get('license')=='apache-2.0'
 repo=meta['id'];rev=meta['sha'];out=R/'models'/role;out.mkdir(parents=True,exist_ok=True);manifest={'repository':repo,'revision':rev,'files':{}}
 for entry in meta['siblings']:
  name=entry['rfilename']
  if not (name.endswith(('.json','.safetensors','.txt','.model')) or name=='README.md'):continue
  if any(x in name for x in ('optimizer','training_args')):continue
  target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():
   print('download',role,name,flush=True)
   partial=target.with_suffix(target.suffix+'.partial');offset=partial.stat().st_size if partial.exists() else 0
   req=urllib.request.Request(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',headers={'Range':f'bytes={offset}-'} if offset else {})
   with urllib.request.urlopen(req,timeout=120) as response,partial.open('ab' if response.status==206 else 'wb') as f:
    chunks=0
    while block:=response.read(1024*1024):
     f.write(block);chunks+=1
     if chunks%16==0:f.flush();os.fdatasync(f.fileno());os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
   target.with_suffix(target.suffix+'.partial').rename(target)
  with target.open('rb') as f:
   sha=hashlib.file_digest(f,'sha256').hexdigest();os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  manifest['files'][name]={'bytes':target.stat().st_size,'sha256':sha}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('complete',role,flush=True)
