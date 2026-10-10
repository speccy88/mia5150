from harness import *
import urllib.request,os
meta=json.loads((R/'evidence/sra-metadata.json').read_text());assert meta['gated'] is False and meta['cardData']['license']=='mit';manifest={'repository':meta['id'],'revision':meta['sha'],'files':{}}
for entry in meta['siblings']:
 name=entry['rfilename']
 if not name.endswith('.json'):continue
 target=R/'data/sra'/name;target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  print('download',name,flush=True)
  partial=target.with_suffix('.partial');offset=partial.stat().st_size if partial.exists() else 0;request=urllib.request.Request(f'https://huggingface.co/datasets/{meta["id"]}/resolve/{meta["sha"]}/{name}',headers={'Range':f'bytes={offset}-'} if offset else {})
  with urllib.request.urlopen(request,timeout=120) as response,partial.open('ab' if response.status==206 else 'wb') as f:
   chunks=0
   while block:=response.read(1024*1024):
    f.write(block);chunks+=1
    if chunks%16==0:f.flush();os.fdatasync(f.fileno());os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  target.with_suffix('.partial').rename(target)
 with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest();os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 manifest['files'][name]={'sha256':sha,'bytes':target.stat().st_size}
save(R/'manifests/sra-files.json',manifest)
