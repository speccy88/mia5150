from harness import *
import urllib.request,os
meta=json.loads((R/'evidence/sra-metadata.json').read_text());assert meta['gated'] is False and meta['cardData']['license']=='mit';manifest={'repository':meta['id'],'revision':meta['sha'],'files':{}}
for entry in meta['siblings']:
 name=entry['rfilename']
 if not name.endswith('.json'):continue
 target=R/'data/sra'/name;target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  print('download',name,flush=True)
  with urllib.request.urlopen(f'https://huggingface.co/datasets/{meta["id"]}/resolve/{meta["sha"]}/{name}',timeout=120) as response,target.with_suffix('.partial').open('wb') as f:
   while block:=response.read(1024*1024):f.write(block);f.flush();os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  target.with_suffix('.partial').rename(target)
 with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest();os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 manifest['files'][name]={'sha256':sha,'bytes':target.stat().st_size}
save(R/'manifests/sra-files.json',manifest)
