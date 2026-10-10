from harness import *
import urllib.request,os
meta=json.loads((R/'evidence/skillreason-bench-metadata.json').read_text());assert meta['gated'] is False;assert meta['sha']=='91e159ea0ee88e6cf8d1f2e4ad46811b64dd281a';readme=(R/'evidence/skillreason-bench-README.md').read_text();assert 'released for research use' in readme;expected=json.loads((R/'evidence/skillreason-bench-stats.json').read_text())['sha256'];manifest={'repository':meta['id'],'revision':meta['sha'],'license_handling':'Local research only; preserve upstream document metadata; do not redistribute corpus','files':{}}
for name in ['queries.jsonl','skills.jsonl']:
 target=R/'data/skillreason-bench'/name;target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  with urllib.request.urlopen(f'https://huggingface.co/datasets/{meta["id"]}/resolve/{meta["sha"]}/data/{name}',timeout=120) as response,target.with_suffix('.partial').open('wb') as f:
   n=0
   while block:=response.read(1024*1024):
    f.write(block);n+=1
    if n%16==0:f.flush();os.fdatasync(f.fileno());os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  target.with_suffix('.partial').rename(target)
 with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest();os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 assert sha==expected[name];manifest['files'][name]={'bytes':target.stat().st_size,'sha256':sha}
save(R/'manifests/skillreason-bench-files.json',manifest);print('Published file hashes verified.')
