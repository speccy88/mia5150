import pathlib,json,urllib.request,hashlib,time
R=pathlib.Path('/home/user/code/mia5150-local');meta=json.loads((R/'evidence/dataset.json').read_text());rev=meta['sha'];manifest={}
for subset in ['skills','queries','qrels']:
 for split in ['train','test']:
  rel=f'data/{subset}/{split}.jsonl';p=R/rel;p.parent.mkdir(exist_ok=True,parents=True)
  if not p.exists():
   with urllib.request.urlopen(f'https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/{rev}/{rel}',timeout=90) as response,p.open('wb') as f:
    while block:=response.read(1024*1024):f.write(block)
  count=0
  with p.open() as f:
   for line in f:json.loads(line);count+=1
  manifest[rel]={'rows':count,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()};print(rel,manifest[rel],flush=True)
(R/'evidence/data-manifest.json').write_text(json.dumps({'revision':rev,'files':manifest},indent=2))
