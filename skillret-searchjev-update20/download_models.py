import json,pathlib,urllib.request,hashlib,time
ROOT=pathlib.Path(__file__).resolve().parent
MODELS=ROOT/'models';MODELS.mkdir(exist_ok=True)
specs=[
 ('SearchJev/SearchJev-0.8B','03c5b638c8fdc570a1d4bf998cebac22c0b53384','adapter', ['adapter/adapter_config.json','adapter/adapter_model.safetensors','label_space.json','config.resolved.yaml']),
 ('Qwen/Qwen3.5-0.8B','2fc06364715b967f1860aea9cf38778875588b17','base',['config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','chat_template.jinja','model.safetensors.index.json','model.safetensors-00001-of-00001.safetensors'])]
manifest=[]
for repo,rev,folder,wanted in specs:
 with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true',timeout=30) as r: meta=json.load(r)
 assert meta['sha']==rev
 (MODELS/(folder+'-metadata.json')).write_text(json.dumps(meta,indent=2))
 files={x['rfilename']:x for x in meta['siblings']}
 total=sum(files[x].get('size',0) for x in wanted if x in files);assert total<2100000000
 for name in wanted:
  if name not in files:
   print('absent optional',repo,name,flush=True);continue
  info=files[name]; size=info.get('size'); expected=info.get('lfs',{}).get('sha256')
  dst=MODELS/folder/pathlib.Path(name).name;dst.parent.mkdir(exist_ok=True)
  if dst.exists():
   h=hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()
   assert (not size or dst.stat().st_size==size) and (not expected or h==expected)
  else:
   temp=dst.with_suffix(dst.suffix+'.partial');hobj=hashlib.sha256();n=0;start=time.monotonic()
   req=urllib.request.Request(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',headers={'User-Agent':'SearchJev-local-pilot/1.0'})
   with urllib.request.urlopen(req,timeout=45) as r,temp.open('wb') as out:
    while True:
     chunk=r.read(1024*1024)
     if not chunk:break
     n+=len(chunk);assert n<=2100000000 and (not size or n<=size)
     out.write(chunk);hobj.update(chunk)
     if n%(128*1024**2)==0:print('download_progress',repo,name,n,round(time.monotonic()-start,1),flush=True)
   h=hobj.hexdigest();assert (not size or n==size) and (not expected or h==expected)
   temp.rename(dst)
  row=dict(repo=repo,revision=rev,file=name,local=str(dst),bytes=dst.stat().st_size,sha256=h,lfs_sha256=expected)
  manifest.append(row);(MODELS/'manifest.json').write_text(json.dumps(manifest,indent=2));print('verified',name,row['bytes'],h,flush=True)
