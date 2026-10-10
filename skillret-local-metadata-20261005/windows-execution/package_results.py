"""One portable ZIP with deterministic inventory; no bulk data/base weights."""
from harness import *
import zipfile,datetime
top_suffixes={'.py','.md','.json','.ini','.txt','.ipynb','.html'};trees={'results','evidence','runs','manifests','checkpoints','plan','official','vendor'};files=[]
for p in sorted(R.rglob('*')):
 if not p.is_file():continue
 relative=p.relative_to(R);parts=relative.parts
 if any(x in parts for x in ['.git','.venv','__pycache__','.pytest_cache']):continue
 if p.suffix.lower() in {'.pt','.pth','.bin','.safetensors','.gguf','.onnx','.npz','.npy','.partial'}:continue
 if p.name in ['PACKAGE_SHA256.json','package-receipt.json','host.jsonl'] or p.suffix in ('.pyc','.zip') and parts[0]!='plan':continue
 include=(len(parts)==1 and p.suffix in top_suffixes) or parts[0] in trees or (parts[0]=='models' and p.name in ('manifest.json','config.json','README.md','chat_template.jinja'))
 if include:
  # Exclude temporary Library transport/authorization material if any exists.
  if any(word in p.name.lower() for word in ['credential','download-spec','materialize-response','access-token']):continue
  files.append(p)
inventory={str(p.relative_to(R)): {'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in files};save(R/'PACKAGE_SHA256.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':inventory,'excluded':'Bulkdata/baseweights/vectorcaches/environments/Git internals/credentials/transport authorization'})
target=R/'MIA5150_Local_Results_2026-10-05.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files+[R/'PACKAGE_SHA256.json']:z.write(p,arcname='mia5150-local/'+str(p.relative_to(R)))
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None
 for name,info in inventory.items():assert hashlib.sha256(z.read('mia5150-local/'+name)).hexdigest()==info['sha256']
receipt={'file':target.name,'bytes':target.stat().st_size,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),'verified_entries':len(inventory)+1};save(R/'package-receipt.json',receipt);print(receipt)
