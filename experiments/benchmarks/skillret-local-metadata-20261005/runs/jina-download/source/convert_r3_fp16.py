"""Bounded tensor-at-a-time FP32-to-FP16 cache; preserve original R3 files."""
from harness import *
import torch,shutil,os,gc,sys
from safetensors import safe_open
from safetensors.torch import save_file
torch.set_num_threads(1);role=sys.argv[1] if len(sys.argv)>1 else 'r3_rerank_06';output_role='r3_rerank_fp16_local' if role=='r3_rerank_06' else role+'_fp16_local';source=R/'models'/role;target=R/'models'/output_role;target.mkdir(exist_ok=True);original=json.loads((source/'manifest.json').read_text());manifest={'repository':original['repository'],'revision':original['revision'],'derivation':'Exact torch.float16 cast, tensor-at-a-time from pinned local safetensors; original unchanged. Same inference precision as planned native FP16 run.','source_files':original['files'],'files':{}};mapping={};total=0;index=0
for f in source.iterdir():
 if f.is_file() and f.name!='manifest.json' and f.suffix!='.safetensors':shutil.copy2(f,target/f.name)
for file in source.glob('*.safetensors'):
 with safe_open(file,framework='pt',device='cpu') as handle:keys=list(handle.keys())
 for key in keys:
  with safe_open(file,framework='pt',device='cpu') as handle:value=handle.get_tensor(key).to(torch.float16).contiguous()
  name=f'model-{index:05d}.safetensors';save_file({key:value},str(target/name),metadata={'format':'pt'});total+=value.numel()*value.element_size();mapping[key]=name;index+=1;del value;gc.collect()
  for p in [file,target/name]:
   with p.open('rb') as f:
    if p.parent==target:os.fsync(f.fileno())
    os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  if index%50==0:print('converted tensors',index,flush=True)
config=json.loads((target/'config.json').read_text());config['dtype']='float16';save(target/'config.json',config);save(target/'model.safetensors.index.json',{'metadata':{'total_size':total},'weight_map':mapping})
for p in target.iterdir():
 if p.is_file() and p.name!='manifest.json':
  with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest();os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  manifest['files'][p.name]={'bytes':p.stat().st_size,'sha256':sha}
save(target/'manifest.json',manifest);print('Converted',index,'tensors;',total,'bytes. Original cache preserved.')
