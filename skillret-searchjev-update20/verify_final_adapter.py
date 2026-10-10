import os,pathlib,json,hashlib
os.environ['USE_HUB_KERNELS']='NO'
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from safetensors.torch import load_file
root=pathlib.Path(__file__).resolve().parent
runs=[p.parent for p in (root/'runs').glob('*/result.json') if json.loads(p.read_text()).get('completed_optimizer_updates')==20]
run=max(runs,key=lambda p:(p/'result.json').stat().st_mtime)
path=run/'checkpoint-update20/adapter_model.safetensors'
state=load_file(str(path),device='cpu')
assert len(state)==372
assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in state.values())
with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
r=dict(checkpoint=str(path),sha256=h,tensors=len(state),parameters=sum(t.numel() for t in state.values()),all_float32=True,all_finite=True,bytes=path.stat().st_size)
pathlib.Path(os.environ['PILOT_RUN'],'checkpoint-integrity.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
