import os,pathlib,json,importlib.util,importlib.metadata,hashlib,time
assert os.environ.get('USE_HUB_KERNELS')=='NO'
assert importlib.util.find_spec('fla') is None
assert importlib.util.find_spec('causal_conv1d') is None
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import transformers,peft
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
from transformers.integrations import hub_kernels as hk
r=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,device=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),hub_kernels_enabled=hk._kernels_enabled,fallback_functions={n: str(getattr(q,n)) for n in ['causal_conv1d_fn','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']},modeling_path=q.__file__,modeling_sha256=hashlib.sha256(pathlib.Path(q.__file__).read_bytes()).hexdigest(),threads=torch.get_num_threads(),fla_spec=str(importlib.util.find_spec('fla')),causal_conv_spec=str(importlib.util.find_spec('causal_conv1d')))
assert r['hub_kernels_enabled'] is False
pathlib.Path(os.environ['PILOT_RUN'],'runtime.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

