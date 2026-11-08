"""Official baseline dense forward with separately identified PEFT adapters."""
import json
from common import BASE,STUDY,CACHE,MODEL_REV,MODEL_SHA,module,atomic_json,digest,file_hash,read

baseline=module('immutable_bge_engine',BASE/'src/engine.py')
Monitor,public,utc=baseline.Monitor,baseline.public,baseline.utc

def attach_lora(encoder, adapter=None):
    from peft import LoraConfig,TaskType,get_peft_model,PeftModel
    base=encoder.flag.model.model
    targets=[name for name,_ in base.named_modules() if name.endswith(('.attention.self.query','.attention.self.value'))]
    assert len(targets)==48,targets
    if adapter is None:
        config=LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,r=8,lora_alpha=16,lora_dropout=.05,
                          target_modules=targets,bias='none',init_lora_weights=True)
        peft=get_peft_model(base,config,autocast_adapter_dtype=True)
    else:peft=PeftModel.from_pretrained(base,str(adapter),is_trainable=False,autocast_adapter_dtype=True)
    encoder.flag.model.model=peft
    return peft,targets

def training_encoder(monitor):
    encoder=baseline.Encoder(CACHE,monitor,frozen=False)
    peft,targets=attach_lora(encoder)
    peft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainable=[(name,p) for name,p in peft.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in name for name,_ in trainable)
    assert all(p.dtype==encoder.torch.float32 for _,p in trainable)
    assert all(p.dtype==encoder.torch.float16 for name,p in peft.named_parameters() if not p.requires_grad)
    atomic_json(STUDY/'results/architecture.json',{'base_revision':MODEL_REV,'checkpoint_sha256':MODEL_SHA,
       'target_modules':targets,'trainable_parameters':sum(p.numel() for _,p in trainable),
       'frozen_parameters':sum(p.numel() for p in peft.parameters() if not p.requires_grad),
       'base_dtype':'FP16','adapter_and_optimizer_dtype':'FP32','pooling':'official CLS + L2',
       'rank':8,'alpha':16,'dropout':.05,'gradient_checkpointing':'nonreentrant',
       'attention':encoder.identity['attention'],'gradient_scaling':True,'backbone_frozen':True,
       'model_loading_s':encoder.load_s,'targets_inspected_at_runtime':True})
    return encoder,peft

class Encoder(baseline.Encoder):
    def __init__(self,cache,monitor,context=8192,frozen=True):
        super().__init__(cache,monitor,context=context,frozen=False)
        selection=read(CACHE/'selection.json');adapter=CACHE/selection['adapter_relative_path']
        assert file_hash(adapter/'adapter_model.safetensors')==selection['adapter_sha256']
        peft,_=attach_lora(self,adapter)
        peft.eval()
        assert not any(p.requires_grad for p in peft.parameters())
        self.identity.update({'protocol_sha256':file_hash(STUDY/'protocol.json'),
            'adapter_sha256':selection['adapter_sha256'],'adapter_dtype':'FP32 projections; FP16 residual',
            'adapter_step':selection['optimizer_step'],'adapter_selected_on':'training-derived validation NDCG@10',
            'peft':__import__('peft').__version__})
        self.root=cache/'vectors'/digest(self.identity);self.root.mkdir(parents=True,exist_ok=True)
        atomic_json(self.root/'identity.json',self.identity)
        atomic_json(STUDY/'results/model.json',{**self.identity,'cache_identity':self.identity,
              'parameter_count':sum(p.numel() for p in peft.parameters()),
              'all_parameters_cuda_fp16':False,'frozen_backbone_cuda_fp16':True,
              'model_loading_s':self.load_s,'os_file_cache_evicted':False})
