"""Publish only compact LoRA weights/configurations; exclude all base weights."""
from common import *
import shutil
from safetensors.numpy import load_file

def check(path,kind):
 cfg=read(path/'adapter_config.json');source=read(STUDY/'manifests'/f'{kind}_source.json');assert cfg['base_model_name_or_path']==source['repository'] and cfg['revision']==source['revision'];assert cfg['r']==8 and cfg['lora_alpha']==16 and cfg['lora_dropout']==.05;a=load_file(str(path/'adapter_model.safetensors'));assert len(a)==112
 for name,value in a.items():
  assert ('.q_proj.' in name or '.v_proj.' in name) and ('lora_A' in name or 'lora_B' in name);assert value.dtype==np.float32 and np.isfinite(value).all();expected=(8,1024) if 'lora_A' in name else (2048,8) if '.q_proj.' in name else (1024,8);assert value.shape==expected,(name,value.shape)
 assert sum(x.size for x in a.values())==1146880
 return {'weight_sha256':sha(path/'adapter_model.safetensors'),'config_sha256':sha(path/'adapter_config.json'),'trainable_parameters':1146880,'tensors':112,'dtype':'float32','base_weights_included':False}

def main():
 l=read(STUDY/'FINAL_EVAL_LOCK.json');manifest={}
 for label,kind,c in [('query','embedding',l['query']),('reranker','reranker',l['reranker']),('query-seed43','embedding',l['secondary_query']),('reranker-seed43','reranker',l['secondary_reranker'])]:
  for stage in ('best','final'):
   src=(CACHE/c['relative_cache_path']).parent/stage;dst=STUDY/'adapters'/label/stage;dst.mkdir(parents=True,exist_ok=True)
   for name in ('adapter_config.json','adapter_model.safetensors'):shutil.copyfile(src/name,dst/name)
   manifest[str(dst.relative_to(STUDY))]=check(dst,kind)
 # Preserve every real candidate's compact best/final adapter as well as selected aliases.
 for family,kind in [('query','embedding'),('reranker','reranker')]:
  for trial in (STUDY/'results'/family).iterdir():
   if not trial.is_dir() or not(trial/'best.json').exists() or not(trial/'summary.json').exists():continue
   for stage in ('best','final'):
    src=CACHE/'checkpoints'/family/trial.name/stage;dst=STUDY/'adapters/attempts'/family/trial.name/stage;dst.mkdir(parents=True,exist_ok=True)
    for name in ('adapter_config.json','adapter_model.safetensors'):shutil.copyfile(src/name,dst/name)
    manifest[str(dst.relative_to(STUDY))]=check(dst,kind)
 save(STUDY/'manifests/adapter_manifest.json',{'adapters':manifest,'no_base_weights':True,'discarded_smoke_and_resume_fixtures':'Not deployment candidates; optimizer states and those discarded fixture weights remain in the external run cache.'});print('Packaged',len(manifest),'compact adapter directories',flush=True)
if __name__=='__main__':main()
