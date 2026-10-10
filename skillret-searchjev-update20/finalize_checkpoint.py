import pathlib,json,hashlib,zipfile,time
root=pathlib.Path(__file__).resolve().parent
run=max((p.parent for p in (root/'runs').glob('*/result.json') if json.loads(p.read_text()).get('completed_optimizer_updates')==20),key=lambda p:(p/'result.json').stat().st_mtime)
checkpoint=run/'checkpoint-update20'
cfgpath=checkpoint/'adapter_config.json'
with (checkpoint/'adapter_model.safetensors').open('rb') as f: weights_before=hashlib.file_digest(f,'sha256').hexdigest()
cfg=json.loads(cfgpath.read_text());original_cfg_sha=hashlib.sha256(cfgpath.read_bytes()).hexdigest()
cfg['base_model_name_or_path']='Qwen/Qwen3.5-0.8B'
cfg['revision']='2fc06364715b967f1860aea9cf38778875588b17'
cfgpath.write_text(json.dumps(cfg,indent=2)+'\n')
provenance=dict(base_repo=cfg['base_model_name_or_path'],base_revision=cfg['revision'],base_safetensors_sha256='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696',initial_adapter_repo='SearchJev/SearchJev-0.8B',initial_adapter_revision='03c5b638c8fdc570a1d4bf998cebac22c0b53384',initial_adapter_sha256='5e3d00afba46dba0715b3a1aed4c82f082678f2992f7912e0d473ccfa966b1f9',optimizer_updates=20,configuration_only_portability_change=dict(original_config_sha256=original_cfg_sha,fields=['base_model_name_or_path','revision']),weights_modified_after_save=False,result='../result.json')
(checkpoint/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
with (checkpoint/'adapter_model.safetensors').open('rb') as f: weights_after=hashlib.file_digest(f,'sha256').hexdigest()
assert weights_before==weights_after
provenance['adapted_safetensors_sha256']=weights_after
(checkpoint/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
