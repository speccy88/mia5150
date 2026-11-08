"""Verify portable adapters contain only the intended finite FP32 LoRA tensors."""
import numpy as np
import re
from safetensors import safe_open
from common import STUDY,MODEL_REV,atomic_json,file_hash,read

def check():
    rows=[];manifest=read(STUDY/'manifests/adapter_manifest.json')
    assert {m['kind'] for m in manifest}=={'best','final'}
    for entry in manifest:
        folder=STUDY/'adapters'/entry['kind'];path=folder/'adapter_model.safetensors'
        assert file_hash(path)==entry['weights_sha256']
        assert file_hash(folder/'adapter_config.json')==entry['config_sha256']
        config=read(folder/'adapter_config.json')
        assert config['base_model_name_or_path']=='BAAI/bge-m3' and config['revision']==MODEL_REV
        assert config['r']==8 and config['lora_alpha']==16 and config['lora_dropout']==.05
        targets=set(config['target_modules'])
        assert targets=={'query','value'} or targets==set(read(STUDY/'results/architecture.json')['target_modules'])
        assert not any(private in (folder/'adapter_config.json').read_text() for private in ('/home/','/Users/'))
        with safe_open(path,framework='np') as archive:
            keys=list(archive.keys());assert len(keys)==96 and all('lora_' in k for k in keys)
            arrays=[archive.get_tensor(k) for k in keys]
            assert sum(a.size for a in arrays)==786432
            assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in arrays)
            observed=set()
            for key,array in zip(keys,arrays):
                match=re.search(r'encoder\.layer\.(\d+)\.attention\.self\.(query|value)\.lora_([AB])\.weight$',key)
                assert match,key
                layer,projection,side=match.groups();observed.add((int(layer),projection,side))
                assert array.shape==((8,1024) if side=='A' else (1024,8))
            assert observed=={(layer,projection,side) for layer in range(24) for projection in ('query','value') for side in ('A','B')}
        rows.append({'kind':entry['kind'],'tensor_count':len(keys),'trainable_parameters':786432,
            'adapter_only':True,'dtype':'float32','finite':True,'portable_pinned_base':True,
            'all_48_encoder_query_value_modules_verified_from_tensors':True,'serialized_target_suffixes':sorted(targets)})
    atomic_json(STUDY/'results/adapter_tensor_validation.json',rows)
    print('Best/final portable adapter tensors passed',flush=True)

if __name__=='__main__':check()
