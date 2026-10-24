"""Fetch only the immutable CPU encoder assets into an isolated runtime directory."""
import argparse,hashlib,json,pathlib,shutil
from huggingface_hub import hf_hub_download
REV='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
def main():
    a=argparse.ArgumentParser();a.add_argument('--runtime-root',required=True);args=a.parse_args();root=pathlib.Path(args.runtime_root);dest=root/'bge-small';dest.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','model.safetensors','onnx/model.onnx']:
        source=hf_hub_download('BAAI/bge-small-en-v1.5',name,revision=REV,cache_dir=str(root/'hf-config-cache'))
        target=dest/pathlib.Path(name).name;shutil.copy2(source,target);manifest[target.name]={'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    assert manifest['model.onnx']['sha256']=='828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35'
    assert manifest['model.safetensors']['sha256']=='3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad'
    assert manifest['tokenizer.json']['sha256']=='d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'
    (root/'ort-manifest.json').write_text(json.dumps({'runtime_version':'1.31.0','model_repository':'BAAI/bge-small-en-v1.5','revision':REV,'serialization':'official onnx/model.onnx','model_sha256':manifest['model.onnx']['sha256'],'bytes':manifest['model.onnx']['bytes']},indent=2))
    (root/'encoder-manifest.json').write_text(json.dumps({'repository':'BAAI/bge-small-en-v1.5','revision':REV,'files':manifest},indent=2))
    print('Immutable CPU encoder assets prepared; selectors are separate downloads in model-provenance.json')
if __name__=='__main__':main()
