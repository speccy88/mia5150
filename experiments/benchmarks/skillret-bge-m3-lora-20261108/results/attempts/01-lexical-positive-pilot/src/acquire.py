"""Download only the pinned training split; reuse verified local base weights."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from common import CACHE,BASE_CACHE,BASE,STUDY,DATA_REV,MODEL_REV,atomic_json,file_hash,verify_baseline,read

def acquire():
    (STUDY/'results').mkdir(exist_ok=True)
    atomic_json(STUDY/'results/baseline_reuse.json',verify_baseline())
    (CACHE/'data').mkdir(parents=True,exist_ok=True)
    for name,expected in read(BASE/'protocol.json')['data_sha256'].items():
        source=BASE_CACHE/'data'/(name+'.jsonl');assert file_hash(source)==expected
        target=CACHE/'data'/(name+'.jsonl')
        if not target.exists():target.symlink_to(source)
        assert file_hash(target)==expected
    for name in ('model','model-acquisition.json'):
        target=CACHE/name
        if not target.exists():target.symlink_to(BASE_CACHE/name,target_is_directory=name=='model')
    url=f'https://huggingface.co/api/datasets/ThakiCloud/SKILLRET/tree/{DATA_REV}/data?recursive=true&limit=1000'
    records=json.load(urllib.request.urlopen(url,timeout=60));out=[]
    for kind in ('skills','queries','qrels'):
        relative=f'data/{kind}/train.jsonl';entry=next(x for x in records if x['path']==relative)
        target=CACHE/'data'/('train-'+kind+'.jsonl')
        if not target.exists():
            source=f'https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/{DATA_REV}/{relative}'
            with urllib.request.urlopen(source,timeout=180) as response,target.with_suffix('.tmp').open('wb') as stream:
                shutil.copyfileobj(response,stream,1024*1024)
            target.with_suffix('.tmp').replace(target)
        assert target.stat().st_size==entry['size']
        sha=file_hash(target)
        if entry.get('lfs'):assert sha==entry['lfs']['oid']
        else:
            h=hashlib.sha1(f"blob {entry['size']}\0".encode())
            with target.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==entry['oid']
        out.append({'component':kind,'split':'train','path':relative,'bytes':entry['size'],
                    'sha256':sha,'upstream_git_blob_oid':entry['oid'],'upstream_lfs':entry.get('lfs')})
        print('Verified training split',kind,entry['size'],flush=True)
    atomic_json(STUDY/'manifests/training_source.json',{'dataset':'ThakiCloud/SKILLRET','revision':DATA_REV,'files':out})

if __name__=='__main__':acquire()
