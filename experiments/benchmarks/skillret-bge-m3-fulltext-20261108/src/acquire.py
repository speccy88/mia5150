"""Download the exact official model outside Git; verify every checkpoint hash."""
import argparse
from pathlib import Path
import hashlib
import json
from common import MODEL_REV,atomic_json


def acquire(cache):
    import requests
    from huggingface_hub import snapshot_download
    response=requests.get(f'https://huggingface.co/api/models/BAAI/bge-m3/tree/{MODEL_REV}?expand=true',timeout=60)
    response.raise_for_status();tree=response.json()
    names=[x['path'] for x in tree if x.get('type')=='file' and x['path'] not in ('.gitattributes','long.jpg')]
    directory=cache/'model'/MODEL_REV
    snapshot_download(repo_id='BAAI/bge-m3',revision=MODEL_REV,local_dir=directory,allow_patterns=names,max_workers=4)
    files=[]
    for entry in tree:
        if entry['path'] not in names:continue
        path=directory/entry['path'];h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        expected=(entry.get('lfs') or {}).get('oid')
        assert path.stat().st_size==entry['size']
        if expected:assert h.hexdigest()==expected
        files.append({'file':entry['path'],'sha256':h.hexdigest(),'bytes':path.stat().st_size,'lfs_hash_verified':bool(expected)})
    atomic_json(cache/'model-acquisition.json',{'model':'BAAI/bge-m3','revision':MODEL_REV,'files':files,'transport':'Official pinned Hugging Face snapshot with independent SHA-256 verification'})
    print('Official model verified',MODEL_REV,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path.home()/'.cache/mia-bgem3')
    acquire(p.parse_args().cache)
