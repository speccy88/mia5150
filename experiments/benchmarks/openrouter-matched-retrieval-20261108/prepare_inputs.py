#!/usr/bin/env python3
"""Download only pinned public data/encoder assets. Never start inference/API calls."""
import argparse,hashlib,os
from pathlib import Path
from urllib.request import urlretrieve

ASSETS = [
    ('test.jsonl','https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/6583d7d2ed07644d0fb8938ed8178f3a7dc42a12/data/skills/test.jsonl','1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81'),
    ('bge-small/model.onnx','https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx','828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35'),
    ('bge-small/tokenizer.json','https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer.json','d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'),
]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    for name,url,expected in ASSETS:
        target=a.runtime/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=expected:raise ValueError('Existing file hash mismatch: '+name)
        else:
            tmp=target.with_suffix(target.suffix+'.partial');urlretrieve(url,tmp)
            if digest(tmp)!=expected:raise ValueError('Downloaded file hash mismatch: '+name)
            os.replace(tmp,target)
        print(name,target.stat().st_size,expected)
if __name__=='__main__':main()
