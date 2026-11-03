"""Acquire the pinned official llama.cpp converter and public source weights."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from fetch_official import ROOT, sha, download_ranges

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'local-skill-benchmark/1.0'}),timeout=90) as r:
        return r.read()

def weights(repo):
    origin=ROOT/'evidence'/'sources'/repo.replace('/','--')
    meta=json.loads((origin/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('gated'):
        raise ValueError('Gated: download prohibited without user acceptance')
    target=ROOT/'conversion-inputs'/repo.replace('/','--')
    target.mkdir(parents=True,exist_ok=True)
    records=[]
    for s in meta['siblings']:
        n=s['rfilename']
        if '/' in n or not (n.endswith(('.safetensors','.json','.model','.txt','.jinja')) or n=='LICENSE'):
            continue
        if n.startswith(('consolidated.','training_args','generation_config')):
            continue
        p=target/n
        if not p.exists():
            part=p.with_suffix(p.suffix+'.part')
            url='https://huggingface.co/'+repo+'/resolve/'+meta['sha']+'/'+n
            if s.get('lfs'):
                download_ranges(url,part,s['lfs']['size'])
            else:
                part.write_bytes(get(url))
            identity=sha(part)
            if s.get('lfs') and (part.stat().st_size!=s['lfs']['size'] or identity!=s['lfs']['sha256']):
                raise ValueError('Source input hash mismatch')
            part.rename(p)
        identity=sha(p)
        if s.get('lfs') and identity!=s['lfs']['sha256']:
            raise ValueError('Existing source input differs')
        records.append({'file':n,'sha256':identity,'size':p.stat().st_size})
    (target/'source.json').write_text(json.dumps({'repo':repo,'revision':meta['sha'],'files':records},indent=2))
    print('Verified conversion source '+repo,flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='converter':
        meta=json.loads(get('https://api.github.com/repos/ggml-org/llama.cpp/commits/b11960'))
        commit=meta['sha']
        p=ROOT/'downloads'/'llama-b11960-source.zip'
        if not p.exists():
            p.write_bytes(get('https://github.com/ggml-org/llama.cpp/archive/'+commit+'.zip'))
        with zipfile.ZipFile(p) as z:
            z.extractall(ROOT/'tools'/'converter')
        (ROOT/'evidence'/'converter-source.json').write_text(json.dumps({'commit':commit,'archive_sha256':sha(p)},indent=2))
        print('Converter commit '+commit,flush=True)
    else:
        weights(sys.argv[1])
