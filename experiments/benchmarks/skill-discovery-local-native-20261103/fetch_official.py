"""Hash-pin downloads from public publisher repositories; preserve originals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent

def download_ranges(url,part,size):
    """Bound each public-file transfer; large unbounded streams stall on this host."""
    start=part.stat().st_size if part.exists() else 0
    if start>size:
        raise ValueError('Partial exceeds publisher size')
    with part.open('ab' if part.exists() else 'xb') as out:
        while start<size:
            previous=start
            end=min(size-1,start+8*1024*1024-1)
            req=urllib.request.Request(url,headers={'User-Agent':'local-skill-benchmark/1.0','Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=30) as r:
                if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                    raise ValueError('Requested public byte range was not honored')
                remaining=end-start+1
                while remaining:
                    data=r.read(min(1024*1024,remaining))
                    if not data:
                        raise ValueError('Incomplete ranged response; partial preserved')
                    out.write(data)
                    start+=len(data)
                    remaining-=len(data)
                out.flush()
            if start==size or start//(512*1024*1024)!=previous//(512*1024*1024):
                print(part.name+f' {start}/{size}',flush=True)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def fetch(repo, filename):
    info=json.loads((ROOT/'evidence'/'sources'/repo.replace('/','--')/'metadata.json').read_text(encoding='utf-8'))
    if info.get('gated'):
        raise ValueError('Gated publisher repository: no download attempted')
    source=next(s for s in info['siblings'] if s['rfilename']==filename)
    expected=source['lfs']
    path=ROOT/'models'/filename
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        part=path.with_suffix(path.suffix+'.part')
        url='https://huggingface.co/'+repo+'/resolve/'+info['sha']+'/'+filename
        download_ranges(url,part,expected['size'])
        if part.stat().st_size!=expected['size'] or sha(part)!=expected['sha256']:
            raise ValueError('Download differs from pinned publisher metadata')
        os.replace(part,path)
    if path.stat().st_size!=expected['size'] or sha(path)!=expected['sha256']:
        raise ValueError('Existing model differs from pinned publisher metadata')
    evidence={'repo':repo,'revision':info['sha'],'file':filename,
              'size':expected['size'],'sha256':expected['sha256'],
              'license':info.get('cardData',{}).get('license')}
    path.with_suffix(path.suffix+'.source.json').write_text(json.dumps(evidence,indent=2))
    print('Verified '+filename,flush=True)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('repo')
    p.add_argument('file')
    a=p.parse_args()
    fetch(a.repo,a.file)
