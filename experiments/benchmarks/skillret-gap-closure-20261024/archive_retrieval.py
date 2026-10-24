"""Publish deterministic gzip preserving every exact uncompressed retrieval byte."""
import gzip,hashlib,json,pathlib,shutil
HERE=pathlib.Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    raw=HERE/'results/retrieval.jsonl';target=raw.with_suffix('.jsonl.gz')
    execution=json.loads((HERE/'results/retrieval-execution.json').read_text());assert execution['cases_total']==52704
    with raw.open('rb') as source,target.open('wb') as destination:
        with gzip.GzipFile(filename='',mode='wb',fileobj=destination,mtime=0,compresslevel=6) as compressed:shutil.copyfileobj(source,compressed,1024**2)
    h=hashlib.sha256()
    with gzip.open(target,'rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    assert h.hexdigest()==sha(raw)
    with raw.open() as f:first=json.loads(next(f))
    files=[HERE/'run.py',HERE/'fast_bm25.py',HERE/'core.py']
    identity=hashlib.sha256(json.dumps([sha(p) for p in files],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert first['source_sha256']==identity,'Do not snapshot a changed retrieval implementation'
    snapshot=HERE/'source-snapshots'/identity;snapshot.mkdir(exist_ok=True)
    for p in files:
        destination=snapshot/p.name
        if destination.exists():assert destination.read_bytes()==p.read_bytes()
        else:shutil.copyfile(p,destination)
    result={'rows':52704,'unique_official_queries':4392,'rankers_per_row':3,'uncompressed_bytes':raw.stat().st_size,'uncompressed_sha256':sha(raw),'gzip_bytes':target.stat().st_size,'gzip_sha256':sha(target),'compression':'gzip mtime0, level6; exact raw bytes preserved; original uncompressed file remains local and ignored'}
    (HERE/'results/retrieval-archive.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
