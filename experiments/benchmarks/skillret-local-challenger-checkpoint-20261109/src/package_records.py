"""Lossless delivery of completed, potentially large per-query journals."""
from common import *
import gzip,shutil

def main():
 assert (STUDY/'results/independent_verification.json').exists()
 assert (STUDY/'results/paired_final_comparisons.json').exists()
 assert not (STUDY/'manifests/record_archive_manifest.json').exists(),'Do not replace published evidence'
 roots=(STUDY/'results',STUDY/'vectors/receipts',STUDY/'manifests')
 paths=set().union(*(set(root.rglob('*.jsonl'))|{Path(str(p)[:-3]) for p in root.rglob('*.jsonl.gz')} for root in roots));manifest={}
 for p in sorted(paths):
  target=Path(str(p)+'.gz');tmp=Path(str(target)+'.tmp');original=CACHE/'publication-uncompressed'/p.relative_to(STUDY)
  source_path=p if p.exists() else original
  threshold=10*1024**2 if p.is_relative_to(STUDY/'manifests') else 1024**2
  if source_path.stat().st_size<threshold and not target.exists():continue
  expected=sha(source_path)
  if not target.exists():
   with source_path.open('rb') as source,tmp.open('wb') as stream:
    with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0,compresslevel=6) as archive:shutil.copyfileobj(source,archive,1024*1024)
   with gzip.open(tmp,'rb') as archive:assert hashlib.file_digest(archive,'sha256').hexdigest()==expected
   tmp.replace(target)
  else:
   with gzip.open(target,'rb') as archive:assert hashlib.file_digest(archive,'sha256').hexdigest()==expected
  manifest[str(target.relative_to(STUDY))]={'sha256':sha(target),'bytes':target.stat().st_size,'original_uncompressed_sha256':expected,'original_uncompressed_bytes':source_path.stat().st_size}
  # Archive original bytes outside the checkout so only one lossless copy is published.
  if p.exists():
   original.parent.mkdir(parents=True,exist_ok=True)
   if original.exists():assert sha(original)==expected;p.unlink()
   else:shutil.move(str(p),original)
 save(STUDY/'manifests/record_archive_manifest.json',{'format':'deterministic gzip; uncompressed SHA256 validated before archival','files':manifest});print('Losslessly packaged',len(manifest),'large journals',flush=True)
if __name__=='__main__':main()
