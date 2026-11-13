from pathlib import Path
import subprocess,json,hashlib,tarfile,time
root=Path('/workspace/metaencoder')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,args in [('report_figures',['python','report_figures.py','--root',str(root)]),('final_diagnostics',['python','final_diagnostics.py','--root',str(root)]),('final_tests',['python','-m','unittest','test_metrics','test_protocol'])]:
 with (root/'logs'/f'{name}_final.log').open('w') as log:
  p=subprocess.run(args,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{name} failed: {p.returncode}')
(root/'manifests'/'finalization.json').write_text(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source':'RunPod','vectors_rewritten':False},indent=2))
(root/'manifests'/'source_hashes_final.sha256').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(root.glob('*.py'))))
e=root/'export'/'evidence.tar.gz'
with tarfile.open(e,'w:gz') as t:
 for p in [root/x for x in ['results','manifests','figures','logs']]+sorted(root.glob('*.py')):
  if p.exists():t.add(p,arcname=p.name)
 t.add(root/'checkpoints'/'decisions_v2',arcname='checkpoints/decisions_v2')
m=root/'export'/'SHA256_MANIFEST.json'
data=json.loads(m.read_text())
data['evidence.tar.gz']={'bytes':e.stat().st_size,'sha256':sha(e)}
m.write_text(json.dumps(data,indent=2))
print(json.dumps({'evidence':data['evidence.tar.gz'],'manifest_bytes':m.stat().st_size}))
