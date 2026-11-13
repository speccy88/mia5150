"""Execute only on RunPod; lossless export and verification manifest."""
from pathlib import Path
import argparse,json,hashlib,shutil,tarfile,time
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;o=r/'export';o.mkdir(exist_ok=True);t=time.time()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
d=np.load(r/'artifacts/documents.npy');q=np.load(r/'artifacts/queries.npy')
assert d.shape==(6006,6656) and q.shape==(4392,6656)
for x in (d,q):assert x.dtype==np.float32 and np.isfinite(x).all() and np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4)
np.savez_compressed(o/'full_embeddings.npz',documents=d,queries=q)
np.savez_compressed(o/'document_chunks.npz',chunks=np.load(r/'artifacts/documents_chunks.npy'),truncated_documents=np.load(r/'artifacts/documents_truncated.npy'))
z={}
for dim in (256,512,1024,2048,4096):
 for name,x in [('documents',d),('queries',q)]:
  v=x[:,:dim].copy();v/=np.linalg.norm(v,axis=1,keepdims=True);z[f'{name}_{dim}']=v
np.savez_compressed(o/'exploratory_dimensions.npz',**z)
for f in (r/'artifacts').glob('*.json'):shutil.copy2(f,o/f.name)
readme='''MetaEncoder v1 SkillRet RunPod embeddings

All arrays are lossless FP32 and L2-normalized. full_embeddings.npz keys documents (6006,6656), queries (4392,6656). Exact row order is documents_ids.json and queries_ids.json. document_chunks.npz contains all document chunk vectors plus the first-chunk truncated document protocol. See documents_chunks_mapping.json for offsets/counts. Query chunk and truncated vectors equal full query vectors; their metadata maps to queries in full_embeddings.npz, not redundant copies.

exploratory_dimensions.npz contains documents_D and queries_D for D=256,512,1024,2048,4096. These are exploratory prefix truncation and L2 renormalization, NOT native MetaEncoder dimensionality support. Full vectors are retained independently.

Use numpy.load(path), access keys, cosine = normalized_queries @ normalized_documents.T. Stable descending sort in original document row order resolves ties. Rankings and source configuration are in evidence.tar.gz. Nine documents exceed 16384 formatted tokens and use lossless character-preserving chunks, normalized mean pooling; query inputs do not exceed the boundary. The fixed-first-chunk arm is separately labeled truncated.

No model weights or credentials included. See source manifests for pinned model/dataset revisions and exact prompts. All inference, vector processing, evaluation, statistics and this export ran on RunPod.
'''
(o/'README.txt').write_text(readme)
# Read back each NPZ on RunPod to verify lossless structure/access.
arrays={}
for f in o.glob('*.npz'):
 with np.load(f) as zf:
  arrays[f.name]={k:{'shape':list(zf[k].shape),'dtype':str(zf[k].dtype),'finite':bool(np.isfinite(zf[k]).all())} for k in zf.files}
(o/'array_metadata.json').write_text(json.dumps(arrays,indent=2))
with tarfile.open(o/'evidence.tar.gz','w:gz') as tf:
 for name in ['results','manifests','figures','logs','runner.py','runner2.py','metrics_existing.py','secondary_runner.py','report_figures.py','export_artifacts.py','test_metrics.py','test_protocol.py','resume_fast.py','final_diagnostics.py']:
  if (r/name).exists():tf.add(r/name,arcname=name)
manifest={f.name:{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(o.iterdir()) if f.is_file() and f.name!='SHA256_MANIFEST.json'}
(o/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'files':len(manifest)+1,'total_bytes':sum(x['bytes'] for x in manifest.values()),'seconds':time.time()-t,'manifest':manifest},indent=2))
