"""Portable public benchmark evidence; no credentials, model weights or publication."""
import hashlib,importlib.metadata,json,platform,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150'
def main():
 provenance=HERE/'model-provenance';provenance.mkdir(exist_ok=True)
 for role,base in [('eg2',Path('/home/user/code/embeddinggemma2-pilot')),('jina',Path('/home/user/code/jina-nano-pilot'))]:
  for name in ['model-manifest.json','review-manifest.json']:
   if (base/name).exists():shutil.copy2(base/name,provenance/(role+'-'+name))
 source=Path('/home/user/code/routing-pilot/source/jevk5');(provenance/'jev-adapter-sha256.json').write_text(json.dumps({name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['__init__.py','gguf.py','prompt.py']},indent=2))
 versions={x.metadata['Name']:x.version for x in importlib.metadata.distributions() if x.metadata.get('Name')};(HERE/'report/environment.json').write_text(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'packages':dict(sorted(versions.items()))},indent=2))
 notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','id':'scope','metadata':{},'source':['# Seed-5150 local scaling evidence\n','Public development, gold-conditioned libraries. Read report/REPORT.md for scope and limitations. No model/API calls are made by this notebook.']},{'cell_type':'code','id':'metrics','metadata':{},'execution_count':None,'outputs':[],'source':['import csv, json\n','from pathlib import Path\n','root = Path.cwd()\n','if not (root / "report").exists(): root = root / "scaling5150"\n','rows = list(csv.DictReader((root / "report/metrics.csv").open()))\n','for r in rows:\n','    if r["size"] == "6006": print(r["kind"], r["model"], r["method"], r["Recall@5"], r["NDCG@10"], r["valid"])']},{'cell_type':'code','id':'audit','metadata':{},'execution_count':None,'outputs':[],'source':['audit = json.loads((root / "report/audit.json").read_text())\n','assert audit["independent_library_audit"]["all_known_official_positives_in_every_library"]\n','assert audit["independent_library_audit"]["nested_sets"]\n','audit["statuses"]']}]}
 for cell in notebook['cells']:
  if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')
 (HERE/'results.ipynb').write_text(json.dumps(notebook,indent=2))
 files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.npy']]
 # Embedding arrays are regenerated or reused locally; manifests and every inference
 # call are included. This keeps evidence portable without bundling cached weights.
 for name in ['guard.py','harness.py','neural.py','structures.py','local_server.py']:
  files.append(ROOT/name)
 for p in (ROOT/'runs').glob('scaling5150-*'):
  if p.is_dir():files.extend(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts)
  elif p.is_file():files.append(p)
 files.extend((ROOT/'evidence').glob('scaling5150-*.json'))
 for role in ['qwen_embed_06','skillret_embed_06','skillrouter_embed_06','skillreason_embed_06','skillret_rerank_06_fp16_local']:
  p=ROOT/'models'/role/'manifest.json'
  if p.exists():files.append(p)
 files=sorted(set(files));hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};dest=ROOT/'deliverables/MIA5150_Seed5150_Local_Scaling.zip';dest.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.relative_to(ROOT))
  z.writestr('CONTENT_SHA256.json',json.dumps(hashes,indent=2))
 with zipfile.ZipFile(dest) as z:
  assert z.testzip() is None
  for path,h in hashes.items():assert hashlib.sha256(z.read(path)).hexdigest()==h
 info={'file':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'files_verified':len(files),'notebook':'Code cells compile; offline report/CSV reader only. No model execution in notebook.','excluded':'Model weights, embedding .npy caches (manifests retained), credentials, unrelated historical results. No external upload.'};(ROOT/'deliverables/MIA5150_Seed5150_Local_Scaling.manifest.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
if __name__=='__main__':main()
