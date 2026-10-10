"""Reviewed pinned Jina native joint-candidate scoring, strictly offline."""
from harness import *
import torch,importlib.util,time,numpy as np,sys
from transformers import AutoTokenizer
torch.set_num_threads(1);p=R/'models/jina_rerank_35';source=R/'evidence/jina-modeling-source.txt'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
spec=importlib.util.spec_from_file_location('reviewed_jina_model',p/'modeling.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
assert hashlib.sha256((p/'modeling.py').read_bytes()).hexdigest()==hashlib.sha256(source.read_bytes()).hexdigest()
model=module.JinaForRanking.from_pretrained(p,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();model._tokenizer=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False,padding_side='left')
queries=json.loads((R/'manifests/scorer-queries.json').read_text())[:32];pool=json.loads((R/'results/skillret_embed_06-header-0.json').read_text())['rankings'];skills={s['id']:view(s,'header') for s in load('skills','test')};result={};raw=[]
for q in queries:
 ids=pool[q['id']][:20];docs=[skills[d] for d in ids];prompt=module.format_docs_prompts_func(q['query'],docs,special_tokens=model.special_tokens);count=len(model._tokenizer.encode(prompt));row={'id':q['id'],'input_tokens':count,'candidate_ids':ids};start=time.perf_counter()
 try:
  if count>8192:raise ValueError('Joint-candidate context exceeds bounded 8192-token pilot cap')
  with torch.inference_mode():out=model._compute_single_batch(q['query'],docs)
  scores=out.scores.flatten().cpu().float().tolist();assert len(scores)==len(ids) and all(np.isfinite(scores));result[q['id']]=[ids[i] for i in sorted(range(len(ids)),key=lambda i:(-scores[i],i))];row.update(scores=scores,status='ok')
 except Exception as e:result[q['id']]=[];row.update(status='failed',error=repr(e))
 row['seconds']=time.perf_counter()-start;raw.append(row);save(R/'results/jina-native.partial.json',{'rankings':result,'raw':raw});print(q['id'],row['status'],row['seconds'],flush=True)
labels=qrels('test');save(R/'results/jina-native.json',{'rankings':result,'raw':raw,'metrics':evaluate({q['id']:labels[q['id']] for q in queries},result),'readout':'Native joint candidate attention + learned projector cosine; same supplied header documents; 8192 total context cap, no independent hidden-state pooling substitute','license':'CC BY-NC 4.0, academic noncommercial project only','reviewed_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
