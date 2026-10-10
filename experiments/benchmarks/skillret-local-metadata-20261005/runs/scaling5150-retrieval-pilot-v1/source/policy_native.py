"""Frozen16-query policy: actual query-to-ranking timing with resident models."""
from harness import *
from neural import Embed,Rerank
import numpy as np,time
policy=json.loads((R/'results/escalation.json').read_text());queries={q['id']:q['query'] for q in json.loads((R/'manifests/scorer-queries.json').read_text())};skills=[]
with (R/'data/skills/test.jsonl').open() as f:
 for line in f:
  s=json.loads(line);skills.append({'id':s['id'],'text':view(s,'header')})
vectors=np.load(R/'cache/skillret_embed_06-header-0-docs.npy');assert len(vectors)==len(skills)
start=time.perf_counter();embed=Embed('skillret_embed_06');strong=Rerank('skillret_rerank_06_fp16_local');startup=time.perf_counter()-start
embed.encode(['warm up'],True,1);strong.score('warm up',['warm up']);rankings={};raw=[]
for qid in policy['evaluation_queries']:
 start=time.perf_counter();status='ok';escalated=False;ids=[];margin=None
 try:
  qvec,_=embed.encode([queries[qid]],True,1);scores=vectors@qvec[0];order=np.argsort(-scores,kind='stable')[:100];margin=float(scores[order[0]]-scores[order[1]]);escalated=margin<=policy['threshold']
  if escalated:
   candidates=order[:20];values,_=strong.score(queries[qid],[skills[i]['text'] for i in candidates]);order=[candidates[i] for i in sorted(range(len(candidates)),key=lambda i:(-values[i],i))]
  ids=[skills[i]['id'] for i in order]
 except Exception as e:status=repr(e)
 rankings[qid]=ids;raw.append({'id':qid,'seconds':time.perf_counter()-start,'status':status,'margin':margin,'escalated':escalated})
 save(R/'results/policy-native.partial.json',{'rankings':rankings,'raw':raw})
labels=qrels('test');seconds=[r['seconds'] for r in raw]
save(R/'results/policy-native.json',{'rankings':rankings,'raw':raw,'metrics':evaluate({q:labels[q] for q in rankings},rankings),'startup_seconds':startup,'latency_seconds':{'mean':float(np.mean(seconds)),'p50':float(np.median(seconds)),'p95':float(np.quantile(seconds,.95))},'escalation_rate':sum(r['escalated'] for r in raw)/len(raw),'frozen_threshold':policy['threshold'],'calibration_ids':policy['calibration_queries'],'note':'Actual warm query encoding/search/conditional native reranking, both models resident; indexing, two-model startup and warmup excluded and identified. Same frozen label-free threshold; no retuning. Local FP16 shard loading only; retained base weights unchanged.'})
