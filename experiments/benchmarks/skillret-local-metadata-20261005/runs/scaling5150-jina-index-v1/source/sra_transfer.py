"""Frozen external transfer screen on full native SRA corpus, no execution."""
from harness import *
from neural import Embed
from json_stream import items
import numpy as np,time,importlib.util
freeze=json.loads((R/'manifests/external_freeze.json').read_text());corpus=[]
for s in items(R/'data/sra/corpus/corpus.json'):corpus.append({'id':s['skill_id'],'name':s['name'],'description':s['description']})
assert len(corpus)==26262 and len({s['id'] for s in corpus})==26262
queries=[];labels={};families={}
for p in sorted((R/'data/sra/instances').glob('*.json')):
 rows=json.loads(p.read_text());selected=sorted(rows,key=lambda row:digest(row['instance_id']))[:4]
 for row in selected:
  qid=row['instance_id'];queries.append({'id':qid,'query':row['question']});labels[qid]={d:1 for d in row['skill_annotations']};families[qid]=p.stem
assert len(queries)==24;allids={s['id'] for s in corpus};assert all(set(g)<=allids for g in labels.values())
save(R/'manifests/sra-panel.json',{'queries':queries,'labels':labels,'families':families,'corpus_ids_hash':digest([s['id'] for s in corpus]),'freeze_hash':digest(freeze),'schema_mapping':'instance_id/question/skill_annotations; eval_data never passed to scorers'})
lex,timing=lexical(corpus,queries,'header',100);save(R/'results/sra-bm25.json',{'rankings':lex,'metrics':evaluate(labels,lex),'timing':timing})
model=Embed('skillret_embed_06');vectors,stats=model.encode([view(s,'header') for s in corpus],batch=1);qv,qstats=model.encode([q['query'] for q in queries],True,1);np.save(R/'cache/sra-skillret-docs.npy',vectors);np.save(R/'cache/sra-skillret-queries.npy',qv);ids=[s['id'] for s in corpus];dense={}
for q,v in zip(queries,qv):dense[q['id']]=[ids[i] for i in np.argsort(-(vectors@v),kind='stable')[:100]]
fusion={q['id']:rrf([dense[q['id']],lex[q['id']]])[:100] for q in queries};systems={'bm25':lex,'skillret':dense,'rrf':fusion};save(R/'results/sra-skillret.json',{'rankings':dense,'metrics':evaluate(labels,dense),'document_stats':stats,'query_stats':qstats});save(R/'results/sra-rrf.json',{'rankings':fusion,'metrics':evaluate(labels,fusion)})
# Source module was read and contains only the native metric arithmetic.
spec=importlib.util.spec_from_file_location('sra_metrics',R/'evidence/sra-native-metrics.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
out={}
for name,rankings in systems.items():
 rows=[{'gold_skill_ids':list(labels[q]),'retrieved':[{'skill_id':d,'score':float(100-i)} for i,d in enumerate(rankings[q])]} for q in labels];native=module.compute_retrieval_metrics(rows,50);ours=evaluate(labels,rankings)['aggregate'];assert abs(native['nDCG@10']-ours['NDCG@10'])<1e-10
 out[name]={'native_metrics':native,'by_family':{family:module.compute_retrieval_metrics([row for q,row in zip(labels,rows) if families[q]==family],50) for family in sorted(set(families.values()))}}
save(R/'results/sra-transfer-summary.json',{'systems':out,'queries':24,'catalog':26262,'precision':'FP16','max_tokens':1024,'batch':1,'frozen_protocol':freeze,'claim':'External transfer diagnostic, not certified blind confirmation. No tuning or downstream task execution.'});print(out,flush=True)
