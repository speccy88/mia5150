"""Native SkillReason-Bench adapter; explicit full corpus or schema smoke."""
from harness import *
from neural import Embed
import numpy as np,sys
mode=sys.argv[1] if len(sys.argv)>1 else 'smoke';assert mode in ('smoke','full');root=R/'data/skillreason-bench';skills=[]
with (root/'skills.jsonl').open() as f:
 for line in f:
  s=json.loads(line);skills.append({'id':s['skill_id'],'name':s['name'],'description':s['description']})
  if mode=='smoke' and len(skills)==32:break
with (root/'queries.jsonl').open() as f:rows=[json.loads(line) for line in f]
queries=sorted(rows,key=lambda x:digest(x['query_id']))[:24];labels={q['query_id']:{d:1 for d in q['positive_ids']} for q in queries};safe=[{'id':q['query_id'],'query':q['query']} for q in queries]
if mode=='smoke':
 save(R/'evidence/skillreason-schema-smoke.json',{'skills_read':len(skills),'queries_available':len(rows),'scorer_keys':list(safe[0]),'all_single_positive':all(len(q['positive_ids'])==1 for q in rows),'note':'Schema-only; no retrieval quality score on truncated corpus.'});print('Schema smoke passed.');sys.exit(0)
assert len(skills)==61228 and len(rows)==3729;assert all(set(g)<={s['id'] for s in skills} for g in labels.values())
save(R/'manifests/skillreason-transfer-panel.json',{'queries':safe,'labels':labels,'domains':{q['query_id']:q['domain'] for q in queries},'protocol':'Frozen first24 hashed IDs; all61228 headers; public research-only external diagnostic, no checkpoint-exposure independence claim.'})
lex,timing=lexical(skills,safe,'header',100);model=Embed('skillret_embed_06');vectors,ds=model.encode([view(s,'header') for s in skills],batch=4);qvec,qs=model.encode([q['query'] for q in safe],True,1);ids=[s['id'] for s in skills];dense={q['id']:[ids[i] for i in np.argsort(-(vectors@v),kind='stable')[:100]] for q,v in zip(safe,qvec)};fusion={q['id']:rrf([dense[q['id']],lex[q['id']]])[:100] for q in safe}
save(R/'results/skillreason-transfer.json',{'systems':{n:{'rankings':r,'metrics':evaluate(labels,r)} for n,r in [('bm25',lex),('skillret',dense),('rrf',fusion)]},'document_stats':ds,'query_stats':qs,'timing':timing,'note':'Single-positive retrieval; nativepool61228, separate from multi-positiveSkillRet and SRA. Raw source documents not redistributed.'})
import subprocess
panel=R/'manifests/skillreason-native-panel.jsonl';panel.write_text(''.join(json.dumps({'query_id':q,'positive_ids':list(g)})+'\n' for q,g in labels.items()))
for name,ranking in [('bm25',lex),('skillret',dense),('rrf',fusion)]:
 prediction=R/f'results/skillreason-native-{name}.jsonl';prediction.write_text(''.join(json.dumps({'query_id':q,'retrieved_skill_ids':r})+'\n' for q,r in ranking.items()));output=R/f'results/skillreason-native-{name}-metrics.json';subprocess.run([sys.executable,str(R/'evidence/skillreason-native-evaluate.py'),'--queries',str(panel),'--predictions',str(prediction),'--output',str(output)],check=True);assert abs(json.loads(output.read_text())['NDCG@10']-evaluate(labels,ranking)['aggregate']['NDCG@10'])<1e-10
