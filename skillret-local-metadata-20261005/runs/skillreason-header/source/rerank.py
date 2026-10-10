from harness import *
from neural import Rerank
import sys,time,json,math
role=sys.argv[1];source=sys.argv[2];kind=sys.argv[3];limit=int(sys.argv[4]) if len(sys.argv)>4 else 32
query=json.loads((R/'manifests/scorer-queries.json').read_text())[:limit];skills={s['id']:s for s in load('skills','test')};pool=json.loads((R/'results'/source).read_text())['rankings'];frozen={q['id']:pool[q['id']][:20] for q in query}
manifest={'source':source,'view':kind,'query_ids':[q['id'] for q in query],'candidate_hash':digest(frozen),'queries_hash':digest(query),'evidence_hash':digest({d:view(skills[d],kind) for p in frozen.values() for d in p}),'candidates':frozen};save(R/'manifests'/f'pool-{pathlib.Path(source).stem}-{kind}-{limit}.json',manifest)
start=time.perf_counter();model=Rerank(role);startup=time.perf_counter()-start;model.score('warmup',['a skill']);raw=[];result={}
for q in query:
 qid=q['id'];ids=frozen[qid];start=time.perf_counter()
 try:
  scores,tokens=model.score(q['query'],[view(skills[d],kind) for d in ids]);assert all(math.isfinite(s) for s in scores);order=sorted(range(len(ids)),key=lambda i:(-scores[i],i));result[qid]=[ids[i] for i in order];row={'id':qid,'scores':scores,'candidate_ids':ids,'raw_token_counts':tokens,'status':'ok'}
 except Exception as e:result[qid]=[];row={'id':qid,'status':'failed','error':repr(e)}
 row['seconds']=time.perf_counter()-start;raw.append(row);print(qid,row['status'],row['seconds'],flush=True)
 save(R/f'results/{role}-{kind}-{limit}.partial.json',{'raw':raw,'rankings':result})
labels=qrels('test');labels={q['id']:labels[q['id']] for q in query}
save(R/f'results/{role}-{kind}-{limit}.json',{'raw':raw,'rankings':result,'metrics':evaluate(labels,result),'pool':manifest,'startup_seconds':startup,'max_length':1024,'precision':'FP16','model_manifest':json.loads((R/'models'/role/'manifest.json').read_text())})
