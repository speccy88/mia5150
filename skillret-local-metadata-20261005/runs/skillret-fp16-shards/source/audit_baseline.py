from harness import *
import collections,subprocess,platform
E=R/'evidence';summary={};ids={}
for split,counts in [('train',(10123,63259,127190)),('test',(6006,4392,7187))]:
 skills=load('skills',split);queries=load('queries',split);rels=load('qrels',split)
 assert tuple(map(len,(skills,queries,rels)))==counts
 si={s['id'] for s in skills};qi={q['id'] for q in queries};assert len(si)==len(skills) and len(qi)==len(queries)
 assert all(x['skill_id'] in si and x['query_id'] in qi for x in rels)
 labels=qrels(split);assert all(set(q['skill_ids'])==set(labels[q['id']]) for q in queries)
 ids[split]=si;summary[split]={'counts':counts,'gold_sizes':dict(collections.Counter(len(v) for v in labels.values())),'query_duplicates':len(queries)-len({q['query'].strip().lower() for q in queries})}
 if split=='test':
  # Public benchmark development panel, never described as unseen confirmation.
  panel=sorted(queries,key=lambda q:digest(q['id']))[:128];save(R/'manifests/public-development.json',{'ids':[q['id'] for q in panel],'exposure':'Public benchmark development; historical exposure unresolved, not pristine holdout.'})
  save(R/'manifests/scorer-queries.json',safe_queries(panel));save(R/'manifests/evaluator-qrels.json',{q['id']:labels[q['id']] for q in panel})
summary['train_test_skill_id_overlap']=len(ids['train']&ids['test']);save(E/'data-audit.json',summary)
skills=load('skills','test');queries=json.loads((R/'manifests/scorer-queries.json').read_text());labels=json.loads((R/'manifests/evaluator-qrels.json').read_text())
for kind in ['header','full']:
 rankings,timing=lexical(skills,queries,kind);save(R/f'results/bm25-locked-{kind}.json',{'rankings':rankings,'timing':timing,'metrics':evaluate(labels,rankings),'view':kind,'dataset':json.loads((E/'data-manifest.json').read_text())['revision']});print(kind,evaluate(labels,rankings)['aggregate'],flush=True)
