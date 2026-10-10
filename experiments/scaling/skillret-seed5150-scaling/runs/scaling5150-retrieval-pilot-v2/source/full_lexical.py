from harness import *
queries=safe_queries(load('queries','test'));skills=[]
with (R/'data/skills/test.jsonl').open() as f:
 for line in f:
  s=json.loads(line);skills.append({k:s.get(k,'') for k in ['id','name','description']})
rankings,timing=lexical(skills,queries,'header');labels=qrels('test');save(R/'results/public-full-bm25.json',{'rankings':rankings,'timing':timing,'metrics':evaluate(labels,rankings),'queries_hash':digest(queries)})
dense=json.loads((R/'results/public-full-skillret_embed_06.json').read_text())['rankings'];fusion={q['id']:rrf([dense[q['id']],rankings[q['id']]])[:100] for q in queries};save(R/'results/public-full-rrf.json',{'rankings':fusion,'metrics':evaluate(labels,fusion),'k':60,'queries_hash':digest(queries)});save(R/'manifests/public-full.json',{'query_ids':[q['id'] for q in queries],'queries_hash':digest(queries),'corpus_ids_hash':digest([s['id'] for s in skills]),'status':'Public full benchmark; historical full-query aggregates already observed; not pristine confirmation.'})
print('All4392 BM25 and RRF saved.',flush=True)
