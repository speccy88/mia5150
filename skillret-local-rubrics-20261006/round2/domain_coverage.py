"""Cached public coverage by gold-skill domain; diagnostic only, not routing."""
import json,sys,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';sys.path.insert(0,str(ROOT))
from harness import qrels
def main():
 dense=json.loads((ROOT/'results/public-full-skillret_embed_06.json').read_text())['rankings'];bm=json.loads((ROOT/'results/public-full-bm25.json').read_text())['rankings'];meta={}
 with (ROOT/'data/skills/test.jsonl').open() as f:
  for line in f:
   s=json.loads(line);meta[s['id']]={'domain':s.get('domain') or 'unspecified','major':s.get('major') or 'unspecified','name':s['name']}
 labels=qrels('test');buckets=collections.defaultdict(list);examples=[];totalpairs=0;miss20=0;recover100=0;recoverbm=0;both=0;query_dense=0;query_bm=0;query_union=0
 for q,g in labels.items():
  gold={d for d,v in g.items() if v>0};groups=collections.defaultdict(set)
  for d in gold:groups[str(meta[d]['domain'])].add(d)
  for domain,subset in groups.items():
   top20=set(dense[q][:20]);top100=set(dense[q][:100]);union=top20|set(bm[q][:10])
   buckets[domain].append({'query':q,'relevant':len(subset),'hit20':len(subset&top20),'hit100':len(subset&top100),'hit_union':len(subset&union)})
  missing=gold-set(dense[q][:20]);totalpairs+=len(gold);miss20+=len(missing);recover100+=len(missing&set(dense[q][:100]));recoverbm+=len(missing&set(bm[q][:10]))
  a=missing&set(dense[q][:100]);b=missing&set(bm[q][:10]);both+=len(a&b);query_dense+=bool(a);query_bm+=bool(b);query_union+=bool(a|b)
  if missing:examples.append({'id':q,'missing_relevant_count':len(missing),'gold_count':len(gold),'recoverable_in_dense100':len(missing&set(dense[q][:100])),'recoverable_in_bm25top10':len(missing&set(bm[q][:10])),'domains':sorted(groups)})
 rows=[]
 for domain,rs in buckets.items():
  pairs=sum(r['relevant'] for r in rs)
  rows.append({'domain':domain,'query_domain_pairs':len(rs),'relevant_pairs':pairs,'micro_recall20':sum(r['hit20'] for r in rs)/pairs,'macro_recall20':sum(r['hit20']/r['relevant'] for r in rs)/len(rs),'completeness20':sum(r['hit20']==r['relevant'] for r in rs)/len(rs),'micro_recall100':sum(r['hit100'] for r in rs)/pairs,'micro_recall_union':sum(r['hit_union'] for r in rs)/pairs})
 rows.sort(key=lambda r:(r['micro_recall20'],r['domain']))
 result={'query_count':len(labels),'relevant_pairs':totalpairs,'missing_pairs_at20':miss20,'queries_with_any_missing_pair':len(examples),'missing_pairs_recovered_at100':recover100,'missing_pairs_recovered_bm25top10':recoverbm,'recovery_overlap_pairs':both,'dense100_only_recovered_pairs':recover100-both,'bm25_only_recovered_pairs':recoverbm-both,'union_recovered_pairs':recover100+recoverbm-both,'neither_recovers_pairs':miss20-(recover100+recoverbm-both),'queries_with_any_dense100_recovery':query_dense,'queries_with_any_bm25_recovery':query_bm,'queries_with_any_union_recovery':query_union,'domains':rows,'missing_queries':examples,'caveat':'Pairs are distinct(query_id,skill_id), not queries. Recovery counts overlap and must not be added without subtracting intersection. Domains taken from positive skill metadata, diagnostic only and unavailable at inference. Queries can belong to multiple domains; do not add domain query counts. Public historically exposed data; no parameter tuning. Candidate coverage is a ceiling, not actual rerank quality.'}
 (HERE/'domain-coverage.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ('domains','missing_queries')},indent=2));print(json.dumps({'lowest_coverage_domains_n_ge20':[r for r in rows if r['query_domain_pairs']>=20][:8]},indent=2))
if __name__=='__main__':main()
