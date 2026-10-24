"""Independent streaming retrieval rescore and exact primary-overlap cross-check."""
import collections,gzip,hashlib,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
def main():
    source=HERE.parents[1]/'scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs'
    gold=collections.defaultdict(set)
    with (source/'qrels-test.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['relevance']>0:gold[r['query_id']].add(r['skill_id'])
    with (HERE/'results/primary200-canonical.jsonl').open() as f:primary=[json.loads(x) for x in f]
    prior={(r['query_id'],r['size'],r['method']):r for r in primary if r['method'] in ['bm25','vector','hybrid'] and r.get('retrieval') is not None}
    seen=set();sources=set();counts=collections.Counter();comparisons=collections.Counter();differences=[];checks=0
    path=HERE/'results/retrieval.jsonl'
    if not path.exists():path=HERE/'results/retrieval.jsonl.gz'
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        for line in f:
            r=json.loads(line);key=(r['query_id'],r['seed'],r['size']);assert key not in seen;seen.add(key);sources.add(r['source_sha256']);counts[(r['seed'],r['size'])]+=1
            for method,data in r['methods'].items():
                ranking=data['ranking5'];g=gold[r['query_id']];assert len(ranking)==len(set(ranking))<=5
                hits=len(g.intersection(ranking));rr=next((1/(i+1) for i,d in enumerate(ranking) if d in g),0.)
                assert data['retrieved_hits']==hits and data['gold_count']==len(g)
                assert abs(data['recall5']-hits/len(g))<1e-12 and abs(data['mrr5']-rr)<1e-12
                assert data['complete5']==float(g<=set(ranking));checks+=1
                old=prior.get((r['query_id'],r['size'],method)) if r['seed']==42 else None
                if old:
                    comparisons[method]+=1
                    if ranking!=old['ranking5'] or any(abs(data[k]-old['retrieval'][k])>1e-12 for k in ['recall5','mrr5']):
                        differences.append({'query_id':r['query_id'],'size':r['size'],'method':method,'primary_ranking5':old['ranking5'],'full_ranking5':ranking,'primary_metrics':old['retrieval'],'full_metrics':{k:data[k] for k in ['recall5','mrr5']},'resolution':'Unresolved diagnostic; do not overwrite primary results or assume equivalence.'})
    assert len(sources)==1 and len(seen)==52704 and all(counts[(seed,n)]==4392 for seed in [42,43,44] for n in [10,100,1000,6006])
    result={'status':'passed' if not differences else 'metrics_passed_overlap_differences_require_review','query_seed_size_rows':len(seen),'ranker_metric_checks':checks,'counts':{str(k):v for k,v in counts.items()},'primary_overlap_available':len(prior),'primary_overlap_checked_by_method':dict(comparisons),'primary_overlap_differences':differences,'scope':'Independent positive-qrel/top-five arithmetic and primary exact-ranking overlap. Search execution/timing is not independently remeasured. Batch8 offline query embeddings versus batch1 primary serving may require numerical diagnostics if differences appear.'}
    (HERE/'report/full-retrieval-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='primary_overlap_differences'}));print('Overlap differences:',len(differences))
if __name__=='__main__':main()
