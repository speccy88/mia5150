"""Non-evaluation runtime/equivalence canary; no task labels or model calls."""
import json,pathlib,time
from core import BM25,digest,render
from fast_bm25 import SparseBM25
HERE=pathlib.Path(__file__).resolve().parent
def main():
    source=HERE.parents[1]/'scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs/skills-metadata.jsonl'
    with source.open() as f:docs={x['id']:render(x) for x in map(json.loads,f)}
    old=BM25(docs);new=SparseBM25(docs)
    fixtures=['Translate a short English paragraph into French.','Find the largest value in a small numeric table.']
    result=[]
    for n in [10,100,1000,6006]:
        catalog=sorted(docs)[:n]
        for query in fixtures:
            timings={}
            for name,engine in [('reference',old),('sparse',new)]:
                elapsed=[]
                for _ in range(3):
                    start=time.perf_counter();ranked=engine.search(query,catalog);elapsed.append(time.perf_counter()-start)
                timings[name]=elapsed
            assert old.search(query,catalog)==new.search(query,catalog)
            result.append({'size':n,'fixture_sha256':digest(query),'ranking100_identical':True,'timings_seconds':timings})
    (HERE/'protocol-amendments/sparse-canary.json').write_text(json.dumps({'scope':'Synthetic non-evaluation fixtures, no expected relevance labels. CPU encoder may be running; timing is calibration only, not paper method latency.','records':result},indent=2));print('Eight real-catalog non-evaluation rank-equivalence checks passed')
if __name__=='__main__':main()
