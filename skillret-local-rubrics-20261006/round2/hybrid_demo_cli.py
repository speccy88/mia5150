"""Working offline CLI demo; never imports a credential source or HTTP client."""
import argparse,json
from pathlib import Path
from hybrid_retrieval import CachedRetriever,CachedJev,HybridRetriever
HERE=Path(__file__).resolve().parent
PANELS={'initial':('jev-typed-12-v1',0),'selective':('selective-jev-8-v1',1),'confirmation':('fusion-confirmation-8-v1',2)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--panel',choices=PANELS,default='confirmation');p.add_argument('--query',default='q-04581');p.add_argument('--simulate-failure',action='store_true');a=p.parse_args()
 folder,index=PANELS[a.panel];data=json.loads((HERE/'hybrid-demo-data.json').read_text());row=next(r for r in data['panels'][index]['rows'] if r['id']==a.query)
 cached=json.loads((HERE/folder/'results.json').read_text())['rows'];adapter=CachedJev(cached)
 if a.simulate_failure:
  class Failure:
   def rank(self,*args):raise RuntimeError('Simulated, no transmission')
  adapter=Failure()
 r=HybridRetriever(CachedRetriever({row['id']:row['baseline']}),data['skills'],adapter).search({'id':row['id'],'query':row['query']})
 names=lambda ids:[{'id':d,'name':data['skills'][d]['name']} for d in ids[:5]]
 print(json.dumps({'query_id':row['id'],'status':r.status,'network_calls':0,'baseline_top5':names(r.baseline),'jev_top5':names(r.reranked),'served_top5':names(r.served),'strict_benchmark_output_count':len(r.strict_benchmark)},indent=2))
if __name__=='__main__':main()
