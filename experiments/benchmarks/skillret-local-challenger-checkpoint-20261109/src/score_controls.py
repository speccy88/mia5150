"""Declared controls derived from fixed saved local scores; no test tuning."""
from common import *

def fusion(ds,rs,w):
 n=len(ds);dr=np.empty(n);rr=np.empty(n);dr[np.argsort(-np.asarray(ds),kind='stable')]=np.arange(n);rr[np.argsort(-np.asarray(rs),kind='stable')]=np.arange(n);return np.argsort(-((1-w)*(1-dr/max(1,n-1))+w*(1-rr/max(1,n-1))),kind='stable')
def main():
 l=read(STUDY/'FINAL_EVAL_LOCK.json');s,q,g=data('test')
 recipes=[('released-reranker','released-reranker-unfused',l['reranker']['candidates'],1.),('trained-reranker','trained-reranker-unfused',l['reranker']['candidates'],1.)]
 if 'released_best_pipeline' in l:
  c=l['released_best_pipeline'];recipes.append(('released-reranker','released-best-pipeline',c['candidates'],c['fusion_weight']))
 for source,name,k,w in recipes:
  records=[]
  for r in rows(STUDY/'results/final'/f'{source}.jsonl'):
   ranking=[]
   if r['status']=='ok':assert len(r['reranker_scores'])>=k;ranking=[r['candidates'][i] for i in fusion(r['dense_scores'][:k],r['reranker_scores'][:k],w)]+r['candidates'][k:]
   records.append({**{field:r[field] for field in ('query_id','candidates','dense_scores','reranker_scores','status','provenance')},'ranking':ranking,'declared_control':{'candidates':k,'fusion_weight':w,'score_source':source},'inference_repeated':False})
  assert len(records)==4392;path=STUDY/'results/final'/f'{name}.jsonl';write_rows(path,records);save(path.with_suffix('.metrics.json'),evaluate(g,{r['query_id']:r['ranking'] for r in records}))
 print('Fixed unfused and released-best controls reconstructed without inference',flush=True)
if __name__=='__main__':main()
