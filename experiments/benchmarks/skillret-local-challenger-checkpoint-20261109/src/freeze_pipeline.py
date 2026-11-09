"""Primary challenger and deployment recommendation chosen on selection only."""
from common import *

def lookup(summary,k,w):
 for key,value in summary.items():
  if int(key.split('-')[0][1:])==k and float(key.split('-')[1][1:])==w:return value
 raise KeyError((k,w))

def main():
 q=read(STUDY/'results/query/SELECTION_LOCK.json');r=read(STUDY/'results/reranker/SELECTION_LOCK.json');ref=read(STUDY/'results/frozen-selection.json')['evaluation']['aggregate']['NDCG@10'];qev=read(STUDY/f'results/query/{q["trial"]}/selection-{q["best_step"]:05d}.json');pure=qev['mixtures']['1']['aggregate']['NDCG@10'];candidates=[{'condition':'adapted-query','ndcg10':pure,'active_local_component':'query adapter'}]
 if q['mixture_weight']>0 and q['mixture_weight']<1:candidates.append({'condition':'mixture-query','ndcg10':q['selection_ndcg10'],'active_local_component':'query adapter plus selection-fitted mixture'})
 if r['fusion_weight']>0:
  candidates.append({'condition':'trained-reranker','ndcg10':r['trained_selection_ndcg10'],'active_local_component':'reranker adapter plus selection-fitted fusion'})
  if q['mixture_weight']>0:
   combined=lookup(read(STUDY/'results/reranker/combined-query/selection.summary.json'),r['candidates'],r['fusion_weight']);candidates.append({'condition':'trained-combined','ndcg10':combined['evaluation']['aggregate']['NDCG@10'],'active_local_component':'query and reranker adapters plus fixed fusion'})
 primary=max(candidates,key=lambda x:x['ndcg10']);released_best=read(STUDY/'results/reranker/policy_selection.json')['selected'];released=lookup(read(STUDY/'results/reranker/selected-released/selection.summary.json'),r['candidates'],r['fusion_weight']);controls=[{'condition':'released-best-pipeline','ndcg10':released_best['ndcg10'],'active_local_component':'selection-fitted fusion; released reranker unadapted','configuration':released_best},{'condition':'archived-base','ndcg10':ref,'active_local_component':None},{'condition':'released-reranker','ndcg10':released['evaluation']['aggregate']['NDCG@10'],'active_local_component':'selection-fitted fusion only; released reranker unadapted'}];recommended=max(controls+candidates,key=lambda x:(x['ndcg10'],x['condition']=='archived-base'))
 save(STUDY/'results/PIPELINE_SELECTION_LOCK.json',{'locked':True,'primary_condition':primary['condition'],'primary':primary,'active_trained_candidates':candidates,'controls':controls,'selection_recommendation':recommended,'selection_only':True,'confirmation_opened':False,'official_test_opened':False,'query_selection_lock_sha256':sha(STUDY/'results/query/SELECTION_LOCK.json'),'reranker_selection_lock_sha256':sha(STUDY/'results/reranker/SELECTION_LOCK.json'),'rule':'Highest selection NDCG among declared active locally-trained challengers, earliest tie; frozen and released controls remain real deployment candidates. Confirmation does not change any model, mixture, candidate count or evidence policy.'})
 print('Primary locked on selection only',primary,flush=True)
if __name__=='__main__':main()
