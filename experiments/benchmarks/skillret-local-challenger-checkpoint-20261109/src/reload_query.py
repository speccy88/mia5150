"""Compare reloaded best query adapter with its original selection rankings."""
from engine import *
def main():
 from train_query import panel
 cfg=read(STUDY/'results/query/SELECTION_LOCK.json');name=cfg['trial'];original=read(STUDY/f'results/query/{name}/selection-{cfg["best_step"]:05d}.json');m,tok=load(adapter=CACHE/cfg['relative_cache_path']);base,bids=load_vectors('train-queries');docs,dids=load_vectors('train-documents');fresh,vec=panel(m,tok,'selection',base,bids,docs,dids,read(STUDY/'results/data_audit.json')['query_budget'],'query/reload-selection.json');reloaded=read(STUDY/'results/query/reload-selection.json')
 for w,ev in fresh.items():
  assert reloaded['rankings'][w]==original['rankings'][w]
  for k,v in ev['aggregate'].items():assert abs(v-original['mixtures'][w]['aggregate'][k])<1e-12
 assert reloaded['adapted_vectors_sha256']==original['adapted_vectors_sha256']
 p=CACHE/'query-selected';p.mkdir(exist_ok=True);np.save(p/'selection-adapted.npy',vec);save(p/'selection-ids.json',read(STUDY/'manifests/split_membership.json')['selection_query_ids'])
 save(STUDY/'results/query/selection_reload_parity.json',{'status':'passed','adapter_sha256':cfg['adapter_sha256'],'query_count':512,'all_five_preregistered_mixture_rankings_identical':True,'adapted_vectors_byte_identical':True,'all_metrics_identical':True})
 print('Reload parity passed512 queries5mixtures',flush=True)
if __name__=='__main__':main()
