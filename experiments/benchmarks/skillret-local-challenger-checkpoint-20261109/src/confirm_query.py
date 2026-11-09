from common import *

def bootstrap(values):
 rng=np.random.default_rng(42);values=np.asarray(values,float);draws=[]
 for start in range(0,10000,100):draws.extend(values[rng.integers(0,len(values),(100,len(values)))].mean(axis=1).tolist())
 return [float(x) for x in np.percentile(draws,[2.5,97.5])]

def main():
 chosen=read(STUDY/'results/query/SELECTION_LOCK.json');root=STUDY/'results/query'
 receipt=root/'confirmation.json';panel_path=root/'confirmation-all-preregistered-mixtures.json';reload=root/'reload_check.json'
 vector_root=CACHE/'query-selected';vectors=vector_root/'confirmation-adapted.npy';ids_path=vector_root/'confirmation-ids.json'
 ids=read(STUDY/'manifests/split_membership.json')['confirmation_query_ids']
 assert len(ids)==len(set(ids))==512
 adapter=CACHE/chosen['relative_cache_path'];adapter_sha=sha(adapter/'adapter_model.safetensors')
 assert adapter_sha==chosen['adapter_sha256']
 assets=(panel_path,reload,vectors,ids_path);existing=[p.exists() for p in assets]
 saved=all(existing)
 if not saved:
  assert not any(existing) and not receipt.exists(),'Preserve partial confirmation evidence and review before any new forward'
  from engine import load,load_vectors
  from train_query import panel
  m,tok=load(adapter=adapter);docs,dids=load_vectors('train-documents');base,bids=load_vectors('train-queries')
  budget=read(STUDY/'results/data_audit.json')['query_budget']
  _,a=panel(m,tok,'confirmation',base,bids,docs,dids,budget,'query/confirmation-all-preregistered-mixtures.json')
  vector_root.mkdir(exist_ok=True);tmp=vector_root/'confirmation-adapted.tmp.npy'
  np.save(tmp,a,allow_pickle=False);tmp.replace(vectors);save(ids_path,ids)
  save(reload,{'status':'passed','adapter_sha256':adapter_sha,'fresh_process':True,'confirmation_query_count':512,'dimension':int(a.shape[1]),'finite':bool(np.isfinite(a).all()),'norm_max_error':float(np.max(np.abs(np.linalg.norm(a,axis=1)-1))),'note':'Fresh inference validates reload operation; exact selection-vector numerical parity is checked separately before final evaluation.'})
 recorded=read(panel_path);a=np.load(vectors,mmap_mode='r',allow_pickle=False);fidelity=read(reload)
 assert recorded['part']=='confirmation' and read(ids_path)==ids
 assert fidelity['status']=='passed' and fidelity['adapter_sha256']==adapter_sha and fidelity['confirmation_query_count']==512
 assert a.shape==(512,1024) and a.dtype==np.float32 and np.isfinite(a).all()
 assert np.max(np.abs(np.linalg.norm(a,axis=1)-1))<1e-5 and digest(a.tobytes().hex())==recorded['adapted_vectors_sha256']
 values=recorded['mixtures'];assert set(values)==set(recorded['rankings'])=={'0','0.25','0.5','0.75','1'}
 for key in values:assert set(values[key]['per_query'])==set(recorded['rankings'][key])==set(ids)
 key=f'{chosen["mixture_weight"]:g}';chosen_eval=values[key];ref=values['0']
 delta=[chosen_eval['per_query'][q]['NDCG@10']-ref['per_query'][q]['NDCG@10'] for q in ref['per_query']]
 ci=bootstrap(delta);point=float(np.mean(delta))
 result={'selected_before_confirmation':chosen,'frozen_ndcg10':ref['aggregate']['NDCG@10'],'chosen_ndcg10':chosen_eval['aggregate']['NDCG@10'],'delta':point,'paired_ci95':ci,'confirmed_point_improvement':point>0,'confirmed_interval_excludes_zero':ci[0]>0,'checkpoint_or_mix_changed_after_confirmation':False,'adapted_vector_npy_sha256':sha(vectors),'panel_sha256':sha(panel_path),'completed_from_saved_inference':saved}
 if receipt.exists():
  old=read(receipt);assert all(old[k]==v for k,v in result.items() if k!='completed_from_saved_inference')
 else:save(receipt,result)
 print('Query confirmation summary verified', 'using saved inference only' if saved else 'after its one inference pass', flush=True)
if __name__=='__main__':main()
