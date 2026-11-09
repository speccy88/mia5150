"""Locked pipeline comparisons, optionally using the selected query adapter."""
from train_reranker import *
import argparse

def reuse_released_selection(cfg,tag):
 selected=read(STUDY/'results/reranker/policy_selection.json')['selected']
 assert (selected['budget'],selected['full_coverage'])==(cfg['budget'],cfg['full_coverage'])
 summary_path=STUDY/selected['summary'];source=summary_path.with_name(summary_path.name.replace('.summary.json','.jsonl'))
 released=rows(source);trained=rows(STUDY/f'results/reranker/{cfg["trial"]}/selection-{cfg["best_step"]:05d}.jsonl')
 assert len(released)==len(trained)==512
 k=cfg['candidates']
 for a,b in zip(released,trained,strict=True):
  assert a['query_id']==b['query_id'] and a['status']==b['status']=='ok'
  assert a['candidates']==b['candidates'] and a['dense_scores']==b['dense_scores']
  assert len(a['reranker_scores'])>=k and len(b['reranker_scores'])>=k and a['selections'][:k]==b['selections'][:k]
 target=STUDY/'results/reranker'/tag
 summary=read(summary_path)
 assert any(key.startswith(f'k{k}-') for key in summary)
 save(target.with_suffix('.summary.json'),summary)
 save(target.parent/'reused_scores.json',{'source_records':source.relative_to(STUDY).as_posix(),'source_records_sha256':sha(source),'source_summary_sha256':sha(summary_path),'same_candidates_and_input_hashes':True,'queries':len(released),'matched_scored_skills':len(released)*k,'inference_repeated':False})
 print('Reused the released selection scores with verified identical candidates and inputs',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--part',choices=['selection','confirmation'],default='selection');p.add_argument('--released',action='store_true');p.add_argument('--adapted-query',action='store_true');p.add_argument('--tag',required=True);a=p.parse_args();cfg=read(STUDY/'results/reranker/SELECTION_LOCK.json');override=None
 if a.part=='selection' and a.released and not a.adapted_query:
  reuse_released_selection(cfg,a.tag)
  return
 if a.adapted_query:
  qc=read(STUDY/'results/query/SELECTION_LOCK.json');weight=qc['mixture_weight'];b,bids=load_vectors('train-queries');bmap={qid:i for i,qid in enumerate(bids)};ids=read(CACHE/'query-selected'/f'{a.part}-ids.json');adapt=np.load(CACHE/'query-selected'/f'{a.part}-adapted.npy');base=np.array([b[bmap[i]] for i in ids]);v=mix_query_vectors(base,adapt,weight)
  override=(v,ids)
 adapter=None if a.released else CACHE/cfg['relative_cache_path'];provenance=None;reuse_source=None
 if a.part=='confirmation':
  from score_reuse import AUDIT,CONFIRM_REFERENCE
  model_hash=sha(STUDY/'manifests/reranker_source.json');adapter_hash=None if a.released else cfg['adapter_sha256']
  if adapter is not None:assert sha(adapter/'adapter_model.safetensors')==adapter_hash
  provenance={'reranker_source_manifest_sha256':model_hash,'reranker_adapter_sha256':adapter_hash}
  gate_path=STUDY/AUDIT
  if a.adapted_query and gate_path.exists():
   gate=read(gate_path)
   if gate['status']=='admitted':
    assert gate['model_source_sha256']==model_hash and gate['adapter_sha256']['released' if a.released else 'seed42']==adapter_hash
    assert gate['batch_size']==admitted_inference_batch()
    reuse_source=CONFIRM_REFERENCE[Path(a.tag).parent.name]
 m,tok=load('reranker',adapter=adapter);score_panel(m,tok,a.part,a.tag,cfg['budget'],max(cfg['candidates'],read(STUDY/'results/reranker/policy_selection.json')['selected']['candidates']) if a.released and not a.adapted_query else cfg['candidates'],cfg['full_coverage'],query_override=override,provenance=provenance,reuse_source=reuse_source)
if __name__=='__main__':main()
