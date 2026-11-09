"""Fresh-process best-adapter logits/rankings versus its original selected state."""
from train_reranker import *

def main():
 cfg=read(STUDY/'results/reranker/SELECTION_LOCK.json');original=rows(STUDY/f'results/reranker/{cfg["trial"]}/selection-{cfg["best_step"]:05d}.jsonl');m,tok=load('reranker',adapter=CACHE/cfg['relative_cache_path']);score_panel(m,tok,'selection','reload-selection/panel.jsonl',cfg['budget'],cfg['candidates'],cfg['full_coverage'],limit=64);fresh=rows(STUDY/'results/reranker/reload-selection/panel.jsonl');old={x['query_id']:x for x in original};maxerr=0;k=cfg['candidates'];weight=cfg['fusion_weight']
 for row in fresh:
  a=old[row['query_id']];assert row['candidates']==a['candidates'] and row['dense_scores']==a['dense_scores'];assert row['selections']==a['selections'][:k];delta=max(abs(x-y) for x,y in zip(row['reranker_scores'][:k],a['reranker_scores'][:k]));maxerr=max(maxerr,delta);assert delta<1e-6
  before=[a['candidates'][i] for i in normalized_fusion(a['dense_scores'][:k],a['reranker_scores'][:k],weight)]+a['candidates'][k:];after=[row['candidates'][i] for i in normalized_fusion(row['dense_scores'][:k],row['reranker_scores'][:k],weight)]+row['candidates'][k:];assert before==after
 ids=read(STUDY/'manifests/split_membership.json')['selection_query_ids'][:64]
 assert len(original)==512 and [r['query_id'] for r in fresh]==ids
 save(STUDY/'results/reranker/adapter_reload_parity.json',{'status':'passed','query_count':64,'query_ids':ids,'fixture_rule':'First 64 published selection IDs, fixed before any trained reranker selection output; full512 used for quality selection','candidate_count':k,'max_score_error':maxerr,'input_selection_identical':True,'rankings_identical':True,'adapter_sha256':cfg['adapter_sha256'],'fresh_process':True})
 print('Reranker fresh-process reload parity passed64 fixed queries',flush=True)
if __name__=='__main__':main()
