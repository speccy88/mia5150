"""Bounded locked official campaign. Run only after reviewing development and the lock."""
from campaign_query import run
from common import *
import subprocess,sys

def cpu(script,args=()):
 p=subprocess.run([sys.executable,str(STUDY/'src'/script),*args]);assert p.returncode==0,(script,p.returncode)

def reconstruct(cap):
 review_path=STUDY/'results/RECONSTRUCTION_REVIEW.json'
 parity_path=STUDY/'results/reconstruction_parity.json'
 diagnostics_path=STUDY/'results/reconstruction_diagnostics.json'
 if review_path.exists():
  # A separately reviewed runtime difference may proceed using the same saved
  # vectors. The archived reference remains the primary contrast.
  review=read(review_path)
  assert review['decision']=='retain_single_reconstruction_and_both_controls' and review['explanation'].strip()
  assert review['evaluation_lock_sha256']==sha(STUDY/'FINAL_EVAL_LOCK.json')
  assert review['strict_parity_receipt_sha256']==sha(parity_path)
  assert review['diagnostics_sha256']==sha(diagnostics_path)
  assert read(parity_path)['status']=='runtime_difference_investigate'
  assert read(diagnostics_path)['query_count']==4392
  assert set(review['retained_array_sha256'])=={'test-documents','test-base-queries'}
  for name,h in review['retained_array_sha256'].items():assert sha(CACHE/'vectors'/name/'vectors.npy')==h,name
  assert review['reconstructed_rankings_sha256']==sha(STUDY/'results/final/reconstructed-base.jsonl')
  print('Using the reviewed single reconstruction; no new reference encodings',flush=True)
  return
 if parity_path.exists() and read(parity_path)['status']=='runtime_difference_investigate':
  if not diagnostics_path.exists():cpu('reconstruction_diagnostics.py')
  cpu('review_reference_rounding.py')
  if review_path.exists():return reconstruct(cap)
  raise RuntimeError('Review the saved ranking differences and vector checksums before proceeding; do not re-embed')
 try:run('official-reconstruct-base-01','evaluate_final.py',['baseline'],cap)
 except AssertionError:
  if parity_path.exists() and read(parity_path)['status']=='runtime_difference_investigate':return reconstruct(cap)
  raise
 cpu('reconstruction_diagnostics.py')

def main():
 complete=STUDY/'results/FINAL_EXECUTION_COMPLETE.json';ready=STUDY/'results/publication_readiness.json'
 if complete.exists():
  if not ready.exists():cpu('publication_readiness.py')
  return
 l=read(STUDY/'FINAL_EVAL_LOCK.json');assert l['locked'] and l['primary_chosen_from_selection_only'];cfg=l['reranker'];probe=rows(STUDY/f'results/reranker/{cfg["trial"]}/selection-{cfg["best_step"]:05d}.jsonl');assert len(probe)==512
 # Actual selected-policy throughput determines a generous finite final-stage limit.
 med=float(np.median([x['timing']['render20']+x['timing']['rerank20'] if cfg['candidates']==20 and x['scored_candidate_count']==50 else x['timing']['render']+x['timing']['rerank'] for x in probe]));qcap=3600;rcap=max(7200,int(3*4392*med+1800));cap={'selected_development_window_and_score_median_seconds':med,'base_reconstruction_seconds_limit':7200,'query_encoding_phase_seconds_limit':qcap,'reranking_phase_seconds_limit':rcap,'scaling_phase_seconds_limit':max(3600,int(rcap*.25)),'latency_sample_seconds_limit':3600,'reasons':'Bounded by measured development throughput with reserve for final query/pool differences; no schedule guarantee'};save(STUDY/'results/final_admission.json',cap)
 reconstruct(cap['base_reconstruction_seconds_limit'])
 run('official-adapted-query-01','evaluate_final.py',['query'],qcap)
 run('official-adapted-query-seed43-01','evaluate_final.py',['query','--secondary'],qcap)
 run('official-released-reranker-01','evaluate_final.py',['reranker','--released'],rcap)
 run('official-trained-reranker-01','evaluate_final.py',['reranker'],rcap)
 run('official-trained-reranker-seed43-01','evaluate_final.py',['reranker','--secondary'],rcap)
 if l['primary_condition']=='trained-combined':
  for name,args in [('official-released-combined-01',['reranker','--combined','--released']),('official-trained-combined-01',['reranker','--combined']),('official-trained-combined-seed43-01',['reranker','--combined','--secondary'])]:run(name,'evaluate_final.py',args,rcap)
 cpu('score_controls.py')
 run('official-scaling-01','scaling.py',[],cap['scaling_phase_seconds_limit'])
 run('official-actual-primary-latency-01','latency.py',[],cap['latency_sample_seconds_limit'])
 cpu('verify.py');cpu('analyze.py');cpu('plots.py');cpu('package_adapters.py');cpu('snapshot.py',['export']);cpu('report.py')
 save(complete,{'status':'complete','evaluation_lock_sha256':sha(STUDY/'FINAL_EVAL_LOCK.json'),'report_sha256':sha(STUDY/'REPORT.md'),'vector_archive_manifest_sha256':sha(STUDY/'vectors/manifest.json'),'publication_verified':False})
 cpu('publication_readiness.py')
 print('Locked official campaign, delivery archive and replay verification complete',flush=True)
if __name__=='__main__':main()
