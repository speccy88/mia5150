"""Bounded policy comparison and local reranker training, no official scores."""
from campaign_query import run
from common import *
import time

def best_record(path,budget,full=False):
 r=read(path);items=[]
 for key,z in r.items():
  k=int(key.split('-')[0][1:]);w=float(key.split('-')[1][1:]);items.append({'budget':budget,'full_coverage':full,'candidates':k,'fusion_weight':w,'ndcg10':z['evaluation']['aggregate']['NDCG@10'],'configuration':key,'summary':str(path.relative_to(STUDY))})
 return max(items,key=lambda x:(x['ndcg10'],-x['candidates'],-x['budget'],-x['fusion_weight']))

def main():
 deadline=time.monotonic()+30000
 while not(STUDY/'results/query/campaign_complete.json').exists():
  assert time.monotonic()<deadline,'Dependency wait expired';time.sleep(15)
 # The query controller writes campaign completion after the guard has reaped
 # its child. Guard the next launch so an unrelated job gets a resume command.
 run('query-reload-check-01','reload_query.py',[],1800)
 run('reranker-parity-01','reranker.py',[],1200)
 parity=read(STUDY/'results/reranker_head_parity.json');times=parity['budget_admission'];batch=int(parity['inference_batch_size']);budget={'measured_admission':times,'development_queries_per_panel':512,'maximum_candidate_count':50,'inference_batch':batch,'per_development_policy_time_limit_seconds':max(3600,int(max(x['inference_batch_profile'][str(batch)]['median_seconds_20_pairs'] for x in times)/20*512*50*2+600)),'pilot_training_steps':100,'full_training_maximum_steps':600,'checkpoint_storage_limit_bytes':1024**3,'all_execution_local':True};save(STUDY/'results/reranker/admission.json',budget)
 run('reranker-train-negative-mining-01','mine_reranker.py',[],3600)
 run('reranker-train-negative-audit-01','verify_mining.py',[],600)
 candidates=[]
 sizes=sorted(x['tokens'] for x in times);assert sizes in ([1024],[1024,2048])
 for size in sizes:
  run(f'released-reranker-dev{size}-01','development_reranker.py',['--mode',f'released{size}'],budget['per_development_policy_time_limit_seconds']);candidates.append(best_record(STUDY/f'results/reranker/released{size}/selection.summary.json',size))
 selected=max(candidates,key=lambda x:(x['ndcg10'],-x['candidates'],-x['budget'],-x['fusion_weight']))
 full_budget=max(sizes)
 run('released-reranker-fullcoverage-dev-01','development_reranker.py',['--mode','full-coverage','--budget',full_budget],3*budget['per_development_policy_time_limit_seconds'])
 full=best_record(STUDY/'results/reranker/released-fullcoverage/selection.summary.json',full_budget,True);selected_rows=rows(STUDY/f'results/reranker/released{selected["budget"]}/selection.jsonl');full_rows=rows(STUDY/'results/reranker/released-fullcoverage/selection.jsonl');selected_times=[x['timing']['rerank20']+x['timing']['render20'] if selected['candidates']==20 else x['timing']['rerank']+x['timing']['render'] for x in selected_rows];ratio=float(np.median([x['timing']['rerank']+x['timing']['render'] for x in full_rows])/np.median(selected_times));promote_full=full['ndcg10']>=selected['ndcg10']+.005 and ratio<=2
 if promote_full:selected=full
 save(STUDY/'results/reranker/policy_selection.json',{'selected':selected,'selected_evidence_candidates':candidates,'full_coverage_candidate':full,'full_vs_selected_median_rerank_time_ratio':ratio,'full_coverage_promoted':promote_full,'rule':'full coverage requires >=.005 selectionNDCG gain and <=2x median measured rendering+scoring time at the selected candidate count'})
 args=['--name','bce-pilot','--steps','100','--budget',selected['budget'],'--candidates',selected['candidates']]
 if selected['full_coverage']:args+=['--full-coverage']
 trainstep=next(x['forward_backward_update_seconds'] for x in times if x['tokens']==selected['budget'])*16;maxpilot=max(1800,int(100*trainstep*2+budget['per_development_policy_time_limit_seconds']+600));save(STUDY/'results/reranker/training_admission.json',{'measured_single_pair_update_seconds':trainstep/16,'pilot_max_steps':100,'pilot_wall_seconds_limit':maxpilot,'full_max_steps':600,'full_wall_seconds_limit':6*maxpilot,'microbatch_pairs':1,'accumulation':16,'full_coverage_recomputation_cost':'additional all-window forward work when full coverage is selected; guarded by wall limit'})
 run('reranker-bce-pilot-01','train_reranker.py',args,maxpilot);best=read(STUDY/'results/reranker/bce-pilot/best.json');name='bce-pilot'
 run('reranker-bce-pilot-matched-audit-01','audit_reranker_pilot.py',[],600)
 if best['ndcg10']>selected['ndcg10']+1e-12:
  full_name='bce-full';args[args.index('bce-pilot')]=full_name;args[args.index('100')]='600';run('reranker-bce-full-01','train_reranker.py',args,6*maxpilot);full_best=read(STUDY/f'results/reranker/{full_name}/best.json')
  if full_best['ndcg10']>best['ndcg10']+1e-12:name=full_name;best=full_best
 folder=CACHE/'checkpoints/reranker'/name/'best';assert sha(folder/'adapter_model.safetensors')==best['adapter_sha256'];config=best['configuration'];chosen={**selected,'trial':name,'relative_cache_path':str(folder.relative_to(CACHE)),'adapter_sha256':best['adapter_sha256'],'best_step':best['step'],'trained_configuration':config,'released_configuration':selected['configuration'],'candidates':int(config.split('-')[0][1:]),'fusion_weight':float(config.split('-')[1][1:]),'trained_selection_ndcg10':best['ndcg10'],'released_selected_ndcg10':selected['ndcg10']};save(STUDY/'results/reranker/SELECTION_LOCK.json',chosen)
 run('trained-reranker-reload-check-01','reload_reranker.py',[],budget['per_development_policy_time_limit_seconds'])
 # Identical selected trained-policy inputs/pools for the unadapted starting control.
 run('released-reranker-selected-control-01','pipeline_panel.py',['--released','--tag','selected-released/selection.jsonl'],budget['per_development_policy_time_limit_seconds'])
 run('trained-combined-query-dev-01','pipeline_panel.py',['--adapted-query','--tag','combined-query/selection.jsonl'],budget['per_development_policy_time_limit_seconds'])
 run('trained-combined-query-dev-audit-01','audit_combined_selection.py',[],600)
 # Confirmation is deferred until both primary routes and the fixed secondary recipes are selected.
 run('pipeline-selection-lock-01','freeze_pipeline.py',[],600)
 save(STUDY/'results/reranker/campaign_complete.json',{'status':'selected','selected':chosen,'official_test_inference_run':False,'confirmation_opened':False})
 print('Reranker development campaign complete',flush=True)
if __name__=='__main__':main()
