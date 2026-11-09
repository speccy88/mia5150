"""Open confirmation once after all recipe and primary selection is frozen."""
from campaign_query import run
from freeze_pipeline import lookup
from confirm_query import bootstrap
from common import *

def paired_effect(start,end):
 assert set(start['per_query'])==set(end['per_query'])
 d=[end['per_query'][q]['NDCG@10']-start['per_query'][q]['NDCG@10'] for q in start['per_query']]
 delta=float(np.mean(d));ci=bootstrap(d)
 return {'ndcg10_delta':delta,'paired_ci95':ci,'positive_interval':delta>0 and ci[0]>0}

def main():
 fixed=read(STUDY/'results/PIPELINE_SELECTION_LOCK.json');assert fixed['locked'] and fixed['selection_only'];assert (STUDY/'results/SECONDARY_SEED_LOCK.json').exists();assert not(STUDY/'results/CONFIRMATION.json').exists()
 cfg=read(STUDY/'results/reranker/SELECTION_LOCK.json')
 if fixed['primary_condition']=='trained-combined' and (cfg['budget'],cfg['candidates'],cfg['full_coverage'])==(2048,50,False):
  run('reranker-shape-preserving-reuse-check-01','score_reuse.py',[],3600)
 run('query-confirmation-01','confirm_query.py',[],1800)
 budget=read(STUDY/'results/reranker/admission.json')['per_development_policy_time_limit_seconds']
 for released,tag in [(True,'released-confirmation'),(False,'trained-confirmation')]:
  args=['--part','confirmation','--tag',tag+'/panel.jsonl']+(['--released'] if released else []);run(tag+'-01','pipeline_panel.py',args,budget)
 if fixed['primary_condition']=='trained-combined':
  for released,tag in [(True,'released-combined-confirmation'),(False,'trained-combined-confirmation')]:
   args=['--part','confirmation','--adapted-query','--tag',tag+'/panel.jsonl']+(['--released'] if released else []);run(tag+'-01','pipeline_panel.py',args,budget)
 baseline=read(STUDY/'results/query/confirmation-all-preregistered-mixtures.json')['mixtures']['0'];out={};matched={}
 for tag in ('released-confirmation','trained-confirmation','released-combined-confirmation','trained-combined-confirmation'):
  p=STUDY/'results/reranker'/tag/'panel.summary.json'
  if not p.exists():continue
  result=lookup(read(p),cfg['candidates'],cfg['fusion_weight'])['evaluation'];matched[tag]=result;e=paired_effect(baseline,result);out[tag]={'ndcg10':result['aggregate']['NDCG@10'],'ndcg10_delta_vs_frozen':e['ndcg10_delta'],'paired_ci95':e['paired_ci95']}
 release_cfg=read(STUDY/'results/reranker/policy_selection.json')['selected'];release_summary=read(STUDY/'results/reranker/released-confirmation/panel.summary.json');release_best=lookup(release_summary,release_cfg['candidates'],release_cfg['fusion_weight'])['evaluation'];qeval=read(STUDY/'results/query/confirmation-all-preregistered-mixtures.json')['mixtures'];qcfg=read(STUDY/'results/query/SELECTION_LOCK.json');mix_key=str(qcfg['mixture_weight']).rstrip('0').rstrip('.') if float(qcfg['mixture_weight']).is_integer() else str(qcfg['mixture_weight']);recommended=fixed['selection_recommendation']['condition'];confirm_values={'archived-base':baseline,'released-best-pipeline':release_best,'released-reranker':lookup(release_summary,cfg['candidates'],cfg['fusion_weight'])['evaluation'],'adapted-query':qeval['1'],'mixture-query':qeval[mix_key],'trained-reranker':lookup(read(STUDY/'results/reranker/trained-confirmation/panel.summary.json'),cfg['candidates'],cfg['fusion_weight'])['evaluation']}
 if 'trained-combined-confirmation' in out:confirm_values['trained-combined']=lookup(read(STUDY/'results/reranker/trained-combined-confirmation/panel.summary.json'),cfg['candidates'],cfg['fusion_weight'])['evaluation']
 effects={'pure-query-vs-own-start':paired_effect(baseline,qeval['1']),'selected-query-mixture-vs-own-start':paired_effect(baseline,qeval[mix_key]),'reranker-vs-own-start':paired_effect(matched['released-confirmation'],matched['trained-confirmation'])}
 if 'trained-combined-confirmation' in matched:
  effects['combined-reranker-vs-own-start']=paired_effect(matched['released-combined-confirmation'],matched['trained-combined-confirmation'])
  effects['combined-query-vs-own-start']=paired_effect(matched['trained-confirmation'],matched['trained-combined-confirmation'])
 required={'adapted-query':['pure-query-vs-own-start'],'mixture-query':['selected-query-mixture-vs-own-start'],'trained-reranker':['reranker-vs-own-start'],'trained-combined':['combined-reranker-vs-own-start','combined-query-vs-own-start']}.get(recommended,[])
 recommended_eval=confirm_values[recommended];vs_baseline=paired_effect(baseline,recommended_eval)
 promotion={'selection_candidate':recommended,'confirmation_ndcg10':recommended_eval['aggregate']['NDCG@10'],'confirmation_delta':vs_baseline['ndcg10_delta'],'paired_ci95':vs_baseline['paired_ci95'],'required_training_effects':required,'promoted':recommended!='archived-base' and vs_baseline['positive_interval'] and all(effects[k]['positive_interval'] for k in required),'rule':'The selection candidate must beat frozen dense with a positive paired interval. Any active local training must also improve on its matched unchanged starting control with a positive paired interval. Otherwise retain frozen dense deployment. Confirmation never changes the research test challenger.'};promotion['deployment_condition']=recommended if promotion['promoted'] else 'archived-base'
 from score_reuse import verify_confirmation_records
 reuse=verify_confirmation_records(STUDY)
 save(STUDY/'results/CONFIRMATION.json',{'selection_lock_sha256':sha(STUDY/'results/PIPELINE_SELECTION_LOCK.json'),'query_confirmation_sha256':sha(STUDY/'results/query/confirmation.json'),'baseline_ndcg10':baseline['aggregate']['NDCG@10'],'pipelines':out,'local_training_effects':effects,'deployment_promotion':promotion,'all_settings_unchanged_after_confirmation':True,'used_for_retuning':False,'offline_score_reuse_verified':reuse})
 print('Single frozen confirmation suite complete; no settings changed',flush=True)
if __name__=='__main__':main()
