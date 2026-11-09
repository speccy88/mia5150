"""Repeat the selected recipes with seed43 before any official result is opened."""
from campaign_query import run
from common import *

def main():
 assert not(STUDY/'FINAL_EVAL_LOCK.json').exists(),'Finish secondary training before final evaluation lock'
 q=read(STUDY/'results/query/SELECTION_LOCK.json');qs=read(STUDY/f'results/query/{q["trial"]}/summary.json');qb=read(STUDY/'results/query/admission.json');qsteps=1200 if q['trial']=='full-selected-lr' else 100;qbudget=qb['full_time_limit_seconds'] if qsteps==1200 else qb['pilot_time_limit_seconds'];run('query-secondary-seed43-01','train_query.py',['--name','secondary-seed43','--seed','43','--lr',qs['lr'],'--steps',qsteps,'--fixed-checkpoint-step',q['best_step']],qbudget)
 qbest=read(STUDY/'results/query/secondary-seed43/best.json');rr=read(STUDY/'results/reranker/SELECTION_LOCK.json');rb=read(STUDY/'results/reranker/training_admission.json');rsteps=600 if rr['trial']=='bce-full' else 100;rtime=rb['full_wall_seconds_limit'] if rsteps==600 else rb['pilot_wall_seconds_limit'];args=['--name','secondary-seed43','--seed','43','--steps',rsteps,'--budget',rr['budget'],'--candidates',rr['candidates'],'--fixed-checkpoint-step',rr['best_step'],'--fixed-repeat-no-dev-scoring']
 if rr['full_coverage']:args+=['--full-coverage']
 run('reranker-secondary-seed43-01','train_reranker.py',args,rtime);rbest=read(STUDY/'results/reranker/secondary-seed43/best.json')
 assert qbest['step']==q['best_step'] and rbest['step']==rr['best_step'] and not rbest['selection_inference_performed']
 for role,steps,fixed in [('query',qsteps,q['best_step']),('reranker',rsteps,rr['best_step'])]:
  trial=read(STUDY/f'results/{role}/secondary-seed43/summary.json')
  assert trial['seed']==43 and trial['schedule_max_steps']==steps and trial['steps']==trial['fixed_checkpoint_step']==fixed
 record={'primary_seed':42,'secondary_seed':43,'query':{**q,'trial':'secondary-seed43','relative_cache_path':'checkpoints/query/secondary-seed43/best','adapter_sha256':qbest['adapter_sha256'],'best_step':qbest['step'],'mixture_weight':q['mixture_weight'],'selection_checkpoint_rule':'fixed primary optimizer step and learning-rate schedule; deployment mixture fixed to primary'},'reranker':{**rr,'trial':'secondary-seed43','relative_cache_path':'checkpoints/reranker/secondary-seed43/best','adapter_sha256':rbest['adapter_sha256'],'best_step':rbest['step'],'selection_checkpoint_rule':'fixed primary optimizer step and learning-rate schedule; deployment pool/window/fusion fixed to primary'},'secondary_checkpoint_steps_fixed_to_primary':True,'test_used_to_choose_seed':False,'deployment_settings_frozen_to_primary':True};save(STUDY/'results/SECONDARY_SEED_LOCK.json',record)
 print('Both selected recipes repeated with seed43 before official scoring',flush=True)
if __name__=='__main__':main()
