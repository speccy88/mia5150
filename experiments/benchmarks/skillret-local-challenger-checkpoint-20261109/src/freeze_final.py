"""Freeze every model/policy before the first new official test forward."""
from common import *
import time

def main():
 target=STUDY/'FINAL_EVAL_LOCK.json';assert not target.exists(),'An existing final lock is immutable; verify it rather than replacing it'
 selection=read(STUDY/'results/PIPELINE_SELECTION_LOCK.json');secondary=read(STUDY/'results/SECONDARY_SEED_LOCK.json');confirmation=read(STUDY/'results/CONFIRMATION.json');q=read(STUDY/'results/query/SELECTION_LOCK.json');r=read(STUDY/'results/reranker/SELECTION_LOCK.json');released=read(STUDY/'results/reranker/policy_selection.json')['selected'];components={}
 r={**r,'trained_configuration':f'k{r["candidates"]}-w{r["fusion_weight"]:g}','released_configuration':released['configuration']}
 for name,c in [('query',q),('reranker',r),('secondary_query',secondary['query']),('secondary_reranker',secondary['reranker'])]:
  folder=CACHE/c['relative_cache_path'];expected=c['adapter_sha256'];assert sha(folder/'adapter_model.safetensors')==expected;components[name]={'relative_cache_path':c['relative_cache_path'],'sha256':expected,'best_step':c['best_step']}
 payload={'locked':True,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'primary_seed':42,'secondary_seed':43,'primary_condition':selection['primary_condition'],'selection_recommendation':selection['selection_recommendation'],'development_promotion':confirmation['deployment_promotion'],'primary_chosen_from_selection_only':True,'test_query_count':4392,'test_skill_count':6006,'test_positive_qrel_count':7187,'test_scope':'all official exact-ID qrels, fixed denominator; failed inference scores zero','query':q,'reranker':r,'secondary_query':secondary['query'],'secondary_reranker':secondary['reranker'],'released_best_pipeline':released,'adapters':components,'embedding_source_sha256':sha(STUDY/'manifests/embedding_source.json'),'reranker_source_sha256':sha(STUDY/'manifests/reranker_source.json'),'dataset_source_sha256':sha(STUDY/'manifests/data_source.json'),'split_membership_sha256':sha(STUDY/'manifests/split_membership.json'),'protocol_sha256':sha(STUDY/'protocol.json'),'pipeline_selection_sha256':sha(STUDY/'results/PIPELINE_SELECTION_LOCK.json'),'confirmation_sha256':sha(STUDY/'results/CONFIRMATION.json'),'secondary_selection_sha256':sha(STUDY/'results/SECONDARY_SEED_LOCK.json'),'source_sha256':{p.name:sha(p) for p in sorted((STUDY/'src').glob('*.py'))},'primary_reference':'archived full-text SkillRet-Embedding-0.6B NDCG@10 .8109902409310652; own starting-checkpoint controls also retained','candidate_rules':'full original ordered6006 skill IDs, stable source-order dense ties, top50; rerank only locked topK; no gold injection or test-derived features','document_encoder':'original checkpoint only; no query adapter and no index rebuild with adaptation','query_encoder':'same original instruction, last nonpadding pooling, F32L2; query-only unmerged adapter; exact selected mixture,0/1useunchanged vectors,fractionalweightednorm','document_limit':32768,'historical_truncation':'preserve both original overlength test documents','renderer':read(STUDY/'manifests/renderer.json'),'fusion':'within-pool stable rank percentiles, fixed selected convex coefficient; original dense rank breaks ties','bootstrap':{'replicates':10000,'seed':42,'unit':'question','interval':'percentile95'},'fixed_controls':['reconstructed-base','adapted-query','mixture-query when distinct','released-reranker at identical trained settings','trained-reranker','released-best-pipeline','released-reranker-unfused','trained-reranker-unfused','secondary seed43 components','released/trained combined if primary selected combined'],'post_test_policy':'No hyperparameter, checkpoint, evidence-window, candidate or fusion selection after official results are opened. Any further tuning requires a new identified development campaign.'}
 old_protocol=OLD/'protocol.json'
 review_path=STUDY/'results/development_review.json';review=read(review_path)
 assert review['status']=='passed' and review['confirmation_receipt_sha256']==payload['confirmation_sha256'] and review['no_official_test_inference']
 payload['development_review_sha256']=sha(review_path)
 pilot_path=STUDY/'results/reranker/pilot_matched_comparison.json';pilot=read(pilot_path)
 assert pilot['status']=='passed' and pilot['queries']==512 and pilot['all_candidates_and_matched_inputs_identical']
 payload['pilot_matched_audit_sha256']=sha(pilot_path)
 if selection['primary_condition']=='trained-combined':
  combined_path=STUDY/'results/reranker/combined_selection_audit.json';combined=read(combined_path)
  assert combined['status']=='passed' and combined['queries']==512 and combined['independent_dense_and_metric_replay']
  assert review['combined_released_and_trained_exact_inputs_equal']
  payload['combined_selection_audit_sha256']=sha(combined_path)
  from score_reuse import AUDIT,AUDIT_STRIDE,REFERENCE
  if (r['budget'],r['candidates'],r['full_coverage'])==(2048,50,False):
   reuse_path=STUDY/AUDIT;reuse=read(reuse_path)
   assert reuse['status'] in ('admitted','use_original_batch_execution') and reuse['development_only']
   assert reuse['model_source_sha256']==payload['reranker_source_sha256']
   assert reuse['adapter_sha256']=={'released':None,'seed42':r['adapter_sha256'],'seed43':secondary['reranker']['adapter_sha256']}
   assert reuse['official_exact_check_stride']==AUDIT_STRIDE
   payload['shape_preserving_score_reuse']={'enabled':reuse['status']=='admitted','audit_path':AUDIT,
       'audit_sha256':sha(reuse_path),'batch_size':reuse['batch_size'],'audit_stride':AUDIT_STRIDE,'sources':REFERENCE,
       'boundary':'Offline evaluation only. Same model/adapter, exact token IDs, original tensor shape; independent uncached latency run is unchanged.'}
 payload['adapter_reload_receipts']={}
 for role,relative,count,cfg in [('query','results/query/selection_reload_parity.json',512,q),('reranker','results/reranker/adapter_reload_parity.json',64,r)]:
  path=STUDY/relative;receipt=read(path)
  assert receipt['status']=='passed' and receipt['query_count']==count and receipt['adapter_sha256']==cfg['adapter_sha256']
  payload['adapter_reload_receipts'][relative]=sha(path)
 payload['archived_protocol_sha256']=sha(old_protocol)
 payload['archived_reference_rankings_sha256']=sha(OLD/'results/baseline_skillret_06b.json')
 payload['official_test_inputs']={f'{name}.jsonl':h for name,h in read(old_protocol)['test_file_sha256'].items()}
 for name,h in payload['official_test_inputs'].items():assert sha(TRAIN_CACHE/name)==h,name
 payload['scaling']={'panel_source':SCALING_PANEL,'panel_sha256':sha(REPO/SCALING_PANEL),
     'pool_source':SCALING_POOLS,'pool_sha256':sha(REPO/SCALING_POOLS),'library_sizes':list(SCALING_SIZES),
     'query_count':20,'dense_conditions':['base','adapted-query','mixture-query'],
     'reranker_conditions':{name:list(spec) for name,spec in scaling_rerankers(selection['primary_condition']).items()}}
 inference={'document_encoder':{'repository':'ThakiCloud/SKILLRET-Embedding-0.6B','revision':MODEL_REV,'adapter':None,'max_length':32768,'pooling':'last nonpadding token','normalize':'float32 L2'},'query_encoder':{'same_base_as_document':True,'adapter':'adapters/query/best','adapter_sha256':q['adapter_sha256'],'apply_adapter_only_to_queries':True,'merge_into_document_encoder':False,'mixture_adapted_weight':q['mixture_weight'],'query_encodings':2 if 0<q['mixture_weight']<1 else 1,'instruction':PREFIX,'complete_query':True},'reranker':{'repository':'ThakiCloud/SKILLRET-Reranker-0.6B','revision':read(STUDY/'manifests/reranker_source.json')['revision'],'adapter':'adapters/reranker/best','adapter_sha256':r['adapter_sha256'],'candidates':r['candidates'],'total_input_budget':r['budget'],'full_coverage':r['full_coverage'],'fusion_weight':r['fusion_weight'],'score':'yes logit minus no logit, final hidden state and frozen two output rows','inference_batch':read(STUDY/'results/reranker_head_parity.json')['inference_batch_size']},'one_resident_model':True,'resource_note':'Switch encoder and reranker sequentially; actual latency sample includes this loading cost','primary_condition':selection['primary_condition'],'recommended_deployment_condition':confirmation['deployment_promotion']['deployment_condition'],'training_success_claim':'Only final paired results and own starting-checkpoint controls establish whether local training helped'}
 inference['conditions']={
  'archived-base':{'query_adapted_weight':0,'reranker':None},
  'adapted-query':{'query_adapted_weight':1,'reranker':None},
  'mixture-query':{'query_adapted_weight':q['mixture_weight'],'reranker':None},
  'released-best-pipeline':{'query_adapted_weight':0,'reranker':{**inference['reranker'],'adapter':None,'adapter_sha256':None,'candidates':released['candidates'],'total_input_budget':released['budget'],'full_coverage':released['full_coverage'],'fusion_weight':released['fusion_weight']}},
  'released-reranker':{'query_adapted_weight':0,'reranker':{**inference['reranker'],'adapter':None,'adapter_sha256':None}},
  'trained-reranker':{'query_adapted_weight':0,'reranker':inference['reranker']},
  'trained-combined':{'query_adapted_weight':q['mixture_weight'],'reranker':inference['reranker']},
 }
 save(STUDY/'INFERENCE_CONFIG.json',inference)
 payload['inference_config_sha256']=sha(STUDY/'INFERENCE_CONFIG.json')
 save(target,payload)
 print('Final evaluation lock created before official forwards',selection['primary_condition'],flush=True)
if __name__=='__main__':main()
