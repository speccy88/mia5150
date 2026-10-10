"""Freeze a response-only validation amendment before any quality readout."""
from pathlib import Path
import sys,time,json
root=Path('/home/user/code/mia5150-local/skillret-embedding-jev-fusion')
sys.path.insert(0,str(root))
import run_jev_rerank as prep
import hosted_jev_client as client
execution=root/'results/hosted-execution'
policy=getattr(prep,'SCORE_RESPONSE_POLICY',None) or getattr(prep,'SCORE_VALIDATION_POLICY',None)
assert policy=='native-fields-arithmetic-diagnostics-v1'
receipt={
 'status':'response_validation_amendment_frozen_before_quality_readout',
 'recorded_at_unix':time.time(),
 'policy':policy,
 'code_policy_constant':policy,
 'validator_source_sha256':prep.file_hash(root/'run_jev_rerank.py'),
 'trigger_response_sha256':'62c8c20b1156425d54f8f7bd88c344099efd2c8381a67f84cedabac7d310a170',
 'trigger_values':{'probabilities':[0,.15,.83,.02,0],'native_score':1.87,'native_confidence':.84},
 'finding':'Reported weighted score equals 1.87. Under nearest-cent intervals mode2 is fixed and MAD <=.185, so confidence >=.8458333; observed .84 requires confidence <=.845. The universal nearest-cent hypothesis is falsified. Cause unknown.',
 'hard_gates':['exact answer IDs','exact five-level rubric/legend and probability keys','numeric finite probability/confidence in [0,1] and native score in [0,4]','pinned actual model','valid usage','exact request/response hashes','ledger provenance','budget and resource limits'],
 'arithmetic_diagnostics':['probability sum and normalization residual','reported versus recomputed weighted score','reported versus documented first-mode MAD confidence','strict consistency','joint nearest-cent feasibility'],
 'all_saved_and_future_responses_same_policy':True,
 'reported_values_preserved':True,'native_ranking_rule_unchanged':True,'request_preparation_unchanged':True,
 'no_metric_based_selection':True,'quality_comparison_metrics_inspected_before_amendment':False,
 'dropped_queries':0,'replayed_paid_requests':0,
 'independent_reviewers':['jev_context_primary_audit','baseline_quality_audit'],
 'limitation':'Evaluates the hosted service as returned; does not claim confidence/probabilities satisfy the documented arithmetic. Confidence remains the originally fixed secondary tie break. All mismatch counts and residuals are reportable.'}
client.write_immutable(execution/'native_field_policy_amendment.json',prep.canonical(receipt)+b'\n')
print(json.dumps({'status':receipt['status'],'source_sha256':receipt['validator_source_sha256'],'api_calls':0,'key_accessed':False}))
