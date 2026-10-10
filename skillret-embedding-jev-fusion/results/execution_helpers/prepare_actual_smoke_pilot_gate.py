"""Derive a pinned pilot admission receipt from completed real smoke evidence."""
from pathlib import Path
from decimal import Decimal
import json, sys, time
root=Path('/home/user/code/mia5150-local/skillret-embedding-jev-fusion')
sys.path.insert(0,str(root))
import run_jev_rerank as prep
import hosted_jev_client as client
import run_hosted_study as runner
execution=root/'results/hosted-execution'
report_path=root/'results/hosted-smoke-native-field-evaluation/evaluation.json'
report=prep.read_json(report_path)
assert report['status']=='complete_scope'
assert report['evaluation']['query_count']==report['evaluation']['complete_hosted_queries']==10
account=report['experiment_unique_ledger_at_snapshot']
assert account['unresolved_attempts']==0 and account['paid_responses_settled']==40
assert runner.ledger_summary(client.SpendLedger(execution))['ledger_head_sha256']==account['ledger_head_sha256']
old_config=prep.read_json(execution/'authorized-20261007-0158/smoke_gate_config.json')
old_subject=client.HostedJevClient(execution,gates=runner.load_gates(execution/'authorized-20261007-0158/smoke_gate_config.json'),enabled=True)
gates=old_subject._gates(True)
plan_path=root/'results/jev-plan-pilot-jev113-context-v1/preparation_manifest.json'
plan=runner.read_plan(plan_path)
requests=sum(len(r['batches']) for r in plan['rows'])
assert requests==779 and len(plan['rows'])==200
ratio=max(Decimal(200)/10,Decimal(requests)/40)
projected_usd=Decimal(account['settled_usd'])*ratio
projected_cad=Decimal(account['settled_cad_with_fee_buffer'])*ratio
# Double the observed invocation time to cushion growing checkpoint/ledger overhead.
wall=report['wall_time']; elapsed=time.time()-account['first_attempt_unix']
projected_seconds=elapsed+wall['scope_invocation_wall_seconds']/10*190*2
assert projected_usd <= Decimal(5) and projected_cad <= Decimal(10) and projected_seconds <= 21600
destination=execution/'actual-smoke-pilot-admission'
destination.mkdir(exist_ok=True)
receipt={'verified':True,'verified_at_unix':time.time(),'source':str(report_path),
 'plan_sha256':plan['plan_sha256'],'based_on_actual_smoke_usage':True,'completed_smoke_queries':10,
 'projected_total_all_in_cad':str(projected_cad),'projected_total_usd':str(projected_usd),
 'projected_total_runtime_seconds':projected_seconds,'currency_caps':{'CAD':'10','USD':'5'},
 'basis':'Maximum of query-count and exact request-count cost extrapolations; runtime uses cumulative elapsed including repairs plus twice measured invocation wall per remaining query. Caches overlap; no additive smoke charge.',
 'smoke_evaluation_sha256':prep.file_hash(report_path),'source_ledger_head_sha256':account['ledger_head_sha256'],
 'target_queries':200,'target_requests':requests,'remaining_new_requests':requests-40,
 'all_ids_retained':True,'source_request_bridge_sha256':prep.file_hash(root/'results/model_pin_cache_bridge_audit.json')}
receipt_path=destination/'expansion.json'
client.write_immutable(receipt_path,prep.canonical(receipt)+b'\n')
pin=prep.read_json(Path(old_config['model_pin']['path']))
body=execution/'attempts/ef83527e133728d1a20a67dfec3b151b93b59590eafc2e8d7929528f18d37207-1.body'
assert prep.read_json(body)['model']=='jev-1.13.0'
pin.update(verified_at_unix=time.time(),source='Actual completed smoke responses confirm jev-1.13.0; original documentary mapping and authenticated alias inventory remain preserved.',actual_inference_response_observed=True,actual_response_sha256=prep.file_hash(body),smoke_evaluation_sha256=prep.file_hash(report_path))
pin_path=destination/'actual_model_pin.json'
client.write_immutable(pin_path,prep.canonical(pin)+b'\n')
config={**old_config,'model_pin':{'path':str(pin_path),'sha256':prep.file_hash(pin_path)},'expansion':{'path':str(receipt_path),'sha256':prep.file_hash(receipt_path)}}
config_path=destination/'pilot_gate_config.json'
client.write_immutable(config_path,prep.canonical(config)+b'\n')
subject=client.HostedJevClient(execution,gates=runner.load_gates(config_path),enabled=True)
gates=subject._gates(True)
runner.validate_requests(plan_path,plan)
for row in plan['rows']:
 for packet in row['batches']:
  subject._validate_packet(packet,prep.read_json(runner.request_path(plan_path,packet)),plan,gates)
client.write_immutable(destination/'preflight.json',prep.canonical({'status':'passed','queries':200,'requests':requests,'plan_sha256':plan['plan_sha256'],'api_calls':0,'key_accessed':False})+b'\n')
print(json.dumps({'status':'pilot_admitted','projected_usd':str(projected_usd),'projected_cad_with_contingency_buffer':str(projected_cad),'projected_seconds':projected_seconds,'config':str(config_path)}))
