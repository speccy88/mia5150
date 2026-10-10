from pathlib import Path
from decimal import Decimal
import sys,time,json
root=Path('/home/user/code/mia5150-local/skillret-embedding-jev-fusion');sys.path.insert(0,str(root))
import run_jev_rerank as prep
import hosted_jev_client as client
import run_hosted_study as runner
run=root/'results/hosted-execution';dest=run/'recovery-after-wsl-restart'
config_path=dest/'pilot_recovery_gate_config.json'
subject=client.HostedJevClient(run,gates=runner.load_gates(config_path),enabled=True)
gates=subject._gates(True)
rows=subject.ledger.rows(); allowed=subject.reviewed_unresolved_attempt_ids(rows)
assert allowed=={'f340d90743f01aac474aed7d9135f15599da30e73aa1f6bb08fed1cf6eb746dd-1'}
account=runner.ledger_summary(subject.ledger)
assert account['paid_responses_settled']==571 and account['unresolved_attempts']==1
reserve_cad,reserve_usd=subject._cost(gates['reserve_input'],2048,gates)
remaining_requests=779-571
cad=Decimal(account['settled_cad_with_fee_buffer'])+Decimal(account['held_cad_with_fee_buffer'])+remaining_requests*reserve_cad
usd=Decimal(account['settled_usd'])+Decimal(account['held_usd'])+remaining_requests*reserve_usd
smoke=prep.read_json(root/'results/hosted-smoke-native-field-evaluation/evaluation.json')
seconds=time.time()-account['first_attempt_unix']+2*54*(smoke['wall_time']['scope_invocation_wall_seconds']/10)+60
assert usd<=5 and cad<=10 and seconds<=21600
plan_path=root/'results/jev-plan-pilot-jev113-context-v1/preparation_manifest.json';plan=runner.read_plan(plan_path)
expansion={'verified':True,'verified_at_unix':time.time(),'source':'Completed actual smoke evaluation and unchanged frozen pilot, current ledger plus conservative full allowance for every remaining unique request and retained lost-request reservation.',
 'plan_sha256':plan['plan_sha256'],'based_on_actual_smoke_usage':True,'completed_smoke_queries':10,
 'projected_total_all_in_cad':str(cad),'projected_total_usd':str(usd),'projected_total_runtime_seconds':seconds,
 'projection_type':'upper_planning_commitment_allowance_not_actual_charge','remaining_unique_requests':remaining_requests,
 'lost_original_allowance_retained_once':True,'known_settled_usd':account['settled_usd'],'held_unknown_usd':account['held_usd'],
 'runtime_basis':'Elapsed from original first reservation, plus twice observed smoke wall per remaining54queries plus60second startup allowance; actual deadline unchanged.',
 'ledger_head_sha256':account['ledger_head_sha256'],'smoke_evaluation_sha256':prep.file_hash(root/'results/hosted-smoke-native-field-evaluation/evaluation.json')}
expansion_path=dest/'recovery_pilot_projection.json';client.write_immutable(expansion_path,prep.canonical(expansion)+b'\n')
config=prep.read_json(config_path);config['expansion']={'path':str(expansion_path),'sha256':prep.file_hash(expansion_path)}
new_config=dest/'pilot_recovery_gate_config_v2.json';client.write_immutable(new_config,prep.canonical(config)+b'\n')
subject=client.HostedJevClient(run,gates=runner.load_gates(new_config),enabled=True);gates=subject._gates(True)
packet=next(b for r in plan['rows'] for b in r['batches'] if b['request_sha256']=='f340d90743f01aac474aed7d9135f15599da30e73aa1f6bb08fed1cf6eb746dd')
subject._validate_packet(packet,prep.read_json(runner.request_path(plan_path,packet)),plan,gates)
print(json.dumps({'status':'recovery_admitted_without_key_or_api','usd_upper_planning_commitment':str(usd),'cad_upper_planning_commitment':str(cad),'projected_total_seconds':seconds,'config':str(new_config)}))
