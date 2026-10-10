"""Publish current local findings from completed immutable smoke/pilot evidence."""
from pathlib import Path
from decimal import Decimal
import sys,json,time,shutil
root=Path('/home/user/code/mia5150-local/skillret-embedding-jev-fusion')
sys.path.insert(0,str(root))
import run_jev_rerank as prep
import hosted_jev_client as client
import run_hosted_study as runner
results=root/'results'; execution=results/'hosted-execution'
smoke=prep.read_json(results/'hosted-smoke-native-field-evaluation/evaluation.json')
pilot=prep.read_json(results/'hosted-pilot-native-field-evaluation/evaluation.json')
report=prep.read_json(results/'hosted-study-findings-final/report_evidence.json')
account=runner.ledger_summary(client.SpendLedger(execution))
assert pilot['status']=='complete_scope' and pilot['evaluation']['complete_hosted_queries']==200
assert account['unresolved_attempts']==1 and account['paid_responses_settled']==779
assert account['paid_api_attempts_reserved']==780
assert account['unresolved_attempt_ids']==['f340d90743f01aac474aed7d9135f15599da30e73aa1f6bb08fed1cf6eb746dd-1']
assert account['ledger_head_sha256']==pilot['ledger_checkpoint_sha256']==report['ledger_head_sha256']
projected_usd=Decimal(account['settled_usd'])*Decimal(4392)/Decimal(200)+Decimal(account['held_usd'])
projected_cad=Decimal(pilot['projection']['projected_total_usage_and_held_cad_with_recorded_fee_buffer'])
projected_seconds=pilot['projection']['projected_experiment_wall_seconds_including_elapsed_gaps']
reasons=[]
if projected_usd>5:reasons.append('projected_total_usd_exceeds_5')
if projected_cad>10:reasons.append('projected_all_in_planning_cad_exceeds_10')
if projected_seconds>21600:reasons.append('projected_cumulative_runtime_exceeds_six_hours')
assert reasons, 'Full expansion may be feasible; do not finalize a blocked-full report without executing the authorized decision.'
decision={'status':'not_admitted','target_queries':4392,'completed_matched_pilot_queries':200,
 'based_on_actual_pilot_usage':True,'projected_total_usd':str(projected_usd),
 'projected_total_cad_with_contingency_and_buffer':str(projected_cad),'projected_total_seconds':projected_seconds,
 'caps':{'USD':'5','CAD':'10','seconds':21600},'reasons':reasons,
 'pilot_evaluation_sha256':prep.file_hash(results/'hosted-pilot-native-field-evaluation/evaluation.json'),
 'ledger_head_sha256':account['ledger_head_sha256'],'full_api_calls':0,
 'projection_caveat':'Linear extrapolation from deterministic 200-query pilot including existing cached smoke; not observed full-study cost. No topup/refill/RunPod or extra budget authorized.',
 'all_pilot_queries_retained':True,'original_lost_request_reservation_retained_once':True,'recorded_at_unix':time.time()}
client.write_immutable(results/'full_expansion_decision.json',prep.canonical(decision)+b'\n')
old=prep.read_json(results/'hosted_stage_status.json')
client.write_immutable(results/'hosted_stage_status_pre_paid_historical.json',prep.canonical(old)+b'\n')
old.update(status='pilot_completed_full_not_admitted_one_billing_unknown',source='Actual immutable smoke/pilot snapshots, guarded runs, and unique experiment ledger; 2026-10-07.')
old['hosted_execution']={'status':'smoke_and_pilot_completed','model_list_calls':1,
 'paid_calls':account['paid_api_attempts_reserved'],'paid_calls_definition':'Reserved transport attempts; 779 confirmed responses and one original dispatch/execution/billing unknown after local restart.',**account,'actual_model_version':'jev-1.13.0',
 'smoke_queries':10,'pilot_queries':200,'full_queries':0,'cached_smoke_requests_reused_without_charge':40,
 'cost_note':'USD is recorded token usage at the published price. CAD is a planning conversion with 25% fee/tax/FX contingency and 10% buffer, not verified account charges. No credit purchase or refill action was taken.'}
old['remaining_requirements']={'full_study':decision,'billing_authorization':'resolved: CAD10 total approved, USD5 reported-credit consumption cap; human explicitly confirms automatic refill is not enabled',
 'account_balance':'Not queried, by user direction; remaining balance is unverified.',
 'additional_user_permission_requested':False}
old['response_validation_policy']={'policy':prep.SCORE_RESPONSE_POLICY,'source_sha256':prep.file_hash(root/'run_jev_rerank.py'),
 'amendment_receipt':'hosted-execution/native_field_policy_amendment.json','uniform_all_saved_and_future_responses':True,
 'native_values_and_ranking_unchanged':True,'diagnostic_counts_in':'hosted-study-findings-final/report_evidence.json'}
old['current_reports']={'findings':'hosted-study-findings-final/findings.md','smoke':'hosted-smoke-native-field-evaluation/evaluation.json','pilot':'hosted-pilot-native-field-evaluation/evaluation.json','full_decision':'full_expansion_decision.json'}
(results/'hosted_stage_status.json').write_bytes(prep.canonical(old)+b'\n')
metrics=pilot['evaluation']['systems']; audit=report['response_validation']['pilot']
ci=pilot['evaluation']['paired_bootstrap']['C_minus_A']['metrics']['NDCG@10']
conclusion=(f"Dense retrieval has the highest pilot NDCG@10 point estimate. Jev reduced NDCG@10 by {metrics['A']['aggregate']['NDCG@10']-metrics['B']['aggregate']['NDCG@10']:.6f}; "
 f"RRF60 changed it by {ci['delta']:.6f} (95% paired bootstrap CI [{ci['ci_low']:.6f}, {ci['ci_high']:.6f}]). "
 f"RRF60 did not demonstrate an NDCG improvement, although its Recall@10 increased from {metrics['A']['aggregate']['Recall@10']:.6f} to {metrics['C']['aggregate']['Recall@10']:.6f}. These conclusions apply to the fixed 200-query pilot.")
lead=(f"The hosted smoke (10 queries) and pilot (200 queries) are complete with jev-1.13.0. All 200 pilot IDs and their fixed dense top-20 pools were retained. "
 f"Pilot NDCG@10 is {metrics['A']['aggregate']['NDCG@10']:.6f} for dense retrieval, {metrics['B']['aggregate']['NDCG@10']:.6f} for Jev, and {metrics['C']['aggregate']['NDCG@10']:.6f} for fixed RRF60. "
 f"The experiment has {account['paid_responses_settled']} usable scoring responses from {account['paid_api_attempts_reserved']} reserved transport attempts, including the 40 smoke responses reused without another charge. Known token usage is USD {account['settled_usd']} at the published rate, or CAD {account['settled_cad_with_fee_buffer']} after planning allowances. One original request lost during a WSL restart remains execution/billing unknown; its full USD {account['held_usd']} / CAD {account['held_cad_with_fee_buffer']} reservation is retained. An identical-payload recovery supplied the first available response without dropping a query. "
 f"The 4,392-query hosted expansion was not run: its pilot-based projection is USD {projected_usd} / CAD {projected_cad}, above the USD 5 / CAD 10 caps. "
 "Read [actual findings](results/hosted-study-findings-final/findings.md), [pilot evaluation](results/hosted-pilot-native-field-evaluation/evaluation.json), and [full expansion decision](results/full_expansion_decision.json).")
lead=conclusion+'\n\n'+lead
readme=(root/'README.md').read_text()
start=readme.index('Current billing reassessment:')
end=readme.index('The portable archive',start)
baseline=readme[readme.index('The full local baseline',start):readme.index('The paid hosted comparison',start)].strip()
readme=readme[:start]+lead+'\n\n'+baseline+'\n\n'+(
 'Two HTTP 200 responses exposed unsupported response-arithmetic assumptions. Both raw responses were preserved and reconciled without replay. The final policy validates structure, model/rubric, ranges, usage, hashes and ledger provenance while recording probability/score/confidence arithmetic as diagnostics. It preserves all native values and the original score/confidence/dense-rank ordering. See [response validation audit](SCORE_RESPONSE_AUDIT.md); the amendment was frozen before comparative quality metrics were inspected.\n\n')+readme[end:]
old_paragraph='The baseline gate has passed and explicit CAD 10 all-in budget approval has been relayed.'
start=readme.index(old_paragraph);end=readme.index('\n\n',start)
readme=readme[:start]+('Budget and process-only key handoff were explicitly approved. The user confirmed automatic refill is disabled and directed acceptance of reported USD 5 signup credit without a balance query. Current public pricing and FX receipts, an additional USD 5 consumption cap, and planning contingencies were enforced. These are not a live account-balance observation. No top-up, refill or RunPod action was taken. Historical billing assessments are superseded by the actual [stage status](results/hosted_stage_status.json).')+readme[end:]
readme=readme.replace('for the completed model-list request.', 'for the authorized model-list and scoring processes.')
readme=readme.replace('supports offline request preparation and strict ingestion of genuine saved responses', 'supports offline request preparation and structurally validated ingestion of genuine saved responses with arithmetic diagnostics')
(root/'README.md').write_text(readme)
(root/'BUDGET_STATUS.md').write_text('# Actual hosted budget status\n\n'+lead+'\n\n'
 'CAD 10 total authorization, process-only saved-key handoff, and the user confirmation that automatic refill is disabled are preserved in the experiment receipts. The user-reported USD 5 signup credit is an additional consumption cap; it is not a verified remaining balance. The key was supplied only as TYPESAFE_API_KEY to the authorized child and removed on exit. No key value was printed or exported.\n\n'
 'Price: USD 0.042 per million input tokens; output free. Reference FX: CAD 1.4226/USD (Bank of Canada, 2026-10-06). The 25% fee/tax/FX contingency and 10% buffer are planning allowances, not observed card fees or invoices. Actual billed account debits were not independently queried. No credit purchase, refill or RunPod was performed.\n\n'
 'The append-only ledger counts unique transport attempts across scopes. Alias/pinned cache reuse costs zero and smoke is included in the pilot total. Earlier parser reservations were settled from their saved HTTP 200 responses and original usage without resending. Separately, a WSL restart lost one in-flight response; the single reviewed identical-payload replacement may duplicate the original provider execution/charge. The original allowance remains fully held and is never fabricated as settled usage or released. All 200 queries are complete, but billing reconciliation is incomplete for that original attempt. See RECOVERY_POLICY.md and the pinned recovery receipt. Historical zero-paid status is preserved in results/hosted_stage_status_pre_paid_historical.json and prior published versions.\n')
for name in ['BILLING_GATE_REASSESSMENT.md','API_READINESS.md','FEES_SOURCE_AUDIT.md']:
 path=root/name; body=path.read_text()
 notice='> Historical assessment. The later authorized smoke and pilot are complete. Current evidence and budget outcome are in [BUDGET_STATUS.md](BUDGET_STATUS.md) and [actual findings](results/hosted-study-findings-final/findings.md). Earlier pending-account requirements below are superseded.\n\n'
 path.write_text(notice+body)
findings=results/'hosted-study-findings-final/findings.md'
heading,body=findings.read_text().split('\n',1)
findings.write_text(heading+'\n\n'+conclusion+'\n\n'+body)
draft=results/'hosted-study-findings/findings.md'
draft.write_text('> Superseded draft. Use [the final findings](../hosted-study-findings-final/findings.md), which correct the label on interrupted-run residual wall-time bounds. Metrics and native rankings are unchanged.\n\n'+draft.read_text())
validation={'status':'passed','command':"../.venv/bin/python -m unittest discover -p 'test_*.py'",'tests':170,'elapsed_seconds':58.404,'exit_code':0,
 'log':'final-hosted-offline-tests.txt','log_sha256':prep.file_hash(results/'final-hosted-offline-tests.txt'),
 'scope':'Final combined local test suite including native response diagnostics, client/runner recovery, evaluation intervals, metric and baseline tests. A subsequent report-label-only correction changes displayed timing-field selection, not metrics or request behavior.',
 'hosted_network_calls_in_tests':0,'source_sha256':{p.name:prep.file_hash(p) for p in root.glob('*.py')}}
client.write_immutable(results/'final-hosted-offline-validation.json',prep.canonical(validation)+b'\n')
print(json.dumps({'status':'pilot_finalized_full_not_admitted','paid_calls':account['paid_responses_settled'],'full_projection_usd':str(projected_usd),'full_projection_cad':str(projected_cad),'response_validation_pair_modes':audit['pair_mode_counts']}))

