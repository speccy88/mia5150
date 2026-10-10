from pathlib import Path
import sys,shutil,json
root=Path('/home/user/code/mia5150-local/skillret-embedding-jev-fusion');sys.path.insert(0,str(root))
import run_jev_rerank as prep
import hosted_jev_client as client
source=Path('/mnt/c/Users/user/Documents/Codex/2026-10-06/task-3')
shutil.copy2(source/'skillret-embedding-jev-fusion/results/pilot-independent-validation.json',root/'results/pilot-independent-validation.json')
for path in [root/'README.md',root/'BUDGET_STATUS.md',root/'results/hosted-study-findings-final/findings.md']:
 text=path.read_text().replace('Jev reduced NDCG@10 by -0.136732','Jev reduced NDCG@10 by 0.136732')
 text=text.replace('Serialized native confidence need not reproduce a self-imposed arithmetic formula from rounded probabilities.', 'The displayed score, confidence and probability fields do not always satisfy the documented arithmetic; the provider gives no exact serialization precision contract.')
 path.write_text(text)
resource=prep.read_json(root/'results/hosted-resource-audit.json')
metrics=[r['logged_telemetry']['metrics'] for r in resource['runs']]
summary={'minimum_host_available_gib':min(m['minimum_host_available_gib'] for m in metrics),
 'minimum_guest_available_gib':min(m['minimum_guest_available_gib'] for m in metrics),
 'maximum_gpu_used_mib':max(m['maximum_gpu_used_mib'] for m in metrics),
 'maximum_gpu_temperature_c':max(m['maximum_gpu_temperature_c'] for m in metrics),
 'maximum_host_cpu_pct':max(m['maximum_host_cpu_pct'] for m in metrics),
 'maximum_host_telemetry_age_seconds':max(m['host_telemetry_age_seconds_maximum'] for m in metrics),
 'positive_swap_counter_increments':sum(m['swap_counters'][k]['positive_increments'] for m in metrics for k in ('pswpin','pswpout')),
 'observed_threshold_violations':sum(len(r['logged_telemetry']['threshold_violations']) for r in resource['runs']),
 'interrupted_run':'12-hosted-pilot-native-fields',
 'coverage_limitation':'These extrema describe saved samples only. Run12 ends abruptly without exit or guard-stop receipt; its unrecorded final interval is not certified compliant. Run13 completed with exit0.'}
client.write_immutable(root/'results/hosted-resource-summary.json',prep.canonical(summary)+b'\n')
path=root/'results/hosted-study-findings-final/findings.md'
with path.open('a') as f:
 f.write('\n## Recorded resource evidence\n\n')
 f.write(f"Across saved hosted-run samples, minimum available memory was {summary['minimum_host_available_gib']:.3f} GiB on the host and {summary['minimum_guest_available_gib']:.3f} GiB in WSL. GPU memory peaked at {summary['maximum_gpu_used_mib']} MiB and temperature at {summary['maximum_gpu_temperature_c']} C; host CPU peaked at {summary['maximum_host_cpu_pct']}%. Host telemetry age stayed below {summary['maximum_host_telemetry_age_seconds']:.3f} seconds in saved samples. No positive swap-counter increments or threshold violations were recorded. ")
 f.write(summary['coverage_limitation']+' See [resource audit](../hosted-resource-audit.json).\n')
helper_dir=root/'results/execution_helpers'
for name in ['reconcile_saved_jev_response.py','reconcile_native_field_response.py','record_native_field_amendment.py','prepare_actual_smoke_pilot_gate.py','evaluate_latest_scope.py','finalize_actual_hosted_stage.py','record_recovery_evidence.py','prepare_recovery_admission.py','recover_exact_stale_locks.py','launch_pilot_recovery.ps1','finish_hosted_evidence_notes.py']:
 shutil.copy2(source/name,helper_dir/name)
client.write_immutable(root/'results/hosted-execution/recovery-after-wsl-restart/windows_launcher.json',prep.canonical(json.loads((source/'pilot-recovery-launcher.json').read_text(encoding='utf-8-sig')))+b'\n')
print(json.dumps({'status':'final_evidence_notes_written','resource_summary':summary,'pilot_independent_audit_sha256':prep.file_hash(root/'results/pilot-independent-validation.json')}))
