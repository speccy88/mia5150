"""Record observed local-process loss and one bounded, reviewed recovery."""
from pathlib import Path
import sys,json,time,hashlib
root=Path('/home/user/code/mia5150-local/skillret-embedding-jev-fusion')
sys.path.insert(0,str(root))
import hosted_jev_client as client
import run_jev_rerank as prep
execution=root/'results/hosted-execution'; dest=execution/'recovery-after-wsl-restart'
dest.mkdir(exist_ok=True)
request_hash='f340d90743f01aac474aed7d9135f15599da30e73aa1f6bb08fed1cf6eb746dd'
aid=request_hash+'-1'; rows=client.SpendLedger(execution).rows()
reservation=next(r for r in rows if r['event']=='reserved' and r['attempt_id']==aid)
assert rows[-1]==reservation and reservation['event_sha256']=='de26e1182d02965af2ca529da83d22a24a513ae6aabdc493cf3c2e3fd26be0ab'
assert not (execution/'attempts'/(aid+'.body')).exists() and not (execution/'attempts'/(aid+'.json')).exists()
assert prep.file_hash(execution/'requests'/(request_hash+'.json'))==request_hash
boot=int(next(l.split()[1] for l in Path('/proc/stat').read_text().splitlines() if l.startswith('btime ')))
now=time.time(); boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot>reservation['utc_unix']
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  comm=(p/'comm').read_text().strip()
  if 'python' not in comm:continue
  args=(p/'cmdline').read_bytes().split(b'\0')
  if any(Path(x.decode(errors='replace')).name in ('resource_guard.py','execute_authorized_jev_study.py') for x in args):matches.append(int(p.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not matches
proof={'kind':'wsl_reboot_observation','observed_at_unix':now,'uptime_seconds':now-boot,
 'boot_id':boot_id,'boot_unix_from_proc_stat':boot,'matching_live_process_count':len(matches),
 'process_match_scope':'Python argv basenames resource_guard.py or execute_authorized_jev_study.py; no environment read.',
 'conclusion':'The previous local worker cannot survive this later kernel boot. Remote request execution/billing remains unknown.',
 'last_recorded_reservation_unix':reservation['utc_unix'],'raw_response_and_metadata_absent':True}
proof_path=dest/'reboot_proof.json'; client.write_immutable(proof_path,prep.canonical(proof)+b'\n')
locks={}
for name in ('budget_ledger.lock','study_runner.lock'):
 data=(execution/name).read_bytes(); copy=dest/(name+'.txt')
 client.write_immutable(copy,data); locks[name]={'sha256':hashlib.sha256(data).hexdigest(),'preserved_copy':str(copy.relative_to(execution))}
receipt={'schema_version':'reviewed-wsl-reboot-recovery-v1','experiment_id':client.EXPERIMENT,
 'verified':True,'verified_at_unix':time.time(),'source':'Existing human CAD10/USD5 study authorization plus current local restart evidence; two independent implementation reviews recommend a single identical-payload recovery with both reservations retained.',
 'reviewed_by':'root coordinator; independent reviewers jev_client_context_integration and hosted_study_runner',
 'request_sha256':request_hash,'original_attempt_id':aid,'replacement_attempt_id':request_hash+'-2',
 'actual_model_version':reservation['actual_model_pin'],'plan_sha256':reservation['plan_sha256'],
 'original_reservation_event_sha256':reservation['event_sha256'],
 'retain_original_reservation':True,'additional_attempts':1,'no_automatic_retry':True,
 'reboot_proof':{'path':str(proof_path.relative_to(execution)),'sha256':prep.file_hash(proof_path)},
 'original_invocation_id':'ad65d43ed3f74704a617d9fbc6843f21','original_invocation_started_at_unix':1791340290.626055,
 'observed_last_action_unix':reservation['utc_unix'],'confirmed_new_wsl_boot_unix':boot,
 'interruption_lower_bound_unix':reservation['utc_unix'],'interruption_upper_bound_unix':boot,
 'first_experiment_attempt_unix':1791338570.549812,'six_hour_deadline_unix':1791360170.549812,
 'authorization_scope':'Complete the already authorized fixed pilot within its existing budget; this is a coordinator recovery review, not a new human approval or evidence of no original charge.',
 'possible_duplicate_billing_disclosed':True,'original_unknown_reservation_usd':reservation['reserve_usd'],
 'original_unknown_reservation_cad':reservation['reserve_cad'],
 'request_bytes_and_candidate_pool_unchanged':True,'use_first_replacement_response_regardless_of_scores':True,
 'new_ambiguous_attempt_stops_execution':True,'locks_preserved':locks}
receipt_path=dest/'recovery.json';client.write_immutable(receipt_path,prep.canonical(receipt)+b'\n')
old=prep.read_json(execution/'actual-smoke-pilot-admission/pilot_gate_config.json')
old['recovery']={'path':str(receipt_path),'sha256':prep.file_hash(receipt_path)}
config_path=dest/'pilot_recovery_gate_config.json';client.write_immutable(config_path,prep.canonical(old)+b'\n')
print(json.dumps({'status':'one_off_recovery_review_recorded','proof':proof,'receipt_sha256':prep.file_hash(receipt_path),'config':str(config_path),'key_accessed':False,'network_calls':0,'locks_removed':False}))
