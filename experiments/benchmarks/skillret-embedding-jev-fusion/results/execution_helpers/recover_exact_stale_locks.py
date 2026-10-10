"""Remove only the two preserved experiment locks after verified local reboot."""
from pathlib import Path
import sys,time,json
root=Path('/home/user/code/mia5150-local/skillret-embedding-jev-fusion');sys.path.insert(0,str(root))
import hosted_jev_client as client
import run_hosted_study as runner
import run_jev_rerank as prep
run=root/'results/hosted-execution';dest=run/'recovery-after-wsl-restart'
receipt=prep.read_json(dest/'recovery.json')
subject=client.HostedJevClient(run,gates=runner.load_gates(dest/'pilot_recovery_gate_config_v2.json'),enabled=True)
subject._gates(True)
rows=subject.ledger.rows();assert rows[-1]['event_sha256']==receipt['original_reservation_event_sha256']
assert subject.reviewed_unresolved_attempt_ids(rows)=={receipt['original_attempt_id']}
boot=int(next(l.split()[1] for l in Path('/proc/stat').read_text().splitlines() if l.startswith('btime ')))
assert boot>rows[-1]['utc_unix']
names=('budget_ledger.lock','study_runner.lock')
for name in names:
 path=run/name;assert path.resolve().parent==run.resolve() and not path.is_symlink()
 assert prep.file_hash(path)==receipt['locks_preserved'][name]['sha256']
 assert prep.file_hash(run/receipt['locks_preserved'][name]['preserved_copy'])==prep.file_hash(path)
record={'status':'exact_stale_locks_reviewed_for_removal','recorded_at_unix':time.time(),
 'current_boot_unix':boot,'previous_local_worker_cannot_survive_later_boot':True,
 'remote_execution_billing_remain_unknown':True,'ledger_changed':False,
 'original_reservation_retained':True,'files':list(names),'receipt_sha256':prep.file_hash(dest/'recovery.json')}
client.write_immutable(dest/'locks_recovery_review.json',prep.canonical(record)+b'\n')
for name in names:(run/name).unlink()
client.write_immutable(dest/'locks_recovered.json',prep.canonical({**record,'status':'exact_stale_locks_removed','removed_at_unix':time.time()})+b'\n')
print(json.dumps({'status':'two_preserved_stale_locks_removed','api_calls':0,'key_accessed':False,'original_reservation_retained':True}))
