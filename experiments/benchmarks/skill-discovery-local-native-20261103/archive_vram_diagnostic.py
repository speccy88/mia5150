"""Preserve the overcommitted diagnostic; create an unattempted capacity-rejection ledger."""
import json
from pathlib import Path
import sqlite3
import bench_native_v2 as b

ROOT=Path(__file__).resolve().parent
alias='qwen3-8b-q4km'
src=(ROOT/'results/local-native-v2'/alias).resolve()
parent=(ROOT/'results/diagnostic-vram-oversubscription').resolve()
dst=parent/alias
workspace=ROOT.resolve()
assert src.is_relative_to(workspace) and dst.is_relative_to(workspace)
assert src.parent==workspace/'results/local-native-v2'
assert not dst.exists()
db=sqlite3.connect('file:'+str(src/'state.sqlite3')+'?mode=ro',uri=True)
assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
acquired=db.execute('SELECT COUNT(*) FROM trials WHERE row_json IS NOT NULL').fetchone()[0]
db.close()
parent.mkdir(exist_ok=True)
src.rename(dst)
record=json.loads((ROOT/'evidence/qwen3-8b-vram-pause.json').read_text())
config_path=ROOT/'configs/native-v2'/(alias+'.json')
cfg=json.loads(config_path.read_text())
cfg.update(startup_error='HARDWARE_VRAM_CAPACITY_EXCEEDED',admission_evidence={
    'declared_cuda_buffers_mib':9097.34,'physical_vram_mib':8192,
    'diagnostic_result_folder':str(dst.relative_to(ROOT)),'diagnostic_acquired_rows':acquired,
    'pending_intents_not_replayed':record['pending_intents_not_replayed'],
    'no_inference_in_capacity_rejection_ledger':True})
b.atomic_write(config_path,json.dumps(cfg,indent=2))
for name,expected in b.SOURCE_HASHES.items():
    assert b.file_hash(ROOT/'data'/name)==expected
b.validate_config(cfg)
skills,queries=b.load_data(ROOT/'data/test.jsonl',ROOT/'data/test_queries.json')
b.run(cfg,skills,queries,ROOT/'results/local-native-v2'/alias)
state_path=ROOT/'results/local-native-v2/suite-status.json'
state=json.loads(state_path.read_text(encoding='utf-8-sig'))
state['state']='paused_for_physical_vram_admission'
state['models'][-1].update(state='blocked',reason='HARDWARE_VRAM_CAPACITY_EXCEEDED',
    diagnostic_folder=str(dst.relative_to(ROOT)),diagnostic_acquired_rows=acquired)
b.atomic_write(state_path,json.dumps(state,indent=2))
print('Preserved '+str(acquired)+' diagnostic rows; capacity ledger has zero inference.',flush=True)
