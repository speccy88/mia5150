"""Preserve and clearly label a no-inference guard-development probe."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = (ROOT/'runtime/qwen3-8b-q4km-native-v2/gpu-proof.json').resolve()
target = source.with_name('gpu-proof-diagnostic-invalid-sum.json')
assert source.is_relative_to(ROOT) and target.is_relative_to(ROOT)
proof = json.loads(source.read_text(encoding='utf-8-sig'))
assert proof['pid'] == 19856 and proof['declared_cuda_buffers_mib'] == 0
log = Path(proof['log_path']).read_text(errors='replace')
assert '/completion' not in log and 'launch_slot_' not in log
assert not target.exists()
digest = hashlib.sha256(source.read_bytes()).hexdigest()
source.rename(target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
record = {'proof': str(target.relative_to(ROOT)), 'sha256': digest,
    'admissible': False, 'inference_attempted': False,
    'reason': 'Startup-only development probe had a zero-sum PowerShell OrderedDictionary aggregation bug. Corrected direct-index summation rejects the same 9097.34 MiB configuration before inference.',
    'corrected_rejection': 'runtime/qwen3-8b-q4km-native-v2/launch-20261103T065008909-c566530a/admission-failure.json'}
(ROOT/'evidence/invalidated-guard-probe-proof.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
