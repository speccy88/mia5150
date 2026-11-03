"""Finish offline evidence and reports after all 18 requested models are resolved."""
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys
import bench_native_v2 as bench

ROOT = Path(__file__).resolve().parent


def main():
    state = json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text())
    jobs = json.loads((ROOT/'local-models.json').read_text())
    assert state['state'] == 'completed' and len(state['models']) == len(jobs) == 18
    for script in ('audit_context_windows.py', 'audit_native_buffers.py', 'export_model_provenance.py',
                   'audit_response_captures.py', 'summarize_wddm_memory.py', 'export_portable_sqlite.py',
                   'write_native_v2_report.py', 'propose_high_context.py', 'scan_handoff_text.py'):
        subprocess.run([sys.executable, str(ROOT/script)], cwd=ROOT, check=True)
    capture = json.loads((ROOT/'evidence/response-capture-audit.json').read_text())
    allocations = json.loads((ROOT/'evidence/native-buffer-audit.json').read_text())
    scorecard = json.loads((ROOT/'evidence/model-scorecard.json').read_text())
    assert len(scorecard['models']) == 18
    for model in scorecard['models']:
        folder = ROOT/'results/local-native-v2'/model['alias']
        with (folder/'details.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 141 and all(len(r) == 45 for r in rows)
        status = json.loads((folder/'status.json').read_text())
        assert status['complete'] and status['recorded_trials'] == 140
        manifest = json.loads((folder/'manifest.json').read_text())
        assert manifest['implementation_sha256']['bench_native_v2.py'] == bench.file_hash(ROOT/'bench_native_v2.py')
    for model in capture['models']:
        if model['suite'] == 'local-native-v2':
            assert not model['unresolved_intents'], model['model']
    for model in allocations['models']:
        if model['state'] == 'completed':
            assert model['comparative_measures_admissible'], model['alias']
            assert model['all_layers_gpu'] and model['kv_gpu'] and model['kv_type'] == 'f16'
            assert model['context'] == 32768
    measured = [m for m in scorecard['models'] if m['measured_responses']]
    blocked = [m for m in scorecard['models'] if not m['measured_responses']]
    output = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'requested_models': 18,
        'protocol': bench.LOCAL_PROTOCOL,
        'implementation_sha256': bench.file_hash(ROOT/'bench_native_v2.py'),
        'runtime': {'build': 'llama.cpp b11960 CUDA 12.4', 'commit': '8642541b255bfb5dc39cd351a178be09013a55d6'},
        'validated_data': {'skills': 6006, 'distinct_names': 5801, 'queries': 20, 'sha256': bench.SOURCE_HASHES},
        'canonical_rows': sum(m['recorded_rows'] for m in scorecard['models']), 'columns': 45,
        'models_with_measured_responses': len(measured),
        'canonical_measured_responses': sum(m['measured_responses'] for m in measured),
        'models_without_measured_responses': [{'model': m['requested'], 'reason': m['state']} for m in blocked],
        'diagnostic_response_accounting': [m for m in capture['models'] if m['suite'] != 'local-native-v2'],
        'checks': 'All 18 ledgers complete; CSV/SQLite/rank/token/candidate review passed; raw response links and source/model/log hashes checked; accepted CUDA footprints fit physical capacity.',
        'profile': '32768 configured context, f16 KV, default CPU_Mapped input embeddings; no transformer fallback or automatic fitting. Not absolute GPU-only.',
        'context_exclusions': '60 per model from configured-window estimate, not intrinsic context limits or measured generation failures.',
        'delivery': ['LOCAL_REPORT.md', 'HIGH_CONTEXT_PROPOSAL.md', 'results/local-native-v2/', 'evidence/model-scorecard.json',
                     'evidence/model-artifact-provenance.json', 'evidence/native-buffer-audit.json',
                     'evidence/response-capture-audit.json', 'evidence/sqlite-portability.json'],
        'publication': 'No email or push performed by this worker; parent coordinates common report and repository integration.'}
    (ROOT/'evidence/final-handoff.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
