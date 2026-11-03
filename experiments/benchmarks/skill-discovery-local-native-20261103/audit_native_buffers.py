"""Offline allocation audit of the exact launches referenced by suite state."""
import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PATTERN = re.compile(r'CUDA0\s+(model|KV|RS|compute) buffer size\s*=\s*([0-9.]+) MiB', re.I)


def main():
    state = json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text(encoding='utf-8-sig'))
    jobs = {j['alias']: j for j in json.loads((ROOT/'local-models.json').read_text())}
    provenance_path = ROOT/'evidence/model-artifact-provenance.json'
    provenance = {a['alias']: a for a in json.loads(provenance_path.read_text())['artifacts']} if provenance_path.exists() else {}
    records = []
    for entry in state['models']:
        record = {k: entry[k] for k in ('alias', 'state', 'reason', 'diagnostic_folder') if k in entry}
        if not entry.get('launch'):
            record['allocation_observed'] = False
            records.append(record)
            continue
        launch = ROOT/entry['launch']
        proof_path = launch/'gpu-proof.json'
        proof = json.loads(proof_path.read_text(encoding='utf-8-sig')) if proof_path.exists() else None
        log_path = launch/'server.stderr.log'
        raw = log_path.read_bytes()
        # A proof seals only the startup prefix; later inference appends are irrelevant.
        sealed = raw[:proof['log_bytes']] if proof else raw
        text = sealed.decode('utf-8', errors='replace')
        buffers = [{'kind': m[1], 'mib': float(m[2]), 'line': m[0]} for m in PATTERN.finditer(text)]
        declared = sum(b['mib'] for b in buffers if b['kind'].lower() != 'compute')
        declared += max((b['mib'] for b in buffers if b['kind'].lower() == 'compute'), default=0)
        record.update(allocation_observed=bool(buffers), launch=entry['launch'],
            declared_cuda_buffers_mib=round(declared, 2), physical_vram_mib=8192,
            physical_capacity_exceeded=declared > 8192,
            cuda_buffer_records=buffers,
            buffer_log_lines=[line for line in text.splitlines() if re.search(r'(CPU_Mapped|CUDA0).*(model|KV|RS|compute) buffer size', line)],
            interpretation='Declared buffers are a necessary capacity check, not a proof of physical residency or zero CPU participation.')
        if proof:
            job = jobs[entry['alias']]
            expected_sha = job.get('sha256')
            if not expected_sha and job.get('path'):
                artifact = Path(job['path'])
                if not artifact.is_absolute():
                    artifact = ROOT/artifact
                source = artifact.with_name(artifact.name+'.source.json')
                if source.exists():
                    expected_sha = json.loads(source.read_text(encoding='utf-8-sig'))['sha256']
                elif entry['alias'] in provenance:
                    public_source = provenance[entry['alias']]
                    expected_sha = public_source.get('sha256') or public_source.get('verified_manifest', {}).get('sha256')
            record.update(context=proof['context_window'], kv_type=proof['kv_type'],
                all_layers_gpu=proof['all_layers_gpu'], kv_gpu=proof['kv_gpu'],
                sealed_startup_hash_valid=hashlib.sha256(sealed).hexdigest() == proof['log_sha256'],
                artifact_hash_matches_verified_source=bool(expected_sha) and expected_sha == proof['model_sha256'],
                artifact_sha256=proof['model_sha256'], executable_sha256=proof['executable_sha256'])
        admission = launch/'admission-failure.json'
        if admission.exists():
            record['startup_admission'] = json.loads(admission.read_text(encoding='utf-8-sig'))
        record['comparative_measures_admissible'] = entry['state'] == 'completed' and bool(proof) and declared <= 8192 and record.get('sealed_startup_hash_valid', False) and record.get('artifact_hash_matches_verified_source', False)
        records.append(record)
    output = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'physical_vram_mib': 8192, 'models': records,
        'scope': 'Transformer/output layers and KV/recurrent states on CUDA. Default input embeddings remain CPU_Mapped. No automatic transformer offload or context fitting.',
        'oversubscription_diagnostics': 'results/diagnostic-vram-oversubscription/'}
    (ROOT/'evidence/native-buffer-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'models': len(records), 'over_capacity': [x['alias'] for x in records if x.get('physical_capacity_exceeded')]}))


if __name__ == '__main__':
    main()
