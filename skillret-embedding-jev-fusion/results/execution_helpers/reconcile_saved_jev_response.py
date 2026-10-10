"""Revalidate and settle one known saved HTTP200 response without network/key access."""
from pathlib import Path
import json
import sys
import time

root = Path('/home/user/code/mia5150-local/skillret-embedding-jev-fusion')
sys.path.insert(0, str(root))
import hosted_jev_client as client
import run_jev_rerank as prep
import run_hosted_study as runner
execution = root / 'results/hosted-execution'
config_path = execution / 'authorized-20261007-0158/smoke_gate_config.json'
subject = client.HostedJevClient(execution, gates=runner.load_gates(config_path), enabled=True)
gates = subject._gates(paid=True)  # Evidence only; this method does not access credentials.
plan_path = root / 'results/jev-plan-smoke-context-v1/preparation_manifest.json'
plan = runner.read_plan(plan_path)
packet = plan['rows'][0]['batches'][0]
request_hash = packet['request_sha256']
assert request_hash == 'ef83527e133728d1a20a67dfec3b151b93b59590eafc2e8d7929528f18d37207'
attempt_id = request_hash + '-1'
with subject.ledger.locked():
    rows = subject.ledger.rows()
    assert len([r for r in rows if r['event'] == 'reserved']) == 1
    assert not any(r['event'] in ('settled', 'documented_no_execution') for r in rows)
    reservation = next(r for r in rows if r['event'] == 'reserved')
    assert reservation['attempt_id'] == attempt_id
    assert reservation['pricing_sha256'] == subject.gates.pricing.sha256
    assert reservation['fx_sha256'] == subject.gates.fx.sha256
    metadata = prep.read_json(execution / 'attempts' / (attempt_id + '.json'))
    raw_bytes = (execution / 'attempts' / (attempt_id + '.body')).read_bytes()
    assert metadata['status_code'] == 200
    assert client.hash_bytes(raw_bytes) == metadata['response_sha256'] == 'c69d7a4175615c73aacbe1966967b265e09a40a27edac1c77fb399eb9559bc4c'
    request_bytes = (execution / 'requests' / (request_hash + '.json')).read_bytes()
    assert client.hash_bytes(request_bytes) == request_hash
    request = prep.loads(request_bytes.decode())
    subject._validate_packet(packet, request, plan, gates)
    raw = raw_bytes.decode('utf-8')
    _, usage = prep.parse_response(prep.loads(raw), packet, gates['model'], gates['contract']['total_context_tokens'])
    assert usage['input_tokens'] == 15821 and usage['output_tokens'] == 74
    actual_cad, actual_usd = subject._cost(usage['input_tokens'], usage['output_tokens'], gates)
    assert actual_cad <= client.dec(reservation['reserve_cad'])
    assert actual_usd <= client.dec(reservation['reserve_usd'])
    evidence = {'status': 'saved_response_revalidated_without_replay', 'attempt_id': attempt_id,
                'original_response_sha256': metadata['response_sha256'],
                'validator_source_sha256': prep.file_hash(root / 'run_jev_rerank.py'),
                'usage': usage, 'actual_usd_at_published_rate': str(actual_usd),
                'actual_cad_with_planning_contingency_and_buffer': str(actual_cad),
                'original_ambiguous_event_preserved': True, 'network_calls': 0, 'key_accessed': False,
                'revalidated_at_unix': time.time(),
                'reason': 'Original HTTP200 included complete raw model, score distributions and usage; validation failure was study tolerance incompatible with observed provider rounding. Native score/confidence values remain unchanged.'}
    client.write_immutable(execution / 'precision_reconciliation.json', prep.canonical(evidence) + b'\n')
    envelope = {'schema_version': prep.SCHEMA_VERSION, 'request_sha256': request_hash,
        'expected_request_sha256': request_hash, 'original_transport_request_sha256': request_hash,
        'status_code': 200, 'raw_response': raw, 'response_sha256': metadata['response_sha256'],
        'elapsed_seconds': metadata['elapsed_seconds'], 'provider_request_id': metadata['provider_request_id'],
        'attempt_id': attempt_id, 'actual_model_version': usage['actual_model_version'],
        'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens'],
        'cost_usd': str(actual_usd), 'all_in_cost_cad_with_buffer': str(actual_cad),
        'reconciliation_receipt_sha256': prep.file_hash(execution / 'precision_reconciliation.json')}
    client.write_immutable(execution / 'raw_responses' / (request_hash + '.json'), prep.canonical(envelope) + b'\n')
    semantic = prep.digest(subject._normalized_payload(request, gates['model']))
    client.write_immutable(execution / 'semantic_cache' / (semantic + '.json'), prep.canonical({
        'normalized_payload_sha256': semantic, 'actual_model_version': gates['model'],
        'original_transport_request_sha256': request_hash, 'response_sha256': metadata['response_sha256'],
        'batch_binding': {'candidate_ids': packet['candidate_ids'], 'question_ids': packet['question_ids']}}) + b'\n')
    subject.ledger.append({'event': 'response_revalidated', 'request_sha256': request_hash,
        'attempt_id': attempt_id, 'utc_unix': time.time(), 'network_calls': 0,
        'evidence_sha256': prep.file_hash(execution / 'precision_reconciliation.json')})
    subject.ledger.append({'event': 'settled', 'request_sha256': request_hash, 'attempt_id': attempt_id,
        'plan_sha256': plan['plan_sha256'], 'utc_unix': reservation['utc_unix'],
        'actual_cad': str(actual_cad), 'actual_usd': str(actual_usd),
        'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens'],
        'actual_model_version': gates['model'], 'response_sha256': metadata['response_sha256'],
        'elapsed_seconds': metadata['elapsed_seconds'], 'reconciled_at_unix': time.time(),
        'reconciliation_receipt_sha256': prep.file_hash(execution / 'precision_reconciliation.json')})
state = runner.cache_state(execution, plan, plan_path)
print(json.dumps({'status': 'reconciled_without_replay', 'ledger': runner.ledger_summary(subject.ledger),
                  'validated_cached_batches': sum(b['status'] == 'validated' for r in state for b in r['batches'])}))
