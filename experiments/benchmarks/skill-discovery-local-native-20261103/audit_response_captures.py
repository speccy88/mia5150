"""Offline linkage of private raw responses, hashes and real usage to SQLite rows."""
import hashlib
import json
from pathlib import Path
import sqlite3

from bench_native_v2 import final_answer

ROOT = Path(__file__).resolve().parent


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    records = []
    for suite in ('local-native-v2', 'diagnostic-vram-oversubscription'):
        for folder in sorted((ROOT/'results'/suite).glob('*')):
            path = folder/'state.sqlite3'
            if not path.exists():
                continue
            db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
            counts = {'suite': suite, 'model': folder.name, 'recorded_rows': 0,
                'captured_generation_responses': 0, 'unresolved_intents': [], 'capture_checks_passed': True}
            for key, row_json, audit_json in db.execute('SELECT key,row_json,audit FROM trials ORDER BY ordinal'):
                if row_json is None:
                    counts['unresolved_intents'].append(key)
                    continue
                counts['recorded_rows'] += 1
                row, audit = json.loads(row_json), json.loads(audit_json)
                measured = row['input_tokens'] != '' and row['input_tokens'] is not None
                if not audit.get('raw_response_path'):
                    assert not measured, (suite, folder.name, key, 'measured response without capture')
                    continue
                stored_path = Path(audit['raw_response_path'])
                # Keep the original audit immutable; permit verification after relocation.
                parts = stored_path.parts
                marker = parts.index('private-responses')
                raw_path = ROOT.joinpath(*parts[marker:]).resolve()
                assert raw_path.is_relative_to(ROOT/'private-responses'), (suite, folder.name, key, 'unexpected capture path')
                raw = json.loads(raw_path.read_text(encoding='utf-8'))
                assert digest(raw['text']) == audit['response_sha256'], (suite, folder.name, key, 'raw hash mismatch')
                parsed, error = final_answer(raw['text'], raw['reasoning_prefix'])
                assert digest(parsed) == audit['final_answer_sha256'], (suite, folder.name, key, 'final hash mismatch')
                assert error == audit['reasoning_error'], (suite, folder.name, key, 'reasoning status mismatch')
                for name in ('input_tokens', 'output_tokens'):
                    assert raw['usage'][name] == row[name], (suite, folder.name, key, name)
                assert raw['reasoning_prefix'] == audit['reasoning_prefix']
                counts['captured_generation_responses'] += 1
            db.close()
            records.append(counts)
    output = {'models': records, 'new_inference_calls': 0,
        'verification': 'Every measured native-v2 response is linked to its captured text SHA-256, parsed-final hash, reasoning status and real token usage. Only counts/hashes are exported, no raw text.',
        'diagnostic_policy': 'VRAM oversubscription responses remain excluded from comparative results; unresolved intents are preserved and never automatically replayed.'}
    (ROOT/'evidence/response-capture-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'models': len(records), 'linked_responses': sum(x['captured_generation_responses'] for x in records), 'checks_passed': True}))


if __name__ == '__main__':
    main()
