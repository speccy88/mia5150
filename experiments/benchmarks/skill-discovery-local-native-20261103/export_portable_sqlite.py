"""Preserve original WAL evidence and export a standalone SQLite diagnostic copy."""
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(db):
    return db.execute('SELECT COUNT(*), SUM(row_json IS NOT NULL), SUM(row_json IS NULL) FROM trials').fetchone()


def main():
    state = json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text())
    assert state['state'] == 'completed', 'No portability export while suite is active.'
    records = []
    for source in sorted((ROOT/'results').glob('*/*/state.sqlite3')):
        wal = source.with_name(source.name+'-wal')
        if not wal.exists() or not wal.stat().st_size:
            continue
        target = source.with_name('state.portable.sqlite3')
        original_hash = sha(source)
        wal_hash = sha(wal)
        src = sqlite3.connect(source.as_uri()+'?mode=ro', uri=True)
        assert src.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        before = counts(src)
        if target.exists():
            dst = sqlite3.connect(target.as_uri()+'?mode=ro', uri=True)
        else:
            dst = sqlite3.connect(target)
            src.backup(dst)
            dst.execute('PRAGMA journal_mode=DELETE')
        assert dst.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert counts(dst) == before
        assert dst.execute('SELECT * FROM trials ORDER BY ordinal').fetchall() == src.execute('SELECT * FROM trials ORDER BY ordinal').fetchall()
        assert dst.execute('SELECT * FROM meta').fetchall() == src.execute('SELECT * FROM meta').fetchall()
        dst.close()
        src.close()
        assert sha(source) == original_hash and sha(wal) == wal_hash, 'Original bytes changed.'
        records.append({'source': str(source.relative_to(ROOT)), 'original_sha256': original_hash,
            'original_wal_sha256': wal_hash, 'portable': str(target.relative_to(ROOT)),
            'portable_sha256': sha(target), 'trial_intents': before[0],
            'recorded_rows': before[1], 'unresolved_intents': before[2],
            'integrity': 'ok', 'original_bytes_preserved': True})
    output = {'exports': records,
        'instructions': 'Original SQLite/WAL/SHM evidence is preserved. state.portable.sqlite3 is a standalone diagnostic copy including committed WAL data and unresolved intents; it needs no WAL sidecar. Completed canonical databases have no nonempty WAL at handoff.'}
    (ROOT/'evidence/sqlite-portability.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'standalone_exports': len(records)}))


if __name__ == '__main__':
    main()
