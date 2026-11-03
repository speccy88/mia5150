"""Create a public integration archive and a separate private response-evidence archive."""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent/'deliverables'
TEXT_SUFFIXES = {'.py', '.ps1', '.md', '.json', '.jsonl', '.txt', '.log', '.csv'}
from scan_handoff_text import PATTERNS


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def inventory(files, base):
    return [{'path': p.relative_to(base).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]


def write_zip(path, files, manifest, prefix='local-benchmark/'):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in files:
            archive.write(source, prefix+source.relative_to(ROOT).as_posix())
        archive.writestr(prefix+'ARCHIVE_CONTENTS.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read(prefix+'ARCHIVE_CONTENTS.json')) == manifest
        for record in manifest['files']:
            digest = hashlib.sha256()
            with archive.open(prefix+record['path']) as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    digest.update(block)
            assert digest.hexdigest() == record['sha256'], record['path']


def main():
    handoff = json.loads((ROOT/'evidence/final-handoff.json').read_text())
    state = json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text())
    assert state['state'] == 'completed' and handoff['canonical_rows'] == 2520
    assert not json.loads((ROOT/'evidence/handoff-text-scan.json').read_text())['findings_without_values']
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    OUT.mkdir(exist_ok=True)
    public = [p for p in ROOT.iterdir() if p.is_file() and p.name != 'ARCHIVE_CONTENTS.json' and (p.suffix in TEXT_SUFFIXES or p.name == '.gitignore')]
    for directory in ('tests', 'configs', 'evidence', 'runtime', 'results'):
        public.extend(p for p in (ROOT/directory).rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.suffix != '.pyc'
            and (p.suffix in TEXT_SUFFIXES or p.name.startswith('state.sqlite3') or p.name == 'state.portable.sqlite3'))
    public = sorted(set(public))
    assert not any(p.suffix.lower() in {'.gguf', '.safetensors', '.exe', '.dll'} for p in public)
    assert not any(p.relative_to(ROOT).parts[0] in {'models', 'tools', 'data', 'private', 'private-responses', '.venv', 'conversion-inputs'} for p in public)
    private = sorted((ROOT/'private-responses').rglob('*.json'))
    private_findings = []
    for source in private:
        text = source.read_text(encoding='utf-8')
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                private_findings.append({'file': source.relative_to(ROOT).as_posix(), 'pattern': name})
    assert not private_findings, 'Credential-shaped private evidence requires local inspection; no values printed.'
    public_manifest = {'kind': 'integration-code-results-notes', 'files': inventory(public, ROOT),
        'excluded': ['model binaries', 'runtime binaries', 'environments', 'private source datasets', 'private raw responses', 'credentials'],
        'private_evidence': 'Separate PRIVATE archive. Do not publish it to GitHub or attach it to ordinary delivery by default.'}
    private_manifest = {'kind': 'PRIVATE-raw-model-response-evidence', 'files': inventory(private, ROOT),
        'instructions': 'Local verification evidence only. Raw model texts, token usage and reasoning-prefix state; no complete prompts or original datasets. Keep separate from repository/public integration.'}
    public_zip = OUT/f'local-code-results-{stamp}.zip'
    private_zip = OUT/f'PRIVATE-local-response-evidence-{stamp}.zip'
    write_zip(public_zip, public, public_manifest)
    write_zip(private_zip, private, private_manifest)
    report = OUT/f'LOCAL_REPORT-{stamp}.md'
    proposal = OUT/f'HIGH_CONTEXT_PROPOSAL-{stamp}.md'
    report.write_bytes((ROOT/'LOCAL_REPORT.md').read_bytes())
    proposal.write_bytes((ROOT/'HIGH_CONTEXT_PROPOSAL.md').read_bytes())
    manifest_file = OUT/f'public-file-manifest-{stamp}.json'
    manifest_file.write_text(json.dumps(public_manifest, indent=2), encoding='utf-8')
    artifacts = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (public_zip, private_zip, report, proposal, manifest_file)]
    checksum = OUT/f'SHA256SUMS-{stamp}.txt'
    checksum.write_text(''.join(a['sha256']+'  '+Path(a['path']).name+'\n' for a in artifacts), encoding='utf-8')
    record = {'created_utc': stamp, 'public_files': len(public), 'private_response_files': len(private),
        'artifacts': artifacts, 'checksum_file': str(checksum),
        'archive_integrity': 'Both ZIP CRC checks and every archived source-file SHA-256 passed.',
        'publication': 'No push or email performed. PRIVATE evidence archive stays separate from integration archive.'}
    output = OUT/f'package-manifest-{stamp}.json'
    output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'manifest': str(output), **record}, indent=2))


if __name__ == '__main__':
    main()
