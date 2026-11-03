"""Scan deliverable text for credential-shaped values without printing matches."""
import datetime
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PATTERNS = {
    'huggingface_token': re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
    'openai_key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'aws_access_id': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'private_key_pem': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}
EXTENSIONS = {'.py', '.ps1', '.md', '.json', '.jsonl', '.txt', '.log', '.csv'}


def main():
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in EXTENSIONS]
    for directory in ('tests', 'configs', 'evidence', 'results', 'runtime'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in EXTENSIONS)
    findings = []
    byte_count = 0
    for path in files:
        raw = path.read_bytes()
        byte_count += len(raw)
        text = raw.decode('utf-8', errors='replace')
        for name, pattern in PATTERNS.items():
            count = len(pattern.findall(text))
            if count:
                findings.append({'file': str(path.relative_to(ROOT)), 'pattern': name, 'count': count})
    output = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'text_files_scanned': len(files), 'bytes_scanned': byte_count, 'findings_without_values': findings,
        'scope': 'Code, configuration, reports, result text and runtime/evidence logs. Weight/tool/environment directories, private inputs and raw private responses are excluded from delivery and this scan. SQLite columns are separately reconciled against CSV plus audited response hashes.',
        'limitation': 'Credential-pattern check, not proof that arbitrary sensitive text cannot exist. Matching values are never logged.'}
    (ROOT/'evidence/handoff-text-scan.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'text_files_scanned': len(files), 'credential_pattern_findings': len(findings)}))
    assert not findings, 'Inspect named files privately before delivery; do not print matching values.'


if __name__ == '__main__':
    main()
