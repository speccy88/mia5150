"""Finalize and archive only the separate context/KV experiment."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'local-benchmark'
OUT=ROOT.parent/'deliverables'
EXT={'.py','.ps1','.md','.json','.jsonl','.txt','.log','.csv','.diff'}
PATTERNS=[re.compile(x) for x in (r'\bhf_[A-Za-z0-9]{30,}\b',r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b',r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def inventory(paths):
    return [{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]


def archive(path,paths,kind):
    records=inventory(paths)
    manifest={'kind':kind,'files':records}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,'profile-benchmark/'+p.relative_to(ROOT).as_posix())
        z.writestr('profile-benchmark/ARCHIVE_CONTENTS.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(path) as z:
        assert json.loads(z.read('profile-benchmark/ARCHIVE_CONTENTS.json'))==manifest
        for item in records:
            h=hashlib.sha256()
            with z.open('profile-benchmark/'+item['path']) as stream:
                for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
            assert h.hexdigest()==item['sha256'],item['path']
    return manifest


def main():
    state=json.loads((ROOT/'evidence/profile-status.json').read_text())
    assert state['state']=='completed'
    subprocess.run([sys.executable,str(ROOT/'review_profiles.py')],cwd=ROOT,check=True)
    review=json.loads((ROOT/'evidence/profile-review.json').read_text())
    assert review['measured_total']<=200
    assert all(s['rows']==20 and not s['unresolved_intents'] for s in review['summary'])
    assert all(not p.stat().st_size for p in (ROOT/'results').rglob('*-wal'))
    preservation=json.loads((ROOT/'evidence/baseline-preservation-after.json').read_text())
    for item in preservation:assert sha(Path(item['path']))==item['expected_sha256']
    plan=json.loads((ROOT/'evidence/profile-plan.json').read_text())
    assert sha(BASE/'bench_native_v2.py')==plan['original_bench_sha256']
    for filename in ('PROFILE_REPORT.md','ACCESS_LLAMA_GEMMA.md'):
        text=(ROOT/filename).read_text(encoding='utf-8')
        assert chr(65533) not in text
        width=None
        for line in text.splitlines():
            if line.startswith('|'):
                n=len(line.split('|'))-2
                if width is None:width=n
                assert n==width
            else:width=None
    final={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':'v3-local-context-kv-real-token-gate',
        'benchmark_sha256':sha(ROOT/'bench_profiles_v3.py'),'columns':45,
        'measured_total':review['measured_total'],'raw_responses_verified':review['raw_responses_verified'],
        'profiles':review['summary'],'paired_quality':review['paired_quality'],
        'baseline_archives_and_code_preserved':True,
        'scope':'N=50/100/250/500 only, same twenty queries, prompts, seed42 and output500. No automatic transformer CPU fallback; default input CPU_Mapped explicitly retained.',
        'access':'Official links and user-only instructions prepared; no login, acceptance or bypass performed.',
        'publication':'No push or email performed.'}
    (ROOT/'evidence/final-summary.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    public=[p for p in ROOT.iterdir() if p.is_file() and p.name!='ARCHIVE_CONTENTS.json' and (p.suffix in EXT or p.name=='.gitignore')]
    for d in ('configs','evidence','results','runtime','tests'):
        public.extend(p for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix in EXT or p.name.startswith('state.sqlite3')))
    public=sorted(set(public))
    private=sorted((ROOT/'private-responses').rglob('*.json'))
    assert len(private)==review['raw_responses_verified']
    for p in public+private:
        if p.suffix not in EXT:continue
        text=p.read_text(encoding='utf-8',errors='replace')
        assert not any(pattern.search(text) for pattern in PATTERNS),'Credential-shaped value; inspect privately, never print matching values.'
        if p.name=='server.stderr.log':assert 'You are evaluating skill discovery performance.' not in text
    OUT.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    pubzip=OUT/f'profile-code-results-{stamp}.zip'
    privzip=OUT/f'PRIVATE-profile-responses-{stamp}.zip'
    pubmanifest=archive(pubzip,public,'public-code-results-notes; no weights, private datasets, raw responses or credentials')
    archive(privzip,private,'PRIVATE-raw-response-verification-only; keep outside GitHub and ordinary delivery')
    report=OUT/f'PROFILE_REPORT-{stamp}.md';report.write_bytes((ROOT/'PROFILE_REPORT.md').read_bytes())
    access=OUT/f'ACCESS_LLAMA_GEMMA-{stamp}.md';access.write_bytes((ROOT/'ACCESS_LLAMA_GEMMA.md').read_bytes())
    filemanifest=OUT/f'profile-file-manifest-{stamp}.json';filemanifest.write_text(json.dumps(pubmanifest,indent=2),encoding='utf-8')
    artifacts=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in (pubzip,privzip,report,access,filemanifest)]
    sums=OUT/f'SHA256SUMS-profiles-{stamp}.txt';sums.write_text(''.join(a['sha256']+'  '+Path(a['path']).name+'\n' for a in artifacts),encoding='utf-8')
    result={'created_utc':stamp,'artifacts':artifacts,'checksum_file':str(sums),'public_files':len(public),
        'private_response_files':len(private),'archive_checks':'All ZIP CRC and source-file SHA256 checks passed.',
        'original_delivery_unchanged':True,'publication':'No push or email; private evidence archive remains separate.'}
    out=OUT/f'profile-package-manifest-{stamp}.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'manifest':str(out),**result},indent=2))


if __name__=='__main__':main()
