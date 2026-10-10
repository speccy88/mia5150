"""Sanitized experiment-only archive, source hashes, and local cache inventory."""
import argparse, hashlib, json, pathlib, re, zipfile

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();root=a.experiment_dir.resolve();dest=a.output.resolve()
    allowed_root={'.py','.md','.txt','.ps1','.json'}
    files=[]
    for path in root.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(root)
        if any(part in {'cache','__pycache__','.pytest_cache','.git','.venv','deliverables'} for part in rel.parts):continue
        if path==dest or path.name.startswith(('.env','host-telemetry')):continue
        if len(rel.parts)==1 and path.suffix not in allowed_root and path.name!='.gitignore':continue
        if len(rel.parts)>1 and rel.parts[0] not in {'results','runs'}:continue
        if path.suffix not in allowed_root|{'.jsonl','.csv','.log','.body'} and path.name!='.gitignore':continue
        files.append(path)
    forbidden=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{50,}'),re.compile(rb'Bearer\s+[A-Za-z0-9_-]{24,}'),re.compile(rb'(?i)(?:api[_-]?key|token|password)\s*[=:]\s*[\x22\x27][A-Za-z0-9_-]{28,}[\x22\x27]')]
    findings=[]
    for path in files:
        body=path.read_bytes()
        if any(pattern.search(body) for pattern in forbidden):findings.append(str(path.relative_to(root)))
    # Never emit a matching secret value; filenames alone identify a review need.
    receipt={'scope':'new experiment allowlisted files only; no environment/credential reads','files_checked':len(files),'pattern_findings_paths_only':findings,'bulk_cache':'retained locally; excluded from archive','status':'passed' if not findings else 'blocked_for_review'}
    (root/'results/secret_scan.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if findings:raise RuntimeError('Secret-pattern review required: '+', '.join(findings))
    files=[path for path in files if path.name not in {'artifact_manifest.json','secret_scan.json'}]+[root/'results/secret_scan.json']
    inventory={str(path.relative_to(root)):{'bytes':path.stat().st_size,'sha256':sha(path)} for path in sorted(files)}
    for directory in sorted((root/'cache').glob('*')):
        if not directory.is_dir():continue
        inventory['LOCAL_ONLY/'+directory.name]={'cache_files':{path.name:{'bytes':path.stat().st_size,'sha256':sha(path)} for path in sorted(directory.iterdir()) if path.is_file()},'included_in_archive':False}
    manifest=root/'results/artifact_manifest.json';manifest.write_text(json.dumps(inventory,indent=2)+'\n');files.append(manifest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    # No overwrite: prior delivery remains independently reproducible.
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files):z.write(path,'skillret-embedding-jev-fusion/'+str(path.relative_to(root)))
    print(json.dumps({'archive':str(dest),'bytes':dest.stat().st_size,'sha256':sha(dest),'files':len(files),'secret_scan':'passed'}))

if __name__=='__main__':main()
