"""Sanitized experiment-only archive, source hashes, and local cache inventory."""
import argparse, csv, hashlib, io, json, pathlib, re, zipfile

# Whole-file hashes reviewed at the root and the run09/run10 source snapshots.
# The only matches in these exact files are header-only negative test strings.
REVIEWED_HEADER_ONLY_TEST_HASHES = {
    '6c661d2cb088d1597bf4702851018e88499b95f269c627742a93bead46d515cb',
    '1d0e4fb00786ff706a017509d993b044278878c23859b837af02add09a749128',
    '8cb0340871b55f6caa43d410037903a8617669068a9ecdc85a6ae3ae830285d6',
    'b16fb03866dd46f82d4646bcc6787fef75d135532f6925fae44be2a8961c0ac0',
}

def reviewed_header_fixture(relative, body, matched):
    rel=pathlib.PurePosixPath(str(relative).replace('\\','/'))
    source_location=(rel.parts==('test_hosted_jev_client.py',) or
                     len(rel.parts)==4 and rel.parts[0]=='runs' and rel.parts[2]=='source'
                     and rel.name=='test_hosted_jev_client.py')
    return (source_location and hashlib.sha256(body).hexdigest() in REVIEWED_HEADER_ONLY_TEST_HASHES
            and re.fullmatch(rb'-----BEGIN (?:RSA )?PRIVATE KEY-----',matched) is not None)

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
    # A public benchmark query contains credential-shaped example values. Keep
    # the original input and CSV locally, but omit that entire query-text cell
    # from the portable export. IDs, rankings and every metric remain exact.
    overrides={};redactions=[];placeholders=[]
    csv_path=root/'results/per_query_results.csv'
    if csv_path in files:
        records=list(csv.DictReader(io.StringIO(csv_path.read_text(encoding='utf-8'))))
        for row in records:
            query=row.get('query','')
            if any(pattern.search(query.encode()) for pattern in forbidden):
                redactions.append({'path':'results/per_query_results.csv','query_id':row['query_id'],'field':'query','original_text_sha256':hashlib.sha256(query.encode()).hexdigest(),'reason':'credential-shaped public benchmark text omitted from portable export; raw local input preserved'})
                row['query']='[Query text omitted from portable export: credential-shaped source examples. Original remains in pinned dataset and local results.]'
        if redactions:
            stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records);overrides[csv_path]=stream.getvalue().encode()
    findings=[]
    for path in files:
        body=overrides.get(path,path.read_bytes())
        for pattern in forbidden:
            for match in pattern.finditer(body):
                # Explicit all-x examples in published skill documentation are
                # placeholders, not authentication material. No other allowlist.
                if re.fullmatch(rb'gh[pousr]_[xX]{30,}',match.group()):
                    placeholders.append({'path':str(path.relative_to(root)),'match_sha256':hashlib.sha256(match.group()).hexdigest(),'reason':'all-x documentation placeholder'})
                elif reviewed_header_fixture(path.relative_to(root),body,match.group()):
                    placeholders.append({'path':path.relative_to(root).as_posix(),'match_sha256':hashlib.sha256(match.group()).hexdigest(),'reviewed_source_sha256':hashlib.sha256(body).hexdigest(),'reason':'exact reviewed synthetic header-only negative test source; no key body'})
                else:findings.append(str(path.relative_to(root)))
    # Never emit a matching secret value; filenames alone identify a review need.
    receipt={'scope':'portable export of new experiment allowlisted files only; no environment/credential reads','files_checked':len(files),'pattern_findings_paths_only':findings,'export_redactions':redactions,'verified_placeholders':placeholders,'bulk_cache':'retained locally; excluded from archive','status':'passed' if not findings else 'blocked_for_review'}
    (root/'results/secret_scan.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if findings:raise RuntimeError('Secret-pattern review required: '+', '.join(findings))
    files=[path for path in files if path.name not in {'artifact_manifest.json','secret_scan.json'}]+[root/'results/secret_scan.json']
    inventory={path.relative_to(root).as_posix():{'bytes':len(overrides[path]) if path in overrides else path.stat().st_size,'sha256':hashlib.sha256(overrides[path]).hexdigest() if path in overrides else sha(path),**({'local_original_sha256':sha(path),'export_text_redacted':True} if path in overrides else {})} for path in sorted(files)}
    for directory in sorted((root/'cache').glob('*')):
        if not directory.is_dir():continue
        inventory['LOCAL_ONLY/'+directory.name]={'cache_files':{path.name:{'bytes':path.stat().st_size,'sha256':sha(path)} for path in sorted(directory.iterdir()) if path.is_file()},'included_in_archive':False}
    manifest=root/'results/artifact_manifest.json';manifest.write_text(json.dumps(inventory,indent=2)+'\n');files.append(manifest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    # No overwrite: prior delivery remains independently reproducible.
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files):
            member='skillret-embedding-jev-fusion/'+path.relative_to(root).as_posix()
            if path in overrides:z.writestr(member,overrides[path])
            else:z.write(path,member)
    print(json.dumps({'archive':str(dest),'bytes':dest.stat().st_size,'sha256':sha(dest),'files':len(files),'secret_scan':'passed'}))

if __name__=='__main__':main()
