"""Bundle only an explicit public allowlist, never private runtime files."""
import argparse,hashlib,json,pathlib,subprocess,zipfile
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def main(output):
    names=['README.md','diversity-design-20261024.md','continuation-20261024.json','diverse-protocol.json','diverse-budget-plan.json','diverse-endpoint-snapshots.json','diverse-request-plan.csv','diverse_contract.py','prepare_diverse.py','run_diverse.py','runner.py','envelope.py','request_log.py','budget.py','publish_status.py','publish_diverse.py','analyze_diverse.py','package_evidence.py']
    members={'hosted/'+name:HERE/name for name in names}
    for p in sorted((HERE/'reports').iterdir()):
        if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.md','.png','.tex']:members['hosted/reports/'+p.name]=p
    for name in ['core.py','ledger.py','evaluator-labels.json','protocol.json','model-provenance.json']:members['study/'+name]=HERE.parent/name
    for p in HERE.parent.glob('test_hosted*.py'):members['tests/'+p.name]=p
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in members.items()}
    manifest={'repository':'https://github.com/speccy88/mia5150','source_commit':commit,'local_data_commit':'0f4e9818b6cefb94e09da30ad76b9ff5ac97999c','files_sha256':hashes,'privacy':'Explicit public source/report allowlist only. No credentials, ciphertext, private generation/account IDs, raw private runtime files, original student-identifying inputs or model binaries.'}
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,path in members.items():z.write(path,name)
        z.writestr('EVIDENCE_MANIFEST.json',json.dumps(manifest,indent=2))
        z.writestr('READ_ME_FIRST.md','# Hosted follow-up evidence\n\nThe common balanced panel contains24 previously exposed queries at N=100. Qwen/SiliconFlow is the active model after the versioned continuation. Earlier DeepSeek/Gemma failures and both UNKNOWN holds remain in the ledger.\n\nStart with hosted/reports/HOSTED_REPORT.md, hosted-selection.csv, hosted-validation.json, diverse-cases.jsonl and api-call-ledger.jsonl. Each canonical export includes golds, catalog/candidate IDs, aliases and raw returned text for independent rescoring. Private provider generation IDs are intentionally excluded. This is owner-run evidence; independent review and human semantic adjudication remain separate. The original visual PDF remains a snapshot of the independently validated local data commit.\n')
    print(json.dumps({'file':output.name,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_commit':commit,'public_files':len(members)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(pathlib.Path(p.parse_args().output))
