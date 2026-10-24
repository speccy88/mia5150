"""Bundle only an explicit public allowlist, never private runtime files."""
import argparse,hashlib,json,pathlib,subprocess,zipfile
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def main(output):
    names=['README.md','diversity-design-20261024.md','continuation-20261024.json','diverse-protocol.json','diverse-budget-plan.json','diverse-endpoint-snapshots.json','diverse-request-plan.csv','diverse_contract.py','prepare_diverse.py','run_diverse.py','runner.py','envelope.py','request_log.py','budget.py','publish_status.py','publish_diverse.py','analyze_diverse.py','package_evidence.py','mistral_contract.py','prepare_mistral.py','mistral-protocol.json','mistral-protocol-metadata-amendment.json','mistral-request-plan.csv','mistral-endpoint-snapshot.json','finalize_hosted.py']
    members={'hosted/'+name:HERE/name for name in names}
    for p in sorted((HERE/'reports').iterdir()):
        if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.md','.png','.tex']:members['hosted/reports/'+p.name]=p
    for p in sorted((HERE/'mistral-reports').iterdir()):
        if p.is_file():members['hosted/mistral-reports/'+p.name]=p
    for p in sorted((HERE.parent/'audit/hosted-qwen').iterdir()):
        if p.is_file():members['audit/hosted-qwen/'+p.name]=p
    for name in ['core.py','ledger.py','evaluator-labels.json','protocol.json','model-provenance.json']:members['study/'+name]=HERE.parent/name
    for p in HERE.parent.glob('test_hosted*.py'):members['tests/'+p.name]=p
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in members.items()}
    manifest={'repository':'https://github.com/speccy88/mia5150','source_commit':commit,'local_data_commit':'0f4e9818b6cefb94e09da30ad76b9ff5ac97999c','files_sha256':hashes,'privacy':'Explicit public source/report allowlist only. No credentials, ciphertext, private generation/account IDs, raw private runtime files, original student-identifying inputs or model binaries.'}
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,path in members.items():z.write(path,name)
        z.writestr('EVIDENCE_MANIFEST.json',json.dumps(manifest,indent=2))
        z.writestr('READ_ME_FIRST.md','# Final hosted follow-up evidence\n\nThe balanced panel contains 24 previously exposed queries at N=100. Qwen completed 120 cases; the separately frozen Mistral/Parasail arm failed its first canary with upstream429 and attempted no benchmark cases. No further generation.150 total POST attempts:147 successful Qwen plus three failed provider canaries. Known charges US$0.02411946; UNKNOWN holds US$0.19; guarded total US$0.21411946. Actual total cost remains unknown.\n\nStart with hosted/reports/HOSTED_REPORT.md and hosted/mistral-reports/terminal-execution.json. Raw Qwen cases include golds, catalog/candidate IDs, aliases and returned text. The independent Qwen audit targets its immutable497ca150 snapshot, before the Mistral attempt; preserve its stated limits and original packet hashes. Public provider receipts are consistency evidence, not account invoice verification. Private generation/account IDs and credentials are excluded. The original visual PDF remains unchanged.\n')
    print(json.dumps({'file':output.name,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_commit':commit,'public_files':len(members)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(pathlib.Path(p.parse_args().output))
