"""Compact review delivery: exclude already validated primary and repeated history."""
import argparse,hashlib,json,pathlib,zipfile
from ledger import canonical_attempts
HERE=pathlib.Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);out=pathlib.Path(p.parse_args().output_dir);out.mkdir(parents=True,exist_ok=True)
    files={}
    for model,expected in [('qwen3_06_q8',1080),('qwen35_4b_q4',600)]:
        with (HERE/f'results/{model}-cases.jsonl').open() as f:rows=canonical_attempts([json.loads(line) for line in f if line.strip()]).values()
        if model=='qwen3_06_q8':rows=[r for r in rows if not(r['panel']=='official_exposed' and r['seed']==42)]
        else:rows=list(rows)
        assert len(rows)==expected and not any(r['status'].startswith('not_run') for r in rows)
        dest=out/(model+'-nonprimary-canonical.jsonl');dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));files[dest.name]=dest
    for name in ['protocol.json','evaluator-labels.json','panel.json','generated-panel.json','generated-label-audit.json','EXECUTION_STATUS.json','MEASUREMENT_CORRECTIONS.md','core.py','aggregate.py']:
        files['study/'+name]=HERE/name
    for p in (HERE/'report').glob('*'):
        if p.suffix in ['.json','.csv','.md']:files['study/report/'+p.name]=p
    for p in (HERE/'results').glob('*execution*.json*'):files['study/results/'+p.name]=p
    files['study/results/persistence-error-receipt.json']=HERE/'results/persistence-error-receipt.json'
    for p in (HERE/'source-snapshots').rglob('*.py'):files['study/'+p.relative_to(HERE).as_posix()]=p
    manifest={name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in files.items()}
    packet=out/'MIA5150_Completed_Local_Sensitivity_2026-10-24.zip'
    with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('README.md','# Completed local sensitivity review\n\nContains 1,080 non-primary small-model cases and 600 4B cases. The previously validated 4,000 primary rows and repeated append-only history are excluded. Reuse unchanged protocol catalogs and public inputs from the primary packet/repository. All selector cells are processed; context exclusions/failures remain. One generated persistence outcome is unavailable, not retried. No hosted or human-label success is claimed.\n')
        z.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2))
        for name,p in files.items():z.write(p,name)
    with zipfile.ZipFile(packet) as z:assert z.testzip() is None
    print(json.dumps({'path':str(packet),'bytes':packet.stat().st_size,'sha256':hashlib.sha256(packet.read_bytes()).hexdigest(),'small_nonprimary_cases':1080,'medium_cases':600}))
if __name__=='__main__':main()
