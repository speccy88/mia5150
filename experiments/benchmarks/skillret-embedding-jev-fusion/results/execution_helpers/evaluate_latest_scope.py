from pathlib import Path
import json,sys
root=Path('/home/user/code/mia5150-local/skillret-embedding-jev-fusion')
sys.path.insert(0,str(root))
import eval_hosted_snapshot as ev
scope=sys.argv[1]; assert scope in ('smoke','pilot')
directory='jev-plan-smoke-context-v1' if scope=='smoke' else 'jev-plan-pilot-jev113-context-v1'
plan=root/'results'/directory/'preparation_manifest.json'
manifest=json.loads(plan.read_text()); execution=root/'results/hosted-execution'
latest=json.loads((execution/'study_progress'/manifest['plan_sha256']/'latest.json').read_text())
assert latest['status']=='completed' and latest['complete_queries']==len(manifest['rows'])
output=root/'results'/('hosted-'+scope+'-native-field-evaluation')
sys.argv=['eval_hosted_snapshot.py','--plan',str(plan),'--run-dir',str(execution),'--snapshot',latest['snapshot_dir'],
 '--baseline',str(root/'results/baseline_skillret_06b.json'),'--qrels',str(root.parent/'data/qrels/test.jsonl'),
 '--output-dir',str(output)]
ev.main()
r=json.loads((output/'evaluation.json').read_text())
print(json.dumps({'scope':scope,'metrics':{s:r['evaluation']['systems'][s]['aggregate'] for s in 'ABC'},
 'ledger':r['experiment_unique_ledger_at_snapshot'],'projection':r['projection']}))
