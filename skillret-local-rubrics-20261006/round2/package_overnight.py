"""Snapshot final reports and nonsecret evidence only; original packages preserved."""
import hashlib,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
reports=['OVERNIGHT_FINDINGS.md','ROUND2_CLASS_DEMO.md','ORDER_SENSITIVITY_REPORT.md','CANDIDATE_BUDGET_REPORT.md','CLAUSE_AND_COVERAGE_REPORT.md','FUSION_CONFIRMATION_REPORT.md','CACHED_FUSION_REPORT.md','HYBRID_ADAPTER.md','FINAL_BUDGET_STATUS.json','PAID_EXPERIMENTS_PAUSED.json']
files=reports+['hybrid-demo.html','hybrid-demo-data.json','hybrid_retrieval.py','probe_contract.py','hybrid_demo_cli.py','test_hybrid_retrieval.py','domain-coverage.json','order-sensitivity-plan.json','candidate_budget_protocol.json','request-clause-protocol.json','fusion-confirmation-plan.json','selective-probe-plan.json','exploration-freeze.json']
for folder in ['jev-typed-12-v1','openrouter-listwise-12-v1','selective-jev-8-v1','fusion-confirmation-8-v1']:
 files.extend(folder+'/'+name for name in ['results.json','evaluation.json'])
files+=['order-sensitivity-2-v1/results.json','order-sensitivity-2-v1/comparison.json','candidate-budget30-v1/freeze.json','candidate-budget30-v1/evaluation.json','candidate-budget30-v1/per-query.json','clause-8-v1/baseline-reproduction.json','clause-8-v1/evaluation.json','clause-8-v1/predictions.json']
out=HERE/'MIA5150_Overnight_Evidence_2026-10-06.zip'
with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(HERE/name,name)
with zipfile.ZipFile(out) as z:assert z.namelist()==files and z.testzip() is None
receipt={'filename':out.name,'bytes':out.stat().st_size,'files':len(files),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'explicit_whitelist':files,'credentials_or_profiles_included':False,'original_packages_unchanged':True}
(HERE/'overnight-package-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='explicit_whitelist'}))
