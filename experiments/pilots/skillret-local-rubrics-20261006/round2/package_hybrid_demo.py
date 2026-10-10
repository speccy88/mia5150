"""Portable demo package built from an explicit nonsecret file whitelist."""
import hashlib,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
files=['hybrid-demo.html','hybrid-demo-data.json','hybrid-demo-verification.json','hybrid_retrieval.py','probe_contract.py','hybrid_demo_cli.py','test_hybrid_retrieval.py','HYBRID_ADAPTER.md','domain-coverage.json','request-clause-protocol.json']
files += [x+'/results.json' for x in ['jev-typed-12-v1','selective-jev-8-v1','fusion-confirmation-8-v1']]
out=HERE/'MIA5150_Hybrid_Cached_Demo.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(HERE/name,name)
with zipfile.ZipFile(out) as z:
 assert z.namelist()==files and z.testzip() is None
 assert not any(name.endswith(('.key','.env')) or '..' in name.split('/') for name in z.namelist())
receipt={'filename':out.name,'files':len(files),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'whitelist':files,'credential_files_included':False,'network_calls':0}
(HERE/'hybrid-package-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
