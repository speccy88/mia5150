import pathlib,json,hashlib,zipfile
root=pathlib.Path(__file__).resolve().parent
evidence=json.loads((root/'evidence-summary.json').read_text())
run=root/'runs'/evidence['selected_training_run']
assert json.loads((run/'result.json').read_text())['completed_optimizer_updates']==20
files=set()
for name in ['SearchJev-SkillRet-Report.txt','SearchJev-Results.txt','evidence-summary.json','paired-quality-analysis.json','pc-release.json','host.jsonl','host-monitor-source.ps1']:
 p=root/name
 if p.is_file():files.add(p)
for p in root.glob('*.py'):files.add(p)
for p in root.glob('*.sh'):files.add(p)
for folder in ['official-review','data-audit','licenses','runs']:
 for p in (root/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.safetensors','.pyc']:
   files.add(p)
for p in [root/'models/manifest.json',root/'models/base-metadata.json',root/'models/adapter-metadata.json',root/'models/base/config.json',root/'models/adapter/adapter_config.json']:
 files.add(p)
files.add(run/'checkpoint-update20/adapter_model.safetensors')
manifest=[]
for p in sorted(files):
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 manifest.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h))
manifest_path=root/'DELIVERABLE-MANIFEST.json';manifest_path.write_text(json.dumps(manifest,indent=2))
files.add(manifest_path)
archive=root/'SearchJev-SkillRet-Pilot.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for p in sorted(files):z.write(p,p.relative_to(root))
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
receipt=dict(file=str(archive),bytes=archive.stat().st_size,sha256=digest,entries=len(files),base_weights_included=False,trained_adapter_included=True)
(root/'package-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
