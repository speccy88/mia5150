from pathlib import Path,PurePosixPath
import zipfile,json,hashlib
root=Path(__file__).resolve().parent/'deliverables';p=root/'MIA5150_Local_Results_2026-10-05.zip'
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None
 inventory=json.loads(z.read('mia5150-local/PACKAGE_SHA256.json'))['files']
 for info in z.infolist():
  path=PurePosixPath(info.filename);assert not path.is_absolute() and '..' not in path.parts
  assert path.suffix.lower() not in {'.pt','.pth','.bin','.safetensors','.gguf','.onnx','.npz','.npy','.partial'}
  assert not {'.git','.venv','.aws','.sandbox-secrets','__pycache__'}.intersection(path.parts)
  if info.filename.endswith('/PACKAGE_SHA256.json'):continue
  relative=info.filename.removeprefix('mia5150-local/');expected=inventory[relative];h=hashlib.sha256()
  with z.open(info) as f:
   while block:=f.read(1024*1024):h.update(block)
  assert h.hexdigest()==expected['sha256'] and info.file_size==expected['bytes']
 receipt={'file':str(p),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'verified_entries':len(z.infolist()),'safe_paths':True,'model_optimizer_binaries_excluded':True,'uploaded':False,'largest_entries':sorted([(x.filename,x.file_size) for x in z.infolist()],key=lambda x:x[1],reverse=True)[:5]}
(root/'LOCAL_VERIFICATION.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
