import pathlib,sys,json,urllib.request,hashlib,subprocess,importlib.metadata,importlib,os
from packaging.requirements import Requirement
from packaging.version import Version,InvalidVersion
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename,canonicalize_name
r=pathlib.Path.home()/'code/embeddinggemma2-pilot';overlay=r/'overlay';overlay.mkdir(exist_ok=True);sys.path.insert(0,str(overlay));uv=pathlib.Path.home()/'code/frognano-benchmark/tools/uv';tags=set(sys_tags());manifest=json.loads((r/'dependency-manifest.json').read_text()) if (r/'dependency-manifest.json').exists() else []
def fetch(url):return json.load(urllib.request.urlopen(url,timeout=60))
def install(name,spec):
 assert name not in ('torch','numpy') and not name.startswith('nvidia-'),(name,spec)
 meta=fetch(f'https://pypi.org/pypi/{name}/json');versions=[]
 for v in meta['releases']:
  try:
   value=Version(v)
   if not value.is_prerelease and value in spec:versions.append(value)
  except InvalidVersion:pass
 version=max(versions,default=None);assert version
 release=fetch(f'https://pypi.org/pypi/{name}/{version}/json');choices=[x for x in release['urls'] if x['filename'].endswith('.whl') and parse_wheel_filename(x['filename'])[3]&tags];assert choices,name
 item=choices[0];assert item['size']<100*1024**2;dest=r/'wheels'/item['filename']
 if not dest.exists():
  with urllib.request.urlopen(item['url'],timeout=60) as src,open(dest,'wb') as out:
   while chunk:=src.read(1024**2):out.write(chunk)
 assert hashlib.file_digest(open(dest,'rb'),'sha256').hexdigest()==item['digests']['sha256']
 subprocess.run([str(uv),'pip','install','--python',sys.executable,'--target',str(overlay),'--no-deps','--no-index',str(dest)],check=True)
 manifest.append({'name':name,'version':str(version),'file':item['filename'],'sha256':item['digests']['sha256'],'url':item['url']});(r/'dependency-manifest.json').write_text(json.dumps(manifest,indent=2));importlib.invalidate_caches()
roots=[Requirement('transformers==5.19.0'),Requirement('sentence-transformers==6.1.0'),Requirement('Pillow>=10.0.1,<=15.0'),Requirement('torchvision==0.23.0')]
for _ in range(8):
 queue=roots.copy();seen=set();requirements={};missing=[]
 while queue:
  req=queue.pop();name=canonicalize_name(req.name);requirements.setdefault(name,[]).append(req)
  if name in seen:continue
  seen.add(name)
  try:dist=importlib.metadata.distribution(name)
  except importlib.metadata.PackageNotFoundError:continue
  for raw in dist.requires or []:
   child=Requirement(raw)
   if child.marker and not child.marker.evaluate({'extra':''}):continue
   queue.append(child)
 for name,reqs in requirements.items():
  try:v=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:v=None
  if v is None or any(v not in req.specifier for req in reqs):
   from packaging.specifiers import SpecifierSet
   missing.append((name,SpecifierSet(','.join(str(x.specifier) for x in reqs))))
 if not missing:break
 print('INSTALL',[(n,str(s)) for n,s in missing],flush=True)
 for name,spec in missing:install(name,spec)
else:raise RuntimeError('Dependency closure not resolved')
effective={n:{'version':importlib.metadata.version(n),'path':str(importlib.metadata.distribution(n)._path)} for n in requirements};(r/'effective-dependencies.json').write_text(json.dumps(effective,indent=2));print('DEPENDENCIES_OK',len(effective),flush=True)
import transformers
from transformers import EmbeddingGemma2Model,EmbeddingGemma2Config
print('NATIVE_IMPORT_OK',transformers.__version__,flush=True)
