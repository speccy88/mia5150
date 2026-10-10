import pathlib,urllib.request,json,zipfile,hashlib,importlib.metadata
r=pathlib.Path.home()/'code/embeddinggemma2-pilot';(r/'wheels').mkdir(exist_ok=True)
def get(url):return urllib.request.urlopen(url,timeout=45).read()
for package in ['transformers','sentence-transformers']:
 data=json.loads(get(f'https://pypi.org/pypi/{package}/json'));(r/(package+'-pypi.json')).write_text(json.dumps(data));print(package,data['info']['version'],data['info']['requires_dist'])
 wheel=next(x for x in data['urls'] if x['filename'].endswith('py3-none-any.whl'));p=r/'wheels'/wheel['filename'];p.write_bytes(get(wheel['url']));assert hashlib.sha256(p.read_bytes()).hexdigest()==wheel['digests']['sha256']
 with zipfile.ZipFile(p) as z:
  hits=[n for n in z.namelist() if 'embedding_gemma2' in n];print('SUPPORT_FILES',hits)
  for n in hits:
   if n.endswith('.py'):
    dest=r/'runtime-review'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
for n in ['torch','transformers','tokenizers','huggingface-hub','sentence-transformers','safetensors']:
 try:print('INSTALLED',n,importlib.metadata.version(n))
 except importlib.metadata.PackageNotFoundError:print('MISSING',n)
model=json.loads(get('https://huggingface.co/api/models/google/embeddinggemma-2?blobs=true'));(r/'model-metadata.json').write_text(json.dumps(model,indent=2));print('ARTIFACTS',[(x['rfilename'],x.get('size'),x.get('lfs')) for x in model['siblings']])
