import pathlib,ast,json,hashlib,importlib.metadata
r=pathlib.Path.home()/'code/jina-nano-pilot';e=pathlib.Path.home()/'code/embeddinggemma2-pilot';manifest=json.loads((r/'review-manifest.json').read_text());review=[]
for item in manifest:
 if not item['file'].endswith('.py'):continue
 p=r/'model'/item['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];tree=ast.parse(p.read_text());imports=[]
 for n in ast.walk(tree):
  if isinstance(n,(ast.Import,ast.ImportFrom)):imports.append(ast.unparse(n))
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in ['exec','eval','open','compile','__import__']
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):assert n.func.attr not in ['system','popen','Popen','unlink','remove','rmdir']
 review.append(item|{'imports':imports})
(r/'code-review.json').write_text(json.dumps({'source':'Pinned official Jina repository; reviewed loader and encoding logic, imports and sensitive-call scan; offline local paths avoid snapshot_download branch; no shell execution or arbitrary eval observed. This is not a formal security audit.','files':review},indent=2))
prefix=(e/'eg2_eval.py').read_text().split('smoke=[]')[0].replace('model.max_seq_length=512','model.max_seq_length=1024').replace("'max_tokens':512","'max_tokens':1024").replace('min(raw,512)','min(raw,1024)').replace('raw>512','raw>1024');(r/'eg2_sensitivity.py').write_text(prefix+(r/'eg2_sensitivity_tail.py').read_text())
print('PACKAGES',{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','huggingface-hub','safetensors']});print('SOURCE_REVIEW_COMPLETE',len(review))
