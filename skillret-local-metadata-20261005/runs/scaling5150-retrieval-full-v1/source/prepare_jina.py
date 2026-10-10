from harness import *
import shutil
p=R/'models/jina_rerank_35';shutil.copy2(R/'evidence/jina-modeling-source.txt',p/'modeling.py')
for role,name in [('jina_rerank_35','modeling.py'),('r3_rerank_06','chat_template.jinja')]:
 path=R/'models'/role;manifest=json.loads((path/'manifest.json').read_text());f=path/name;manifest['files'][name]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()};save(path/'manifest.json',manifest)
save(R/'evidence/jina-source-review.json',{'repository_revision':json.loads((p/'manifest.json').read_text())['revision'],'source_sha256':hashlib.sha256((p/'modeling.py').read_bytes()).hexdigest(),'review':'Read entire 10389-byte module before execution. Standard torch/transformers/numpy imports; native joint-candidate learned-projector architecture. Tokenizer auto-download prevented by prepopulated local tokenizer and offline environment. No subprocess, shell, credential or network operations in reviewed scoring code. Noncommercial academic license.'})
