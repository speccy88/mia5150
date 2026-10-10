"""Single sequential queue, guarded stages with fail-closed stop."""
import subprocess,pathlib,json,sys
R=pathlib.Path(__file__).resolve().parent;py=str(R/'.venv/bin/python')
stages=[('anchoring',[py,'adaptation_anchored.py']),('structures',[py,'structures.py']),('field-tfidf',[py,'field_tfidf.py']),('escalation',[py,'escalation.py']),('scale',[py,'scale.py']),('jev-skillret',[py,'quantized.py','jevk5','8']),('plumb-skillret',[py,'quantized.py','plumb','8']),('prompt-scale',[py,'prompt_scale.py']),('full-qwen',[py,'public_full.py','qwen_embed_06']),('full-skillret',[py,'public_full.py','skillret_embed_06'])]
for tag,cmd in stages:
 out=R/'runs'/tag
 if out.exists():
  marker=out/'exit.json'
  if marker.exists() and json.loads(marker.read_text()).get('returncode')==0:continue
  raise RuntimeError(f'Existing unverified run: {out}')
 subprocess.run(['python3','reclaim.py'],cwd=R,check=True)
 status=subprocess.run(['python3','guard.py',tag]+cmd,cwd=R).returncode
 print(tag,status,flush=True)
 if status:sys.exit(status)
