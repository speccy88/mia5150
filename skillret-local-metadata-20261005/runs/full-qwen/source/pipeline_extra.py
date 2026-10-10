"""Second bounded sequential batch, every heavyweight action guarded."""
import subprocess,pathlib,json,sys,time
R=pathlib.Path(__file__).resolve().parent;py=str(R/'.venv/bin/python')
stages=[('r3-direct-gpu',[py,'rerank.py','r3_rerank_06','skillret_embed_06-header-0.json','header','32']),('jina-download',[py,'fetch_models.py','jina_rerank_35']),('jina-source',[py,'prepare_jina.py']),('jina-native',[py,'jina_native.py']),('fullbody-smoke',[py,'neural.py','skillret_embed_06','full','32']),('fullbody-skillret',[py,'neural.py','skillret_embed_06','full']),('field-six',[py,'field_dense.py']),('sra-download',[py,'fetch_sra.py']),('qwen8-download',[py,'fetch_qwen8.py']),('qwen8-rerank',[py,'qwen8_rerank.py'])]
for tag,cmd in stages:
 out=R/'runs'/tag
 if out.exists():
  marker=out/'exit.json'
  if marker.exists() and json.loads(marker.read_text()).get('returncode')==0:continue
  print('Blocked existing failed run',tag,flush=True);continue
 subprocess.run(['python3','reclaim.py'],cwd=R,check=True)
 status=subprocess.run(['python3','guard.py',tag]+cmd,cwd=R).returncode
 print(tag,status,flush=True)
 if status:
  print('Stopped; inspect failure before continuing.',flush=True);sys.exit(status)
