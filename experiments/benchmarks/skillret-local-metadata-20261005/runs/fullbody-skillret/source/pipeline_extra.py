"""Second bounded sequential batch, every heavyweight action guarded."""
import subprocess,pathlib,json,sys,time
R=pathlib.Path(__file__).resolve().parent;py=str(R/'.venv/bin/python')
stages=[('r3-direct-gpu',[py,'rerank.py','r3_rerank_06','skillret_embed_06-header-0.json','header','32']),('full-lexical',[py,'full_lexical.py']),('sra-download',[py,'fetch_sra.py']),('jina-download',[py,'fetch_models.py','jina_rerank_35']),('jina-source',[py,'prepare_jina.py']),('jina-native',[py,'jina_native.py']),('fullbody-smoke',[py,'neural.py','skillret_embed_06','full','32']),('fullbody-skillret',[py,'neural.py','skillret_embed_06','full']),('field-six',[py,'field_dense.py']),('qwen8-download',[py,'fetch_qwen8.py']),('qwen8-rerank',[py,'qwen8_rerank.py']),('llm-candidates',[py,'llm_candidates.py'])]
stages.insert(0,('pytest-install',['/home/user/code/frognano-benchmark/tools/uv','pip','install','--python',py,'pytest==8.4.2']))
stages.insert(1,('contract-tests',[py,'-m','pytest','-q']))
stages.insert(9,('lora-replay-clean',[py,'lora_pilot.py','replay-clean']))
stages.extend([('large-choice-jev',[py,'large_choice.py','jevk5']),('large-choice-plumb',[py,'large_choice.py','plumb']),('lora-transfer-base',[py,'lora_transfer.py','base']),('lora-transfer-hard',[py,'lora_transfer.py','hard']),('lora-transfer-replay-clean',[py,'lora_transfer.py','replay-clean'])])
stages.extend([('prompt-32768',[py,'prompt_scale.py','32768','2']),('sra-transfer',[py,'sra_transfer.py'])])
stages=[(tag,cmd) for tag,cmd in stages if tag!='lora-transfer-replay-clean']
stages.insert(2,('r3-fp16-cache',[py,'convert_r3_fp16.py']))
stages.insert(3,('r3-fp16-rerank',[py,'rerank.py','r3_rerank_fp16_local','skillret_embed_06-header-0.json','header','32']))
stages.insert(0,('paired-statistics-checkpoint',[py,'statistics_report.py']))
stages.insert(0,('training-audit',[py,'audit_training.py']))
for tag,cmd in stages:
 out=R/'runs'/tag
 if out.exists():
  marker=out/'exit.json'
  if marker.exists() and json.loads(marker.read_text()).get('returncode')==0:continue
  print('Blocked existing failed run',tag,flush=True);continue
 subprocess.run(['python3','reclaim.py'],cwd=R,check=True)
 status=subprocess.run(['python3','gate_run.py',tag]+cmd,cwd=R).returncode
 print(tag,status,flush=True)
 if status:
  print('Stopped; inspect failure before continuing.',flush=True);sys.exit(status)
