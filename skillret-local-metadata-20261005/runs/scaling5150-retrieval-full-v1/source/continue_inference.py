"""Preserve blocked replay; run independent inference stages sequentially."""
import pathlib,subprocess,sys,json
R=pathlib.Path(__file__).resolve().parent;py=str(R/'.venv/bin/python');prefix=['env','MIA_LORA_BASE_ROLE=skillret_rerank_06_fp16_local',py]
for tag,cmd in [('lora-transfer-base',prefix+['lora_transfer.py','base']),('lora-transfer-hard',prefix+['lora_transfer.py','hard'])]:
 p=R/'runs'/tag
 if p.exists():
  if (p/'exit.json').exists() and json.loads((p/'exit.json').read_text())['returncode']==0:continue
  print('Existing failed inference stage',tag,flush=True);continue
 subprocess.run(['python3','reclaim.py'],cwd=R,check=True);status=subprocess.run(['python3','gate_run.py',tag]+cmd,cwd=R).returncode;print(tag,status,flush=True)
 if status:sys.exit(status)
sys.exit(subprocess.run(['python3','pipeline_extra.py'],cwd=R).returncode)
