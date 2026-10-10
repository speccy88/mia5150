import pathlib,subprocess,sys,json
R=pathlib.Path(__file__).resolve().parent;py=str(R/'.venv/bin/python')
stages=[('lora-replay-clean-resumed',[py,'lora_pilot.py','replay-clean-resumed']),('lora-transfer-base',[py,'lora_transfer.py','base']),('lora-transfer-hard',[py,'lora_transfer.py','hard']),('lora-transfer-replay-clean-resumed',[py,'lora_transfer.py','replay-clean-resumed'])]
for tag,cmd in stages:
 p=R/'runs'/tag
 if p.exists():assert (p/'exit.json').exists() and json.loads((p/'exit.json').read_text())['returncode']==0;continue
 subprocess.run(['python3','reclaim.py'],cwd=R,check=True);status=subprocess.run(['python3','guard.py',tag]+cmd,cwd=R).returncode;print(tag,status,flush=True)
 if status:sys.exit(status)
sys.exit(subprocess.run(['python3','pipeline_extra.py'],cwd=R).returncode)
