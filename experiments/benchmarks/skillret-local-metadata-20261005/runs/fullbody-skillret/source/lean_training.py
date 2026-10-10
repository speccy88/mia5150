import pathlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parent;py=str(R/'.venv/bin/python');prefix=['env','MIA_LORA_BASE_ROLE=skillret_rerank_06_fp16_local',py]
stages=[('lora-replay-clean-lean',prefix+['lora_pilot.py','replay-clean-lean']),('lora-transfer-base',prefix+['lora_transfer.py','base']),('lora-transfer-hard',prefix+['lora_transfer.py','hard']),('lora-transfer-replay-clean-lean',prefix+['lora_transfer.py','replay-clean-lean'])]
for tag,cmd in stages:
 subprocess.run(['python3','reclaim.py'],cwd=R,check=True);status=subprocess.run(['python3','gate_run.py',tag]+cmd,cwd=R).returncode;print(tag,status,flush=True)
 if status:sys.exit(status)
sys.exit(subprocess.run(['python3','pipeline_extra.py'],cwd=R).returncode)
