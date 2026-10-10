import pathlib,subprocess,sys,json
R=pathlib.Path(__file__).resolve().parent;py=str(R/'.venv/bin/python');prefix=['env','MIA_LORA_BASE_ROLE=skillret_rerank_06_fp16_local',py]
stages=[('skillret-fp16-shards',[py,'convert_r3_fp16.py','skillret_rerank_06']),('lora-replay-clean-sharded',prefix+['lora_pilot.py','replay-clean-sharded']),('lora-transfer-base',prefix+['lora_transfer.py','base']),('lora-transfer-hard',prefix+['lora_transfer.py','hard']),('lora-transfer-replay-clean-sharded',prefix+['lora_transfer.py','replay-clean-sharded'])]
for tag,cmd in stages:
 p=R/'runs'/tag
 if p.exists():assert (p/'exit.json').exists() and json.loads((p/'exit.json').read_text())['returncode']==0;continue
 subprocess.run(['python3','reclaim.py'],cwd=R,check=True);status=subprocess.run(['python3','guard.py',tag]+cmd,cwd=R).returncode;print(tag,status,flush=True)
 if status:sys.exit(status)
sys.exit(subprocess.run(['python3','pipeline_extra.py'],cwd=R).returncode)
