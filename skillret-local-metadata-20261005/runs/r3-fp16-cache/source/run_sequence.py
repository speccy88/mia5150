"""Sequential local stages. Stops on failure; never changes resource limits."""
import subprocess,pathlib,sys,json
R=pathlib.Path('/home/user/code/mia5150-local');py=str(R/'.venv/bin/python')
stages={
 'cpu':[('anchoring',[py,'adaptation_anchored.py']),('structures',[py,'structures.py']),('field-tfidf',[py,'field_tfidf.py'])],
 'specialists':[('skillrouter-header',[py,'neural.py','skillrouter_embed_06','header']),('skillreason-header',[py,'neural.py','skillreason_embed_06','header']),('skillrouter-rerank',[py,'rerank.py','skillrouter_rerank_06','skillret_embed_06-header-0.json','header','32']),('skillreason-rerank',[py,'rerank.py','skillreason_rerank_06','skillret_embed_06-header-0.json','header','32']),('r3-rerank',[py,'rerank.py','r3_rerank_06','skillret_embed_06-header-0.json','header','32'])],
 'training':[('lora-smoke',[py,'lora_smoke.py']),('lora-hard',[py,'lora_pilot.py','hard']),('lora-replay',[py,'lora_pilot.py','replay'])],
 'medium':[('jev-skillret',[py,'quantized.py','jevk5','8']),('plumb-skillret',[py,'quantized.py','plumb','8'])],
 'scale':[('scale',[py,'scale.py'])]
}
for tag,cmd in stages[sys.argv[1]]:
 subprocess.run(['python3','reclaim.py'],cwd=R,check=True)
 out=R/'runs'/tag
 if out.exists():raise RuntimeError(f'Refusing to overwrite {out}')
 status=subprocess.run(['python3','guard.py',tag]+cmd,cwd=R).returncode
 print(tag,status,flush=True)
 if status:sys.exit(status)
