"""Distinguish available implementations from completed and blocked executions."""
from harness import *
stages={
 'Original plan/data/split and metric audit':['contract-root','training-audit'],
 'Full4392 public comparison':['full-lexical','full-skillret-direct'],
 'Native R3 sharded FP16':['r3-fp16-cache','r3-fp16-rerank'],
 'Native Jina joint reranking':['jina-source','jina-native'],
 'Full-body1024 retrieval':['fullbody-smoke','fullbody-skillret'],
 'Six-channel field retrieval':['field-six'],
 'Actual hard-negative LoRA':['lora-hard-streamed'],
 'Hard-negative LoRA transfer':['lora-transfer-base','lora-transfer-hard'],
 'Corrected replay (interrupted checkpoints only)':['lora-replay-clean-lean'],
 'Quantized8B native reranking':['qwen8-download','qwen8-rerank'],
 'Local4B ID ranking and pairwise':['llm-candidates'],
 'Native multipass option choice':['large-choice-jev','large-choice-plumb'],
 '32K full-catalog prompt diagnostic':['prompt-32768'],
 'SRA frozen24-query full-catalog transfer':['sra-transfer'],
 'SkillReason frozen24-query full-catalog transfer':['skillreason-data','skillreason-schema','skillreason-external'],
 'Frozen16-query actual escalation latency':['policy-native'],
}
rows=[]
for name,tags in stages.items():
 receipts={}
 for tag in tags:
  p=R/'runs'/tag
  if (p/'guard-stop.json').exists():receipts[tag]={'status':'resource-blocked','receipt':json.loads((p/'guard-stop.json').read_text())}
  elif (p/'exit.json').exists():
   r=json.loads((p/'exit.json').read_text());receipts[tag]={'status':'completed' if r['returncode']==0 else 'failed','receipt':r}
  elif (R/f'evidence/gate-{tag}.json').exists() and not json.loads((R/f'evidence/gate-{tag}.json').read_text()).get('ready'):
   receipts[tag]={'status':'resource-blocked-before-launch','receipt':'evidence/gate-'+tag+'.json'}
  else:receipts[tag]={'status':'not-run-or-unfinished'}
 rows.append({'component':name,'runs':receipts,'status':'tested' if all(r['status']=='completed' for r in receipts.values()) else 'blocked-or-incomplete'})
save(R/'results/scope-receipts.json',rows)
lines=['# Execution status','','This table describes recorded execution, not only code availability. Read IMPLEMENTATION_MAP.md for methodological boundaries.','','| Component | Status | Run receipts |','|---|---|---|']
for row in rows:lines.append('| '+row['component']+' | '+row['status']+' | '+', '.join(k+': '+v['status'] for k,v in row['runs'].items())+' |')
lines+=['','Implemented and tested earlier: locked lexical and0.6B retrieval, same-pool rerankers, grouped lexical adaptation, selection/domain/escalation controls,8K prompting and nested catalog scaling. Raw receipts and exact source snapshots remain in runs/.','', 'Deferred: paid APIs/RunPod,8B BF16 reference, full32K dense evidence, large multiseed training and full-query reranking tournament. Conditional SPLADE/late interaction/generated views, learned routing and labelled dependency/activation methods were not promoted or claimed implemented without their required evidence. Certified untouched final confirmation remains unavailable.']
(R/'EXECUTION_STATUS.md').write_text('\n'.join(lines))
