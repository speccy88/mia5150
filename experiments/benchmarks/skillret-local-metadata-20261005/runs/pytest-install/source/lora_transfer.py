"""Training-disjoint public-task transfer, same full512 evidence and pool."""
from harness import *
from neural import Rerank
from peft import PeftModel
import time,sys
mode=sys.argv[1];assert mode in ('base','hard','replay-clean');queries=json.loads((R/'manifests/scorer-queries.json').read_text())[:32];pool=json.loads((R/'results/skillret_embed_06-header-0.json').read_text())['rankings'];needed={d for q in queries for d in pool[q['id']][:20]};skills={}
with (R/'data/skills/test.jsonl').open() as f:
 for line in f:
  s=json.loads(line)
  if s['id'] in needed:skills[s['id']]=view(s,'full')
model=Rerank('skillret_rerank_06',max_length=512)
if mode!='base':model.model=PeftModel.from_pretrained(model.model,R/f'checkpoints/lora-{mode}',is_trainable=False,local_files_only=True).get_base_model().eval()
rankings={};raw=[]
for q in queries:
 ids=pool[q['id']][:20];start=time.perf_counter()
 try:scores,tokens=model.score(q['query'],[skills[d] for d in ids]);rankings[q['id']]=[ids[i] for i in sorted(range(len(ids)),key=lambda i:(-scores[i],i))];status='ok'
 except Exception as e:scores=[];tokens=[];rankings[q['id']]=[];status=repr(e)
 raw.append({'id':q['id'],'candidate_ids':ids,'scores':scores,'tokens':tokens,'seconds':time.perf_counter()-start,'status':status});print(q['id'],status,flush=True)
labels=qrels('test');save(R/f'results/lora-transfer-{mode}.json',{'rankings':rankings,'raw':raw,'metrics':evaluate({q['id']:labels[q['id']] for q in queries},rankings),'pool_hash':digest({q['id']:pool[q['id']][:20] for q in queries}),'note':'Full512, train/test skill IDs disjoint, public32-query transfer diagnostic; public query feedback already consumed, not pristineOOD confirmation.'})
