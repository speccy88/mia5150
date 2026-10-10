from harness import *
from local_server import server,post
import time,math
queries=json.loads((R/'manifests/scorer-queries.json').read_text())[:8];pool=json.loads((R/'results/skillret_embed_06-header-0.json').read_text())['rankings'];needed={d for q in queries for d in pool[q['id']][:20]};skills={}
with (R/'data/skills/test.jsonl').open() as f:
 for line in f:
  row=json.loads(line)
  if row['id'] in needed:skills[row['id']]=view(row,'header')
rankings={};raw=[];model=R/'models/qwen8_gguf/Qwen3-Reranker-8B.Q4_K_M.gguf'
with server(model,'qwen8',2048):
 for q in queries:
  ids=pool[q['id']][:20];scores=[];counts=[];start=time.perf_counter();status='ok'
  try:
   for d in ids:
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';text=prefix+'<Instruct>: Given a skill search query, judge whether the skill document is relevant and useful for the query\n<Query>: '+q['query']+'\n<Document>: '+skills[d]
    tok=post('/tokenize',{'content':text,'add_special':False,'parse_special':True})['tokens'];end=post('/tokenize',{'content':suffix,'add_special':False,'parse_special':True})['tokens'];counts.append(len(tok)+len(end));tok=tok[:1024-len(end)]+end
    out=post('/completion',{'prompt':tok,'n_predict':1,'n_probs':256,'temperature':0,'cache_prompt':False});lp={x['token']:x['logprob'] for x in out['completion_probabilities'][0]['top_logprobs']};assert 'yes' in lp and 'no' in lp,'Native yes/no logits missing';scores.append(lp['yes']-lp['no'])
   rankings[q['id']]=[ids[i] for i in sorted(range(len(ids)),key=lambda i:(-scores[i],i))]
  except Exception as e:rankings[q['id']]=[];status=repr(e)
  raw.append({'id':q['id'],'scores':scores,'candidate_ids':ids,'raw_tokens':counts,'seconds':time.perf_counter()-start,'status':status});save(R/'results/qwen8-q4.partial.json',{'rankings':rankings,'raw':raw});print(q['id'],status,flush=True)
labels=qrels('test');save(R/'results/qwen8-q4.json',{'rankings':rankings,'raw':raw,'metrics':evaluate({q['id']:labels[q['id']] for q in queries},rankings),'manifest':json.loads((model.parent/'manifest.json').read_text()),'note':'Native yes/no logit difference; 1024-token cap; third-partyQ4_K_M deployment diagnostic, not official BF16 reference.'})
