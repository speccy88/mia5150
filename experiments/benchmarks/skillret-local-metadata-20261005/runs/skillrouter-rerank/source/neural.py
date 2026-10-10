"""Pinned local FP16 adapters; no remote code, no qrels in scoring."""
from harness import R,load,view,save,digest
import numpy as np,json,pathlib,time,sys,os
PROMPTS={'qwen_embed_06':'Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: ','skillret_embed_06':'Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: ','skillrouter_embed_06':'Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task\nQuery:','skillreason_embed_06':'Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task\nQuery: '}
class Embed:
 def __init__(self,role,max_length=1024):
  import torch
  from transformers import AutoModel,AutoTokenizer
  torch.set_num_threads(1);self.torch=torch;self.role=role;self.max_length=max_length
  p=R/'models'/role;self.tok=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False,padding_side='left')
  self.model=AutoModel.from_pretrained(p,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda').eval()
  for fpath in p.glob('*.safetensors'):
   with fpath.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 def encode(self,texts,query=False,batch=4):
  torch=self.torch;vectors=[];tokens=[];times=[]
  for i in range(0,len(texts),batch):
   text=[(PROMPTS[self.role] if query else '')+t for t in texts[i:i+batch]];lengths=[len(x) for x in self.tok(text,add_special_tokens=True,truncation=False)['input_ids']];tokens.extend(lengths)
   start=time.perf_counter();inputs=self.tok(text,padding=True,truncation=True,max_length=self.max_length,return_tensors='pt').to('cuda')
   with torch.inference_mode():
    out=self.model(**inputs).last_hidden_state[:,-1];out=torch.nn.functional.normalize(out,dim=1);vectors.append(out.float().cpu().numpy())
   times.append(time.perf_counter()-start)
   if i%100==0:print('encode',i,len(texts),flush=True)
  return np.concatenate(vectors),{'raw_token_counts':tokens,'truncated_count':sum(n>self.max_length for n in tokens),'batch_seconds':times,'max_length':self.max_length}
class Rerank:
 def __init__(self,role,max_length=1024):
  import torch
  from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification
  torch.set_num_threads(1);self.torch=torch;self.role=role;self.max_length=max_length;p=R/'models'/role
  self.tok=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False,padding_side='left');config=json.loads((p/'config.json').read_text());self.classifier='SequenceClassification' in ' '.join(config.get('architectures',[]))
  cls=AutoModelForSequenceClassification if self.classifier else AutoModelForCausalLM
  self.model=cls.from_pretrained(p,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda').eval()
  self.yes=self.tok.convert_tokens_to_ids('yes');self.no=self.tok.convert_tokens_to_ids('no')
  system='Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
  self.prefix=self.tok.encode(f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n',add_special_tokens=False)
  self.suffix=self.tok.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
 def inputs(self,q,d):
  if self.role=='r3_rerank_06':
   instruction='Given a user request, retrieve the agent skill that solves it.'
   messages=[{'role':'system','content':instruction},{'role':'query','content':q},{'role':'document','content':d}]
   ids=self.tok.apply_chat_template(messages,tokenize=True)
   marker=self.tok.apply_chat_template(messages[:-1]+[{'role':'document','content':'__DOC_BOUNDARY__'}],tokenize=False)
   suffix=self.tok.encode(marker.split('__DOC_BOUNDARY__')[1],add_special_tokens=False);assert ids[-len(suffix):]==suffix
   self.last_tokens=len(ids);ids=ids[:self.max_length-len(suffix)]+suffix if len(ids)>self.max_length else ids
   return self.tok.pad({'input_ids':[ids]},padding=True,return_tensors='pt').to('cuda')
  if self.classifier:return self.tok(q,d,truncation=True,max_length=self.max_length,return_tensors='pt').to('cuda')
  instruction='Given a skill search query, judge whether the skill document is relevant and useful for the query'
  if self.role in ('skillrouter_rerank_06','skillreason_rerank_06'):instruction='Given a task description, judge whether the skill document is relevant and useful for completing the task'
  sep='\n\n' if self.role=='skillrouter_rerank_06' else '\n'
  text=f'<Instruct>: {instruction}{sep}<Query>: {q}{sep}<Document>: {d}'
  ids=self.tok.encode(text,add_special_tokens=False);self.last_tokens=len(ids)+len(self.prefix)+len(self.suffix);ids=self.prefix+ids[:self.max_length-len(self.prefix)-len(self.suffix)]+self.suffix
  return self.tok.pad({'input_ids':[ids]},padding=True,return_tensors='pt').to('cuda')
 def score(self,q,docs):
  torch=self.torch;scores=[];tokens=[]
  with torch.inference_mode():
   for d in docs:
    inputs=self.inputs(q,d)
    if self.classifier:v=self.model(**inputs).logits.flatten()[0]
    else:
     h=self.model.model(**inputs).last_hidden_state[:,-1:];logits=self.model.lm_head(h)[0,0];v=logits[self.yes]-logits[self.no]
    scores.append(float(v));tokens.append(getattr(self,'last_tokens',int(inputs['input_ids'].numel())))
  return scores,tokens
def embed_main(role,kind,limit=0):
 skills=[]
 with (R/'data/skills/test.jsonl').open() as f:
  for line in f:
   s=json.loads(line);skills.append({'id':s['id'],'text':view(s,kind)})
   if limit and len(skills)>=limit:break
 queries=json.loads((R/'manifests/scorer-queries.json').read_text());queries=queries[:4] if limit else queries
 start=time.perf_counter();m=Embed(role);startup=time.perf_counter()-start
 m.encode(['warm up']);docs,docstats=m.encode([s['text'] for s in skills]);qvec,qstats=m.encode([q['query'] for q in queries],True,batch=1)
 cache=R/'cache';cache.mkdir(exist_ok=True);np.save(cache/f'{role}-{kind}-{limit}-docs.npy',docs);np.save(cache/f'{role}-{kind}-{limit}-queries.npy',qvec)
 rankings={};lat=[]
 for q,v in zip(queries,qvec):
  start=time.perf_counter();scores=docs@v;order=np.argsort(-scores,kind='stable')[:100];rankings[q['id']]=[skills[i]['id'] for i in order];lat.append(time.perf_counter()-start)
 save(R/f'results/{role}-{kind}-{limit}.json',{'rankings':rankings,'startup_seconds':startup,'document_stats':docstats,'query_stats':qstats,'search_seconds':lat,'model_manifest':json.loads((R/'models'/role/'manifest.json').read_text()),'corpus_hash':digest([s['text'] for s in skills]),'queries_hash':digest(queries),'precision':'FP16','attention':'sdpa','batch':4,'cpu_threads':1,'view':kind,'smoke_only':bool(limit)})
if __name__=='__main__':embed_main(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 0)
