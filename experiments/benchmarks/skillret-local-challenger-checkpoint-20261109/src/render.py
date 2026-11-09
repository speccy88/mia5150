"""Label-blind selected evidence from original skill_md with source offsets."""
from common import *
from collections import Counter
import re
from functools import lru_cache
INSTRUCTION='Given a skill search query, judge whether the skill document is relevant and useful for the query'
PREFIX_R='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
SHORT_WINDOWS='Keep an entire short source window when it fits; only truncated fragments need at least 32 tokens.'
def words(x):return re.findall(r'\w+',x.casefold())

def prepare_renderer():
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(CACHE/'models/reranker',local_files_only=True,padding_side='left');s,q,g=data('train');df=Counter()
 for x in s:df.update(set(words(x['skill_md'])))
 idf={w:math.log(1+(len(s)-n+.5)/(n+.5)) for w,n in sorted(df.items())};save(CACHE/'renderer/idf.json',idf)
 save(STUDY/'manifests/renderer.json',{'rule':'BM25-like frozen lexical evidence over ALL original body windows; rank spans by sum IDF(qword)*tf/(tf+1.2), choose greedily within residual token budget, restore original source order; query/suffix retained, metadata capped 192 tokens; overlap32; source spans exact offsets','idf_corpus':'all 10123 training-origin skills, no relevance labels','idf_sha256':sha(CACHE/'renderer/idf.json'),'window_tokens':256,'overlap_tokens':32,'metadata_cap':192,'budgets':[1024,2048],'full_coverage':'all dynamic-budget body windows, sequential max aggregation per skill; no individual window positive labels','determinism':'IDF keys and unique query terms are lexicographically sorted; lexical sums and source-order ties are independent of Python hash randomization.','short_window_policy':SHORT_WINDOWS})
 Renderer(tok,'train')
 print('Renderer prepared',flush=True)

class Renderer:
 def __init__(self,tok,split='train'):
  self.tok=tok;self.idf=read(CACHE/'renderer/idf.json');assert sha(CACHE/'renderer/idf.json')==read(STUDY/'manifests/renderer.json')['idf_sha256'];self.records={};self.split=split
  self.record_dir=CACHE/'renderer'/f'{split}-records';self.record_dir.mkdir(parents=True,exist_ok=True)
  self.prefix=tok.encode(PREFIX_R,add_special_tokens=False);self.suffix=tok.encode(SUFFIX,add_special_tokens=False)
  p=CACHE/'renderer'/f'{split}-window-index.json'
  if p.exists():self.records=read(p);return
  skills,_,_=data(split)
  manifest=[]
  for x in skills:
   body=x['skill_md'];enc=tok(body,add_special_tokens=False,return_offsets_mapping=True,truncation=False);ids=enc['input_ids'];offsets=enc['offset_mapping'];spans=[]
   for start in range(0,max(1,len(ids)),224):
    end=min(len(ids),start+256)
    if not offsets:break
    a=offsets[start][0];b=offsets[end-1][1];spans.append({'start':a,'end':b,'token_start':start,'tokens':ids[start:end],'term_counts':dict(Counter(words(body[a:b]))),'sha256':digest(body[a:b])})
    if end==len(ids):break
   meta=(x.get('name') or '').strip()+' | '+(x.get('description') or '').strip()+' | '
   record={'body_sha256':digest(body),'metadata_ids':tok.encode(meta,add_special_tokens=False),'body_characters':len(body),'body_ids':ids,'offsets':offsets,'windows':spans}
   manifest.append({'skill_id':x['id'],'body_sha256':digest(body),'body_characters':len(body),'windows':[{k:v for k,v in a.items() if k in ('start','end','sha256')} for a in spans]})
   save(self.record_dir/(x['id']+'.json'),record);self.records[x['id']]={'record_sha256':sha(self.record_dir/(x['id']+'.json'))}
  save(p,self.records);write_rows(STUDY/'manifests'/f'{split}-body-windows.jsonl',manifest)
 @lru_cache(maxsize=32)
 def record(self,sid):
  p=self.record_dir/(sid+'.json');assert sha(p)==self.records[sid]['record_sha256'];return read(p)
 def query_context(self,query):
  cached=getattr(self,'last_query_context',None)
  if cached is None or cached[0]!=query:
   head=self.prefix+self.tok.encode(f'<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: ',add_special_tokens=False)
   cached=(query,head,sorted(set(words(query))),digest(query));self.last_query_context=cached
  return cached[1:]
 def render(self,query,sid,budget=1024,full=False):
  t=self.tok;r=self.record(sid);head,qwords,query_hash=self.query_context(query);remaining=budget-len(head)-len(self.suffix);assert remaining>=64,'Full query and suffix cannot fit'
  meta=r['metadata_ids'][:min(192,remaining//3)];room=remaining-len(meta);windows=r['windows']
  if not hasattr(self,'body_separator'):self.body_separator=t.encode('\n\n',add_special_tokens=False)
  sep=self.body_separator
  if full:
   # Dynamic windows cover every token, using exactly the same rendering for all skills.
   pieces=[];raw=r['body_ids'];offsets=r['offsets'];stride=max(1,room-32)
   for start in range(0,max(1,len(raw)),stride):
    end=min(len(raw),start+room);spans=[] if not offsets else [[offsets[start][0],offsets[end-1][1]]]
    pieces.append((head+meta+raw[start:end]+self.suffix,{'spans':spans,'body_characters':r['body_characters'],'metadata_tokens_omitted':len(r['metadata_ids'])-len(meta),'full_coverage':True,'query_sha256':query_hash,'body_sha256':r['body_sha256']}))
    if end==len(raw):break
   return pieces
  scores=[sum(self.idf.get(w,0)*c.get(w,0)/(c.get(w,0)+1.2) for w in qwords) for c in (x['term_counts'] for x in windows)];chosen=[];used=0
  for i in sorted(range(len(windows)),key=lambda i:(-scores[i],i)):
   n=min(len(windows[i]['tokens']),room-used-(len(sep) if chosen else 0))
   if n<=0 or (n<32 and n<len(windows[i]['tokens'])):continue
   chosen.append((i,n));used+=n+(len(sep) if len(chosen)>1 else 0)
   if room-used<=len(sep):break
  body=[];spans=[]
  for i,n in sorted(chosen):
   if body:body+=sep
   w=windows[i];body+=w['tokens'][:n]
   # Re-encode offsets are monotone and exact; prefix shortening records the true end.
   end=w['end'] if n==len(w['tokens']) else r['offsets'][w['token_start']+n-1][1]
   spans.append([w['start'],end])
  ids=head+meta+body+self.suffix;assert len(ids)<=budget
  return [(ids,{'spans':spans,'body_characters':r['body_characters'],'metadata_tokens_omitted':len(r['metadata_ids'])-len(meta),'full_coverage':False,'query_sha256':query_hash,'body_sha256':r['body_sha256']})]
