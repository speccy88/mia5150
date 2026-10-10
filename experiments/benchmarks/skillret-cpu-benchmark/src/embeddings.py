from pathlib import Path
import json,time,hashlib
import numpy as np
from common import ROOT,RESULTS,save
class Embedder:
 def __init__(self,slug):
  import torch
  from transformers import AutoTokenizer,AutoModel
  self.slug=slug;self.maxlen=2048 if slug=='embeddinggemma-int8' else 512;self.path=ROOT/'models'/slug;self.tok=AutoTokenizer.from_pretrained(self.path,local_files_only=True,trust_remote_code=False);torch.set_num_threads(4)
  manifest=json.loads((ROOT/'docs/model_sources.json').read_text())['models']; source=next(x for x in manifest if x['key']=={'bge-small':'bge-small-en-v1.5','e5-small':'multilingual-e5-small','embeddinggemma-int8':'embeddinggemma-300m-int8'}[slug])
  self.meta={'repository':source['repo'],'revision':source['revision'],'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'slug':slug,'threads':4,'device':'CPU','max_length':self.maxlen,'normalization':'L2'}
  if slug=='embeddinggemma-int8':
   import onnxruntime as ort
   o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1;o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;o.add_session_config_entry('session.intra_op.allow_spinning','0');o.add_session_config_entry('session.inter_op.allow_spinning','0')
   self.session=ort.InferenceSession(str(self.path/'onnx/model_quantized.onnx'),o,providers=['CPUExecutionProvider']);self.inputs={x.name for x in self.session.get_inputs()};self.meta.update(runtime='onnxruntime',precision='INT8 public ONNX conversion',pooling='exported sentence_embedding',ort_allow_spinning=False)
  else:self.model=AutoModel.from_pretrained(self.path,local_files_only=True,trust_remote_code=False).eval().to('cpu');self.meta.update(runtime='torch',precision='float32',pooling='CLS' if slug=='bge-small' else 'masked mean')
 def format(self,text,kind):
  if self.slug=='bge-small':return ('Represent this sentence for searching relevant passages: '+text) if kind=='query' else text
  if self.slug=='e5-small':return ('query: ' if kind=='query' else 'passage: ')+text
  return ('task: search result | query: '+text) if kind=='query' else 'title: none | text: '+text
 def encode(self,texts,kind='query',batch_size=8):
  import torch
  output=[]
  for start in range(0,len(texts),batch_size):
   batch=[self.format(x,kind) for x in texts[start:start+batch_size]];tok=self.tok(batch,padding=True,truncation=True,max_length=self.maxlen,return_tensors='np' if self.slug=='embeddinggemma-int8' else 'pt')
   if self.slug=='embeddinggemma-int8':v=self.session.run(['sentence_embedding'],{k:a.astype(np.int64) for k,a in tok.items() if k in self.inputs})[0]
   else:
    with torch.inference_mode():
     h=self.model(**tok).last_hidden_state
     if self.slug=='bge-small':v=h[:,0]
     else:
      mask=tok['attention_mask'].unsqueeze(-1);v=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
     v=v.float().numpy()
   v=np.asarray(v,dtype=np.float32);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12);output.append(v)
  return np.concatenate(output)
 def cached(self,texts,kind,tag):
  token_lengths=[len(x) for x in self.tok([self.format(t,kind) for t in texts],truncation=False,padding=False)['input_ids']]
  token_audit={'raw_tokens':sum(token_lengths),'truncated_count':sum(x>self.maxlen for x in token_lengths),'max_raw_tokens':max(token_lengths),'max_length':self.maxlen}
  p=RESULTS/'embeddings';p.mkdir(exist_ok=True);file=p/f'{self.slug}_{tag}.npy';meta=p/f'{self.slug}_{tag}.json'
  h=hashlib.sha256(json.dumps({'texts':texts,'kind':kind,'model':self.meta},sort_keys=True).encode()).hexdigest();dim=768 if self.slug=='embeddinggemma-int8' else 384;start=0;previous_seconds=0
  if meta.exists():
   old=json.loads(meta.read_text());assert old['fingerprint']==h;start=old['completed'];previous_seconds=old['encoding_seconds'];a=np.load(file,mmap_mode='r+')
  else:a=np.lib.format.open_memmap(file,mode='w+',dtype=np.float32,shape=(len(texts),dim))
  t=time.perf_counter()
  for i in range(start,len(texts),16):
   a[i:i+16]=self.encode(texts[i:i+16],kind,16)
   if i%160==0 or i+16>=len(texts):
    a.flush();done=min(i+16,len(texts));save(meta,{'fingerprint':h,'completed':done,'count':len(texts),'encoding_seconds':previous_seconds+time.perf_counter()-t,'batch_size':16,'token_audit':token_audit,'shape':list(a.shape),'model':self.meta});print(self.slug,tag,done,'/',len(texts),flush=True)
  a.flush();return np.asarray(a),json.loads(meta.read_text())
