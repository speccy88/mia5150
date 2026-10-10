"""Offline GGUF tensor extent check and hash; never downloads or loads model."""
import hashlib,json,math,os,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150';p=ROOT/'models/qwen8_gguf/Qwen3-Reranker-8B.Q4_K_M.gguf'
def main():
 def number(f,fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
 def string(f):
  n=number(f,'Q');assert n<10000000;return f.read(n).decode('utf8')
 formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
 def value(f,t):
  if t==8:return string(f)
  if t==9:
   sub=number(f,'I');n=number(f,'Q');assert n<10000000
   for _ in range(n):value(f,sub)
   return {'array_count':n}
  return number(f,formats[t])
 with p.open('rb') as f:
  assert f.read(4)==b'GGUF';version=number(f,'I');assert version==3;nt=number(f,'Q');nk=number(f,'Q');assert nt<100000 and nk<100000;metadata={}
  for _ in range(nk):key=string(f);metadata[key]=value(f,number(f,'I'))
  tensors=[];blocks={0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292)}
  for _ in range(nt):
   name=string(f);nd=number(f,'I');assert 0<nd<8;dims=[number(f,'Q') for _ in range(nd)];typ=number(f,'I');offset=number(f,'Q');block,size=blocks[typ];n=math.prod(dims);assert n%block==0;tensors.append({'name':name,'dims':dims,'type':typ,'offset':offset,'bytes':n//block*size})
  alignment=metadata.get('general.alignment',32);data_offset=(f.tell()+alignment-1)//alignment*alignment
  ordered=sorted(tensors,key=lambda t:t['offset']);last=0
  for t in ordered:assert t['offset']>=last;last=t['offset']+t['bytes']
  extent=data_offset+last;assert extent<=p.stat().st_size and p.stat().st_size-extent<alignment
  f.seek(0);h=hashlib.sha256();offset=0
  while b:=f.read(8*1024*1024):h.update(b);os.posix_fadvise(f.fileno(),offset,len(b),os.POSIX_FADV_DONTNEED);offset+=len(b)
 meta=json.loads((ROOT/'evidence/qwen8_rerank_gguf.json').read_text());assert meta['gated'] is False and meta['cardData']['license']=='apache-2.0'
 out={'file':str(p),'size':p.stat().st_size,'sha256':h.hexdigest(),'version':version,'tensor_count':nt,'tensor_extent':extent,'gguf_extent_valid':True,'architecture':metadata.get('general.architecture'),'context_length':metadata.get('qwen3.context_length'),'parameters_from_tensors':sum(math.prod(t['dims']) for t in tensors),'source_repository':meta['id'],'source_revision':meta['sha'],'license':'apache-2.0','verification':'All local tensor offsets/byte extents valid, complete file hashed. Cached download provenance; no independent publisher LFS hash available. Third-party quantization, not official BF16 reproduction. No network calls.'};(HERE/'qwen8-integrity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
