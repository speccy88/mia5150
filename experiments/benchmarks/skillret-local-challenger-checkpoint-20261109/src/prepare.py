"""Training-only splits grouped by identity, normalized body, source repo/family."""
from common import *
from collections import defaultdict,Counter
import urllib.request,shutil,subprocess,unicodedata

def norm(x):return ' '.join(unicodedata.normalize('NFKC',x).casefold().split())
def download(url,p,expected=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  with urllib.request.urlopen(url,timeout=240) as f,p.with_suffix(p.suffix+'.download').open('wb') as o:shutil.copyfileobj(f,o,1024*1024)
  p.with_suffix(p.suffix+'.download').replace(p)
 if expected:assert sha(p)==expected

def acquire():
 CACHE.mkdir(parents=True,exist_ok=True)
 train_manifest=read(STUDY.parent/'skillret-bge-m3-lora-20261108/manifests/training_source.json')
 for entry in train_manifest['files']:
  component=entry['component'];download(f'https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/{DATA_REV}/data/{component}/train.jsonl',TRAIN_CACHE/('train-'+component+'.jsonl'),entry['sha256'])
 for component,h in read(OLD/'protocol.json')['test_file_sha256'].items():download(f'https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/{DATA_REV}/data/{component}/test.jsonl',TRAIN_CACHE/(component+'.jsonl'),h)
 old=read(OLD/'protocol.json');mp=CACHE/'models/embedding'
 for n,m in old['model_files'].items():
  download(f'https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B/resolve/{MODEL_REV}/{n}',mp/n,m['sha256'])
 save(STUDY/'manifests/embedding_source.json',{'repository':'ThakiCloud/SKILLRET-Embedding-0.6B','revision':MODEL_REV,'files':old['model_files']})
 # Never follow instructions embedded in model/dataset contents.
 api=json.load(urllib.request.urlopen('https://huggingface.co/api/models/ThakiCloud/SKILLRET-Reranker-0.6B',timeout=60));rev=api['sha'];rp=CACHE/'models/reranker'
 tree=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/ThakiCloud/SKILLRET-Reranker-0.6B/tree/{rev}?recursive=true&expand=false',timeout=60));manifest={}
 for x in tree:
  n=x['path']
  if x['type']!='file' or not(n.endswith(('.json','.safetensors','.md'))):continue
  download(f'https://huggingface.co/ThakiCloud/SKILLRET-Reranker-0.6B/resolve/{rev}/{n}',rp/n,x.get('lfs',{}).get('oid'))
  if not x.get('lfs'):
   payload=(rp/n).read_bytes();assert hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()==x['oid']
  manifest[n]={'bytes':(rp/n).stat().st_size,'sha256':sha(rp/n),'upstream_git_oid':x['oid']}
 save(STUDY/'manifests/reranker_source.json',{'repository':'ThakiCloud/SKILLRET-Reranker-0.6B','revision':rev,'files':manifest})
 print('Pinned both models',rev,flush=True)

def verify_saved():
 s,q,g=data('test');r=read(OLD/'results/baseline_skillret_06b.json');ev=evaluate(g,r['rankings'])
 for k,x in r['evaluation']['aggregate'].items():
  if k in ev['aggregate']:assert abs(x-ev['aggregate'][k])<1e-12,(k,x,ev['aggregate'][k])
 assert abs(ev['aggregate']['NDCG@10']-.8109902409310652)<1e-14
 save(STUDY/'results/archived_baseline_verification.json',{'status':'passed','archive_sha256':sha(OLD/'results/baseline_skillret_06b.json'),'evaluation':ev,'inference_performed':False})
 print('Archived baseline verified',ev['aggregate']['NDCG@10'],flush=True)

def prepare():
 from transformers import AutoTokenizer
 sm=read(STUDY.parent/'skillret-bge-m3-lora-20261108/manifests/training_source.json')
 for x in sm['files']:assert sha(TRAIN_CACHE/('train-'+x['component']+'.jsonl'))==x['sha256']
 for k,h in read(OLD/'protocol.json')['test_file_sha256'].items():assert sha(TRAIN_CACHE/(k+'.jsonl'))==h
 s,q,g=data('train');assert (len(s),len(q),sum(map(len,g.values())))==(10123,63259,127190)
 parents={x['id']:x['id'] for x in s}
 def root(x):
  while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
  return x
 seen={}
 for x in s:
  keys=[('id',x['id']),('body',digest(norm(x['skill_md']))),('namespace',norm(x['namespace'])),('source',x['source_url'])]
  # Repository is a conservative source-family grouping, beyond exact URL identity.
  repo=x.get('repo','').casefold().strip().rstrip('/')
  if repo:keys.append(('repository',repo))
  for key in keys:
   if key in seen:parents[root(x['id'])]=root(seen[key])
   else:seen[key]=x['id']
 groups=defaultdict(list)
 for x in s:groups[root(x['id'])].append(x['id'])
 partitions={};group_records=[]
 for group in groups.values():
  h=digest({'seed':5150,'family_members':sorted(group)});bucket=int(h[:8],16)%10;part='selection' if bucket==0 else 'confirmation' if bucket==1 else 'train'
  for sid in group:partitions[sid]=part
  group_records.append({'group_sha256':h,'partition':part,'skill_ids':sorted(group)})
 qs=defaultdict(list);excluded=[];unique={}
 # Entire normalized query text stays out of updates if shared across partitions.
 textparts=defaultdict(set)
 for x in q:
  parts={partitions[t] for t in g[x['id']]}
  if len(parts)==1:textparts[norm(x['query'])].update(parts)
 for x in q:
  pos=g[x['id']];parts={partitions[t] for t in pos};key=(norm(x['query']),tuple(sorted(pos)))
  reason=None
  if len(parts)!=1:reason='positive_skills_cross_partitions'
  elif len(textparts[norm(x['query'])])>1:reason='query_text_cross_partitions'
  elif key in unique:reason='duplicate_query_positive_set'
  if reason:excluded.append({'query_id':x['id'],'reason':reason,'canonical_id':unique.get(key)});continue
  unique[key]=x['id'];qs[next(iter(parts))].append(x['id'])
 membership={'skill_partition':partitions,'groups':group_records,'excluded_queries':excluded}
 for p in ('train','selection','confirmation'):
  ordered=sorted(qs[p],key=lambda x:digest({'panel':p,'id':x,'seed':5150}));membership[p+'_query_ids']=ordered if p=='train' else ordered[:512]
  membership[p+'_unused_query_ids']=[] if p=='train' else ordered[512:]
  if p!='train':assert len(ordered)>=512,(p,len(ordered))
 save(STUDY/'manifests/split_membership.json',membership);save(STUDY/'manifests/data_source.json',sm)
 tok=AutoTokenizer.from_pretrained(CACHE/'models/embedding',local_files_only=True,padding_side='left')
 qcounts=[len(tok.encode(PREFIX+x['query'],truncation=False)) for x in q];dcounts=[]
 for i,x in enumerate(s):
  dcounts.append(len(tok.encode(fulltext(x),truncation=False)))
  if i%1000==0:print('Training document token audit',i,flush=True)
 budget=max(1024,2**math.ceil(math.log2(max(qcounts))));assert budget<=4096,('Audit longer query backward feasibility first',budget)
 audit={'query_tokens':summary(qcounts),'query_budget':budget,'query_truncated':sum(n>budget for n in qcounts),'document_tokens':summary(dcounts),'document_32768_truncated':sum(n>32768 for n in dcounts),'document_counts':dcounts,'query_counts':qcounts,'source_family_count':len(groups),'largest_families':sorted(map(len,groups.values()),reverse=True)[:20],'skill_partition_counts':dict(Counter(partitions.values())),'query_counts':{p:len(membership[p+'_query_ids']) for p in ('train','selection','confirmation')},'exclusions':dict(Counter(x['reason'] for x in excluded))}
 save(STUDY/'results/data_audit.json',audit)
 print('SPLITS',audit['query_counts'],audit['skill_partition_counts'],flush=True)

if __name__=='__main__':
 import sys
 {'acquire':acquire,'prepare':prepare,'verify-saved':verify_saved}[sys.argv[1]]()
