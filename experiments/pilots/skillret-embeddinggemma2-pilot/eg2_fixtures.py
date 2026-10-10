import pathlib,json,hashlib,numpy as np,shutil
r=pathlib.Path.home()/'code/embeddinggemma2-pilot';m=pathlib.Path.home()/'code/mia5150-local';dest=r/'fixtures';dest.mkdir(exist_ok=True)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
old=pathlib.Path.home()/'code/category-pilot/embedding/results'
for name in ['fixtures.json','quality.json','summary.json','command.json']:shutil.copy2(old/name,dest/('qwen-small-'+name))
skills=rows(m/'data/skills/test.jsonl');ids=[s['id'] for s in skills];index={d:i for i,d in enumerate(ids)};texts=[' | '.join((s.get(k) or '').strip() for k in ['name','description']) for s in skills]
allq=read(m/'manifests/scorer-queries.json');oldcontrol=read(m/'results/qwen_embed_06-header-0.json');assert digest(allq)==oldcontrol['queries_hash'];assert digest(texts)==oldcontrol['corpus_hash']
selected=sorted(allq,key=lambda q:hashlib.sha256(('eg2-pilot-v1:'+q['id']).encode()).hexdigest())[:32];qindex={q['id']:i for i,q in enumerate(allq)}
bm=read(m/'results/public-full-bm25.json')['rankings'];qw=oldcontrol['rankings'];pool=set()
for q in selected:pool.update(qw[q['id']][:5]);pool.update(bm[q['id']][:5])
pool.update(sorted(ids,key=lambda d:hashlib.sha256(('eg2-pool-v1:'+d).encode()).hexdigest())[:32]);pool=sorted(pool);di=[index[d] for d in pool];qi=[qindex[q['id']] for q in selected]
controls={};cachehash={}
for role in ['qwen_embed_06','skillret_embed_06']:
 dp=m/f'cache/{role}-header-0-docs.npy';qp=m/f'cache/{role}-header-0-queries.npy';dv=np.load(dp,mmap_mode='r');qv=np.load(qp,mmap_mode='r');assert dv.shape[0]==len(ids) and qv.shape[0]==len(allq)
 score=qv[qi]@dv[di].T;np.save(dest/(role+'-pool-scores.npy'),score);ranking={q['id']:[pool[j] for j in np.argsort(-score[i],kind='stable')] for i,q in enumerate(selected)}
 if role=='qwen_embed_06':assert all(ranking[q['id']][0]==qw[q['id']][0] for q in selected)
 controls[role]=ranking
 for p in [dp,qp]:cachehash[str(p)]=hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
gold={q['id']:{} for q in selected}
for x in rows(m/'data/qrels/test.jsonl'):
 if x['query_id'] in gold:gold[x['query_id']][x['skill_id']]=int(x['relevance'])
fixture={'protocol':'32 queries selected by SHA-256 of eg2-pilot-v1:qID from frozen 128-query scorer set; shared pool union Qwen top5, BM25 top5,32 hash-selected distractors. No qrels used to select queries or documents. Restricted-corpus reranking/complementarity pilot, not full-catalog retrieval.','queries':selected,'documents':[{'id':d,'text':texts[index[d]]} for d in pool],'qrels':gold,'controls':controls,'qwen_original_top100':{q['id']:qw[q['id']] for q in selected},'bm25_original_top100':{q['id']:bm[q['id']] for q in selected},'original_corpus_count':len(ids),'source_corpus_hash':oldcontrol['corpus_hash'],'source_queries_hash':oldcontrol['queries_hash'],'cache_sha256':cachehash,'source_qwen_model_manifest':oldcontrol['model_manifest']}
(dest/'skillret-pilot.json').write_text(json.dumps(fixture,indent=2));(dest/'manifest.json').write_text(json.dumps({p.name:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest() for p in dest.iterdir() if p.is_file()},indent=2));print('FROZEN',len(selected),len(pool),'queries/docs','gold coverage',sum(len(set(g)&set(pool)) for g in gold.values()),sum(len(g) for g in gold.values()))
