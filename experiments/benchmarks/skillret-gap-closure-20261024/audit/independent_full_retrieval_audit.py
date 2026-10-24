"""Independent streaming saved-rank audit; no experiment-code imports or inference."""
import argparse,collections,csv,gzip,hashlib,itertools,json,pathlib,random
HERE=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=pathlib.Path,default=HERE/'raw_library/MIA5150_Full4392_Retrieval_2026-10-24.jsonl.gz')
p.add_argument('--primary-root',type=pathlib.Path,default=HERE.parent/'candidate_ed12adad')
p.add_argument('--report',type=pathlib.Path,default=HERE/'report/full-retrieval.csv')
p.add_argument('--output',type=pathlib.Path,default=HERE/'independent_full_retrieval_validation.json')
a=p.parse_args();P=a.primary_root;I=P/'evidence_extract/public-inputs';S=P/'evidence_extract/study'
def read(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as f:return [json.loads(x) for x in f if x.strip()]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
protocol=read(S/'protocol.json');gold=collections.defaultdict(set)
for filename,sha in protocol['source_hashes'].items():assert hashlib.sha256((I/filename).read_bytes()).hexdigest()==sha
for r in rows(I/'qrels-test.jsonl'):
 if r['relevance']>0:gold[r['query_id']].add(r['skill_id'])
queries=rows(I/'queries-test.jsonl');ids=sorted(r['id'] for r in rows(I/'skills-metadata.jsonl'));idset=set(ids)
assert len(ids)==len(idset)==6006 and len(gold)==len(queries)==4392 and sum(map(len,gold.values()))==7187
assert all(set(q['skill_ids'])==gold[q['id']] for q in queries)
primary_path=P/'raw_library/MIA5150_Primary200_Raw_2026-10-24.jsonl'
assert hashlib.sha256(primary_path.read_bytes()).hexdigest()=='6b77f74019790a27243ca2e243aa6626acddbe8070c69930658192de8b4302f2'
primary=rows(primary_path)
prior={(r['query_id'],r['size'],r['method']):r for r in primary if r['method'] in ['bm25','vector','hybrid'] and r.get('retrieval') is not None}
assert len(prior)==2400
b=a.raw.read_bytes();gzsha=hashlib.sha256(b).hexdigest();gitsha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
assert len(b)==15054930 and gzsha=='a55bfecfa99508cebcc0d6c0238575c60bfbd919d46c2e69654f66dcd0b53a35' and gitsha=='268b1281e2851a811aab7be4cdcc32d363c88536'
checks=collections.Counter();seen=set();counts=collections.Counter();sources=collections.Counter();sums={};basefull={};previous=None;rawsha=hashlib.sha256();rawbytes=0;overlap=collections.Counter()
with gzip.open(a.raw,'rb') as f:
 for line in f:
  rawsha.update(line);rawbytes+=len(line);r=json.loads(line);q,seed,n=r['query_id'],r['seed'],r['size'];key=(q,seed,n)
  assert key not in seen and q in gold and seed in [42,43,44] and n in [10,100,1000,6006];seen.add(key);counts[(seed,n)]+=1
  assert r['case_id']==digest(['retrieval',protocol['protocol_sha256'],q,seed,n])
  if previous!=(q,seed):
   rest=sorted(idset-gold[q]);random.Random(int(digest([seed,q,'membership']),16)).shuffle(rest);ordered=sorted(gold[q])+rest;previous=(q,seed)
  library=ordered[:n];assert len(library)==len(set(library))==n and gold[q]<=set(library) and digest(library)==r['catalog_sha256'];checks['catalog_memberships']+=1
  assert set(r['methods'])=={'bm25','vector','hybrid'};sources[r['source_sha256']]+=1
  for method,d in r['methods'].items():
   ranking=d['ranking5'];assert len(ranking)==len(set(ranking))==5 and set(ranking)<=set(library)
   hits=len(gold[q]&set(ranking));rr=next((1/(i+1) for i,x in enumerate(ranking) if x in gold[q]),0.);recall=hits/len(gold[q]);complete=int(gold[q]<=set(ranking))
   for k,v in {'retrieved_hits':hits,'gold_count':len(gold[q]),'recall5':recall,'mrr5':rr,'complete5':complete}.items():assert abs(d[k]-v)<1e-12,(key,method,k)
   assert d['search_seconds']>=0;checks['ranker_metric_checks']+=1
   aggregate=sums.setdefault((seed,n,method),{'count':0,'recall':0.,'mrr':0.,'hits':0,'gold':0,'search':0.})
   for k,v in [('count',1),('recall',recall),('mrr',rr),('hits',hits),('gold',len(gold[q])),('search',d['search_seconds'])]:aggregate[k]+=v
   if seed==42 and (q,n,method) in prior:
    old=prior[(q,n,method)];assert ranking==old['ranking5']
    for k in ['recall5','mrr5','complete5','retrieved_hits','gold_count']:assert abs(d[k]-old['retrieval'][k])<1e-12
    overlap[method]+=1
   if n==6006:
    if seed==42:basefull[(q,method)]=ranking
    else:assert ranking==basefull[(q,method)];checks['full_catalog_seed_invariance']+=1
assert len(seen)==52704 and seen==set(itertools.product(gold,[42,43,44],[10,100,1000,6006]))
assert len(sources)==1 and rawbytes==91864324 and rawsha.hexdigest()=='dcf3630cf9d8fade4375e335c43aaf070ee507ce0482579276efd8df23ca60e1'
for r in csv.DictReader(a.report.open()):
 k=(int(r['seed']),int(r['size']),r['method']);s=sums[k];assert s['count']==int(r['queries'])==4392 and s['gold']==7187
 for column,v in [('macro_recall5',s['recall']/s['count']),('micro_positive_pair_recall5',s['hits']/s['gold']),('mrr5',s['mrr']/s['count']),('search_seconds_mean',s['search']/s['count'])]:assert abs(float(r[column])-v)<1e-12,(k,column)
 checks['aggregate_cells']+=1
assert checks['aggregate_cells']==36 and dict(overlap)=={'bm25':800,'vector':800,'hybrid':800}
result={'status':'passed','commit':'0f4e9818b6cefb94e09da30ad76b9ff5ac97999c','source_library_id':'libfile_1c718d66ccc88191a5519e6cb6a93a4e','unique_queries':4392,'positive_qrels':7187,'query_seed_size_rows':len(seen),'checks':dict(checks),'primary_overlap_checks':dict(overlap),'primary_overlap_differences':0,'raw_gzip':{'bytes':len(b),'sha256':gzsha,'git_blob_sha1':gitsha},'uncompressed':{'bytes':rawbytes,'sha256':rawsha.hexdigest()},'execution_source_sha256':dict(sources),'scope':'Independent full-rank positive-qrel rescore, catalog/ID/grid validation, 36 aggregates and all 2400 primary overlaps; no inference or search reexecution','limits':['All requests were previously exposed; this is full-benchmark breadth, not fresh-holdout evidence','Small catalogs are gold-conditioned; catalog membership is naturally identical across seeds at N6006','Repeated seeds and sizes do not increase the number of independent queries','Search-only timings are reproduced as recorded; sparse BM25 and cached batch query embeddings are distinct from primary end-to-end timing','No hosted calls, human semantic adjudication, native activation, or execution success established']}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
