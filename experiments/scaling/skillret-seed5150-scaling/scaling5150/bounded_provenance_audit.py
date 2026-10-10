"""Bounded CPU-only post-run provenance audit; no torch/model/network imports."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,mmap,resource,shutil,signal,time,itertools
from pathlib import Path
import numpy as np
ROOT=Path('/home/user/code/mia5150-local');HERE=ROOT/'scaling5150';PACKAGE=Path('/mnt/c/Users/user/Documents/Codex/2026-10-06/task-2/staging');OUT=PACKAGE/'scaling5150/provenance-audit';CHUNK=1024*1024
def read(p):return json.loads(p.read_text())
def objhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def hhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def check_ram():
 guard_check()
 available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
 if available<2*1024**3:raise RuntimeError('Stop own audit: guest free below2GiB')
 return available
def filehash(p):
 check_ram()
 h=hashlib.sha256();offset=0
 with p.open('rb') as f:
  while b:=f.read(CHUNK):
   h.update(b);os.posix_fadvise(f.fileno(),offset,len(b),os.POSIX_FADV_DONTNEED);offset+=len(b)
   if offset%(16*CHUNK)==0:check_ram()
 return h.hexdigest()
def release(*arrays):
 for a in arrays:
  if hasattr(a,'_mmap'):a._mmap.madvise(mmap.MADV_DONTNEED)
def array_info(p):
 a=np.load(p,mmap_mode='r',allow_pickle=False);finite=True;normlo=float('inf');normhi=0.
 for lo in range(0,len(a),64):
  check_ram()
  b=np.asarray(a[lo:lo+64]);finite=finite and bool(np.isfinite(b).all());n=np.linalg.norm(b.astype(np.float64),axis=1);normlo=min(normlo,float(n.min()));normhi=max(normhi,float(n.max()));release(a)
 return {'path':str(p),'sha256':filehash(p),'shape':list(a.shape),'dtype':str(a.dtype),'all_finite':finite,'norm_min':normlo,'norm_max':normhi},a
def copy_evidence(source,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 with source.open('rb') as fi,dest.open('wb') as fo:shutil.copyfileobj(fi,fo,CHUNK)
 return {'path':str(dest.relative_to(PACKAGE)),'bytes':dest.stat().st_size,'sha256':filehash(dest)}
def main():
 def timeout(*args):raise TimeoutError('CPU audit180second cap')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(180);started=time.monotonic();OUT.mkdir(exist_ok=False);receipt={'status':'in_progress','scope':'CPU-only, single BLAS thread;1MiB file hashing;64-vector finite/norm chunks;24 bounded ranking replays; no inference, re-encoding, downloads or model-weight hashing.','arrays':{},'models':{},'gaps':[]};save=lambda:(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));save();check_ram()
 manifest=read(HERE/'manifest.json');texts=read(HERE/'documents.json');oldqueries=read(ROOT/'manifests/scorer-queries.json');skills=[];oldtext=[];original=ROOT/'data/skills/test.jsonl';projection=OUT/'public-inputs/skills-metadata.jsonl';projection.parent.mkdir(parents=True,exist_ok=True);rawhash=hashlib.sha256()
 read_offset=0;advised=0
 with original.open('rb') as fi,projection.open('w') as fo:
  for line in fi:
   rawhash.update(line);s=json.loads(line);row={k:s[k] for k in ['id','name','description','major']};skills.append(row);oldtext.append(s['name'].strip()+' | '+s['description'].strip());fo.write(json.dumps(row,ensure_ascii=False)+'\n');read_offset+=len(line)
   if read_offset-advised>=8*CHUNK:os.posix_fadvise(fi.fileno(),advised,read_offset-advised,os.POSIX_FADV_DONTNEED);advised=read_offset;check_ram()
 assert rawhash.hexdigest()==manifest['source_hashes']['skills/test.jsonl'];ids=[s['id'] for s in skills];idpos={d:i for i,d in enumerate(ids)};changed=[i for i,d in enumerate(ids) if oldtext[i]!=texts[d]];assert len(changed)==13
 receipt['public_inputs']={'source_corpus_bytes':original.stat().st_size,'source_corpus_sha256_verified':rawhash.hexdigest(),'metadata_projection':{'bytes':projection.stat().st_size,'sha256':filehash(projection),'records':len(skills),'fields':['id','name','description','major'],'omitted':'skill_md bodies; full117.6MB original corpus not bundled; archived metadata alone cannot reproduce full original-file hash.'},'files':[copy_evidence(ROOT/'data/queries/test.jsonl',OUT/'public-inputs/queries-test.jsonl'),copy_evidence(ROOT/'data/qrels/test.jsonl',OUT/'public-inputs/qrels-test.jsonl'),copy_evidence(ROOT/'manifests/scorer-queries.json',OUT/'public-inputs/scorer-queries-128.json')]}
 assert receipt['public_inputs']['files'][0]['sha256']==manifest['source_hashes']['queries/test.jsonl'] and receipt['public_inputs']['files'][1]['sha256']==manifest['source_hashes']['qrels/test.jsonl']
 publicqueries={q['id']:q['query'] for q in (json.loads(line) for line in (OUT/'public-inputs/queries-test.jsonl').read_text().splitlines())};assert all(publicqueries[q['id']]==q['query'] for q in oldqueries)
 selected=sorted(oldqueries,key=lambda q:hashlib.sha256(f'5150\0panel\0{q["id"]}'.encode()).hexdigest())[:20];assert selected==manifest['queries'];receipt['panel_selection_verified']=True
 receipt['ordered_bindings']={'document_ids_sha256':objhash(ids),'uncapped_header_texts_sha256':objhash(oldtext),'capped_header_texts_sha256':objhash([texts[d] for d in ids]),'historical128queries_sha256':objhash(oldqueries),'selected20queries_sha256':objhash(selected),'serialization':'UTF8 json sort_keys=True,separators=(comma,colon),ensure_ascii=False; array order retained.'}
 receipt['changed_headers']=[{'index':i,'id':ids[i],'old_text_sha256':hashlib.sha256(oldtext[i].encode()).hexdigest(),'new_text_sha256':hashlib.sha256(texts[ids[i]].encode()).hexdigest(),'old_chars':len(oldtext[i]),'new_chars':len(texts[ids[i]])} for i in changed]
 retrieval=read(HERE/'retrieval-full/results.json');saved={(r['method'],r['case_id']):r['ranking'] for r in retrieval['rows']};replay_cases=[c for c in manifest['cases'] if c['query_id'] in manifest['generative_query_ids'] and c['size'] in [50,6006]];fixture=read(Path('/home/user/code/embeddinggemma2-pilot/fixtures/skillret-pilot.json'));all_replays=[]
 for indexdir in sorted(HERE.glob('index-*')):
  check_ram();im=read(indexdir/'manifest.json');role=im['role'];print('audit',role,flush=True);di,dv=array_info(indexdir/'docs.npy');qi,qv=array_info(indexdir/'queries.npy');receipt['arrays'][role+'/output_docs']=di;receipt['arrays'][role+'/output_queries']=qi;assert im['document_ids']==ids and im['query_ids']==[q['id'] for q in selected];detail={'ordered_ids_match':True,'ordered_query_ids_match':True,'output_manifest_sha256':filehash(indexdir/'manifest.json')}
  if 'source_cache_sha256' in im:
   dpath=ROOT/f'cache/{role}-header-0-docs.npy';qpath=ROOT/f'cache/{role}-header-0-queries.npy';old=read(ROOT/f'results/{role}-header-0.json');oi,ov=array_info(dpath);oqi,oqv=array_info(qpath);receipt['arrays'][role+'/source_docs']=oi;receipt['arrays'][role+'/source_queries']=oqi;assert oi['sha256']==im['source_cache_sha256'][dpath.name] and oqi['sha256']==im['source_cache_sha256'][qpath.name];assert old['corpus_hash']==hhash(oldtext) and old['queries_hash']==hhash(oldqueries);assert [ids[i] for i in changed]==im['changed_document_ids'];unchanged_equal=0;changes=[]
   for lo in range(0,len(ids),64):
    a=np.asarray(dv[lo:lo+64]);b=np.asarray(ov[lo:lo+64]);eq=np.all(a==b,axis=1)
    for j,same in enumerate(eq):
     i=lo+j
     if i in changed:changes.append({'id':ids[i],'row_changed':not bool(same),'source_row_sha256':hashlib.sha256(b[j].tobytes()).hexdigest(),'output_row_sha256':hashlib.sha256(a[j].tobytes()).hexdigest()})
     else:unchanged_equal+=bool(same)
    release(dv,ov)
   qpositions={q['id']:i for i,q in enumerate(oldqueries)};qequal=sum(np.array_equal(qv[i],oqv[qpositions[q['id']]]) for i,q in enumerate(selected));assert unchanged_equal==5993 and qequal==20;currentmodel=read(ROOT/'models'/role/'manifest.json');assert old['model_manifest']==im['source_model_manifest']==currentmodel
   detail.update(source_hashes_match_recorded=True,historical_text_and_query_bindings_match=True,unchanged_rows_bitwise_equal=unchanged_equal,query_rows_bitwise_equal=qequal,changed_rows=changes,source_model_manifest_equal=True,revision=currentmodel['revision']);release(ov,oqv)
  else:
   base=Path('/home/user/code/embeddinggemma2-pilot' if role.startswith('eg2') else '/home/user/code/jina-nano-pilot');hist=base/('evaluation-compatible' if role.startswith('eg2') else 'evaluation');oi,ov=array_info(hist/'skillret-docs.npy');oqi,oqv=array_info(hist/'skillret-queries-0.npy');receipt['arrays'][role+'/historical_docs']=oi;receipt['arrays'][role+'/historical_queries']=oqi;pilot=HERE/('embedding-pilot-'+role);pi=read(pilot/'manifest.json');pdi,pdv=array_info(pilot/'docs.npy');pqi,pqv=array_info(pilot/'queries.npy');receipt['arrays'][role+'/new_pilot_docs']=pdi;receipt['arrays'][role+'/new_pilot_queries']=pqi
   sources={d['id']:(d['text'],ov[i],'historical') for i,d in enumerate(fixture['documents'])};qsources={q['id']:(q['query'],oqv[i],'historical') for i,q in enumerate(fixture['queries'])}
   for i,d in enumerate(pi['document_ids']):sources[d]=(texts[d],pdv[i],'new_pilot')
   for i,qid in enumerate(pi['query_ids']):qsources[qid]=(publicqueries[qid],pqv[i],'new_pilot')
   reused=[];qreused=[]
   for i,d in enumerate(ids):
    if d in sources and sources[d][0]==texts[d]:assert np.array_equal(dv[i],sources[d][1]);reused.append({'id':d,'source':sources[d][2]})
   for i,q in enumerate(selected):
    if q['id'] in qsources and qsources[q['id']][0]==q['query']:assert np.array_equal(qv[i],qsources[q['id']][1]);qreused.append({'id':q['id'],'source':qsources[q['id']][2]})
   assert len(reused)==im['reused_document_vectors']==299 and len(qreused)==im['reused_query_vectors']==4
   runtime=read(hist/'runtime.json');download_manifest=read(base/'model-manifest.json');assert filehash(base/'model-manifest.json')==im['model_manifest_sha256'];tokenizer=[]
   for f in download_manifest:
    if any(t in f['file'] for t in ['tokenizer','special_tokens','vocab','merges']):
     target=base/'model'/f['file']
     if target.is_file():actual=filehash(target);tokenizer.append({'file':f['file'],'revision':f['revision'],'sha256':actual,'matches_download_manifest':actual==f['sha256']})
   expected_limit=512 if role.startswith('eg2') else 1024;assert runtime['dtype']=='float32' and im['dtype']=='FP32' and runtime['max_tokens']==im['max_tokens']==expected_limit and runtime['attention']==im['attention']=='eager'
   oldcode=base/('eg2_eval.py' if role.startswith('eg2') else 'jina_eval.py');copy_evidence(oldcode,OUT/role/oldcode.name);copy_evidence(hist/'runtime.json',OUT/role/'historical-runtime.json');copy_evidence(hist/'calls.jsonl',OUT/role/'historical-calls.jsonl');copy_evidence(base/'model-manifest.json',OUT/role/'model-download-manifest.json')
   hashes_record=hist/'final-audit.json';earlier_hashes=None
   if hashes_record.exists():
    earlier_hashes=read(hashes_record).get('vectors',{});assert all(earlier_hashes.get(name,{}).get('sha256')==info['sha256'] for name,info in [('skillret-docs.npy',oi),('skillret-queries-0.npy',oqi)]);copy_evidence(hashes_record,OUT/role/'historical-final-audit.json')
   detail.update(reused_document_rows_bitwise_equal=len(reused),reused_query_rows_bitwise_equal=len(qreused),reused_document_mapping=reused,reused_query_mapping=qreused,source_fixture_sha256=filehash(base.parent/'embeddinggemma2-pilot/fixtures/skillret-pilot.json'),historical_array_hash_receipt_verified=earlier_hashes is not None,current_download_revision=sorted({f['revision'] for f in download_manifest}),model_manifest_matches_scaling_record=True,tokenizer_files=tokenizer,declared_compatibility={'dtype':'float32','attention':'eager','max_tokens':expected_limit,'normalization':'L2; EG2 normalize_embeddings=True; Jina reviewed model.encode uses F.normalize','query_prefix':'task: search result | query: ' if role.startswith('eg2') else 'Query: ','document_prefix':'title: none | text: ' if role.startswith('eg2') else 'Document: ','historical_runtime':runtime['packages'],'scaling_runtime_evidence':'same EG2 overlay/current root environment and explicit code; fresh per-run package snapshot absent for indexes'},provenance_gap='Historical per-run runtime does not record model revision, tokenizer hashes or immutable worker-source hash. Current downloaded-file manifests and source are compatible, but cannot retrospectively prove exact historical loaded bytes/software. No new inference performed to resolve this.')
   receipt['gaps'].append(role+': '+detail['provenance_gap']);release(ov,oqv,pdv,pqv)
  # Recompute at most4 rankings/model:2 fixed queries x sizes50/6006.
  for c in replay_cases:
   library=sorted(c['skill_ids']);v=qv[im['query_ids'].index(c['query_id'])];scores=np.empty(len(library),dtype=np.float32)
   for lo in range(0,len(library),64):
    positions=[idpos[d] for d in library[lo:lo+64]];block=np.asarray(dv[positions]);scores[lo:lo+len(positions)]=block@v;release(dv)
   ranking=[library[i] for i in sorted(range(len(library)),key=lambda i:(-float(scores[i]),library[i]))][:100];expected=saved[(role,c['case_id'])];all_replays.append({'role':role,'case_id':c['case_id'],'rank_count':len(ranking),'exact_top100_match':ranking==expected,'replayed_ranking_sha256':objhash(ranking),'saved_ranking_sha256':objhash(expected)})
  receipt['models'][role]=detail;release(dv,qv);save()
 receipt['score_to_ranking_replays']=all_replays;receipt['all_replays_match']=all(r['exact_top100_match'] for r in all_replays)
 # Exact two-sided paired sign-flip test, fixed20-query cohort; zeros stay zero.
 result=read(HERE/'retrieval-full/results.json');b=result['metrics']['bm25/6006']['per_query'];h=result['metrics']['hybrid_rrf60/6006']['per_query'];diff=[h[q['id']]['Recall@5']-b[q['id']]['Recall@5'] for q in selected];nonzero=[x for x in diff if abs(x)>1e-12];observed=abs(sum(nonzero));extreme=sum(abs(sum(s*x for s,x in zip(signs,nonzero)))>=observed-1e-12 for signs in itertools.product([-1,1],repeat=len(nonzero)));receipt['paired_signflip']={'metric':'Recall@5','library':6006,'n':20,'nonzero':len(nonzero),'positive':sum(x>0 for x in nonzero),'zeros':20-len(nonzero),'two_sided_p':extreme/(2**len(nonzero)),'interpretation':'p=.125; bootstrap95% interval[.05,.35] remains descriptive. No general winner or independent confirmation claim.'}
 source=Path('/home/user/code/routing-pilot/source');license_text=(source/'LICENSE').read_text();assert 'Apache License' in license_text and 'Version 2.0' in license_text;receipt['jev_source']=[copy_evidence(source/'LICENSE',OUT/'jevk5-source/LICENSE')]+[copy_evidence(source/'jevk5'/name,OUT/'jevk5-source/jevk5'/name) for name in ['__init__.py','gguf.py','prompt.py']];receipt['jev_output_limit']='Saved JevK5 outputs are processed scores/probabilities, not original pre-normalization raw logits/top-logprob responses. Missing raw responses are not reconstructed.'
 receipt.update(status='audited_preliminary',independently_complete_provenance=False,wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,source_arrays_modified=False,new_inference_calls=0,new_downloads=0);save();signal.alarm(0);print(json.dumps({'status':receipt['status'],'arrays':len(receipt['arrays']),'replay_count':len(all_replays),'all_replays_match':receipt['all_replays_match'],'new_inference_calls':0,'wall_seconds':receipt['wall_seconds'],'peak_rss_bytes':receipt['peak_rss_bytes'],'paired_signflip':receipt['paired_signflip'],'gaps':receipt['gaps']}))
# Invoked by run_audit.py after guard setup.
