"""Fresh read-only evidence audit. No model/GPU/network imports or calls."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import importlib.util, json, time, resource, hashlib, random, itertools, math, sys
from pathlib import Path
WORK=Path('/mnt/c/Users/user/Documents/Codex/2026-10-06/task-2')
ROOT=Path('/home/user/code/mia5150-local')
HERE=ROOT/'scaling5150'
OUT=WORK/'staging/scaling5150/provenance-audit'
resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(90,90))
def swap():
 return {s.split()[0]:int(s.split()[1]) for s in Path('/proc/vmstat').read_text().splitlines() if s.startswith(('pswpin ','pswpout '))}
initial_swap=swap(); samples=[]; last_check=0
def guard():
 global last_check
 now=time.time()
 if now-last_check<.25:return
 h=json.loads((WORK/'host-live.json').read_text(encoding='utf-8-sig'))
 available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
 s=swap()
 if now-h['epoch']>8 or h['available_bytes']<2*1024**3 or available<2*1024**3 or s!=initial_swap:
  raise RuntimeError('Audit paused: stale host memory, <2GiB host/guest reserve, or active swap')
 samples.append({'epoch':now,'host_available_bytes':h['available_bytes'],'guest_available_bytes':available,'swap':s})
 last_check=now
guard()
spec=importlib.util.spec_from_file_location('bounded_audit',WORK/'staging/scaling5150/bounded_provenance_audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.guard_check=guard
frozen=['protocol.json','manifest.json','documents.json','evaluator-qrels.json','retrieval-full/results.json']
before={p:a.filehash(HERE/p) for p in frozen}
a.main()
r=a.read(OUT/'receipt.json');m=a.read(HERE/'manifest.json');evalq=a.read(HERE/'evaluator-qrels.json')
# Gold validation is independent of harness.py and saved audit assertions.
gold={};qrel_count=0
with (ROOT/'data/qrels/test.jsonl').open() as f:
 for line in f:
  row=json.loads(line);qrel_count+=1
  if row['relevance']>0:gold.setdefault(row['query_id'],{})[row['skill_id']]=row['relevance']
for q in m['queries']:assert evalq[q['id']]==gold[q['id']]
ids=[]
with (OUT/'public-inputs/skills-metadata.jsonl').open() as f:
 for line in f:ids.append(json.loads(line)['id'])
allids=set(ids);previous={};case_map={}
for c in m['cases']:
 guard();qid=c['query_id'];positives=set(gold[qid]);negative=sorted(allids-positives)
 seed=int(hashlib.sha256((str(m['seed'])+'\0'+qid).encode()).hexdigest(),16)
 random.Random(seed).shuffle(negative)
 expected=positives|set(negative[:c['size']-len(positives)])
 actual=c['skill_ids'];assert len(actual)==len(set(actual))==c['size']
 assert set(actual)==expected and positives<=expected and previous.get(qid,set())<=expected
 display=sorted(expected,key=lambda d:(hashlib.sha256(('5150\0display\0'+qid+'\0'+d).encode()).hexdigest(),d))
 assert actual==display
 previous[qid]=expected;case_map[c['case_id']]=c
res=a.read(HERE/'retrieval-full/results.json');headline={};byquery={};row_checks=0
for row in res['rows']:
 c=case_map[row['case_id']];rank=row['ranking'];assert row['query_id']==c['query_id'] and row['size']==c['size']
 assert len(rank)==len(set(rank)) and set(rank)<=set(c['skill_ids'])
 row_checks+=1
 if row['size']==6006:
  qid=row['query_id'];rr=sum(d in gold[qid] for d in rank[:5])/len(gold[qid])
  byquery.setdefault(row['method'],{})[qid]=rr
for method,qs in byquery.items():headline[method]=sum(qs.values())/len(qs)
assert headline['bm25']==.625 and headline['skillret_embed_06']==.7 and headline['hybrid_rrf60']==.8
diff=[byquery['hybrid_rrf60'][q['id']]-byquery['bm25'][q['id']] for q in m['queries']]
nz=[x for x in diff if x!=0];obs=abs(sum(nz))
p=sum(abs(sum(s*x for s,x in zip(signs,nz)))>=obs-1e-12 for signs in itertools.product([-1,1],repeat=len(nz)))/(2**len(nz))
assert p==.125 and len(nz)==4 and all(x>0 for x in nz)
assert before=={p:a.filehash(HERE/p) for p in frozen}
# Verify the included Jev source against pre-existing review pins.
review=a.read(HERE/'model-provenance/jev-adapter-sha256.json')
assert all(a.filehash(OUT/'jevk5-source/jevk5'/name)==sha for name,sha in review.items())
r['all_arrays_finite']=all(x['all_finite'] for x in r['arrays'].values())
r['tokenizer_download_hashes_all_match']=all(f['matches_download_manifest'] for x in r['models'].values() for f in x.get('tokenizer_files',[]))
r['changed_header_rows_all_different']=all(x['row_changed'] for x in r['models'].values() for x in x.get('changed_rows',[]))
assert r['all_arrays_finite'] and r['tokenizer_download_hashes_all_match'] and r['changed_header_rows_all_different']
r['fresh_independent_gold_audit']={'source':'existing local upstream data/qrels/test.jsonl, hash verified to frozen manifest','qrel_rows':qrel_count,'queries_with_qrels':len(gold),'selected_gold_dictionaries_equal':20,'nested_gold_complete_cases':len(case_map),'display_order_matches':True,'retrieval_rows_membership_unique_checked':row_checks,'headline_R5_from_saved_rankings':headline,'two_sided_exact_signflip_p':p,'positive':4,'ties':16,'limits':'Completeness relative to pinned official positives, not exhaustive semantic relevance. Original upstream network origin was not revalidated.'}
r['frozen_inputs_before_after_sha256']=before
r['jev_matches_existing_review_pins']=True
r['guard']={'host_guest_min_available_bytes':2*1024**3,'address_space_cap_bytes':1024**3,'cpu_seconds_cap':90,'nice':os.getpriority(os.PRIO_PROCESS,0),'chunk_bytes':1024**2,'vector_chunk_rows':64,'swap_initial':initial_swap,'samples':samples,'no_model_modules_loaded':not any(x in sys.modules for x in ['torch','transformers','sentence_transformers'])}
r['audit_script_sha256']={p.name:a.filehash(p) for p in [WORK/'run_audit.py',WORK/'staging/scaling5150/bounded_provenance_audit.py']}
r['fresh_audit_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
r['peak_rss_bytes_including_gold_audit']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
(OUT/'receipt.json').write_text(json.dumps(r,indent=2))
print(json.dumps({'fresh':True,'arrays':len(r['arrays']),'replays':len(r['score_to_ranking_replays']),'all_replays_match':r['all_replays_match'],'gold_cases':len(case_map),'rows':row_checks,'signflip_p':p,'peak_rss_bytes':r['peak_rss_bytes_including_gold_audit'],'receipt_sha256':a.filehash(OUT/'receipt.json')}))
