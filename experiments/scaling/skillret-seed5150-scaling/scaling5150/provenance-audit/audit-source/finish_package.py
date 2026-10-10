"""Small compatibility checks and streamed archive update; no model execution."""
import json, hashlib, zipfile, shutil, resource, collections, re, time
from pathlib import Path
WORK=Path('/mnt/c/Users/user/Documents/Codex/2026-10-06/task-2')
ROOT=Path('/home/user/code/mia5150-local'); HERE=ROOT/'scaling5150'
STAGE=WORK/'staging'; P=STAGE/'scaling5150/provenance-audit'
resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(90,90))
initial_swap={s.split()[0]:int(s.split()[1]) for s in Path('/proc/vmstat').read_text().splitlines() if s.startswith(('pswpin ','pswpout '))}
last_guard=0
def guard():
 global last_guard
 now=time.time()
 if now-last_guard<.25:return
 h=json.loads((WORK/'host-live.json').read_text(encoding='utf-8-sig'))
 available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
 current_swap={s.split()[0]:int(s.split()[1]) for s in Path('/proc/vmstat').read_text().splitlines() if s.startswith(('pswpin ','pswpout '))}
 if now-h['epoch']>8 or h['available_bytes']<2*1024**3 or available<2*1024**3 or current_swap!=initial_swap:raise RuntimeError('Package paused: memory guard or active swap')
 last_guard=now
guard()
def read(p):return json.loads(p.read_text())
def sha(p):
 guard()
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024**2):guard();h.update(b)
 return h.hexdigest()
def objsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def copy(src,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 with src.open('rb') as fi,dest.open('wb') as fo:shutil.copyfileobj(fi,fo,1024**2)
 return {'path':str(dest.relative_to(STAGE)),'bytes':dest.stat().st_size,'sha256':sha(dest)}
r=read(P/'receipt.json');manifest=read(HERE/'manifest.json')
for role,folder in [('eg2_text270m_fp32','embeddinggemma2-pilot'),('jina_nano239m_fp32','jina-nano-pilot')]:
 base=Path('/home/user/code')/folder
 pi=read(HERE/('embedding-pilot-'+role)/'manifest.json');im=read(HERE/('index-'+role)/'manifest.json')
 fields=['role','view_sha256','model_manifest_sha256','dtype','attention','batch','max_tokens']
 assert all(pi[k]==im[k] for k in fields)
 assert im['view_sha256']==manifest['text_sha256'] and im['batch']==1
 assert len(pi['document_ids'])==len(set(pi['document_ids']))==r['arrays'][role+'/new_pilot_docs']['shape'][0]
 assert len(pi['query_ids'])==len(set(pi['query_ids']))==r['arrays'][role+'/new_pilot_queries']['shape'][0]
 assert all(r['arrays'][role+'/'+suffix]['shape'][1]==768 for suffix in ['new_pilot_docs','new_pilot_queries','output_docs','output_queries','historical_docs','historical_queries'])
 files=read(base/'model-manifest.json');pins={x['file']:x for x in files};checks=[]
 for name in (['config_sentence_transformers.json','config.json'] if role.startswith('eg2') else ['config_sentence_transformers.json','config.json','modeling_jina_embeddings_v5.py','adapters/retrieval/adapter_config.json']):
  path=base/'model'/name
  assert sha(path)==pins[name]['sha256']
  checks.append(copy(path,P/role/'current-config'/name))
 rt=read(P/role/'historical-runtime.json')
 if role.startswith('eg2'):
  prompts=read(base/'model/config_sentence_transformers.json')['prompts']
  assert prompts['SearchQuery']==rt['prompts']['SearchQuery']=='task: search result | query: '
  assert prompts['Document']==rt['prompts']['Document']=='title: none | text: '
  native='SentenceTransformer.encode(normalize_embeddings=True); historical runtime prompts match current pinned prompt config'
 else:
  code=(base/'model/modeling_jina_embeddings_v5.py').read_text()
  assert 'prefix = "Query: " if prompt_name == "query" else "Document: "' in code
  assert 'embeddings = F.normalize(pooled, p=2, dim=-1)' in code
  assert 'self.set_adapter([task])' in code
  assert rt['active_adapters']==['retrieval']
  review={x['file']:x for x in read(base/'review-manifest.json')}
  assert sha(base/'model/modeling_jina_embeddings_v5.py')==review['modeling_jina_embeddings_v5.py']['sha256']
  native='Reviewed native encode uses Query:/Document: prefixes, retrieval adapter and F.normalize(pooled,p=2,dim=-1)'
 detail=r['models'][role]
 detail['pilot_full_config_comparison']={'equal_fields':{k:im[k] for k in fields},'pilot_manifest_sha256':sha(HERE/('embedding-pilot-'+role)/'manifest.json'),'pilot_ordered_document_ids_sha256':objsha(pi['document_ids']),'pilot_ordered_query_ids_sha256':objsha(pi['query_ids']),'array_rows_ids_unique_and_dimension768':True,'pilot_reused_documents':pi['reused_document_vectors'],'pilot_reused_queries':pi['reused_query_vectors'],'current_config_files_match_pinned_download_hashes':checks,'normalization_prefix_source_evidence':native}
 detail['immediate_source_mapping_note']='pilot_array is the immediate source, not necessarily a new producer: pilot itself reused 2 documents and 1 query and encoded 30 documents and 1 query.'
 for key in ['reused_document_mapping','reused_query_mapping']:
  for row in detail[key]:
   if row['source']=='new_pilot':row['source']='pilot_array'
# Independently replay equal-weight RRF60 from existing rank channels, all 120 cases.
rows=read(HERE/'retrieval-full/results.json')['rows'];rankings={(x['case_id'],x['method']):x['ranking'] for x in rows}
rrf=[]
for case in manifest['cases']:
 cid=case['case_id'];scores={}
 for method in ['skillret_embed_06','bm25']:
  for rank,d in enumerate(rankings[cid,method],1):scores[d]=scores.get(d,0)+1/(60+rank)
 output=sorted(scores,key=lambda d:(-scores[d],d))[:100]
 assert output==rankings[cid,'hybrid_rrf60']
 rrf.append({'case_id':cid,'ranking_sha256':objsha(output)})
r['hybrid_rrf60_replay']={'exact_matches':len(rrf),'cases':rrf,'source':'saved SkillRet and BM25 top100; no BM25 reindex or model execution'}
r['config_supplement_script_sha256']=sha(WORK/'finish_package.py')
(P/'receipt.json').write_text(json.dumps(r,indent=2))
copy(WORK/'run_audit.py',P/'audit-source/run_audit.py')
copy(WORK/'run_guarded.ps1',P/'audit-source/run_guarded.ps1')
copy(WORK/'finish_package.py',P/'audit-source/finish_package.py')
copy(WORK/'protocol-source-review.md',P/'audit-source/protocol-source-review.md')
copy(WORK/'host-audit.jsonl',P/'audit-source/host-audit.jsonl')
baseline=Path('/mnt/c/Users/user/Documents/Codex/2026-10-05/task-2/deliverables/MIA5150_Seed5150_Local_Scaling.zip')
assert sha(baseline)=='0cbea2f04adb73eb0d51c7faa2827e84a881fe5d4bffa554d24b1e9e55ac6e3b'
with zipfile.ZipFile(baseline) as z:
 oldhash=read_manifest=json.loads(z.read('CONTENT_SHA256.json'))
 assert len(oldhash)==1656
 report=z.read('scaling5150/report/REPORT.md').decode()
report=report.replace('# MIA5150: seed-5150 local library scaling','# MIA5150: seed-5150 local library scaling — audited preliminary',1)
banner='**Audited preliminary:** stored array integrity, reuse equality and bounded ranking replay are verified. EG2/Jina historical producer provenance remains incomplete: their per-run records do not bind every source vector file to the exact loaded model revision, tokenizer bytes and immutable worker source. No re-encoding was performed.'
report=report.replace('\n\n','\n\n'+banner+'\n\n',1)
report=report.replace('Per-request logical tokens, raw model responses, cache flags and timings are in the raw selection/reranking results.','Per-request logical tokens, selector HTTP responses, cache flags and timings are saved. JevK5 retains processed scores/probabilities, not original completion responses or pre-normalization raw logits/top-logprobs. Missing responses were not reconstructed.')
receipt_sha=sha(P/'receipt.json')
addendum=f'''\n## Read-only evidence audit\n\nA fresh low-priority audit checked the existing local sources and arrays. It made no GPU, inference, training or model-network calls and installed nothing. It used 1 MiB streamed hashes and 64-row vector chunks, one BLAS thread, a 1 GiB address-space cap, and live 2 GiB host/guest RAM reserves with no swap-counter increases during the audit. Peak RSS including gold verification was {r['peak_rss_bytes_including_gold_audit']/2**20:.2f} MiB. No source arrays, frozen protocols, predictions, metrics or failure cases were changed.\n\n- **28 arrays:** SHA-256, shape, dtype, finite checks and norm ranges are recorded. All are finite. Ordered document-ID, uncapped/capped-text, historical 128-query and selected 20-query hashes are explicit.\n- **Four cached 0.6B encoders:** each retains 5,993 document rows and all 20 query rows bit-for-bit. All 13 changed header identities were independently derived from the 1,024-character cap; their source/output vector-row and old/new text hashes are recorded. Source array hashes match existing cache pins.\n- **EG2 and Jina:** each has 299 reused document rows and 4 query rows matching its existing source arrays bit-for-bit. Pilot/full role, text view, model-manifest hash, dtype, attention, batch and maximum-token settings agree. Current tokenizer/config hashes match pinned downloads. Prefix and L2 normalization behavior are checked against available native source/config; EG2's historical runtime also records matching prefixes. The pilot-array mapping names the immediate source; that pilot itself contains 2 reused document rows and 1 reused query row. Available compatibility evidence cannot establish the exact historical loaded model/tokenizer/worker bytes. Jina has no prior historical array-hash receipt; its source arrays are hashed now.\n- **Ranking replay:** 24/24 dense array-to-ranking checks match (six encoders × two fixed queries × sizes 50 and 6006). All 120 equal-weight RRF60 hybrid rankings also match when independently rebuilt from saved BM25 and SkillRet rank channels. These are bounded consistency checks, not fresh model benchmarks or a full independent BM25 reindex.\n- **Pinned gold:** all 20 evaluator gold dictionaries match the local upstream qrels; all 120 seeded, nested libraries and their display order reconstruct exactly; all 2,040 retrieval rows have unique in-library IDs. Recomputed Recall@5 at 6006 is BM25 0.625, SkillRet 0.700 and hybrid 0.800. Completeness is relative to pinned official positives, not exhaustive semantic relevance. Upstream network origin was not revalidated.\n- **Public inputs:** exact test queries (6,053,507 bytes), qrels (661,204 bytes), the 128-query panel (157,802 bytes), and the 6,006-row corpus metadata projection are included. The original 117,606,037-byte corpus was stream-hashed and matched its pin. Full skill bodies and vector arrays are omitted to keep the archive modest; the projection cannot reconstruct the original corpus-file hash, and a fresh vector replay requires the existing local arrays.\n- **JevK5:** existing reviewed adapter files match their earlier review pins and are included with the Apache 2.0 license. The saved field named `raw` contains processed adapter scores/probabilities, not original completion responses or pre-normalization logits.\n\nThe exact two-sided paired sign-flip test for hybrid versus BM25 Recall@5 at 6006 gives **p = 0.125** (4 positive differences, 16 ties). The paired bootstrap 95% interval [0.05, 0.35] remains descriptive; this exposed 20-query panel does not establish a general winner or independent confirmation.\n\nReceipt: `scaling5150/provenance-audit/receipt.json`, SHA-256 `{receipt_sha}`. `independently_complete_provenance=false` remains explicit. The source audit is read-only; this package adds evidence and corrects report provenance language only.\n'''
report+=addendum
rp=STAGE/'scaling5150/report/REPORT.md';rp.parent.mkdir(parents=True,exist_ok=True);rp.write_text(report)
(WORK/'AUDIT_SUMMARY.md').write_text(banner+'\n'+addendum)
dest=WORK/'MIA5150_Seed5150_Local_Scaling.zip';newhash={};preserved=[]
overrides={str(p.relative_to(STAGE)):p for p in STAGE.rglob('*') if p.is_file()}
# No binary model/vector files or unrelated files are selected.
assert all(not p.endswith(('.npy','.gguf','.safetensors','.bin')) for p in overrides)
with zipfile.ZipFile(baseline) as source,zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as target:
 for name,expected in oldhash.items():
  guard()
  h=hashlib.sha256()
  with source.open(name) as f:
   while b:=f.read(1024**2):h.update(b)
  assert h.hexdigest()==expected
  if name in overrides:continue
  with source.open(name) as fi,target.open(name,'w') as fo:shutil.copyfileobj(fi,fo,1024**2)
  newhash[name]=expected;preserved.append(name)
 for name,path in sorted(overrides.items()):
  target.write(path,name);newhash[name]=sha(path)
 target.writestr('CONTENT_SHA256.json',json.dumps(newhash,indent=2))
with zipfile.ZipFile(dest) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(newhash)+1
 for name,expected in newhash.items():
  guard()
  h=hashlib.sha256()
  with z.open(name) as f:
   while b:=f.read(1024**2):h.update(b)
  assert h.hexdigest()==expected
 assert all(newhash[p]==oldhash[p] for p in oldhash if p!='scaling5150/report/REPORT.md')
info={'file':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size,'original_zip_sha256':sha(baseline),'receipt_sha256':receipt_sha,'files_verified':len(newhash),'original_files_unchanged':len(preserved),'original_files_changed':['scaling5150/report/REPORT.md'],'frozen_results_preserved':True,'source_arrays_not_modified':True,'new_inference_calls':0,'library_file_id':'libfile_b85b1eda59908191b682cfcf0c6e966b','expected_current_version':1,'independently_complete_provenance':False}
(WORK/'PACKAGE_RECEIPT.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
