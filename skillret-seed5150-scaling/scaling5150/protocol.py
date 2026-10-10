"""Seed5150 gold-conditioned libraries. Gold is evaluator/construction-only."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150';SEED=5150;SIZES=[10,50,100,250,1000,6006]
def sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def hash_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_new(path,obj):
 text=json.dumps(obj,indent=2,ensure_ascii=False)
 if path.exists():assert json.loads(path.read_text())==obj,'Immutable artifact mismatch'
 else:path.write_text(text)
def library(ids,gold,qid,size,seed=SEED):
 if len(ids)!=len(set(ids)) or not set(gold)<=set(ids) or not len(gold)<=size<=len(ids):raise ValueError('Invalid library contract')
 negatives=sorted(set(ids)-set(gold));rng=random.Random(int.from_bytes(hashlib.sha256(f'{seed}\0{qid}'.encode()).digest(),'big'));rng.shuffle(negatives)
 chosen=sorted(set(gold))+negatives[:size-len(set(gold))]
 return sorted(chosen,key=lambda d:(hashlib.sha256(f'{seed}\0display\0{qid}\0{d}'.encode()).hexdigest(),d))
def render(skill):return (skill['name'].strip()+' | '+skill['description'].strip())[:1024]
def main():
 HERE.mkdir(exist_ok=True);sources={f'{x}/test.jsonl':hash_file(ROOT/f'data/{x}/test.jsonl') for x in ['skills','queries','qrels']}
 skills=[json.loads(x) for x in (ROOT/'data/skills/test.jsonl').read_text().splitlines()];queries=[json.loads(x) for x in (ROOT/'data/queries/test.jsonl').read_text().splitlines()];rels=[json.loads(x) for x in (ROOT/'data/qrels/test.jsonl').read_text().splitlines()]
 ids=[s['id'] for s in skills];qids={q['id'] for q in queries};assert len(ids)==len(set(ids))==6006 and len(queries)==len(qids)==4392 and len(rels)==7187
 labels={};seen=set()
 for r in rels:
  assert r['query_id'] in qids and r['skill_id'] in set(ids) and r['relevance']==1 and (r['query_id'],r['skill_id']) not in seen
  seen.add((r['query_id'],r['skill_id']));labels.setdefault(r['query_id'],{})[r['skill_id']]=1
 assert set(labels)==qids and max(map(len,labels.values()))==3
 cache_queries=json.loads((ROOT/'manifests/scorer-queries.json').read_text());lookup={q['id']:q['query'] for q in queries}
 assert all(set(q)=={'id','query'} and lookup[q['id']]==q['query'] for q in cache_queries)
 panel=sorted(cache_queries,key=lambda q:hashlib.sha256(f'{SEED}\0panel\0{q["id"]}'.encode()).hexdigest())[:20]
 cases=[]
 for q in panel:
  previous=set()
  for n in SIZES:
   lib=library(ids,labels[q['id']],q['id'],n);assert len(lib)==len(set(lib))==n and previous<=set(lib) and set(labels[q['id']])<=set(lib);previous=set(lib)
   cases.append({'case_id':q['id']+f'/random/{n}','query_id':q['id'],'size':n,'condition':'random','skill_ids':lib})
 text={s['id']:render(s) for s in skills};majors={s['id']:s['major'] for s in skills};assert len(set(majors.values()))==6
 manifest={'seed':SEED,'queries':panel,'sizes':SIZES,'cases':cases,'view':'name | description, rendered once and capped at1024characters; no body text. BGE128-token restriction not claimed for non-BGE encoders. Model token limits and truncations separately recorded.','text_sha256':sha(text),'library_sha256':sha(cases),'source_hashes':sources,'generative_query_ids':[q['id'] for q in panel[:2]],'reranker_query_ids':[q['id'] for q in panel[:2]],'reranker_sizes':[50,1000,6006],'pilot_query_ids':[q['id'] for q in panel[:2]],'pilot_sizes':[10,50,250],'condition':'Per-query gold-conditioned nested distractor sets; independent stable hash display order. Public historically exposed development, not full-catalog discovery or untouched confirmation.'}
 plan={'version':'scaling5150-v1','manifest_sha256':sha(manifest),'sources':'Supplied proposal_1.pdf and research plan use10/100/1000/6006; reviewer reports repository041f2ce uses50/250/1000. Union grid preserves both; user5150 supersedes historicalseed42/extra seeds.','query_panel':'20 deterministic SHA256-selected entries from existing128-query development/cache manifest; no selection by outcome or gold count. Expensive matched generative panel is first2 only, reported separately.','method_map':{
 'bm25':'mandatory lexical control, rebuild IDF within each library, ties by skillID',
 'qwen_embed_06/skillret_embed_06':'primary small-vector anchors; existing full-corpus caches reused only for byte-identical rendered text; changed13documents re-encoded before fair comparison',
 'skillrouter_embed_06/skillreason_embed_06':'cached specialist challengers, same view-alignment rule',
 'eg2_text270m_fp32/jina_nano239m_fp32':'new small retrieval extensions, native prompts and existing reviewed local deployments; full6006document index required, restricted270documentpilot not reused as corpus',
 'hybrid_rrf60':'proposal hybrid operationalization: equal rank fusion of BM25 and SkillRet top100; no learned weights',
 'domain_vector_lm':'proposal domain routing: same local selector predicts one of6major categories; vector restricted to it; empty-domain failure, no global fallback',
 'domain_vector_centroid':'separate historical centroid-routing ablation, not substituted for LM routing',
 'field_tfidf':'fixed equal name/description field cosine, local-library IDF; extension, not six-channel MLP reproduction',
 'roundrobin20/dense20_novel10':'recent candidate-pool constructions, separately matched to dense20/dense30 controls; include candidate ceilings',
 'mmr/dsr':'fixed structural selection diagnostics on common SkillRet top20, MMRlambda0.8/DSRblend0.9, budget5; do not label top5 baseline exact final selection',
 'query_clause_bm25':'deterministic clause/RRF ablation retained as planned extension after baseline pilot; no LLM rewriting',
 'full_listing':'all actual library items in one common-selector prompt; overflow explicit zero/failure, no hidden shortlist/chunking',
 'common_selector':'Qwen3.5-4B Q4,0-5opaque local aliases mapped to originalIDs after validation; same prompt for BM25/vector/domain/hybrid top5 and full listing',
 'small_rerank':'SkillRet-Reranker0.6B common SkillRet top20 at50/1000/6006 on2queries; no gold insertion',
 'medium_rerank':'JevK5-4B Q5 on same pools; Qwen8B Q4 is a fit challenger only after cached weight provenance/integrity and guarded load canary; Plumb reserved pending resources',
 'hosted_jev':'pending/notrun: no new paid API calls in this experiment',
 'bge_small/clm':'historical proposal controls not currently located/verified in this executor; not silently renamed Qwen/SkillRet',
 'lora/training/graph_dependencies/activation':'previous separate experiments or unsupported by current annotations; no fresh training or invented dependency labels',
 'hard_overlap':'planned separate condition, not mixed into this random-library primary curve'},
 'tiers':'Small is role-specific239M/270M/0.6B retrieval and0.6B reranking. Medium candidates are4B quantized selector/typed reranker plus8B Q4 native reranker fit challenger. No global-best or weights-only fit claim. Record KV/context, hostRAM, VRAM, startup and failures.',
 'metrics':['candidateRecall20/50/100','Recall5/10/15','NDCG5/10/15','MAP5/10/15 official convention','MRR5','Completeness5/10/15','selected-setprecision/recall/F1/exactmatch/size','planned/attempted/valid denominators','invalidIDs/duplicates/overflow/failures','queryembedding/retrieval/rerank/routing/selection wall and CPU times','coldstartup/index seconds and bytes/amortization','logical input/output/routing tokens versus physical cache calls','guardhost/guestRAM/GPUmemory/temp/swap'],
 'tie_policy':'Retrieval scores descending then original skillID lexical; fusion same; pointwise reranker scores descending then frozen pool order. Display hash independent of labels. Returned selector order preserved for ranking; selected set evaluated separately.',
 'gold_boundary':'All official positive qrels included only in controlled LIBRARY, never forced into retrieved shortlist. Scorers receive query text and shared document text; model prompts use label-independent aliases, no gold flags/counts/IDs lists. Evaluator qrels separate artifact.',
 'scope':'No new downloads, paid calls, parallelGPU, broad tuning or Git push. Pilot first; resource failure halts own workload, settings unchanged; record unattempted arms explicitly. No retries of identical failed inference.'}
 plan['protocol_sha256']=sha(plan)
 for name,value in [('manifest.json',manifest),('protocol.json',plan),('documents.json',text),('major-map.json',majors),('evaluator-qrels.json',{q['id']:labels[q['id']] for q in panel})]:write_new(HERE/name,value)
 print(json.dumps({'protocol_sha256':plan['protocol_sha256'],'query_count':20,'cases':len(cases),'sizes':SIZES,'generative_queries':manifest['generative_query_ids'],'gold_count_distribution':{k:sum(len(labels[q['id']])==k for q in panel) for k in [1,2,3]},'char_truncated_docs':sum(len(s['name'].strip()+' | '+s['description'].strip())>1024 for s in skills),'all_gold_in_libraries':True,'new_inference_calls':0}))
if __name__=='__main__':main()
