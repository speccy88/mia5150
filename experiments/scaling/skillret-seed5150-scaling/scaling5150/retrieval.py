"""Common-view, shared-library exact retrieval; no qrels enter scorers."""
import argparse,json,sys,time,re,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150';sys.path.insert(0,str(ROOT))
from harness import evaluate,set_metrics,rrf,mmr
from structures import dsr
from protocol import sha,write_new

def clauses(query):
 return list(dict.fromkeys(x.strip() for x in re.split(r'[;\n]+|(?<=[.!?])\s+',query) if len(x.strip())>=20))[:3]

def sorted_scores(ids,scores):return [ids[i] for i in sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))]
def roundrobin(a,b,k):
 out=[]
 for i in range(max(len(a),len(b))):
  for r in [a,b]:
   if i<len(r) and r[i] not in out:out.append(r[i])
   if len(out)==k:return out
 return out
def bm25(query,ids,texts):
 import bm25s
 start=time.perf_counter();corpus=bm25s.tokenize([texts[d] for d in ids],stopwords='en',show_progress=False);index=bm25s.BM25();index.index(corpus,show_progress=False);index_time=time.perf_counter()-start
 nbytes=sum(v.nbytes for v in index.scores.values() if hasattr(v,'nbytes'));start=time.perf_counter();cpu=time.process_time();qt=bm25s.tokenize(query,stopwords='en',show_progress=False)
 if not qt.vocab:qt=bm25s.tokenize('UNMATCHED_SENTINEL',stopwords=[],show_progress=False)
 names,scores=index.retrieve(qt,corpus=ids,k=len(ids),n_threads=1,show_progress=False);ranking=[str(d) for d,v in sorted(zip(names[0],scores[0]),key=lambda x:(-float(x[1]),str(x[0])))];elapsed=time.perf_counter()-start
 return ranking,{'index_seconds':index_time,'index_bytes':nbytes,'search_seconds':elapsed,'cpu_seconds':time.process_time()-cpu,'query_embedding_seconds':0,'index_plus_query_seconds':index_time+elapsed,'amortized_1000_queries_seconds':index_time/1000+elapsed}
def field_tfidf(query,ids,texts):
 import scipy.sparse as sp
 def counts(t):return collections.Counter(re.findall(r'(?u)\b\w\w+\b',t.lower()))
 def matrix(cs,vocab,idf):
  row=[];col=[];val=[]
  for i,c in enumerate(cs):
   for w,n in c.items():
    if w in vocab:row.append(i);col.append(vocab[w]);val.append(n)
  x=sp.csr_matrix((np.array(val,dtype=np.float32),(row,col)),shape=(len(cs),len(vocab)))
  x=x.multiply(idf).tocsr();norm=np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel());return sp.diags(1/np.maximum(norm,1e-12))@x
 start=time.perf_counter();arrays=[];score=np.zeros(len(ids),dtype=np.float64)
 for part in [0,1]:
  docs=[texts[d].split(' | ',1)[part] if ' | ' in texts[d] else texts[d] for d in ids];cs=[counts(d) for d in docs];vocab={w:i for i,w in enumerate(sorted({w for c in cs for w in c}))};df=collections.Counter(w for c in cs for w in c);idf=np.array([np.log((len(ids)+1)/(df[w]+1))+1 for w in vocab]);x=matrix(cs,vocab,idf);arrays.append((vocab,idf,x))
 index_time=time.perf_counter()-start;start=time.perf_counter()
 for vocab,idf,x in arrays:score+=(matrix([counts(query)],vocab,idf)@x.T).toarray()[0]/2
 return sorted_scores(ids,score),{'index_seconds':index_time,'search_seconds':time.perf_counter()-start,'index_bytes':sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for _,_,x in arrays)}

def main():
 a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');args=a.parse_args();m=json.loads((HERE/'manifest.json').read_text());texts=json.loads((HERE/'documents.json').read_text());assert sha(texts)==m['text_sha256'];major=json.loads((HERE/'major-map.json').read_text());queries={q['id']:q['query'] for q in m['queries']}
 cases=[c for c in m['cases'] if not args.pilot or (c['query_id'] in m['pilot_query_ids'] and c['size'] in m['pilot_sizes'])]
 tag='retrieval-pilot-v2' if args.pilot else 'retrieval-full';out=HERE/tag;out.mkdir(exist_ok=False);indexes={}
 if not args.pilot:write_new(HERE/'clause-amendment.json',{'version':'scaling5150-clause-v1','query_only':True,'rule':'First3 distinct clauses of at least20characters split on semicolon/newline or sentence punctuation; whole-query BM25 plus each clause BM25 top100, equal RRF60. No query rewriting, qrels or tuning.','timing':'Current simple implementation rebuilds the same lexical index for each clause; all rebuild/search time included, no optimized serving claim.','clauses':{q:clauses(t) for q,t in queries.items()}})
 for p in sorted(HERE.glob('index-*')):
  if not (p/'manifest.json').exists():continue
  info=json.loads((p/'manifest.json').read_text())
  if info.get('view_sha256')!=m['text_sha256']:continue
  docs=np.load(p/'docs.npy',mmap_mode='r');qv=np.load(p/'queries.npy',mmap_mode='r');indexes[info['role']]={'docs':docs,'query':qv,'ids':{d:i for i,d in enumerate(info['document_ids'])},'qids':{q:i for i,q in enumerate(info['query_ids'])},'info':info}
 assert 'skillret_embed_06' in indexes
 rows=[]
 for case in cases:
  qid=case['query_id'];query=queries[qid];ids=sorted(case['skill_ids']);systems={};costs={};base_cpu=time.process_time()
  systems['bm25'],costs['bm25']=bm25(query,ids,texts)
  if not args.pilot:
   clause_start=time.perf_counter();channels=[systems['bm25'][:100]];clause_indexes=costs['bm25']['index_seconds']
   for part in clauses(query):
    if part==query:continue
    cr,cc=bm25(part,ids,texts);channels.append(cr[:100]);clause_indexes+=cc['index_seconds']
   systems['query_clause_bm25']=rrf(channels);costs['query_clause_bm25']={'total_seconds':costs['bm25']['index_plus_query_seconds']+time.perf_counter()-clause_start,'index_seconds':clause_indexes,'channel_count':len(channels),'note':'Includes repeated per-clause index construction; no hidden preprocessing excluded.'}
  systems['field_tfidf'],costs['field_tfidf']=field_tfidf(query,ids,texts)
  for name,x in indexes.items():
   start=time.perf_counter();v=x['query'][x['qids'][qid]];positions=[x['ids'][d] for d in ids];docs=x['docs'][positions];scores=docs@v;systems[name]=sorted_scores(ids,scores);elapsed=time.perf_counter()-start
   info=x['info'];qtime=info.get('historical_query_seconds',info.get('query_seconds',[0]*len(info['query_ids'])))[x['qids'][qid]]
   costs[name]={'search_seconds':elapsed,'query_embedding_seconds':qtime,'query_embedding_timing_source':'historical cached forward' if 'historical_query_seconds' in info else 'fresh index-build query forward','index_bytes':int(docs.nbytes),'full_corpus_index_forward_seconds':info.get('old_index_forward_seconds',info.get('document_forward_seconds')),'index_note':'One shared corpus embedding index; subset gather/search measured now. Cached query vectors do not count as fresh serving inference.'}
   if name=='skillret_embed_06':
    dense=systems[name];docmap={d:docs[i] for i,d in enumerate(ids)};rel={d:float(scores[i]) for i,d in enumerate(ids)};pool=dense[:min(20,len(ids))]
    start=time.perf_counter();groups=sorted(set(major[d] for d in ids));winner=max(groups,key=lambda g:(np.mean([rel[d] for d in ids if major[d]==g]),g));systems['domain_centroid']=[d for d in dense if major[d]==winner];costs['domain_centroid']={'search_seconds':elapsed+time.perf_counter()-start,'chosen_major':winner,'routing':'embedding centroid, not LM domain routing'}
    start=time.perf_counter();systems['mmr5']=mmr(pool,rel,docmap,.8,5,rel[pool[0]]-.2);costs['mmr5']={'search_seconds':elapsed+time.perf_counter()-start}
    start=time.perf_counter();systems['dsr5']=dsr(pool,1/(1+np.exp(-np.array([rel[d] for d in pool]))),np.stack([docmap[d] for d in pool]),v,5,.9);costs['dsr5']={'search_seconds':elapsed+time.perf_counter()-start}
  dense=systems['skillret_embed_06'];lex=systems['bm25'];start=time.perf_counter();systems['hybrid_rrf60']=rrf([dense[:100],lex[:100]]);costs['hybrid_rrf60']={'search_seconds':costs['bm25']['search_seconds']+costs['skillret_embed_06']['search_seconds']+time.perf_counter()-start}
  systems['dense20']=dense[:20];systems['dense30']=dense[:30];systems['roundrobin20']=roundrobin(lex[:100],dense[:100],min(20,len(ids)));systems['dense20_novel10']=dense[:20]+[d for d in lex[:100] if d not in dense[:20]][:10]
  for name,ranking in systems.items():
   assert len(ranking)==len(set(ranking)) and set(ranking)<=set(ids)
   rows.append({'case_id':case['case_id'],'query_id':qid,'size':case['size'],'method':name,'status':'ok','ranking':ranking[:100],'effective_candidate_count':min(100,len(ranking)),'timing':costs.get(name,{'timing_note':'Candidate extraction from cached parent ranking; not independent serving timing.'})})
  (out/'progress.json').write_text(json.dumps({'rows':rows}));print(case['case_id'],len(systems),'methods',flush=True)
 # Evaluator boundary: all predictions are persisted before labels are loaded.
 labels=json.loads((HERE/'evaluator-qrels.json').read_text());metrics={}
 for method in sorted({x['method'] for x in rows}):
  for size in sorted({x['size'] for x in rows}):
   subset=[x for x in rows if x['method']==method and x['size']==size];qs=[c['query_id'] for c in cases if c['size']==size];ranks={x['query_id']:x['ranking'] for x in subset};lab={q:labels[q] for q in qs};ev=evaluate(lab,ranks)
   oracle={q:sorted(ranks.get(q,[])[:20],key=lambda d:(-labels[q].get(d,0),d)) for q in qs};ev['candidate20_oracle']=evaluate(lab,oracle)['aggregate'];ev['fixed_top5_set_metrics']={q:set_metrics(ranks.get(q,[])[:5],labels[q]) for q in qs};ev['planned_queries']=len(qs);ev['attempted_queries']=len(subset);ev['valid_queries']=sum(x['status']=='ok' for x in subset)
   if method in ['dense20','dense30','roundrobin20','dense20_novel10']:
    ev['whole_candidate_pool_oracle']=evaluate(lab,{q:sorted(ranks.get(q,[]),key=lambda d:(-labels[q].get(d,0),d)) for q in qs})['aggregate'];ev['whole_candidate_pool_recall']=sum(len(set(ranks.get(q,[]))&set(labels[q]))/len(labels[q]) for q in qs)/len(qs)
   metrics[f'{method}/{size}']=ev
 result={'rows':rows,'metrics':metrics,'manifest_sha256':sha(m),'indexes_used':list(indexes),'case_count':len(cases),'note':'Gold-conditioned shared libraries; rankings and fixed-top5 set diagnostic are not learned/final LLM selection. Query-embedding timing is cached historical/fresh-index evidence, not timed on every retrieval replay.'};(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'cases':len(cases),'rows':len(rows),'methods':len({r['method'] for r in rows})}))
if __name__=='__main__':main()
