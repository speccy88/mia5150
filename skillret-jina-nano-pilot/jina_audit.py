import pathlib,sys,json,numpy as np,statistics,hashlib,shutil,datetime
r=pathlib.Path.home()/'code/jina-nano-pilot';d=r/sys.argv[1];f=r/'fixtures'
def read(p):return json.loads(p.read_text())
def save(n,x):(d/(n+'.json')).write_text(json.dumps(x,indent=2))
def metrics(labels,rankings):
 rows={}
 for q,g in labels.items():
  gold={k for k,v in g.items() if v>0};rank=rankings[q];out={}
  for k in [1,5,10,20]:out[f'Recall@{k}']=len(set(rank[:k])&gold)/len(gold) if gold else 0.;out[f'Completeness@{k}']=float(bool(gold) and gold.issubset(rank[:k]))
  out['MRR']=next((1/(i+1) for i,x in enumerate(rank) if x in gold),0.);rows[q]=out
 return {'aggregate':{k:statistics.mean(x[k] for x in rows.values()) for k in next(iter(rows.values()))},'per_query':rows}
def rank(qv,dv,queries,ids):
 scores=qv@dv.T;return {q:[ids[j] for j in np.argsort(-scores[i],kind='stable')] for i,q in enumerate(queries)}
small=read(f/'qwen-small-fixtures.json');dv=np.load(d/'small-docs.npy');labels={str(i):{j:1} for i,j in enumerate(small['relevant_document_indices'])};smallmetrics=[];smallrank=[]
for repeat in range(3):
 rv=rank(np.load(d/f'small-queries-{repeat}.npy'),dv,list(labels),list(range(len(dv))));smallrank.append(rv);smallmetrics.append(metrics(labels,rv))
old=pathlib.Path.home()/'code/category-pilot/embedding/results'
q=np.array([read(old/f'query-{i:02}.json')['response']['data'][0]['embedding'] for i in range(20)]);v=np.array([read(old/f'document-{i:02}.json')['response']['data'][0]['embedding'] for i in range(28)]);q/=np.linalg.norm(q,axis=1,keepdims=True);v/=np.linalg.norm(v,axis=1,keepdims=True);control=metrics(labels,rank(q,v,list(labels),list(range(28))));assert abs(control['aggregate']['Recall@1']-.9)<1e-9 and abs(control['aggregate']['MRR']-.95)<1e-9
pilot=read(f/'skillret-pilot.json');ids=[x['id'] for x in pilot['documents']];queries=[x['id'] for x in pilot['queries']];pd=np.load(d/'skillret-docs.npy');pr=rank(np.load(d/'skillret-queries-0.npy'),pd,queries,ids);pr2=rank(np.load(d/'skillret-queries-1.npy'),pd,queries,ids);allrank=pilot['controls']|{'jina_nano':pr}
rrf={}
for q in queries:
 score={}
 for channel in [pr[q],allrank['qwen_embed_06'][q]]:
  for i,x in enumerate(channel):score[x]=score.get(x,0)+1/(60+i+1)
 rrf[q]=sorted(score,key=lambda x:(-score[x],x))
allrank['qwen_plus_jina_rrf']=rrf;pm={n:metrics(pilot['qrels'],rv) for n,rv in allrank.items()};ceiling={q:len(set(g)&set(ids))/len(g) for q,g in pilot['qrels'].items()};comp={}
for k in [5,10]:
 gained=lost=0;win=loss=tie=0;union={}
 for q in queries:
  gold=set(pilot['qrels'][q]);a=set(allrank['qwen_embed_06'][q][:k])&gold;b=set(pr[q][:k])&gold;gained+=len(b-a);lost+=len(a-b);win+=len(b)>len(a);loss+=len(b)<len(a);tie+=len(a)==len(b);union[q]=list(dict.fromkeys(allrank['qwen_embed_06'][q][:k]+pr[q][:k]))
 comp[str(k)]={'new_relevant_query_document_pairs_vs_qwen':gained,'lost_relevant_pairs_vs_qwen':lost,'queries_better':win,'queries_worse':loss,'queries_equal':tie,'union_recall_mean':statistics.mean(len(set(union[q])&set(pilot['qrels'][q]))/len(pilot['qrels'][q]) for q in queries),'union_max_budget':2*k,'qwen_same_max_budget_recall':pm['qwen_embed_06']['aggregate'][f'Recall@{2*k}']}
calls=[json.loads(x) for x in (d/'calls.jsonl').read_text().splitlines()];stats={}
for prefix in ['smoke','small/document','small/query','skillret/document','skillret/query']:
 rows=[x for x in calls if x['id'].startswith(prefix+'/')];stats[prefix]={'calls':len(rows),'wall_sum':sum(x['wall_seconds'] for x in rows),'cpu_sum':sum(x['cpu_seconds'] for x in rows),'wall_median':statistics.median(x['wall_seconds'] for x in rows),'wall_p95':float(np.percentile([x['wall_seconds'] for x in rows],95)),'wall_min':min(x['wall_seconds'] for x in rows),'wall_max':max(x['wall_seconds'] for x in rows),'max_raw_tokens':max(x['raw_tokens'] for x in rows),'truncated_calls':sum(x['truncated'] for x in rows)}
tele=[json.loads(x) for x in (d/'telemetry.jsonl').read_text().splitlines()];active=[x for x in tele if x.get('process')];resources={'peak_vram_mib':max(x.get('gpu',{}).get('memory_mib',0) for x in tele),'max_gpu_utilization_percent':max(x.get('gpu',{}).get('utilization_percent',0) for x in tele),'max_gpu_temperature_c':max(x.get('gpu',{}).get('temperature_c',0) for x in tele),'min_host_available_gib':min(x['host']['available_bytes'] for x in tele)/2**30,'min_guest_available_gib':min(x['wsl_memory']['MemAvailable'] for x in tele)/2**30,'peak_process_rss_gib':max(x['process']['rss_bytes'] for x in active)/2**30,'max_host_cpu_percent':max(x['host']['cpu_percent'] for x in tele),'swap_pages':sum(sum(x['swap_delta_pages'].values()) for x in tele),'sample_count':len(tele),'process_sample_count':len(active),'guard_stops':[p.name for p in d.glob('guard*.json')]}
summary={'small':{'jina_nano':smallmetrics[0]['aggregate'],'qwen_historical_audited':control['aggregate'],'rankings_identical_across_repeats':smallrank[0]==smallrank[1]==smallrank[2]},'skillret':{'queries':len(queries),'documents':len(ids),'original_corpus':pilot['original_corpus_count'],'metrics':{n:x['aggregate'] for n,x in pm.items()},'macro_recall_ceiling':statistics.mean(ceiling.values()),'gold_pairs_in_pool':sum(len(set(g)&set(ids)) for g in pilot['qrels'].values()),'total_gold_pairs':sum(len(g) for g in pilot['qrels'].values()),'rankings_identical_across_repeats':pr==pr2,'complementarity':comp},'timing':stats,'resources':resources,'first_call':calls[0],'load':read(d/'load.json'),'import':read(d/'import.json'),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
save('summary',summary);save('skillret-rankings',allrank);save('skillret-metrics',pm);save('small-metrics',smallmetrics);save('small-rankings',smallrank);save('resources-summary',resources)
out=pathlib.Path('/mnt/c/Users/user/Documents/Codex/2026-10-05/task/jina-nano-results');out.mkdir(exist_ok=True)
for p in r.iterdir():
 if p.is_file():shutil.copy2(p,out/p.name)
 elif p.name not in ['model','overlay','wheels','__pycache__']:shutil.copytree(p,out/p.name,dirs_exist_ok=True)
print(json.dumps(summary,indent=2))

