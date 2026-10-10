import pathlib,json,numpy as np,hashlib,statistics,shutil
r=pathlib.Path.home()/'code/embeddinggemma2-pilot';d=r/'evaluation-compatible';m=pathlib.Path.home()/'code/mia5150-local';read=lambda p:json.loads(p.read_text());p=read(r/'fixtures/skillret-pilot.json');checks={}
checks['qwen_pool_top5_matches_original_all32']=all(p['controls']['qwen_embed_06'][q['id']][:5]==p['qwen_original_top100'][q['id']][:5] for q in p['queries']);assert checks['qwen_pool_top5_matches_original_all32']
checks['vectors']={}
for f in sorted(d.glob('*.npy')):
 a=np.load(f);assert np.isfinite(a).all();err=float(np.max(np.abs(np.linalg.norm(a,axis=-1)-1)));assert err<1e-4;checks['vectors'][f.name]={'shape':list(a.shape),'max_unit_norm_error':err,'sha256':hashlib.file_digest(open(f,'rb'),'sha256').hexdigest()}
checks['small_repeat_max_absolute_difference']=max(float(np.max(np.abs(np.load(d/'small-queries-0.npy')-np.load(d/f'small-queries-{i}.npy')))) for i in [1,2]);checks['skillret_repeat_max_absolute_difference']=float(np.max(np.abs(np.load(d/'skillret-queries-0.npy')-np.load(d/'skillret-queries-1.npy'))))
calls=[json.loads(x) for x in (d/'calls.jsonl').read_text().splitlines()];assert len(calls)==431;checks['calls']=len(calls);checks['truncated']=[x|{'query_id':p['queries'][int(x['id'].split('/')[-1])]['id']} for x in calls if x['truncated']]
old=read(m/'results/qwen_embed_06-header-0.json');scorer=read(m/'manifests/scorer-queries.json');mapping={q['id']:i for i,q in enumerate(scorer)};counts=old['query_stats']['raw_token_counts'];checks['cached_qwen_selected_query_max_tokens']=max(counts[mapping[q['id']]] for q in p['queries']);checks['cached_qwen_selected_truncated_queries']=sum(counts[mapping[q['id']]]>1024 for q in p['queries'])
checks['runtime_wheels']=[x for x in read(r/'dependency-manifest.json') if x['name'] in ['transformers','sentence-transformers','torchvision','huggingface-hub','tokenizers']]
checks['inference_wall_sum']=sum(x['wall_seconds'] for x in calls);checks['inference_cpu_sum']=sum(x['cpu_seconds'] for x in calls)
(d/'final-audit.json').write_text(json.dumps(checks,indent=2));shutil.copy2(d/'final-audit.json',pathlib.Path('/mnt/c/Users/user/Documents/Codex/2026-10-05/task/embeddinggemma2-results/evaluation-compatible/final-audit.json'));print(json.dumps({k:v for k,v in checks.items() if k!='vectors'},indent=2))
