import pathlib,json,numpy as np
r=pathlib.Path.home()/'code/mia5150-local'
for p in sorted((r/'cache').glob('qwen_embed_06-header-0-*.npy')):print(p.name,np.load(p,mmap_mode='r').shape)
for p in sorted((r/'results').glob('*qwen*')):
 if p.stat().st_size>50*1024**2:continue
 d=json.loads(p.read_text());print(p.name,list(d)[:10],len(d.get('rankings',{})))
print('ROWS',[(sub,sum(1 for _ in (r/f'data/{sub}/test.jsonl').open())) for sub in ['queries','skills','qrels']])
