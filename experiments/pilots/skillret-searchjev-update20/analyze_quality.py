import json,pathlib,random,math,statistics
root=pathlib.Path(__file__).resolve().parent
run=max((p.parent for p in (root/'runs').glob('*/result.json') if json.loads(p.read_text()).get('completed_optimizer_updates')==20),key=lambda p:(p/'result.json').stat().st_mtime)
before=json.loads((run/'baseline.json').read_text())['predictions']
after=json.loads((run/'adapted-update20.json').read_text())['predictions']
assert [(r['query_id'],r['skill_id'],r['y']) for r in before]==[(r['query_id'],r['skill_id'],r['y']) for r in after]
qids=list(dict.fromkeys(r['query_id'] for r in before))
groups={q:[] for q in qids}
for a,b in zip(before,after):groups[a['query_id']].append((a,b))
rng=random.Random(271828)
intervals={}
for metric in ['loss','ce','brier','correct']:
 diffs=[]
 for _ in range(4000):
  rows=[pair for q in rng.choices(qids,k=len(qids)) for pair in groups[q]]
  diffs.append(sum(b[metric]-a[metric] for a,b in rows)/len(rows))
 diffs.sort()
 intervals['accuracy' if metric=='correct' else metric]=dict(delta=sum(b[metric]-a[metric] for a,b in zip(before,after))/len(before),query_cluster_bootstrap_percentile_95=[diffs[99],diffs[3899]])
prompt=json.loads((run/'prompt-audit.json').read_text())
r=dict(eval_queries=len(qids),pairs=len(before),resamples=4000,seed=271828,intervals=intervals,before_correct_after_wrong=sum(a['correct']==1 and b['correct']==0 for a,b in zip(before,after)),before_wrong_after_correct=sum(a['correct']==0 and b['correct']==1 for a,b in zip(before,after)),caveat='Small exposed balanced diagnostic. Bootstrap conditions on the chosen16 queries and cannot establish unseen-test or full-corpus retrieval performance.',prompt_truncation={split:dict(prompts=sum(r['split']==split for r in prompt),query_truncated=sum(r['split']==split and r['query_tokens_kept']<r['query_tokens_original'] for r in prompt),skill_header_truncated=sum(r['split']==split and r['skill_tokens_kept']<r['skill_tokens_original'] for r in prompt),max_tokens=max(r['tokens'] for r in prompt if r['split']==split)) for split in ['train','eval']})
(root/'paired-quality-analysis.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
