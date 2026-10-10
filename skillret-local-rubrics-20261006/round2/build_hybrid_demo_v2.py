"""Append completed BM25/rubric evidence to the preserved cached class demo."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
from varied_prompts import VARIANTS
def main():
 data=json.loads((HERE/'hybrid-demo-data.json').read_text());labels=qrels('test');plan=json.loads((HERE/'varied-screen-plan.json').read_text());run=json.loads((HERE/'varied-screen-v3/results.json').read_text());queries={x['id']:x['query'] for x in plan['dev_queries']}
 for panel in data['panels']:panel.update(baseline_label='SkillRet dense',rerank_label='Hosted Jev Score')
 for v,depth in [(v,20) for v in VARIANTS]+[('original',10)]:
  rows=[]
  for source in run['rows']:
   if source['depth']!=depth or v not in source['variants']:continue
   q=source['id'];value=source['outputs'][v];rows.append({'id':q,'query':queries[q],'baseline':source['pool'],'reranked':value['ranking'],'fused':value['fusion'],'status':source['status'],'gold':[s for s,n in labels[q].items() if n>0]})
  data['panels'].append({'title':f'BM25 top{depth}: {v}','baseline_label':f'BM25 top{depth}','rerank_label':f'Hosted Jev: {v}','note':'Six-query exposed development screen. Original top20 reaches Recall5 pool ceiling; no new rubric beat original Recall5. No variant qualified for confirmation. Depth10 is a separate pool-size factor.','rows':rows})
 needed={d for panel in data['panels'] for row in panel['rows'] for d in row['baseline']}
 with (ROOT/'data/skills/test.jsonl').open() as f:
  for line in f:
   s=json.loads(line)
   if s['id'] in needed:data['skills'][s['id']]={k:s[k] for k in ('id','name','description')}
 for panel in data['panels']:
  for row in panel['rows']:
   q=row['id'];row['metrics']={k:evaluate({q:labels[q]},{q:ranking})['aggregate'] for k,ranking in [('baseline',row['baseline']),('reranked',row['reranked']),('fusion',row['fused'])]}
   row['pool_covered']=len(set(row['gold'])&set(row['baseline']));row['pool_recall']=row['pool_covered']/len(row['gold'])
   row['oracle_recall5']=min(5,row['pool_covered'])/len(row['gold']);row['oracle_recall10']=min(10,row['pool_covered'])/len(row['gold'])
   delta=row['metrics']['reranked']['Recall@5']-row['metrics']['baseline']['Recall@5'];row['outcome']='No rerank' if row['status']!='ok' else 'R5 gain' if delta>1e-12 else 'R5 loss' if delta< -1e-12 else 'R5 tie'
 data['evidence']='Shared commitments $1.87/$4; reservations retained. Hosted screen:24 successful calls,660 finite questions,$0.007060914 metered use; no confirmation calls. This viewer uses cached evidence only, costs nothing to open and performs no network requests. No cash charge is inferred from metered use. Historical panels and the BM25 panel are different cohorts; do not average them together.'
 data['screen_summary']='BM25 screen (six queries): original Recall5 0.5833 vs BM25 0.5278. Original reaches pool ceiling. Action/anchored 0.5278; role Choice/constraint 0.5833. No new rubric improves original Recall5. Secondary NDCG variation is exploratory, not a confirmed winner.'
 text=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
 template=(HERE/'hybrid_demo_v2.html').read_text();(HERE/'hybrid-demo-v2.html').write_text(template.replace('__DATA__',text));(HERE/'hybrid-demo-v2-data.json').write_text(json.dumps(data,indent=2))
 print(json.dumps({'panels':len(data['panels']),'display_cases':sum(len(p['rows']) for p in data['panels']),'new_unique_queries':6,'network_calls':0}))
if __name__=='__main__':main()
