"""Offline per-query metric, call-accounting and fallback-semantics audit."""
import collections,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2';sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
from varied_prompts import VARIANTS
def main():
 r=json.loads((HERE/'varied-screen-v3/results.json').read_text());p=json.loads((HERE/'prepared-varied-screen.json').read_text());d=json.loads((HERE/'varied-screen-v3/development.json').read_text());labels=qrels('test');rows=[]
 counts=collections.Counter(x['tag'] for x in r['rows']);assert dict(counts)==dict.fromkeys(['original_action','anchored_role','constraint','depth10'],6)
 assert sum(len(x['request']['questions']) for x in p['dev'])==660
 for row in r['rows']:
  q=row['id'];gold={s for s,v in labels[q].items() if v>0};pool=row['pool'];covered=len(gold&set(pool));assert gold
  for variant,value in row['outputs'].items():
   for mode,key in [('direct','ranking'),('fusion','fusion')]:
    rank=value[key];metrics={'Recall@5':len(gold&set(rank[:5]))/len(gold),'Recall@10':len(gold&set(rank[:10]))/len(gold),'Completeness@5':float(gold.issubset(rank[:5])),'Completeness@10':float(gold.issubset(rank[:10]))}
    arm=variant+('_depth10' if row['depth']==10 else '')+'_'+mode
    for k,v in metrics.items():assert abs(d['metrics'][arm]['per_query'][q][k]-v)<1e-12
    baseline5=len(gold&set(pool[:5]))/len(gold);original5=d['metrics']['original_direct']['per_query'][q]['Recall@5']
    rows.append({'query_id':q,'depth':row['depth'],'rubric':variant,'mode':mode,'positive_qrels':len(gold),'pool_positives':covered,'pool_recall':covered/len(gold),'oracle_recall5':min(5,covered)/len(gold),'oracle_recall10':min(10,covered)/len(gold),**metrics,'delta_recall5_vs_BM25':metrics['Recall@5']-baseline5,'delta_recall5_vs_original_top20':metrics['Recall@5']-original5})
 # Explicitly distinguish serving fallback from strict benchmark failure.
 q=r['rows'][0]['id'];gold={q:labels[q]};baseline=r['rows'][0]['pool'];strict=evaluate(gold,{q:[]})['aggregate'];served=evaluate(gold,{q:baseline})['aggregate']
 assert strict['Recall@5']==strict['Recall@10']==strict['NDCG@10']==0
 assert len(rows)==72
 out={'per_query_rows':rows,'accounting':{'development_queries':6,'depth20_rubrics':5,'depth20_questions':600,'depth10_original_questions':60,'total_typed_questions':660,'batches_per_query':4,'calls':24,'successful_calls':24,'failed_calls':0,'confirmation_calls':0,'batch_counts':dict(counts)},'failure_semantics':{'actual_failed_development_queries':0,'strict_benchmark':'An invalid or unattempted rubric/query has an empty ranking and stays in the full query denominator; Recall/NDCG/completeness are zero.','production_fallback':'May serve original BM25 pool after API failure; report this separately as fallback, never as successful Jev ranking.','demo_simulation':'Toggling failure changes only the display, shows strict benchmark zero and served BM25 metrics separately, and never modifies saved results.','example_id':q,'strict_empty_metrics':strict,'served_baseline_metrics':served},'verification':'All72 query/rubric/mode metric rows match direct set-count recall and completeness; 24 requests account for660 typed questions.'}
 (HERE/'VARIED_SCREEN_AUDIT.json').write_text(json.dumps(out,indent=2))
 with (HERE/'VARIED_SCREEN_PER_QUERY.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps({'metric_rows_verified':72,'typed_questions':660,'calls':24,'failures':0,'strict_failure_metric_test':'passed'}))
if __name__=='__main__':main()
