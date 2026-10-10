"""Offline fixed-denominator scoring of the stopped OpenRouter probe."""
import sys,json,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness import qrels,evaluate
from probe_contract import digest
from jev_probe import PIN
def main():
 here=ROOT/'round2';out=here/'openrouter-listwise-12-v1'
 r=json.loads((out/'results.json').read_text());p=json.loads((here/'exploration-freeze.json').read_text());assert p.pop('protocol_sha256')==PIN and digest(p)==PIN
 packets={x['id']:x for x in json.loads((here/'prepared-openrouter-requests.json').read_text())};ids=[q['id'] for q in p['queries']]
 assert [x['id'] for x in r['rows']]==ids and r['protocol_sha256']==PIN
 ranks={};ok=[]
 for row in r['rows']:
  q=row['id'];assert row['candidate_ids']==p['candidate_ids'][q]
  if row['status']=='ok':
   assert set(row['ranking'])==set(row['candidate_ids']) and len(row['ranking'])==20
   assert row['request_sha256']==digest(packets[q]['request']);ok.append(q)
  else:assert row['ranking']==[]
  ranks[q]=row['ranking']
 labels=qrels('test');labels={q:labels[q] for q in ids};base=evaluate(labels,p['candidate_ids']);scored=evaluate(labels,ranks)
 completed={'baseline':evaluate({q:labels[q] for q in ok},p['candidate_ids'])['aggregate'],'rerank':evaluate({q:labels[q] for q in ok},ranks)['aggregate']} if ok else {}
 result={'audit':'Frozen12 IDs/pools/request hashes verified; four valid permutations, one failed attempt, seven unattempted; no retries.','baseline':base,'rerank_fixed_denominator':scored,'successful_subset_descriptive_only':completed,'attempts':r['attempts'],'known_reported_cost_usd':r['known_reported_cost_usd'],'reserved_usd':r['reserved_usd'],'median_success_latency_seconds':statistics.median(x['seconds'] for x in r['rows'] if x['status']=='ok'),'results_sha256':hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),'caveat':'Stopped run, not a completed model-quality comparison. Eight empty outputs retain denominator12. Successful subset is selection-biased and only descriptive. HTTP status was not retained by original safe logger; no error body retained or retry performed.'}
 (out/'evaluation.json').write_text(json.dumps(result,indent=2))
 summary={k:v for k,v in result.items() if k not in ('baseline','rerank_fixed_denominator','successful_subset_descriptive_only')}
 summary.update(baseline_ndcg10=base['aggregate']['NDCG@10'],fixed_denominator_ndcg10=scored['aggregate']['NDCG@10'],success_subset_ndcg10={k:v['NDCG@10'] for k,v in completed.items()})
 (out/'REPORT.md').write_text('# OpenRouter stopped exploratory probe\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
