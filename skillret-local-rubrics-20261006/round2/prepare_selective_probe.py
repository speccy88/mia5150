"""Prepare one different policy probe from cached rankings; never reads qrels."""
import json
from pathlib import Path
from probe_contract import digest
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2'
def main():
 cohort=json.loads((HERE/'next-confirmation-8.json').read_text());assert cohort.pop('manifest_sha256')==digest(cohort)
 dense=json.loads((ROOT/'results/public-full-skillret_embed_06.json').read_text())['rankings']
 lexical=json.loads((ROOT/'results/public-full-bm25.json').read_text())['rankings']
 rows=[];selected=0
 for q in cohort['queries']:
  qid=q['id'];overlap=len(set(dense[qid][:5])&set(lexical[qid][:5]));trigger=overlap<=2;escalate=trigger and selected<4
  if escalate:selected+=1
  rows.append({'id':qid,'query':q['query'],'candidate_ids':dense[qid][:20],'dense_lexical_top5_overlap':overlap,'gate_triggered':trigger,'escalate':escalate,'quota_excluded':trigger and not escalate})
 obj={'protocol_version':'selective-escalation-v1','cohort_sha256':digest(cohort),'rows':rows,'hypothesis':'Selective paid evidence only when dense and BM25 top5 overlap is at most2; preserve dense on other queries. On escalated queries apply fixed equal-weight RRF60 of dense and Jev Score ranks.','threshold_status':'Fixed once before any new cohort labels or Jev results; not tuned.','quota':'At most4 calls in pre-frozen cohort order; not confidence-ranked after seeing results.','model':'jev-1.13.0','candidate_count':20,'baseline':'Cached SkillRet top20 for all8 queries; same pool and denominator; full labels only after results are frozen.','failure_policy':'No automatic retry; stop on first uncertain failure. Policy falls back to cached dense; additionally report failed/uncertain call rate and no-fallback pipeline metrics. Do not treat no-call dense outputs as missing.','planned_metrics':['NDCG@10 all8','Completeness@5 all8','Recall@20 matched coverage','call_fraction','API_latency_including_failed_calls','known_cost_plus_unreconciled_reservation'],'inference_calls_made':0,'exposure':cohort['exposure'],'budget':'No reservation or execution yet; previous total1.60 remains. Existing balance and quote checks required before a future authorized run.'}
 obj['protocol_sha256']=digest(obj);path=HERE/'selective-probe-plan.json'
 if path.exists():assert json.loads(path.read_text())==obj
 else:path.write_text(json.dumps(obj,indent=2))
 print(json.dumps({'planned_queries':8,'planned_api_calls':selected,'selected_ids':[r['id'] for r in rows if r['escalate']],'protocol_sha256':obj['protocol_sha256'],'network_calls':0,'labels_read':False},indent=2))
if __name__=='__main__':main()
