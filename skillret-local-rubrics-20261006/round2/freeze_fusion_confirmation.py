"""Freeze one out-of-panel fixed-fusion test without labels."""
import hashlib,json
from pathlib import Path
from probe_contract import digest
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'round2'
def main():
 excluded={q['id'] for q in json.loads((ROOT/'manifests/scorer-queries.json').read_text())}
 excluded.update(q['id'] for q in json.loads((HERE/'next-confirmation-8.json').read_text())['queries'])
 queries=[json.loads(line) for line in (ROOT/'data/queries/test.jsonl').read_text().splitlines()]
 chosen=sorted(({'id':q['id'],'query':q['query']} for q in queries if q['id'] not in excluded),key=lambda q:hashlib.sha256(('mia5150-fixed-rrf-confirmation-v1:'+q['id']).encode()).hexdigest())[:8]
 dense=json.loads((ROOT/'results/public-full-skillret_embed_06.json').read_text())['rankings']
 p={'queries':chosen,'candidate_ids':{q['id']:dense[q['id']][:20] for q in chosen},'hypothesis':'Unconditional fixed equal-weight RRF60 of cached SkillRet and Jev Score ranks improves NDCG@10; ties preserve dense order.','primary_metric':'NDCG@10 all8','secondary_metrics':['Completeness@5','Recall@20 matched pool','pool oracle NDCG@10','failures','API latency','cost'],'model':'jev-1.13.0','question_type':'Score only,20 fixed header-only questions per query','dataset_revision':'6583d7d2ed07644d0fb8938ed8178f3a7dc42a12','selection':'First8 salted SHA256 query IDs with salt mia5150-fixed-rrf-confirmation-v1:, exclude all128 prior scorer IDs and the eight selective-panel IDs; no qrels inspection.','parameters':'One hypothesis, RRF60 equal weight; no sweep, no rerun, no reordering candidates from labels.','failure_policy':'Stop at first uncertain failure; failed/unattempted predictions empty, all8 retained; no provider fallback or retries.','calls_max':8,'input_tokens_max_total':512000,'reserved_usd':'0.10','exposure':'New to this round and excluded previous rerank cohorts, but historically exposed public benchmark, including full4392 dense evaluation. Not a genuine blind final test.'}
 p['protocol_sha256']=digest(p);path=HERE/'fusion-confirmation-plan.json'
 if path.exists():assert json.loads(path.read_text())==p
 else:path.write_text(json.dumps(p,indent=2))
 print(json.dumps({'ids':[q['id'] for q in chosen],'protocol_sha256':p['protocol_sha256'],'labels_read':False,'network_calls':0}))
if __name__=='__main__':main()
