"""Independently reconstruct frozen libraries and audit pinned qrels/metrics."""
import hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'scaling5150';sys.path.insert(0,str(ROOT))
from harness import evaluate,official_metrics
def main():
 m=json.loads((HERE/'manifest.json').read_text());skills=[json.loads(x) for x in (ROOT/'data/skills/test.jsonl').read_text().splitlines()];ids={x['id'] for x in skills};rels=[json.loads(x) for x in (ROOT/'data/qrels/test.jsonl').read_text().splitlines()];gold={}
 for r in rels:gold.setdefault(r['query_id'],set()).add(r['skill_id'])
 assert len(rels)==7187==sum(map(len,gold.values())) and len(gold)==4392 and max(map(len,gold.values()))==3
 evaluator=json.loads((HERE/'evaluator-qrels.json').read_text());assert all(set(evaluator[q['id']])==gold[q['id']] for q in m['queries'])
 previous={};count=0
 for c in m['cases']:
  qid=c['query_id'];negative=sorted(ids-gold[qid]);seed_bytes=str(m['seed']).encode()+b'\x00'+qid.encode();seed_int=int(hashlib.sha256(seed_bytes).hexdigest(),16);random.Random(seed_int).shuffle(negative);expected=set(gold[qid])|set(negative[:c['size']-len(gold[qid])]);actual=c['skill_ids'];assert len(actual)==len(set(actual))==c['size'] and set(actual)==expected and gold[qid]<=expected
  assert previous.get(qid,set())<=expected;previous[qid]=expected
  independent_order=sorted(expected,key=lambda d:(hashlib.sha256(b'5150\x00display\x00'+qid.encode()+b'\x00'+d.encode()).hexdigest(),d));assert actual==independent_order;count+=1
 # Hand-built multi-positive, missing, duplicate and cutoff cases.
 labels={'both':{'a':1,'b':1},'half':{'a':1,'b':1},'six':{'a':1},'missing':{'a':1}}
 rankings={'both':['a','x','y','b'],'half':['a','a'],'six':['x','y','z','w','v','a']};ev=evaluate(labels,rankings);assert ev['per_query']['both']['Recall@5']==1 and ev['per_query']['half']['Recall@5']==.5 and ev['per_query']['six']['MRR@5']==0 and ev['per_query']['missing']['Recall@5']==0 and ev['query_count']==4
 official=official_metrics(labels,{q:list(dict.fromkeys(rankings.get(q,[]))) for q in labels})
 for key,value in official.items():assert abs(value-ev['aggregate'][key])<=0.0000051,(key,value,ev['aggregate'][key])
 report={'library_cases_verified_independently':count,'qrels_total':7187,'queries_with_qrels':4392,'max_gold':3,'all_known_official_positives_in_every_library':True,'nested_sets':True,'unique_ids':True,'label_independent_display_order':True,'hand_metric_tests_passed':True,'official_hand_metrics':official,'fixed_query_denominator':4,'gold_semantics':'Completeness means all pinned official positives preserved; does not assert all semantically valid alternatives were exhaustively judged.','untouched_confirmation_used':False,'source_hash_checks':{name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==value for name,value in m['source_hashes'].items()}}
 assert all(report['source_hash_checks'].values());(HERE/'independent-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
