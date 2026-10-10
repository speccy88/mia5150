"""Map and compare scores/ranks; intentionally no relevance labels."""
import hashlib,json,statistics
from pathlib import Path
from probe_contract import digest
HERE=Path(__file__).resolve().parent;OUT=HERE/'order-sensitivity-2-v1'
def inversion_count(a,b):
 pos={d:i for i,d in enumerate(b)}
 return sum(pos[a[i]]>pos[a[j]] for i in range(len(a)) for j in range(i+1,len(a)))
def main():
 items=json.loads((HERE/'prepared-order-sensitivity.json').read_text());r=json.loads((OUT/'results.json').read_text());p=json.loads((HERE/'order-sensitivity-plan.json').read_text());assert p.pop('protocol_sha256')==digest(p)==r['protocol_sha256']
 assert len(r['rows'])==len(items)==2;rows=[]
 for old,new in zip(items,r['rows']):
  assert old['id']==new['id'];row={'id':old['id'],'status':new['status']}
  if new['status']=='ok':
   assert new['request_sha256']==digest(old['request'])
   ds=[abs(a-b) for a,b in zip(old['original_scores'],new['mapped_scores'])]
   row.update(score_count=20,score_change_epsilon=1e-9,changed_scores=sum(d>1e-9 for d in ds),max_absolute_score_change=max(ds),mean_absolute_score_change=statistics.mean(ds),jev_ranking_identical=old['original_jev_ranking']==new['jev_ranking'],jev_inverted_pairs=inversion_count(old['original_jev_ranking'],new['jev_ranking']),jev_top5_overlap=len(set(old['original_jev_ranking'][:5])&set(new['jev_ranking'][:5])),fusion_identical=old['original_fusion']==new['fusion'],fusion_inverted_pairs=inversion_count(old['original_fusion'],new['fusion']),fusion_top1_same=old['original_fusion'][0]==new['fusion'][0],fusion_top5_overlap=len(set(old['original_fusion'][:5])&set(new['fusion'][:5])))
  rows.append(row)
 result={'rows':rows,'known_usage_usd':r['known_usage_usd'],'input_tokens':r['known_input_tokens'],'committed_usd':'1.77','paid_tests_stopped':True,'caveat':p['caveat'],'protocol_sha256':r['protocol_sha256'],'raw_results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()}
 (OUT/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
