"""Disjoint primary error partition from retained raw case evidence."""
import collections,csv,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
def main():
    with (HERE/'results/primary200-canonical.jsonl').open() as f:rows=[json.loads(x) for x in f]
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['size'],row['method'])].append(row)
    result=[]
    for (size,method),items in groups.items():
        counts=collections.Counter()
        for r in items:
            if r['status']=='context_overflow':reason='context_excluded'
            elif r['status']=='invalid_or_empty_domain':reason='invalid_or_empty_route'
            elif r.get('candidate_recall') is None:reason='pipeline_stage_unavailable'
            elif r['candidate_recall']<1:reason='pool_missing_at_least_one_gold'
            elif r['status']!='ok':reason='output_or_transport_failure_with_complete_pool'
            elif r['metrics']['exact']==1:reason='exact_success'
            else:reason='selection_error_despite_complete_pool'
            counts[reason]+=1
        assert sum(counts.values())==200
        admitted=[r for r in items if r.get('candidate_recall')==1 and r['status']!='context_overflow']
        result.append({'size':size,'method':method,'planned':len(items),**{reason:counts[reason] for reason in ['context_excluded','invalid_or_empty_route','pipeline_stage_unavailable','pool_missing_at_least_one_gold','output_or_transport_failure_with_complete_pool','selection_error_despite_complete_pool','exact_success']},'complete_pool_admitted_cases':len(admitted),'exact_per_complete_pool_admitted':sum(r['metrics']['exact'] for r in admitted)/len(admitted) if admitted else None,'scope':'Disjoint first-failure partition; descriptive attribution, not a causal intervention. Generated/model sensitivity excluded.'})
    with (HERE/'report/error-pipeline.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    print('20 disjoint primary partitions each sum to200')
if __name__=='__main__':main()
