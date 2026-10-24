"""Independent mechanical provenance/label checks, without inventing semantics."""
import collections,json,pathlib,re
HERE=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def main():
    src=HERE.parents[1]/'scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs'
    queries=rows(src/'queries-test.jsonl');lookup={q['id']:q for q in queries};qrels=rows(src/'qrels-test.jsonl');truth=collections.defaultdict(set)
    for row in qrels:truth[row['query_id']].add(row['skill_id'])
    panel=read(HERE/'panel.json');generated=read(HERE/'generated-panel.json');labels=read(HERE/'evaluator-labels.json')
    old={q['id'] for q in read(src/'scorer-queries-128.json')};allsets={q['id']:set(re.findall(r'\w+',q['query'].casefold())) for q in queries}
    near=[]
    for q in panel:
        assert set(labels[q['id']])==truth[q['id']]==set(lookup[q['id']]['skill_ids'])
        a=allsets[q['id']]
        for other,b in allsets.items():
            if other==q['id']:continue
            # Jaccard cannot reach threshold if length ratio is lower.
            if min(len(a),len(b))<.85*max(len(a),len(b)):continue
            similarity=len(a&b)/len(a|b)
            if similarity>=.85:near.append({'query_id':q['id'],'other_id':other,'token_jaccard':similarity})
    checks=[]
    for q in generated:
        expected=set().union(*(truth[x] for x in q['constituent_ids']));assert expected==set(labels[q['id']])
        checks.append({'query_id':q['id'],'constituent_count':len(q['constituent_ids']),'gold_count':len(expected),'mechanical_union_valid':True,'human_semantic_labels':'pending','sequence_validity':'not established'})
    result={'status':'passed mechanical checks','public_query_count':len(queries),'primary_queries':len(panel),'primary_overlap_with_prior128':sum(q['id'] in old for q in panel),'all_primary_previously_exposed':True,'exact_duplicate_texts_in_official':len(queries)-len({q['query'] for q in queries}),'near_duplicate_threshold':.85,'near_duplicate_measure':'Unicode word-token Jaccard, casefold, compared against all official queries excluding sameID','near_duplicate_pairs':near,'primary_strata':dict(collections.Counter(len(labels[q['id']]) for q in panel)),'generated_checks':checks,'label_limits':'Positive qrels are not exhaustive human semantic judgments; no human agreement, fresh holdout or verified dependency labels are fabricated.'}
    (HERE/'label-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['near_duplicate_pairs','generated_checks']}))
if __name__=='__main__':main()
