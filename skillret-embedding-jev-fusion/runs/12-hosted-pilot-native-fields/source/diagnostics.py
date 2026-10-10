"""Local descriptive diagnostics; never changes model inputs or requests APIs."""
import argparse, csv, json, pathlib, statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,default=pathlib.Path('/home/user/code/mia5150-local'));p.add_argument('--experiment-dir',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent);a=p.parse_args()
    out=a.experiment_dir/'results';audit=json.loads((out/'token_audit.json').read_text())
    with (a.base/'data/skills/test.jsonl').open() as f:skills=[json.loads(line) for line in f]
    qrels={}
    with (a.base/'data/qrels/test.jsonl').open() as f:
        for line in f:
            r=json.loads(line);qrels.setdefault(r['query_id'],set()).add(r['skill_id'])
    clipped=[]
    for skill,n in zip(skills,audit['document_raw_tokens']):
        if n>audit['max_seq_length']:
            clipped.append({'skill_id':skill['id'],'name':skill['name'],'original_characters':len(' | '.join((skill.get(k) or '').strip() for k in ('name','description','skill_md'))),'original_tokens':n,'encoded_tokens':audit['max_seq_length'],'discarded_tokens':n-audit['max_seq_length'],'policy':'official 32768-token right truncation; raw full text retained','gold_queries':sum(skill['id'] in values for values in qrels.values())})
    with (out/'truncation_audit.csv').open('w',newline='') as f:
        fields=['skill_id','name','original_characters','original_tokens','encoded_tokens','discarded_tokens','policy','gold_queries'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clipped)
    path=out/'baseline_skillret_06b.json'
    if path.exists():
        baseline=json.loads(path.read_text())
        if baseline.get('status')=='completed':
            per=baseline['evaluation']['per_query'];rows={}
            for count in (1,2,3):
                selected=[q for q,values in qrels.items() if len(values)==count]
                rows[str(count)]={'query_count':len(selected),'metrics':{metric:statistics.mean(per[q][metric] for q in selected) for metric in per[selected[0]]}}
            (out/'baseline_by_relevant_count.json').write_text(json.dumps(rows,indent=2)+'\n')
            pairs=[]
            for qid in sorted(qrels):
                ranks={sid:i+1 for i,sid in enumerate(baseline['rankings'][qid])}
                for sid in sorted(qrels[qid]):
                    rank=ranks.get(sid)
                    pairs.append({'query_id':qid,'skill_id':sid,'dense_rank_if_top50':rank,'in_top20':int(rank is not None and rank<=20),'in_top50':int(rank is not None and rank<=50)})
            assert len(pairs)==7187
            for k in (20,50):assert sum(row[f'in_top{k}'] for row in pairs)==baseline['evaluation']['candidate_coverage'][str(k)]['relevant_pairs_found']
            with (out/'relevant_pair_coverage.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    print(json.dumps({'truncated_documents':len(clipped),'affected_gold_queries':sum(bool(set(x['skill_id'] for x in clipped)&values) for values in qrels.values())}))

if __name__=='__main__':main()
