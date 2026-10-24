"""Export immutable primary rows in small, independently scoreable JSONL cells."""
import collections,hashlib,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
def main():
    labels=json.loads((HERE/'evaluator-labels.json').read_text());out=HERE/'report/primary-audit';out.mkdir(exist_ok=True)
    cells=collections.defaultdict(list)
    with (HERE/'results/primary200-canonical.jsonl').open('rb') as f:
        for line in f:
            row=json.loads(line);assert row['panel']=='official_exposed' and row['seed']==42 and not row['status'].startswith('not_run')
            data={k:row[k] for k in ['case_id','query_id','seed','size','method','status','selected_ids','candidate_ids','gold_count','metrics']}
            data.update(gold_ids=labels[row['query_id']],source_row_sha256=hashlib.sha256(line.rstrip(b'\r\n')).hexdigest())
            cells[(row['size'],row['method'])].append(data)
    files={}
    for (size,method),items in cells.items():
        assert len(items)==200
        p=out/f'{size}-{method}.jsonl';p.write_text(''.join(json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n' for r in items))
        assert p.stat().st_size<500000
        files[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'queries':len(items)}
    (out/'manifest.json').write_text(json.dumps({'cases':sum(len(v) for v in cells.values()),'source':'results/primary200-canonical.jsonl','source_row_hash':'SHA256 of exact UTF8 source row without CR/LF line terminator','scoring':'status=ok: strict set comparison; otherwise decision exact/precision/recall/F1 zero. Unknown output cardinality precludes unqualified micro precision. Listing candidate_ids=null means full catalog.','files':files},indent=2))
    print('Exported',len(cells),'cells,4000 cases; largest bytes',max(v['bytes'] for v in files.values()))
if __name__=='__main__':main()
