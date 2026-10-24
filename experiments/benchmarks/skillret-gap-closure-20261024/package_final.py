"""Self-contained compact evidence and canonical local raw delivery; no models/keys."""
import argparse,hashlib,json,pathlib,re,zipfile
from ledger import canonical_attempts
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);parser.add_argument('--main-commit',required=True);parser.add_argument('--wiki-commit',required=True);args=parser.parse_args()
    assert all(re.fullmatch('[0-9a-f]{40}',x) for x in [args.main_commit,args.wiki_commit])
    out=pathlib.Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);validation=json.loads((HERE/'report/full-retrieval-validation.json').read_text());assert validation['status']=='passed'
    rows=[]
    for model,expected in [('qwen3_06_q8',5080),('qwen35_4b_q4',600)]:
        with (HERE/f'results/{model}-cases.jsonl').open() as f:current=list(canonical_attempts([json.loads(x) for x in f]).values())
        assert len(current)==expected and not any(r['status'].startswith('not_run') for r in current);rows.extend(current)
    raw=out/'MIA5150_Local_Selection_Raw_2026-10-24.jsonl';raw.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    files={}
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.txt','.gz','.ps1']:files[p.relative_to(ROOT).as_posix()]=p
    for folder in ['report','audit','source-snapshots','protocol-amendments','hosted']:
        for p in (HERE/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.lock']:files[p.relative_to(ROOT).as_posix()]=p
    for p in (HERE/'results').iterdir():
        if p.is_file() and not p.name.endswith('-cases.jsonl') and p.name!='retrieval.jsonl':files[p.relative_to(ROOT).as_posix()]=p
    source=ROOT/'experiments/scaling/skillret-seed5150-scaling/scaling5150/provenance-audit/public-inputs'
    for name in ['skills-metadata.jsonl','queries-test.jsonl','qrels-test.jsonl']:files[(source/name).relative_to(ROOT).as_posix()]=source/name
    for p in [ROOT/'README.md',ROOT/'EXPERIMENTS.md',ROOT/'docs/SkillRet-Benchmark-Catalog.md',ROOT/'paper/final_report.tex',ROOT/'paper/project_proposal.tex',ROOT/'paper/references.bib',ROOT/'paper/neurips_2026.sty',ROOT/'paper/presentation/research_draft.tex']:
        files[p.relative_to(ROOT).as_posix()]=p
    files['raw/'+raw.name]=raw
    manifest={name:{'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in files.items()}
    mainurl='https://github.com/speccy88/mia5150/tree/'+args.main_commit
    intro=f'''# Completed local evidence — 24 October 2026

Code publication: {mainurl}
Wiki commit: {args.wiki_commit}

The local selector grid is complete: 5,080 small-model cases and 600 separate 4B sensitivity cases, with all context exclusions and failures retained. There are 200 exposed official primary requests, a reused balanced 24-query sensitivity subset and 12 generated silver probes. Repeats are not new independent requests. The full retrieval pass contains 52,704 query/seed/size rows over all 4,392 exposed official requests, with three rankers per row. Its separate owner-run arithmetic and primary-overlap check passed. External independent reviewers validated the recorded local selector outputs; their receipts retain their narrower exact scopes.

Start with experiments/benchmarks/skillret-gap-closure-20261024/report/RESEARCH_DRAFT.md, GAP_CLOSURE.md, report/*.csv and the PNG/SVG figures. The raw local canonical export is in raw/. Exact original primary bytes and compressed full-retrieval raw evidence remain under the study's results/. Raw repeated attempt history is preserved in Git; it is not duplicated in this compact ZIP. FILE_MANIFEST.json hashes every bundled file. Code, panels, evaluator labels, exact catalogs, model pins, source snapshots, test sources, execution/resource receipts and sanitized public inputs support reproduction without private Library originals.

At 6,006 skills, the primary small selector recovers 9/200 exact sets with BM25, 7/200 with vector and 11/200 with hybrid. Hybrid-minus-BM25 uncertainty spans zero. Hybrid's pool contains all gold skills in 81/200 cases, showing substantial selection losses even when retrieval succeeds. A single-domain route cannot cover every gold domain for 31% of this primary panel. Full listing exceeds the configured 8K context at 1,000 and 6,006 skills. Distinguish successful serving latency, sparse full-query search timing, startup/indexing, failure wall time and tokenized overflow preflight.

Recommendations: retain matched comparisons and strict opaque IDs; improve selection after candidate coverage; treat domain routing as conditional on domain purity; avoid bulk listing beyond measured context; obtain independent human requests and semantic adjudication before generalization claims. Qwen model deployments differ in architecture/training/quantization. Generated labels are silver unions, not human annotation, workflow order or execution evidence. The existing public queries are exposed, and positive qrels may be incomplete. No new native activation or skill execution is claimed.

The cheaper DeepSeek/DeepInfra hosted stage is prepared separately within US$1.10 and 600 requests. It remains blocked on locating the user's credential: no authenticated or paid request, hosted score or spending is claimed. Paper and static presentation sources are research drafts; their compilation was not confirmed. The parent publication owner produces the polished PDF. Original private proposal/assessments, student identifiers, keys, encrypted-key files, model binaries, environments and machine-specific paths are excluded.
'''
    packet=out/'MIA5150_Final_Local_Evidence_2026-10-24.zip'
    with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('START_HERE.md',intro);z.writestr('FILE_MANIFEST.json',json.dumps({'main_commit':args.main_commit,'wiki_commit':args.wiki_commit,'canonical_local_cases':5680,'full_retrieval_rows':52704,'files':manifest},indent=2))
        for name,p in files.items():z.write(p,name)
    with zipfile.ZipFile(packet) as z:assert z.testzip() is None
    receipt={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in [packet,raw]};(out/'final-delivery-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
