"""Prepare checked study artifacts and update repository indexes after full execution."""
from common import *
import re

def main():
 assert not(STUDY/'FILE_MANIFEST.json').exists(),'Publication artifacts already frozen'
 l=read(STUDY/'FINAL_EVAL_LOCK.json');v=read(STUDY/'results/independent_verification.json');replay=read(STUDY/'results/vector_replay.json');assert v['status']==replay['status']=='passed';required={'reconstructed-base','adapted-query','released-reranker','trained-reranker','trained-reranker-seed43','scaling',l['primary_condition']};assert required<=set(replay['conditions'])
 scope=l['scaling'];scaling_count=scope['query_count']*len(scope['library_sizes'])*(len(scope['dense_conditions'])+len(scope['reranker_conditions']))
 assert replay['conditions']['scaling']['rankings_identical']==scaling_count
 ready=read(STUDY/'results/publication_readiness.json')
 assert ready['status']=='passed' and ready['lock_sha256']==sha(STUDY/'FINAL_EVAL_LOCK.json') and ready['report_sha256']==sha(STUDY/'REPORT.md')
 assert ready['vector_replay_sha256']==sha(STUDY/'results/vector_replay.json') and ready['independent_verification_sha256']==sha(STUDY/'results/independent_verification.json')
 assert set(replay['development_query_panels'])=={'selection','confirmation','secondary_selection'} and all(r['rankings_identical']==2560 for r in replay['development_query_panels'].values())
 assert len(list((STUDY/'figures').glob('*.png')))>=7 and len(list((STUDY/'figures').glob('*.svg')))>=7
 p=l['primary_condition'];metrics=read(STUDY/'results/final'/f'{p}.metrics.json')['aggregate'];d=read(STUDY/'results/paired_final_comparisons.json')[p];point=metrics['NDCG@10'];delta=d['ndcg10_delta']*100;lo,hi=[x*100 for x in d['ci95']];summary_line=f'Locked **{p}** NDCG@10 is **{point:.6f}** versus **0.810990** frozen ({delta:+.3f} pp, paired 95% interval [{lo:+.3f}, {hi:+.3f}]).';rel=STUDY.relative_to(REPO).as_posix();report=rel+'/REPORT.md';name=STUDY.name;guide='docs/publication/SKILLRET-LOCAL-CHALLENGER-REPRODUCTION.md'
 vr=STUDY/'vectors/README.md';vr.write_text(f'''# Lossless SkillRet-0.6B vector replay

`dense-vectors-v1.npz` contains the normalized float32 document and query
matrices, IDs, exact input-text hashes and qrels. It preserves the frozen full
training corpus, all cached training/development query anchors and the primary
selection, primary confirmation, and secondary selection adapted query vectors,
the rebuilt
original 32K test index, all official base and adapted query vectors, and the
historical scaling panel. The secondary seed's official vectors are retained.
No raw query or skill text and no base model weights are embedded in the archive.

The reranker conditions are reproducible from the saved per-candidate scores.
This is score replay, not a claim to run the transformer without its weights.
The `receipts/` directory holds actual cache contracts and timing journals.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/speccy88/mia5150.git
cd mia5150
git lfs pull --include='{rel}/**'
cd {rel}
python3.12 -m venv /tmp/skillret-vector-replay
/tmp/skillret-vector-replay/bin/pip install numpy==2.2.6
OPENBLAS_NUM_THREADS=1 /tmp/skillret-vector-replay/bin/python src/snapshot.py replay
```

Replay itself needs no model, GPU, raw dataset, original cache or network
request. It checks the full official rankings and metrics, all {scaling_count:,} scaling
query/configuration cases, all five query mixture weights on each retained
development panel, and all completed development reranker policies from their saved scores.
The archive's SHA256 and per-array shape/source
bindings are in `manifest.json`. Larger score and timing journals use lossless
gzip, with original-byte checks in `manifests/record_archive_manifest.json`.

To restore immutable cached arrays for actual training or inference, choose a
fresh external `SKILLRET_CHALLENGER_CACHE` and run
`python src/snapshot.py hydrate --subset train` (or `test` / `all`). Hydration
verifies the restored NPY and journal bytes and never replaces different
existing data. Transformer execution still requires the pinned models and
raw dataset described in the study README.
''')
 readme=STUDY/'README.md';s=readme.read_text();old='Execution is in progress. Prepared code is not evidence that a stage has run.\nOnly completed receipts and scored query records establish actual scope.';assert old in s;s=s.replace(old,summary_line+'\n\nRead [the results](REPORT.md) and [the model-free replay guide](vectors/README.md).')
 s=s.replace('python src/guard.py query-pilot python src/cli.py train-query --lr 3e-6 --steps 100\n','')
 s+='''

For a full retraining reproduction, use an isolated checkout and preserve the
published `results`, `adapters`, `vectors`, `FINAL_EVAL_LOCK.json` and inference
configuration in a separate archive before writing a new run in this directory.
Use a fresh external cache or hydrate the frozen arrays from the publication.
The recorded seeds and lock identify the original scientific result; never
replace the archived results with a rerun.

After the frozen index is available, the campaign controllers are:

```bash
python src/development_reranker.py --mode prepare
python src/campaign_query.py
python src/campaign_reranker.py
python src/secondary.py
python src/confirm_campaign.py
python src/review_development.py
python src/freeze_final.py
python src/run_final.py
```

The controllers use new attempt directories and retain failed receipts. If a
model stage stops, inspect its receipt and resume from the corresponding atomic
checkpoint with the same trial name, recipe and total step budget. The first
query fixture's deterministic-backward repair is documented in the source audit.
''';readme.write_text(s)
 catalog=REPO/'docs/SkillRet-Benchmark-Catalog.md';s=catalog.read_text();m=re.search(r'\*\*(\d+) studies and bounded pilots\*\*',s);n=int(m.group(1));s=s[:m.start(1)]+str(n+1)+s[m.end(1):];s+=f'\n## {n+1}. Local SkillRet-0.6B challenger\n\n[Report](../{report}) · [Model-free vector replay](../{rel}/vectors/README.md)\n\nLocal RTX 2060 SUPER 8 GB / i3-3225. Anchored query-only LoRA against the original 32K frozen index, task-specific reranker LoRA and fixed development-selected fusion. {summary_line}\n\nComplete 4,392-query / 6,006-skill exact-ID official test, with separate starting-checkpoint controls and an exposed 20-query, eight-pool scaling panel. Lossless vectors, compact adapters and per-query score evidence are retained. Development holds out source families from local updates; prior checkpoint and official-test exposure limit the claim.\n';catalog.write_text(s)
 rootreadme=REPO/'README.md';s=rootreadme.read_text().replace(f'[{n}-study benchmark catalog]',f'[{n+1}-study benchmark catalog]',1);marker='## BGE-M3 LoRA:';idx=s.index(marker);s=s[:idx]+f'## Local SkillRet-0.6B challenger\n\n{summary_line} The [completed study]({report}) separates query-only changes, the released reranker and locally trained reranking. [Download and replay]({guide}) the saved vectors and scores without embedding again.\n\n'+s[idx:];rootreadme.write_text(s)
 ex=REPO/'EXPERIMENTS.md';s=ex.read_text().replace(f'[{n}-study benchmark catalog]',f'[{n+1}-study benchmark catalog]',1);marker='| [skillret-bge-m3-lora-20261108]';idx=s.index(marker);s=s[:idx]+f'| [{name}]({report}) | Anchored query-only LoRA, reranker LoRA, fixed fusion, complete scores, adapters and lossless vectors | {summary_line} All 4,392 official queries and eight exact scaling pools retained. Starting controls and exposure limits reported. |\n'+s[idx:];ex.write_text(s)
 bench=REPO/'experiments/benchmarks/README.md';s=bench.read_text();idx=s.index('| BGE-M3 LoRA');s=s[:idx]+f'| Local SkillRet-0.6B query and reranker adaptation | [Report, adapters and vector replay]({name}/REPORT.md) |\n'+s[idx:];bench.write_text(s)
 gp=REPO/guide;gp.write_text(f'# Local SkillRet challenger reproduction\n\nUse the [lossless vector and score replay guide](../../{rel}/vectors/README.md) for CPU-only reproduction of the complete official evaluation and the exact historical scaling pools. See the [measured report](../../{report}) for the locked comparison and training attribution.\n\nThe study is published directly on `main`. The compact adapters and full indexed vectors use Git LFS. Original model weights, optimizer states and raw source corpora remain outside Git.\n')
 attr=REPO/'.gitattributes';s=attr.read_text();s+=f'\n# Local SkillRet challenger: lossless vectors, compact adapters, and score evidence.\n/{rel}/** -text\n/{rel}/vectors/*.npz filter=lfs diff=lfs merge=lfs -text\n/{rel}/**/*.jsonl.gz filter=lfs diff=lfs merge=lfs -text\n/{rel}/results/*.csv whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n';attr.write_text(s)
 paths=[p for p in STUDY.rglob('*') if p.is_file() and not any(part in ('__pycache__','.pytest_cache') for part in p.parts) and p.suffix not in ('.pyc','.tmp')];assert all(not p.is_symlink() for p in paths);assert not any(p.name in ('model.safetensors','pytorch_model.bin','recovery.pt') for p in paths)
 index={str(p.relative_to(STUDY)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)};save(STUDY/'FILE_MANIFEST.json',{'scope':'All published study files except this manifest itself; hydrate Git LFS before checking','files':index});paths.append(STUDY/'FILE_MANIFEST.json')
 pub=REPO/'docs/publication/PUBLICATION_MANIFEST.sha256';lines=pub.read_text().splitlines();headers=[x for x in lines if x.startswith('#')];items={}
 for line in lines:
  if not line or line.startswith('#'):continue
  h,name=line.split('  ',1);items[name]=h
 for p in paths+[rootreadme,ex,bench,catalog,gp,attr]:items[p.relative_to(REPO).as_posix()]=sha(p)
 pub.write_text('\n'.join(headers+[f'{h}  {name}' for name,h in sorted(items.items())])+'\n');print('Prepared checked publication files',len(paths),flush=True)
if __name__=='__main__':main()
