"""Reproduce supplementary tables from previously independently validated raw cases."""
import argparse,collections,csv,hashlib,json,pathlib
import numpy as np
R=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--primary-root',type=pathlib.Path,default=R.parent/'candidate_ed12adad')
p.add_argument('--packet-root',type=pathlib.Path,default=R.parent/'candidate_e8690e35/evidence_extract')
p.add_argument('--report',type=pathlib.Path,default=R/'report')
p.add_argument('--output',type=pathlib.Path,default=R/'supplementary_table_validation.json')
a=p.parse_args();P=a.primary_root;S=a.packet_root
def rows(p):
 with p.open() as f:return [json.loads(x) for x in f if x.strip()]
files=[P/'raw_library/MIA5150_Primary200_Raw_2026-10-24.jsonl',S/'qwen3_06_q8-nonprimary-canonical.jsonl',S/'qwen35_4b_q4-nonprimary-canonical.jsonl']
for path,sha in zip(files,['6b77f74019790a27243ca2e243aa6626acddbe8070c69930658192de8b4302f2','51cc8227aa7f1b7357db0de81402781697a8d64bb52568bc8bcf97f3e8021b1f','93b484be6c3a10bdf5cd3d3e1366ab107ec7476bc5349a733ed2233de5e691ff']):assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
pr=rows(files[0]);rr=pr+rows(files[1])+rows(files[2]);protocol=json.loads((S/'study/protocol.json').read_text());labels=json.loads((S/'study/evaluator-labels.json').read_text());qids=sorted(protocol['sensitivity_query_ids'])
groups={k:[q for q in qids if len(labels[q])==k] for k in [1,2,3]};rng=np.random.default_rng(20261024);draw={k:rng.integers(0,8,(2000,8)) for k in groups}
lookup={(r['model'],r['query_id'],r['seed'],r['size'],r['method']):r for r in rr};checked=collections.Counter()
for file,kind in [('seed-query-bootstrap.csv','seed'),('model-paired-bootstrap.csv','model')]:
 for row in csv.DictReader((a.report/file).open()):
  n,m,metric=int(row['size']),row['method'],row['metric'];vals={}
  for q in qids:
   values=[lookup[('qwen3_06_q8',q,seed,n,m)]['metrics'][metric] for seed in [42,43,44]]
   vals[q]=np.mean(values) if kind=='seed' else lookup[('qwen35_4b_q4',q,42,n,m)]['metrics'][metric]-values[0]
  distribution=sum(np.array([vals[q] for q in group])[draw[k]].mean(axis=1) for k,group in groups.items())/3
  if kind=='seed':
   columns=['balanced_macro_seed_averaged','query_bootstrap_ci95_low','query_bootstrap_ci95_high']
   seedmeans=[np.mean([lookup[('qwen3_06_q8',q,seed,n,m)]['metrics'][metric] for q in qids]) for seed in [42,43,44]]
   assert abs(float(row['seed_mean_sample_sd'])-np.std(seedmeans,ddof=1))<1e-12
  else:columns=['medium_minus_small_balanced_macro','paired_query_bootstrap_ci95_low','paired_query_bootstrap_ci95_high']
  for column,value in zip(columns,[np.mean(list(vals.values())),np.quantile(distribution,.025),np.quantile(distribution,.975)]):assert abs(float(row[column])-value)<1e-12
  checked[kind+'_bootstrap_cells']+=1
for row in csv.DictReader((a.report/'error-pipeline.csv').open()):
 items=[r for r in pr if r['size']==int(row['size']) and r['method']==row['method']];counts=collections.Counter()
 for r in items:
  if r['status']=='context_overflow':key='context_excluded'
  elif r['status']=='invalid_or_empty_domain':key='invalid_or_empty_route'
  elif r.get('candidate_recall') is None:key='pipeline_stage_unavailable'
  elif r['candidate_recall']<1:key='pool_missing_at_least_one_gold'
  elif r['status']!='ok':key='output_or_transport_failure_with_complete_pool'
  elif r['metrics']['exact']==1:key='exact_success'
  else:key='selection_error_despite_complete_pool'
  counts[key]+=1
 assert sum(counts.values())==200 and all(int(row[k])==v for k,v in counts.items());checked['disjoint_error_cells']+=1
result={'status':'passed','commit':'0f4e9818b6cefb94e09da30ad76b9ff5ac97999c','checks':dict(checked),'scope':'Independently reconstructed bootstrap arithmetic and disjoint first-failure counts from already independently validated raw cases; descriptive nominal sensitivity, no multiplicity-adjusted or causal model-size claim.'}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
