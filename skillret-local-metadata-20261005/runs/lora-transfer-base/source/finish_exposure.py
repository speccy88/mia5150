from harness import *
ledger=json.loads((R/'evidence/exposure-ledger.json').read_text());old=json.loads((R/'evidence/historical-200-ids.json').read_text());current=[q['id'] for q in json.loads((R/'manifests/scorer-queries.json').read_text())]
ledger['historical_jev'].update(status='Pinned private-fork README and200 query IDs read. Original recorded200-query result not rerun. HistoricalRecall5.7092 vs BM25.6058 on BM25top20, not comparable to newdense pools.',source_commit='041f2ceccd3f43e9f841289541b68083c795df73',overlap_with_new128=sorted(set(old)&set(current)))
ledger['historical_lora'].update(status='Pinned fork README read. Differentdatasetrevision a050ad233a504a43135bafe8cdf45574052b5729; already-selected256val. Recall5.473307->.509766 atstep100. A40BF16 peak9.956GB not reproduced on8GBGPU. Never merged with this adaptation.')
ledger['historical_full_public']={'source':'evidence/historical-cpu.md','status':'Historical retained observations consumed full4392query aggregates; raw runs not on this executor. Newly completed full-query comparison is explicitly public, not blind.'}
ledger['final_test']={'status':'No pristine independent final set certified. SRA frozen transfer protocol remains an external diagnostic because checkpoint exposure cannot be proven absent.'}
ledger['superseded_lora_replay']={'files':'results/lora-replay-*.json','reason':'Original384example random cap omitted8 of91 unique labelled positivepairs. Preserved but excluded from supported gain claims. replay-clean preserves allpositives before sampling negatives.','audit':'evidence/lora-positive-coverage.json'}
save(R/'evidence/exposure-ledger.json',ledger)
cells=[{'cell_type':'markdown','metadata':{},'source':['# Bounded native LoRA: inspect before and after\n','This notebook reads saved results. Training commands below are text, not automatically executed. Base weights are unchanged; adapters are reversible.\n']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['import json\n','from harness import R\n','for mode in ["hard", "replay-clean"]:\n','    for stage in ["base", "adapted"]:\n','        p = R / f"results/lora-{mode}-{stage}.json"\n','        if p.exists(): print(mode, stage, json.loads(p.read_text())["metrics"]["aggregate"]["NDCG@10"])\n']},{'cell_type':'markdown','metadata':{},'source':['## Explicit training command\n','After reading START_HERE and starting telemetry, use a fresh output directory and unique run tag:\n','```bash\npython3 guard.py UNIQUE .venv/bin/python lora_pilot.py replay-clean\n```\n','Old replay was superseded because a random cap omitted positive pairs. The corrected run asserts complete positive coverage.\n']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['stats=json.loads((R/"results/paired-statistics.json").read_text())\n','{k:v for k,v in stats.items() if k.startswith("lora")}\n']}]
save(R/'training_demo.ipynb',{'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5})
import io,contextlib
receipts={}
for filename in ['analysis.ipynb','training_demo.ipynb']:
 p=R/filename;book=json.loads(p.read_text());namespace={};number=0
 for i,cell in enumerate(book['cells']):
  cell['id']=f'cell-{i}'
  if 'IPython.display' in ''.join(cell.get('source',[])):cell['source']=['print(R / "demo.html")\n']
  if cell['cell_type']=='code':
   capture=io.StringIO()
   with contextlib.redirect_stdout(capture):exec(compile(''.join(cell['source']),f'{filename}/cell{i}','exec'),namespace)
   number+=1;cell['execution_count']=number;cell['outputs']=[{'output_type':'stream','name':'stdout','text':capture.getvalue().splitlines(True)}] if capture.getvalue() else []
 save(p,book);receipts[filename]={'executed_code_cells':number,'method':'Sequential Python code-cell smoke, no notebook frontend or kernel launch; no training side effects.'}
save(R/'evidence/notebook-smoke.json',receipts)
