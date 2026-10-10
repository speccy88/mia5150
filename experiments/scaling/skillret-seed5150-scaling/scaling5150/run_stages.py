"""Sequential bounded stages; stop at first failed guard, preserve every run."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STAGES={
 'indexes':[(f'index-{r}-v1',['scaling5150/prepare_cached_index.py',r]) for r in ['qwen_embed_06','skillrouter_embed_06','skillreason_embed_06']],
 'retrieval':[('retrieval-full-v1',['scaling5150/retrieval.py'])],
 'integrity':[('qwen8-integrity-v1',['scaling5150/inspect_qwen8.py'])],
 'selectors':[(f'selection-{r}-{mode}-v1',['scaling5150/selection.py','--model',r]+(['--pilot'] if mode=='pilot' else [])) for mode in ['pilot','full'] for r in ['small_qwen06','medium_qwen35']],
 'rerankers':[(f'rerank-{r}-v1',['scaling5150/rerank_local.py','--model',r]) for r in ['small_skillret','medium_jevk5','fit_qwen8']]}
def main():
 p=argparse.ArgumentParser();p.add_argument('group',choices=STAGES);a=p.parse_args()
 for tag,args in STAGES[a.group]:
  run='scaling5150-'+tag
  if (ROOT/'runs'/run).exists():raise RuntimeError('Refuse existing run '+run)
  print(json.dumps({'starting':run,'command':args}),flush=True)
  result=subprocess.run(['python3','guard.py',run,'.venv/bin/python',*args],cwd=ROOT)
  print(json.dumps({'finished':run,'returncode':result.returncode}),flush=True)
  if result.returncode:sys.exit(result.returncode)
if __name__=='__main__':main()
