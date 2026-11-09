"""Copy audited public telemetry and exact source snapshots, without private paths."""
from common import *
import shutil

def main():
 out=STUDY/'results/execution-receipts';out.mkdir(parents=True,exist_ok=True);summary_rows={}
 for run in sorted((CACHE/'runs').iterdir()):
  if not run.is_dir() or not(run/'exit.json').exists():continue
  dst=out/run.name;dst.mkdir(exist_ok=True);source=dst/'source';source.mkdir(exist_ok=True);names=('command.json','exit.json','telemetry.jsonl','source-sha256.json')
  for filename in names:
   if(run/filename).exists():shutil.copyfile(run/filename,dst/filename)
  manifest=read(run/'source-sha256.json') if (run/'source-sha256.json').exists() else {}
  for name,h in manifest.items():
   assert sha(run/'source'/name)==h;shutil.copyfile(run/'source'/name,source/name)
  summary_rows[run.name]={'exit':read(run/'exit.json'),'source_file_count':len(manifest),'source_execution':'launch snapshot captured' if run.name in ('embedding-parity-01','index-pilot-01','index-full-01') else 'executed from frozen source copy','raw_local_log_policy':'Unredacted process logs stay in the external cache; public receipts contain no personal paths.'}
 save(out/'manifest.json',summary_rows);print('Copied complete and failed supervised receipts',len(summary_rows),flush=True)
if __name__=='__main__':main()
