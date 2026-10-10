"""Wait only for this project's known sequential specialist queue."""
import pathlib,time,json,subprocess,sys
R=pathlib.Path(__file__).resolve().parent
tags=['skillrouter-header','skillreason-header','skillrouter-rerank','skillreason-rerank','r3-rerank']
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
 for tag in tags:
  p=R/'runs'/tag
  if (p/'guard-stop.json').exists():raise RuntimeError(f'{tag} resource guard stopped')
  if (p/'exit.json').exists() and json.loads((p/'exit.json').read_text())['returncode']!=0:raise RuntimeError(f'{tag} failed')
 if all((R/'runs'/tag/'exit.json').exists() for tag in tags):break
 time.sleep(2)
else:raise TimeoutError('Specialist queue did not complete within30min')
sys.exit(subprocess.run(['python3','pipeline_remaining.py'],cwd=R).returncode)
