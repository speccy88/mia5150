import pathlib,time,json,subprocess,sys
R=pathlib.Path(__file__).resolve().parent;tags=['anchoring','structures','field-tfidf','escalation','scale','jev-skillret','plumb-skillret','prompt-scale','full-qwen','full-skillret'];deadline=time.monotonic()+3600
while time.monotonic()<deadline:
 for tag in tags:
  p=R/'runs'/tag
  if (p/'guard-stop.json').exists():raise RuntimeError(f'{tag} guard stopped')
  if (p/'exit.json').exists() and json.loads((p/'exit.json').read_text())['returncode']!=0:raise RuntimeError(f'{tag} failed')
 if all((R/'runs'/tag/'exit.json').exists() for tag in tags):break
 time.sleep(2)
else:raise TimeoutError('Remaining queue exceeded1hour')
sys.exit(subprocess.run(['python3','pipeline_extra.py'],cwd=R).returncode)
