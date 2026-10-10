"""Final external diagnostic queue waits for the known previous queue."""
import pathlib,time,json,subprocess,sys
R=pathlib.Path(__file__).resolve().parent;py=str(R/'.venv/bin/python');deadline=time.monotonic()+10800
while time.monotonic()<deadline:
 p=R/'runs/sra-transfer'
 if (p/'guard-stop.json').exists():raise RuntimeError('SRA resource guard stopped')
 if (p/'exit.json').exists():
  assert json.loads((p/'exit.json').read_text())['returncode']==0;break
 time.sleep(2)
else:raise TimeoutError('Previous queue not complete within3hours')
for tag,cmd in [('skillreason-data',[py,'fetch_skillreason_bench.py']),('skillreason-schema',[py,'skillreason_transfer.py','smoke']),('skillreason-external',[py,'skillreason_transfer.py','full'])]:
 subprocess.run(['python3','reclaim.py'],cwd=R,check=True);status=subprocess.run(['python3','guard.py',tag]+cmd,cwd=R).returncode;print(tag,status,flush=True)
 if status:sys.exit(status)
