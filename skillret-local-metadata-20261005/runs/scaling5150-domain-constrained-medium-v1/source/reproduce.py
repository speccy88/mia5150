"""Create an isolated reproduction root without overwriting delivered evidence."""
import argparse,pathlib,shutil,subprocess,os,json,sys
p=argparse.ArgumentParser();p.add_argument('stage',choices=['bm25','embedding','reranker','lora-hard']);p.add_argument('--name',required=True);p.add_argument('--run',action='store_true');args=p.parse_args();source=pathlib.Path(__file__).resolve().parent
assert args.name and all(c.isalnum() or c in '-_' for c in args.name),'Use letters, digits, hyphen or underscore';target=source/'reproductions'/args.name;target.mkdir(parents=True,exist_ok=False)
for f in source.iterdir():
 if f.is_file() and f.suffix in ('.py','.ini'):shutil.copy2(f,target/f.name)
for name in ['data','models','official','.venv','vendor']:
 if (source/name).exists():(target/name).symlink_to(source/name,target_is_directory=True)
for name in ['evidence','manifests']:
 shutil.copytree(source/name,target/name)
(target/'results').mkdir();(target/'cache').mkdir()
for name in ['skillret_embed_06-header-0.json','training-bm25-clean-header.json','training-bm25-clean-full.json']:
 if (source/'results'/name).exists():shutil.copy2(source/'results'/name,target/'results'/name)
py=str(source/'.venv/bin/python');role='skillret_rerank_06_fp16_local' if (source/'models/skillret_rerank_06_fp16_local/manifest.json').exists() else 'skillret_rerank_06'
commands={'bm25':[py,'audit_baseline.py'],'embedding':[py,'neural.py','skillret_embed_06','header'],'reranker':[py,'rerank.py',role,'skillret_embed_06-header-0.json','header','32'],'lora-hard':['env','MIA_LORA_BASE_ROLE='+role,py,'lora_pilot.py','hard']};command=['python3','gate_run.py','reproduction-'+args.stage]+commands[args.stage];(target/'reproduction-command.json').write_text(json.dumps({'source':str(source),'target':str(target),'command':command,'source_inputs_linked_read_only_by_workflow':['data','models','official','vendor'],'warning':'Do not run downloaders/converters in this linked reproduction root; selected commands never write base model/data files.'},indent=2));print(str(target));print(json.dumps(command))
if args.run:
 env=os.environ.copy();env['MIA5150_ROOT']=str(target);sys.exit(subprocess.run(command,cwd=target,env=env).returncode)
