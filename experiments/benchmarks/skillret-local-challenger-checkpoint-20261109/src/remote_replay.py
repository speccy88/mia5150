"""Verify actual published GitHub code and LFS payloads in a fresh checkout."""
from common import *
import argparse,subprocess,time

AUDITED_REPLAY=r'''
import pathlib, runpy, sys
src=pathlib.Path(sys.argv[1]); forbidden=[str(pathlib.Path.home()/'.cache'/p) for p in ('mia-local-challenger','mia-bge-lora','mia-bge-lora-lead','mia-bgem3','mia-qwen')]
def guard(event,args):
 if event.startswith('socket.'):raise RuntimeError('Network forbidden during portable replay')
 if event=='import' and args and str(args[0]).split('.')[0] in {'torch','transformers','peft','huggingface_hub','FlagEmbedding'}:raise RuntimeError('Model dependency forbidden during replay')
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  p=pathlib.Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve();s=str(p)
  if any(s==f or s.startswith(f+'/') for f in forbidden):raise RuntimeError('Original model/data cache forbidden during replay')
sys.addaudithook(guard);sys.path.insert(0,str(src));sys.argv=[str(src/'snapshot.py'),'replay'];runpy.run_path(sys.argv[0],run_name='__main__')
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('commit');p.add_argument('--python',required=True,type=Path);a=p.parse_args();assert len(a.commit)==40 and all(c in '0123456789abcdef' for c in a.commit);parent=CACHE/'remote-verification';parent.mkdir(parents=True,exist_ok=True);clone=parent/('published-'+a.commit[:12]);assert not clone.exists(),'Use a new cache for another independent fresh clone';rel=STUDY.relative_to(REPO).as_posix();env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
 subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse','--branch','main','https://github.com/speccy88/mia5150.git',str(clone)],env=env,check=True)
 def git(*args):return subprocess.check_output(['git',*args],cwd=clone,env=env,text=True).strip()
 assert git('rev-parse','HEAD')==a.commit,'Fresh main clone did not resolve to the expected publication commit';git('sparse-checkout','set',rel);git('lfs','pull','--include='+rel+'/**');target=clone/rel;manifest=read(target/'FILE_MANIFEST.json')
 for name,meta in manifest['files'].items():p=target/name;assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256'],name
 env.pop('PYTHONPATH',None);env.pop('SKILLRET_STUDY_ROOT',None);env.pop('SKILLRET_EXECUTION_SOURCE',None);env['PYTHONNOUSERSITE']='1';env['SKILLRET_CHALLENGER_CACHE']=str(clone/'nonexistent-external-cache');env['SKILLRET_TRAIN_CACHE']=str(clone/'nonexistent-raw-data');begin=time.perf_counter();subprocess.run([str(a.python),'-c',AUDITED_REPLAY,str(target/'src')],cwd=target,env=env,check=True)
 replay_path=target/'results/vector_replay.json'
 assert sha(replay_path)==manifest['files']['results/vector_replay.json']['sha256'],'Fresh replay differs from the published replay receipt'
 receipt={'status':'passed','actual_fresh_github_main_clone':True,'commit':a.commit,'verified_published_files':len(manifest['files']),'replay_seconds':time.perf_counter()-begin,'no_model_gpu_network_or_original_data_cache_during_replay':True,'replay_receipt_byte_identical':True,'replayed':read(replay_path)};save(parent/('verified-'+a.commit[:12]+'.json'),receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
