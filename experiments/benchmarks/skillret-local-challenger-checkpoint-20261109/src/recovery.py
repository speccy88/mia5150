"""Archive post-checkpoint journal rows and restore the saved best pointer."""
from common import *
import shutil,time

def reconcile(output,folder,state,kind):
 output=Path(output);folder=Path(folder);archive=output/'interrupted'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());archive.mkdir(parents=True,exist_ok=True);step=state['step'];journal=output/'steps.jsonl'
 if journal.exists():
  allrows=rows(journal);keep=[x for x in allrows if x['step']<=step];later=[x for x in allrows if x['step']>step]
  if later:write_rows(archive/'uncheckpointed-steps.jsonl',later);write_rows(journal,keep)
  assert len({x['step'] for x in keep})==len(keep)
 for p in output.glob('selection-*'):
  number=int(p.name.split('-')[1].split('.')[0])
  if number>step:shutil.move(str(p),archive/p.name)
 beststep=state['beststep']
 if beststep:
  source=folder/f'step-{beststep:05d}';assert (source/'adapter_model.safetensors').exists()
  (folder/'best').mkdir(exist_ok=True)
  for name in ('adapter_model.safetensors','adapter_config.json'):shutil.copyfile(source/name,folder/'best'/name)
  previous=output/'best.json'
  if previous.exists():shutil.copyfile(previous,archive/'best-pointer-before-recovery.json')
  if kind=='query':
   record=read(output/f'selection-{beststep:05d}.json');scores={w:r['aggregate']['NDCG@10'] for w,r in record['mixtures'].items()};weight=max(scores,key=lambda w:(scores[w],-float(w)));best={'step':beststep,'ndcg10':state['best'],'weight':float(weight),'adapter_sha256':sha(folder/'best/adapter_model.safetensors'),'mixture_scores':scores}
  else:
   record=read(output/f'selection-{beststep:05d}.summary.json');scores={w:r['evaluation']['aggregate']['NDCG@10'] for w,r in record.items()};key=max(scores,key=lambda w:scores[w]);best={'step':beststep,'ndcg10':state['best'],'configuration':key,'adapter_sha256':sha(folder/'best/adapter_model.safetensors')}
  save(previous,best)
 save(archive/'recovery.json',{'restored_checkpoint_step':step,'restored_best_step':beststep,'kind':kind,'reason':'Explicit resume; later journal/validation output retained here, not silently mixed into continuation'})
