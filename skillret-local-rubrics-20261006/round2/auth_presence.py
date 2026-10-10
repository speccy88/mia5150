"""Presence-only report: never emits credential contents or contacts providers."""
import os,json
from pathlib import Path
keys=('RUNPOD_API_KEY','OPENROUTER_API_KEY','TYPESAFE_API_KEY','JEV_API_KEY','JEVK5_API_KEY')
report={'environment':{k:bool(os.environ.get(k)) for k in keys},'files':{}}
for root in ('mia5150-local','routing-pilot','category-pilot','frognano-benchmark'):
 p=Path('/home/user/code')/root/'.env';entry={'exists':p.is_file()}
 entry['contents_inspected']=False
 report['files'][str(p)]=entry
print(json.dumps(report,indent=2))
