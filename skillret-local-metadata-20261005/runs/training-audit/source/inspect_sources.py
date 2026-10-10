import json,pathlib,urllib.request,hashlib,zipfile,subprocess
R=pathlib.Path('/home/user/code/mia5150-local'); E=R/'evidence';E.mkdir(exist_ok=True)
z=R/'plan/SkillRet_Research_Codex_Plan_2026-10-05.zip'
assert z.stat().st_size==182489
assert hashlib.sha256(z.read_bytes()).hexdigest()=='a4f2ca4604a1b7c6120f8533c80eca1d2fee540b75a6c6d1975287af8cf89a0d'
def get(url):
 with urllib.request.urlopen(url,timeout=45) as f:return f.read()
models=json.loads((R/'plan/SkillRet_Research_Codex_Plan/registries/models.json').read_text())
for role,model in [('dataset','datasets/ThakiCloud/SKILLRET')]+[(x['id'],'models/'+x['model_id']) for x in models if x['id'] in ['qwen_embed_06','skillret_embed_06','skillret_rerank_06','skillrouter_embed_06','skillrouter_rerank_06','r3_rerank_06','skillreason_embed_06','skillreason_rerank_06','jina_rerank_35']]:
 try:
  info=json.loads(get('https://huggingface.co/api/'+model));(E/(role+'.json')).write_text(json.dumps(info,indent=2))
  mid=model.split('/',1)[1];prefix='datasets/' if role=='dataset' else ''
  card=get('https://huggingface.co/'+prefix+mid+'/resolve/'+info['sha']+'/README.md');(E/(role+'.md')).write_bytes(card)
  print(role,info['sha'],info.get('gated'),info.get('cardData',{}).get('license'),flush=True)
 except Exception as e:print(role,repr(e),flush=True)
(E/'official-pin.txt').write_text(subprocess.check_output(['git','-C',str(R/'official'),'rev-parse','HEAD'],text=True))
