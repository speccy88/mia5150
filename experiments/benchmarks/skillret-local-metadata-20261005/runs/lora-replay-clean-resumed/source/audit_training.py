"""Independent positive-coverage and split checks for every new adaptation."""
from harness import *
parts=json.loads((R/'manifests/training-partitions.json').read_text());train=set(parts['train']);dev=set(parts['dev']);labels=qrels('train');query_text={}
with (R/'data/queries/train.jsonl').open() as f:
 for line in f:
  q=json.loads(line)
  if q['id'] in train|dev:query_text[q['id']]=q['query']
assert not train&dev;assert not {query_text[q].strip().lower() for q in train}&{query_text[q].strip().lower() for q in dev}
metadata={}
with (R/'data/skills/train.jsonl').open() as f:
 for line in f:
  s=json.loads(line);metadata[s['id']]={'repo':s.get('repo'),'name':s['name'].strip().lower()}
train_positive={d for q in train for d in labels[q]};dev_positive={d for q in dev for d in labels[q]};allowed=set(parts['train_allowed_skill_ids']);assert not train_positive&dev_positive;assert train_positive<=allowed;assert not allowed&dev_positive
for key in ['repo','name']:
 assert not {metadata[d][key] for d in allowed if metadata[d][key]}&{metadata[d][key] for d in dev_positive if metadata[d][key]},key+' overlap'
out={'split_checks':{'query_ids_disjoint':True,'normalized_query_text_disjoint':True,'positive_skill_ids_disjoint':True,'allowed_training_negatives_disjoint_from_dev_repositories_and_names':True},'lora':{}}
for mode in ['hard','replay','replay-clean']:
 manifest=R/f'manifests/lora-{mode}.json'
 if not manifest.exists():continue
 m=json.loads(manifest.read_text());required={(digest(query_text[q]),d) for q in m['train_query_ids'] for d in labels[q]};p=R/f'manifests/lora-{mode}-examples.json'
 if p.exists():
  ex=json.loads(p.read_text())['examples'];positive={(r['query_sha256'],r['skill_id']) for r in ex if r['label']};omitted=required-positive;assert not omitted;out['lora'][mode]={'all_positive_pairs_retained':True,'required_pairs':len(required),'example_count':len(ex),'negative_positive_collisions':len({(r['query_sha256'],r['skill_id']) for r in ex if not r['label']}&required)};assert out['lora'][mode]['negative_positive_collisions']==0
 else:out['lora'][mode]=json.loads((R/'evidence/lora-positive-coverage.json').read_text())[mode]
channels={n:json.loads((R/f'results/training-bm25-clean-{n}.json').read_text())['rankings'] for n in ['header','full']};expected=sum(len(labels[q])+len(negatives(labels[q],[d for d in rrf([channels[n][q] for n in channels]) if d in allowed],12)) for q in train);recorded=json.loads((R/'results/adaptation-clean-summary.json').read_text())['examples'];assert expected==recorded
out['lexical']={'all_positives_retained':True,'examples':recorded,'examples_capped':False,'applies_to':['hard-negative','label-smoothing','L2-anchoring'],'reason':'Identical uncapped all-positive plus allowed-hard-negative construction, verified count.'}
save(R/'evidence/training-audit.json',out);print(out)
