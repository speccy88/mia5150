from harness import *
import random
labels=qrels('train');queries={q['id']:q['query'] for q in load('queries','train')};parts=json.loads((R/'manifests/training-partitions.json').read_text());allowed=set(parts['train_allowed_skill_ids']);channels={n:json.loads((R/f'results/training-bm25-clean-{n}.json').read_text())['rankings'] for n in ['header','full']};report={}
for mode in ['hard','replay']:
 ids=json.loads((R/f'manifests/lora-{mode}.json').read_text())['train_query_ids'];examples=[];required=set()
 for qid in ids:
  query=queries[qid];gold=labels[qid];required.update((query,d) for d in gold);pool=rrf([channels[n][qid] for n in channels]);examples.extend((query,d,1.) for d in gold);examples.extend((query,d,0.) for d in negatives(gold,[d for d in pool if d in allowed],4))
  if mode=='replay':examples.append((query,next(iter(gold)),1.));examples.append((query,next(d for d in sorted(allowed,key=lambda d:digest(qid+d)) if d not in gold),0.))
 random.Random(5150).shuffle(examples);retained=examples[:384];seen={(q,d) for q,d,y in retained if y};report[mode]={'before_cap':len(examples),'after_cap':len(retained),'required_positive_pairs':len(required),'omitted_positive_pairs':len(required-seen),'all_positives_preserved':required<=seen}
save(R/'evidence/lora-positive-coverage.json',report);print(report)
