"""Full public-query comparison on fixed cached document embeddings."""
from harness import *
from neural import Embed
import numpy as np,sys,time
role=sys.argv[1];kind='header';queries=safe_queries(load('queries','test'));ids=[]
with (R/'data/skills/test.jsonl').open() as f:
 for line in f:ids.append(json.loads(line)['id'])
docs=np.load(R/f'cache/{role}-{kind}-0-docs.npy');m=Embed(role);vectors,stats=m.encode([q['query'] for q in queries],True,batch=1);rankings={};lat=[]
for q,v in zip(queries,vectors):
 start=time.perf_counter();scores=docs@v;rankings[q['id']]=[ids[i] for i in np.argsort(-scores,kind='stable')[:100]];lat.append(time.perf_counter()-start)
save(R/f'results/public-full-{role}.json',{'rankings':rankings,'query_stats':stats,'search_seconds':lat,'metrics':evaluate(qrels('test'),rankings),'queries_hash':digest(queries),'note':'All 4392 corrected public test queries; public benchmark comparison, not blind confirmation. Frozen native header adapter FP16 max1024.'})
print('Full public query result saved.',flush=True)
