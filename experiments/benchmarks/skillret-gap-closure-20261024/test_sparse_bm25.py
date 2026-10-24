import random
from core import BM25
from fast_bm25 import SparseBM25

def test_sparse_matches_frozen_dense_reference_across_memberships_and_ties():
    rng=random.Random(5150);vocabulary=['a','b','c','é','東京','repeat','COMMON']
    docs={f'd{i:04d}':' '.join(rng.choices(vocabulary,k=rng.randrange(0,35))) for i in range(160)}
    reference=BM25(docs);optimized=SparseBM25(docs)
    for n in [1,10,100,160]:
        for _ in range(8):
            catalog=rng.sample(list(docs),n)
            for query in ['no_match','a a repeat','COMMON é 東京','']:
                assert optimized.search(query,catalog)==reference.search(query,catalog)
