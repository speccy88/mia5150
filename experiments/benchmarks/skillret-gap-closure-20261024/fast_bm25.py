"""Sparse postings optimization of the frozen BM25 formula, not a new method.

Primary selector timing keeps its original implementation. This equivalent
search-only implementation is versioned separately for the full-query pass.
"""
import math,re
from collections import Counter
import numpy as np
from core import BM25

class SparseBM25(BM25):
    def __init__(self,documents):
        super().__init__(documents)
        self.postings={t:(np.fromiter(counts.keys(),dtype=np.int64),np.fromiter(counts.values(),dtype=np.float64)) for t,counts in self.inverted.items()}
    def search(self,query,ids,depth=100):
        ix=np.fromiter((self.pos[d] for d in ids),dtype=np.int64,count=len(ids));lengths=self.lengths[ix];avg=lengths.mean() or 1
        denominator=1.5*(.25+.75*lengths/avg);scores=np.zeros(len(ix));local=np.full(len(self.ids),-1,dtype=np.int64);local[ix]=np.arange(len(ix))
        for term,qtf in Counter(re.findall(r'\w+',query.casefold())).items():
            posting=self.postings.get(term)
            if posting is None:continue
            global_positions,frequencies=posting;positions=local[global_positions];valid=positions>=0;positions=positions[valid];tf=frequencies[valid];df=len(positions)
            scores[positions]+=qtf*math.log1p((len(ix)-df+.5)/(df+.5))*tf*2.5/(tf+denominator[positions])
        return [ids[i] for i in sorted(range(len(ids)),key=lambda i:(-scores[i],ids[i]))[:depth]]
