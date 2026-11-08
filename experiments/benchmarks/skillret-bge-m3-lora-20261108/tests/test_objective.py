import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import pytest
from objective import contrastive_loss,negative_ids
from prepare_data import normalize,split_queries,near_rows

def test_known_multiple_positives_are_not_negatives():
    assert negative_ids(['a','b','c','d'],{'a','c'},{'a','b','c'})==['b']

def test_loss_matches_explicit_probability_and_gradients():
    query=torch.tensor([[1.,0.]],requires_grad=True)
    docs=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
    loss=contrastive_loss(query,docs,torch.tensor([[True,False]]),temperature=1)
    assert loss.item()==pytest.approx(torch.log1p(torch.exp(torch.tensor(-1.))).item())
    loss.backward();assert torch.isfinite(query.grad).all() and torch.isfinite(docs.grad).all()
    assert docs.grad.abs().sum()>0

def test_multi_positive_mass_and_mask():
    q=torch.tensor([[1.,0.]])
    d=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
    all_positive=contrastive_loss(q,d,torch.tensor([[True,True,False]]),temperature=1)
    expected=torch.logsumexp(torch.tensor([1.,0.,-1.]),0)-torch.logsumexp(torch.tensor([1.,0.]),0)
    assert all_positive.item()==pytest.approx(expected.item())
    masked=contrastive_loss(q,d,torch.tensor([[True,True,False]]),candidate_mask=torch.tensor([[True,True,False]]))
    assert masked.item()==pytest.approx(0)

def test_missing_positive_is_rejected():
    with pytest.raises(AssertionError):contrastive_loss(torch.ones(1,2),torch.ones(2,2),torch.zeros(1,2,dtype=torch.bool))

def test_group_split_rejects_crossing_positives_and_test_overlap():
    qs=[{'id':x} for x in ('train','validation','cross','excluded')]
    gold={'train':{'a':1},'validation':{'b':1},'cross':{'a':1,'b':1},'excluded':{'a':1}}
    train,valid,reasons=split_queries(qs,gold,{'a'},{'b'},{'excluded'})
    assert [q['id'] for q in train]==['train'] and [q['id'] for q in valid]==['validation']
    assert sum(reasons.values())==2

def test_near_audit_last_block_does_not_include_reference_rows():
    left=[{'id':'train','query':'deterministic robust skill retrieval'}]
    right=[{'id':'test','query':'deterministic robust skill retrieval'}]
    found=near_rows(left,right);assert len(found)==1 and found[0]['left_query_id']=='train'
    assert normalize('  A\nB ')==normalize('a b')
