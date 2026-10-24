import pytest
pytest.importorskip('fcntl')
from hosted.budget import Budget,BudgetStop

def test_reservations_survive_restart_and_do_not_duplicate(tmp_path):
    p=tmp_path/'ledger.jsonl';b=Budget(p,1.1,1.65,1.5,600);b.reserve('first','.4')
    with pytest.raises(BudgetStop):Budget(p,1.1,1.65,1.5,600).reserve('first','.4')
    with pytest.raises(BudgetStop):b.reserve('second','.8')
    b.settle('first','.1');b.reserve('second','.8')

def test_unknown_charge_stays_reserved_and_halts(tmp_path):
    b=Budget(tmp_path/'ledger.jsonl',1.1,1.65,1.5,600);b.reserve('first','.2')
    with pytest.raises(BudgetStop):b.settle('first',None)
    assert str(b.state()[1])=='0.2'
    with pytest.raises(BudgetStop):b.reserve('second','.01')

def test_cad_and_call_caps_are_independent(tmp_path):
    b=Budget(tmp_path/'one.jsonl',10,1.5,1.5,600)
    with pytest.raises(BudgetStop):b.reserve('one','1.01')
    b=Budget(tmp_path/'two.jsonl',10,15,1.5,1);b.reserve('one','.01')
    with pytest.raises(BudgetStop):b.reserve('two','.01')

def test_known_zero_cost_is_not_replaced_by_reserve(tmp_path):
    b=Budget(tmp_path/'one.jsonl',1.1,1.65,1.5,600);b.reserve('one','.1');b.settle('one',0)
    assert b.state()[1]==0

def test_model_allocations_share_one_global_ledger(tmp_path):
    b=Budget(tmp_path/'ledger.jsonl',2,3,1.5,600,{'A':'.60','B':'.65'},legacy_model='A')
    b.reserve('first','.55',model='A');b.settle('first','.55')
    with pytest.raises(BudgetStop):b.reserve('over-a','.06',model='A')
    b.reserve('second','.64',model='B');b.settle('second','.64')
    assert str(b.state()[1])=='1.19'

def test_parallel_reservations_allow_only_one_unsettled_attempt(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def reserve(i):
        try:Budget(tmp_path/'shared.jsonl',2,3,1.5,600).reserve(str(i),'.01');return True
        except BudgetStop:return False
    with ThreadPoolExecutor(max_workers=8) as pool:assert sum(pool.map(reserve,range(8)))==1
    with pytest.raises(BudgetStop):Budget(tmp_path/'shared.jsonl',2,3,1.5,600).reserve('restart','.01')

def test_bound_is_unknown_spend_and_survives_restart_with_larger_hold(tmp_path):
    from decimal import Decimal
    p=tmp_path/'bound.jsonl';b=Budget(p,2,3,1.5,600,{'A':'.60','B':'.65'},legacy_model='A')
    b.reserve('old','.00051456',model='A')
    with pytest.raises(BudgetStop):b.settle('old',None)
    b.bound_unknown('old','.14','a'*64)
    b=Budget(p,2,3,1.5,600,{'A':'.60','B':'.65'},legacy_model='A')
    state,held=b.state();assert held==Decimal('.14')
    assert state['old']['actual_usd'] is None and state['old']['kind']=='bounded_unknown'
    with pytest.raises(BudgetStop):b.reserve('old','.001',model='A')
    b.reserve('different-model','.01',model='B')
    with pytest.raises(BudgetStop):b.settle('different-model',None)
    with pytest.raises(BudgetStop):b.reserve('third','.01',model='B')
    assert len(b.events())==5

def test_bound_requires_prior_unknown_and_cannot_shrink_it(tmp_path):
    b=Budget(tmp_path/'b.jsonl',2,3,1.5,600);b.reserve('one','.2')
    with pytest.raises(BudgetStop):b.bound_unknown('one','.3','a'*64)
    with pytest.raises(BudgetStop):b.settle('one',None)
    with pytest.raises(BudgetStop):b.bound_unknown('one','.1','a'*64)
    with pytest.raises(BudgetStop):b.bound_unknown('one','.3','not-a-hash')
