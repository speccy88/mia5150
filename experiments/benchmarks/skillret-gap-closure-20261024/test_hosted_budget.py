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
