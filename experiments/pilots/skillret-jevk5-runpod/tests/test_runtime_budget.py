import copy,subprocess,sys
import scripts.train as train
from skillret_decider.core import digest

def test_runtime_budget_override_does_not_change_resume_contract():
    cfg={'training':{'max_wall_seconds':2700,'max_steps':100}}
    original=copy.deepcopy(cfg);before=digest(cfg)
    resolver=getattr(train,'resolve_wall_seconds',None)
    assert callable(resolver), 'Training has no separate administrative runtime budget'
    assert resolver(cfg,7200)==7200
    assert resolver(cfg,None)==2700
    assert cfg==original and digest(cfg)==before

def test_cli_exposes_administrative_wall_seconds():
    r=subprocess.run([sys.executable,'-m','scripts.train','--help'],capture_output=True,text=True)
    assert r.returncode==0
    assert '--wall-seconds' in r.stdout
