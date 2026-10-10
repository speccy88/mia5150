import copy, json
from pathlib import Path
import pytest
import torch
from torch import nn
from skillret_decider.model import soft_target_ce

def test_soft_ce_expected_value_and_gradient():
    x=torch.zeros((1,3),requires_grad=True)
    y=torch.tensor([[.5,.5,0.]])
    loss=soft_target_ce(x,y)
    assert loss.item()==pytest.approx(1.0986123)
    loss.backward()
    assert x.grad.tolist()[0]==pytest.approx([-1/6,-1/6,1/3])

def test_padded_column_zero_gradient_and_finite_loss():
    x=torch.tensor([[2.,1.,900.]],requires_grad=True)
    y=torch.tensor([[1.,0.,0.]])
    loss=soft_target_ce(x,y,torch.tensor([[True,True,False]]));loss.backward()
    assert torch.isfinite(loss) and x.grad[0,2]==0

@pytest.mark.parametrize('target',[[[.5,.4]],[[1.1,-.1]],[[float('nan'),0.]]])
def test_bad_targets_fail(target):
    with pytest.raises(ValueError): soft_target_ce(torch.zeros((1,2)),torch.tensor(target))

class ToyLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=nn.Linear(4,2,bias=False)
        self.base.weight.requires_grad_(False)
        self.a=nn.Parameter(torch.randn(2,4)*.1)
        self.b=nn.Parameter(torch.zeros(2,2))
    def forward(self,x): return self.base(x)+x@self.a.T@self.b.T

def run_toy():
    torch.manual_seed(42);torch.set_num_threads(1)
    model=ToyLoRA();base=model.base.weight.detach().clone()
    x=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    targets=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    opt=torch.optim.AdamW([model.a,model.b],lr=.05,weight_decay=0)
    initial=float(soft_target_ce(model(x),targets).detach())
    for _ in range(120):
        opt.zero_grad();loss=soft_target_ce(model(x),targets);loss.backward();opt.step()
    final=float(soft_target_ce(model(x),targets).detach())
    return model,opt,x,targets,base,initial,final

def test_toy_lora_trains_and_frozen_base_is_unchanged(tmp_path):
    m,opt,x,y,base,initial,final=run_toy()
    assert final<initial*.1 and torch.equal(base,m.base.weight)
    path=tmp_path/'own_checkpoint.pt';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},path)
    restored=ToyLoRA();state=torch.load(path,weights_only=True);restored.load_state_dict(state['model'])
    assert torch.equal(m(x),restored(x))
    # Exact resume for this deterministic CPU toy optimizer update.
    ropt=torch.optim.AdamW([restored.a,restored.b],lr=.05,weight_decay=0);ropt.load_state_dict(state['optimizer'])
    for model,optimizer in [(m,opt),(restored,ropt)]:
        optimizer.zero_grad();soft_target_ce(model(x),y).backward();optimizer.step()
    assert torch.equal(m(x),restored(x))

def test_letter_readout_matches_full_head_and_uses_last_real_token():
    from types import SimpleNamespace
    from skillret_decider.model import LetterReader
    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__(); self.embedding=nn.Embedding(32,4)
        def forward(self,input_ids,attention_mask,use_cache):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    class TinyBase(nn.Module):
        def __init__(self):
            super().__init__();self.model=TinyDecoder();self.lm_head=nn.Linear(4,32,bias=False)
    class Tok:
        def encode(self,letter,add_special_tokens=False):return [ord(letter)-ord('A')]
    b=TinyBase();reader=LetterReader(b,Tok())
    ids=torch.tensor([[2,4,0],[3,6,8]]);mask=torch.tensor([[1,1,0],[1,1,1]])
    actual=reader(ids,mask)
    h=b.model.embedding(ids)
    expected=b.lm_head(torch.stack([h[0,1],h[1,2]]))[:,:16]
    assert torch.allclose(actual,expected)
    actual.sum().backward()
    assert b.lm_head.weight.grad is None and b.model.embedding.weight.grad is not None
