"""JevK5-compatible differentiable letter-logit readout (no autoregressive generation).
GPU integration must be validated on the target RunPod stack before a long run.
"""
from __future__ import annotations
import os, sys, json, hashlib, time
# Match upstream's conservative PyTorch fallback when FLA backward support is uncertain.
# Enable only in a separately recorded pilot with a compatible Triton/FLA stack.
if os.environ.get('SKILLRET_ENABLE_FLA', '0') != '1':
    sys.modules['fla'] = None
else:
    from importlib.metadata import version
    from packaging.version import Version
    if Version(version('triton')) < Version('3.7.1'):
        raise RuntimeError('FLA backward is not approved with Triton < 3.7.1; unset SKILLRET_ENABLE_FLA')
    version('flash-linear-attention')  # Fail explicitly if requested but absent.
import torch
from torch import nn
from skillret_decider.core import digest, prediction_from_logits

TARGET_NAMES = ('q_proj','k_proj','v_proj','o_proj','in_proj_qkvz','in_proj_ba',
                'in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj')

def soft_target_ce(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor | None = None):
    """Multi-positive listwise cross-entropy; categorical mass, NOT independent relevance probabilities."""
    if logits.shape != targets.shape or logits.ndim != 2: raise ValueError('Bad loss shapes')
    if valid is None: valid = torch.ones_like(targets, dtype=torch.bool)
    if valid.shape != logits.shape or not valid.any(-1).all(): raise ValueError('No valid options')
    if not torch.isfinite(logits[valid]).all() or not torch.isfinite(targets).all(): raise ValueError('Nonfinite loss input')
    if (targets < 0).any() or (targets[~valid] != 0).any(): raise ValueError('Invalid target mass')
    if not torch.allclose(targets.sum(-1), torch.ones_like(targets.sum(-1)), atol=1e-5):
        raise ValueError('Targets must sum to one per query')
    # Finite sentinel avoids 0 * -inf NaNs in padded columns.
    masked = logits.float().masked_fill(~valid, -1e9)
    return -(targets.float() * torch.log_softmax(masked, dim=-1)).sum(-1).mean()

def trainable_fingerprint(model) -> str:
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad:
            h.update(name.encode()); h.update(param.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_backbone(config: dict):
    import transformers
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.get_device_properties(0).major < 8:
        raise RuntimeError('Reference requires a BF16-capable CUDA GPU. Do not silently use CPU, FP16 or QLoRA.')
    spec = config['model']
    revision = spec['revision']
    if len(revision) != 40: raise ValueError('Require a full model commit SHA')
    hfconfig = transformers.AutoConfig.from_pretrained(spec['id'], revision=revision, trust_remote_code=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec['id'], revision=revision, trust_remote_code=False)
    cls = transformers.AutoModelForCausalLM
    if hfconfig.model_type in {'qwen3_5', 'qwen3_5_text'}:
        cls, hfconfig = transformers.Qwen3_5ForCausalLM, hfconfig.get_text_config()
    model = cls.from_pretrained(spec['id'], revision=revision, config=hfconfig,
        dtype=torch.bfloat16, device_map={'':'cuda:0'}, attn_implementation='sdpa', trust_remote_code=False)
    if any(p.device.type != 'cuda' for p in model.parameters()): raise RuntimeError('CPU model spill is not accepted')
    if getattr(model,'is_quantized',False): raise RuntimeError('Quantization is not the BF16 reference')
    model.config.use_cache = False
    for p in model.parameters(): p.requires_grad_(False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return model, tokenizer

def attach_lora(model, config: dict):
    from peft import LoraConfig, get_peft_model
    present = {name.rsplit('.',1)[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    targets = [x for x in TARGET_NAMES if x in present]
    if not targets: raise RuntimeError('No expected attention projections found; stop instead of silently training the wrong modules')
    c = config['lora']
    model = get_peft_model(model, LoraConfig(r=c['rank'], lora_alpha=c['alpha'],
        lora_dropout=c['dropout'], target_modules=targets, bias='none'))
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    return model, targets

class LetterReader(nn.Module):
    def __init__(self, backbone, tokenizer):
        super().__init__(); self.backbone = backbone; self.tokenizer = tokenizer
        letters = [tokenizer.encode(c, add_special_tokens=False) for c in 'ABCDEFGHIJKLMNOP']
        if any(len(t) != 1 for t in letters): raise ValueError('Expected single-token uppercase answer letters')
        base = backbone.get_base_model() if hasattr(backbone,'get_base_model') else backbone
        if base.lm_head.bias is not None: raise ValueError('Unexpected biased LM head: implement and validate before use')
        self.register_buffer('slot_weight',base.lm_head.weight[[t[0] for t in letters]].detach().clone(),persistent=False)
    def forward(self, ids, mask):
        base = self.backbone.get_base_model() if hasattr(self.backbone,'get_base_model') else self.backbone
        hidden = base.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = mask.long().sum(-1)-1
        if (last < 0).any(): raise ValueError('Empty prompt')
        return hidden[torch.arange(len(ids),device=ids.device),last] @ self.slot_weight.T

def encode_example(tokenizer, example: dict, config: dict) -> dict:
    from jevk5.prompt import messages, decision_options
    c = config['tokenization']
    audit = {'query_id':example['id'], 'query_truncated':False, 'options_truncated':0}
    def clip(text, maximum, field):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > maximum:
            if field=='query': audit['query_truncated']=True
            else: audit['options_truncated'] += 1
            return tokenizer.decode(ids[:maximum], skip_special_tokens=False)
        return text
    state = clip(example['state'], c['query_max_tokens'], 'query')
    pairs = decision_options(example['question'])
    if not 2 <= len(pairs) <= 16: raise ValueError('Expected 2..16 typed options')
    texts = [clip(text,c['option_max_tokens'],'option') for _,text in pairs]
    prompt = tokenizer.apply_chat_template(messages(state,example['question']['instructions'],texts),
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids = tokenizer.encode(prompt,add_special_tokens=False)
    if len(ids) > c['max_sequence_tokens']:
        raise ValueError(f"{example['id']}: {len(ids)} tokens exceeds budget. Adjust per-field caps in a NEW protocol; do not drop rows.")
    audit.update(input_tokens=len(ids), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                 prompt_policy_sha256=digest(c))
    targets = None
    if 'gold_probs' in example:
        if set(example['gold_probs']) != {k for k,_ in pairs}: raise ValueError('Incomplete target dictionary')
        targets = [float(example['gold_probs'][k]) for k,_ in pairs]
        if any(t<0 for t in targets) or abs(sum(targets)-1)>1e-6: raise ValueError('Invalid target distribution')
    return {'ids':ids,'count':len(pairs),'targets':targets,'audit':audit}

def collate(records, pad_id):
    maximum = max(len(r['ids']) for r in records)
    ids = torch.full((len(records),maximum),pad_id,dtype=torch.long,device='cuda')
    mask = torch.zeros_like(ids)
    target = torch.zeros((len(records),16),dtype=torch.float32,device='cuda')
    valid = torch.zeros_like(target,dtype=torch.bool)
    for i,r in enumerate(records):
        ids[i,:len(r['ids'])] = torch.tensor(r['ids'],device='cuda'); mask[i,:len(r['ids'])]=1
        valid[i,:r['count']]=True
        if r['targets'] is not None: target[i,:r['count']] = torch.tensor(r['targets'],device='cuda')
    return ids, mask, target, valid

@torch.inference_mode()
def predict(reader, examples, encoded, batch_size=1, deadline=None):
    reader.eval(); output=[]
    for start in range(0,len(examples),batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Evaluation process deadline reached; partial output is not an accepted score")
        batch=encoded[start:start+batch_size]
        ids,mask,_,_=collate(batch,reader.tokenizer.pad_token_id)
        torch.cuda.synchronize(); t=time.perf_counter()
        logits=reader(ids,mask).float()
        torch.cuda.synchronize(); seconds=time.perf_counter()-t
        for i,(e,r) in enumerate(zip(examples[start:start+batch_size],batch)):
            item=prediction_from_logits(e,logits[i,:r['count']].cpu().tolist())
            item.update(input_tokens=r['audit']['input_tokens'],batch_seconds=seconds,
                        batch_size=len(batch),seconds_per_query_amortized=seconds/len(batch))
            output.append(item)
    return output
