"""Train a new LoRA on merged JevK5; evaluate against fixed natural BGE candidates.
Requires explicit approval. Never creates, stops or terminates a RunPod Pod.
"""
from __future__ import annotations
import argparse, json, math, os, random, shutil, subprocess, time, traceback, platform
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata as md
import numpy as np
from skillret_decider.core import (read_jsonl, write_jsonl, write_json, sha256_file, digest,
    evaluate_predictions, baseline_predictions)

def verify_data(path: Path):
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['is_test']: raise ValueError('Never train or select a checkpoint on the official test split')
    for name, checksum in manifest['files'].items():
        if sha256_file(path/name) != checksum: raise ValueError(f'Data hash mismatch: {name}')
    return manifest

def resolve_wall_seconds(cfg, override):
    """Administrative time allowance is not a model/data/optimizer contract change."""
    return cfg['training']['max_wall_seconds'] if override is None else override

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--resume',type=Path)
    p.add_argument('--approve-gpu-run',action='store_true')
    p.add_argument('--wall-seconds',type=int,help='Explicit administrative process allowance; leaves checkpoint contract unchanged')
    args=p.parse_args()
    if not args.approve_gpu_run: p.error('Set --approve-gpu-run only after agreeing the Pod budget and external teardown plan.')
    if args.out.exists(): raise FileExistsError('Use a NEW output directory, even when resuming')
    cfg=json.loads(args.config.read_text()); dm=verify_data(args.data)
    for key in ('seed','train_limit','validation_limit','candidates','retriever'):
        if cfg[key]!=dm['config'][key]: raise ValueError(f'Data/config mismatch: {key}')
    args.out.mkdir(parents=True)
    write_json(args.out/'config.json',cfg)
    write_json(args.out/'data_manifest.json',dm)
    wall_seconds=resolve_wall_seconds(cfg,args.wall_seconds)
    if wall_seconds<=0: p.error('--wall-seconds must be positive')
    write_json(args.out/'runtime_budget.json',{'wall_seconds':wall_seconds,'original_config_wall_seconds':cfg['training']['max_wall_seconds'],'override_requested':args.wall_seconds,'checkpoint_contract_changed':False})
    started=time.monotonic(); deadline=started+wall_seconds
    from skillret_decider.model import (torch, load_backbone, attach_lora, LetterReader, encode_example,
        collate, soft_target_ce, predict, trainable_fingerprint)
    seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    manifest={'config_sha256':digest(cfg),'data_sha256':digest(dm),'started_utc':datetime.now(timezone.utc).isoformat(),
        'precision':'BF16 unquantized backbone; PEFT may promote adapters to FP32',
        'fla_enabled':os.environ.get('SKILLRET_ENABLE_FLA','0')=='1',
        'python':platform.python_version(),'versions':{},'source_status':'new experiment; no historical rows edited'}
    for name in ('torch','transformers','peft','accelerate','huggingface-hub','jevk5','triton','flash-linear-attention'):
        try: manifest['versions'][name]=md.version(name)
        except md.PackageNotFoundError: manifest['versions'][name]=None
    manifest['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True).stdout
    write_json(args.out/'environment.json',manifest)
    train=read_jsonl(args.data/'train.jsonl'); dev=read_jsonl(args.data/'validation.jsonl')
    labels=json.loads((args.data/'validation_labels.json').read_text())
    if {e['id'] for e in train}&{e['id'] for e in dev}: raise ValueError('Train/dev leakage')
    if any('gold_probs' in e or 'expected' in e for e in dev): raise ValueError('Inference dev file should not contain targets')
    model,tok=load_backbone(cfg)
    encoded_train=[encode_example(tok,e,cfg) for e in train]
    encoded_dev=[encode_example(tok,e,cfg) for e in dev]
    write_jsonl(args.out/'tokenization_audit.jsonl',(dict(split=s,**r['audit']) for s,rs in [('train',encoded_train),('validation',encoded_dev)] for r in rs))
    # Verify the zero-initialized new LoRA preserves the merged starting checkpoint.
    base_reader=LetterReader(model,tok).eval()
    ids,mask,_,_=collate(encoded_dev[:1],tok.pad_token_id)
    with torch.inference_mode(): reference=base_reader(ids,mask).float().cpu()
    del base_reader
    model,targets=attach_lora(model,cfg); reader=LetterReader(model,tok).eval()
    with torch.inference_mode(): initial=reader(ids,mask).float().cpu()
    if not torch.allclose(reference,initial,atol=0.02,rtol=0.002):
        raise RuntimeError('New LoRA does not reproduce the merged base logits; stop before training')
    trainable=[v for v in model.parameters() if v.requires_grad]
    if not trainable or any('lora_' not in n for n,v in model.named_parameters() if v.requires_grad):
        raise RuntimeError('Unexpected trainable tensors; this run is LoRA-only')
    write_json(args.out/'admission.json',{'targets':targets,'trainable_parameters':sum(v.numel() for v in trainable),
        'total_parameters':sum(v.numel() for v in model.parameters()),'zero_lora_max_logit_difference':float((reference-initial).abs().max()),
        'cuda_resident':all(v.device.type=='cuda' for v in model.parameters()),'quantized':False,
        'gradient_checkpointing':cfg['training']['gradient_checkpointing']})
    optimizer=torch.optim.AdamW(trainable,lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    step=0; epoch=0; cursor=0; trained_queries=0; best=None; before_hash=trainable_fingerprint(model)
    def log(row):
        with (args.out/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'elapsed_s':time.monotonic()-started,**row},allow_nan=False)+'\n'); f.flush()
    def validate(label):
        predictions=predict(reader,dev,encoded_dev,cfg['training']['eval_batch_size'],deadline)
        metrics=evaluate_predictions(predictions,labels,[e['id'] for e in dev])
        write_jsonl(args.out/f'{label}.predictions.jsonl',predictions)
        write_json(args.out/f'{label}.metrics.json',metrics)
        log({'event':'validation','step':step,'label':label,'metrics':metrics['mean']})
        return metrics
    def key(metrics): return (metrics['mean']['recall@5'],metrics['mean']['mrr@15'])
    def save_checkpoint():
        cp=args.out/'checkpoints'/f'step-{step:06d}'
        if cp.exists(): return cp
        cp.mkdir(parents=True)
        model.save_pretrained(cp/'adapter',safe_serialization=True)
        tok.save_pretrained(cp/'tokenizer')
        state={'step':step,'epoch':epoch,'cursor':cursor,'trained_queries':trained_queries,'optimizer':optimizer.state_dict(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'best':best,'contract':manifest['config_sha256'],
            'data_contract':manifest['data_sha256']}
        torch.save(state,cp/'training_state.pt')
        write_json(cp/'manifest.json',{'step':step,'config':cfg,'data_sha256':manifest['data_sha256'],
            'training_state_sha256':sha256_file(cp/'training_state.pt'),
            'adapter_files':{p.relative_to(cp/'adapter').as_posix():sha256_file(p) for p in (cp/'adapter').rglob('*') if p.is_file()},
            'warning':'training_state.pt is a trusted, locally generated pickle; never load an untrusted replacement'})
        log({'event':'checkpoint','step':step,'path':str(cp)})
        return cp
    try:
        if args.resume:
            from peft import set_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights
            cm=json.loads((args.resume/'manifest.json').read_text())
            if cm['training_state_sha256'] != sha256_file(args.resume/'training_state.pt'): raise ValueError('Resume checksum mismatch')
            for name,checksum in cm['adapter_files'].items():
                if sha256_file(args.resume/'adapter'/name)!=checksum: raise ValueError('Resume adapter checksum mismatch')
            # Only load checkpoints produced by this project, not third-party pickle files.
            state=torch.load(args.resume/'training_state.pt',map_location='cpu',weights_only=False)
            if state['contract']!=manifest['config_sha256'] or state['data_contract']!=manifest['data_sha256']:
                raise ValueError('Resume must use exactly the same data and configuration')
            status=set_peft_model_state_dict(model,load_peft_weights(args.resume/'adapter',device='cpu'))
            if status.unexpected_keys: raise RuntimeError(f'Unexpected adapter keys: {status.unexpected_keys}')
            optimizer.load_state_dict(state['optimizer'])
            for st in optimizer.state.values():
                for k,v in st.items():
                    if torch.is_tensor(v) and k != 'step': st[k]=v.to('cuda')
            step,epoch,cursor,trained_queries,best=[state[k] for k in ('step','epoch','cursor','trained_queries','best')]
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
            log({'event':'resumed','source':str(args.resume),'step':step,'epoch':epoch,'cursor':cursor})
        else:
            write_json(args.out/'validation.bge.metrics.json',evaluate_predictions(baseline_predictions(dev),labels))
            zero=validate('validation.zero-shot')
            best={'key':key(zero),'step':0,'adapter':None,'metrics':zero['mean'],'source':'unadapted_merged_checkpoint',
                  'predictions':str((args.out/'validation.zero-shot.predictions.jsonl').resolve())}
            save_checkpoint()
        tcfg=cfg['training']; group_size=tcfg['microbatch_size']*tcfg['accumulation_steps']
        limit=tcfg['max_steps']
        while epoch < tcfg['epochs'] and (limit is None or step<limit):
            order=list(range(len(train))); random.Random(seed+epoch).shuffle(order)
            while cursor<len(order) and (limit is None or step<limit):
                if time.monotonic()>=deadline: raise TimeoutError('Training process limit reached; last complete checkpoint retained')
                group=order[cursor:cursor+group_size]; reader.train(); optimizer.zero_grad(set_to_none=True)
                losses=[]; t=time.monotonic()
                for b in range(0,len(group),tcfg['microbatch_size']):
                    ix=group[b:b+tcfg['microbatch_size']]
                    ids,mask,target,valid=collate([encoded_train[i] for i in ix],tok.pad_token_id)
                    loss=soft_target_ce(reader(ids,mask),target,valid)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                    (loss*(len(ix)/len(group))).backward(); losses.append((float(loss.detach()),len(ix)))
                grads=[v.grad for v in trainable if v.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads): raise RuntimeError('Missing/nonfinite LoRA gradients')
                if step==0 and not any(bool(g.abs().sum()>0) for g in grads): raise RuntimeError('All LoRA gradients are zero')
                norm=torch.nn.utils.clip_grad_norm_(trainable,tcfg['gradient_clip'])
                optimizer.step(); step+=1; cursor+=len(group); trained_queries+=len(group)
                torch.cuda.synchronize()
                log({'event':'train','step':step,'epoch':epoch,'cursor':cursor,'queries':len(group),
                    'loss':sum(v*n for v,n in losses)/len(group),'gradient_norm':float(norm),
                    'step_s':time.monotonic()-t,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                evaluate_now=bool(tcfg['eval_every_steps'] and step%tcfg['eval_every_steps']==0) or cursor==len(order) or (limit is not None and step==limit)
                if evaluate_now:
                    m=validate(f'validation.step-{step:06d}')
                    if key(m)>tuple(best['key']):
                        best={'key':key(m),'step':step,'adapter':str((args.out/'checkpoints'/f'step-{step:06d}'/'adapter').resolve()),
                              'metrics':m['mean'],'source':'validation_selected_lora',
                              'predictions':str((args.out/f'validation.step-{step:06d}.predictions.jsonl').resolve())}
                if evaluate_now or step%tcfg['save_every_steps']==0: save_checkpoint()
            if cursor>=len(order): epoch+=1; cursor=0
        if step and trainable_fingerprint(model)==before_hash and args.resume is None:
            raise RuntimeError('Trainable weights did not change')
        if best is None: raise RuntimeError('No validation result for checkpoint selection')
        # Copy the selected adapter so this run remains portable even when resumed elsewhere.
        if best['adapter']:
            shutil.copytree(best['adapter'],args.out/'selected_adapter')
            best['adapter']='selected_adapter'
        shutil.copyfile(best['predictions'],args.out/'selected_validation_predictions.jsonl')
        best['predictions']='selected_validation_predictions.jsonl'
        selection={'status':'completed_requested_schedule','model':cfg['model'],'selected':best,
            'config_sha256':digest(cfg),'data_sha256':digest(dm),'dataset_revision':dm['dataset']['revision'],
            'retriever':dm['retriever'],'test_used_for_selection':False,
            'adapter_files':{p.relative_to(args.out/'selected_adapter').as_posix():sha256_file(p) for p in (args.out/'selected_adapter').rglob('*') if p.is_file()} if best['adapter'] else {},
            'selection_rule':'validation Recall@5, then MRR@15; unadapted baseline may remain selected'}
        write_json(args.out/'selection.json',selection)
        write_json(args.out/'training_summary.json',{'steps':step,'train_query_exposures':trained_queries,
            'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'budget_warning':'THIS PROCESS DOES NOT STOP POD BILLING. Export evidence and terminate the exact experiment Pod.'})
        print(json.dumps(selection,indent=2))
    except BaseException as exc:
        # Accepted checkpoints are complete optimizer boundaries. Never save a half-accumulated gradient state.
        write_json(args.out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'step':step,
            'traceback':traceback.format_exc(),'accepted_training':False,
            'warning':'Pod may still be billing. Controller must export existing evidence then terminate it.'})
        raise

if __name__ == '__main__': main()
