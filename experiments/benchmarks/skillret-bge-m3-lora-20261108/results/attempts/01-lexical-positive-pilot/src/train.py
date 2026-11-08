"""CUDA LoRA: parity, real-data smoke, train-only mining, pilot and gated full run."""
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
from common import *
from engine import Monitor,public,utc,training_encoder
from objective import contrastive_loss,negative_ids

R=STUDY/'results'

def append(path,row):
    with path.open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())

def frozen_hash(model):
    """Hash every frozen tensor with bounded transfer buffers, not a RAM model copy."""
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:continue
        h.update(name.encode());flat=p.detach().reshape(-1)
        for start in range(0,flat.numel(),2**22):h.update(flat[start:start+2**22].cpu().numpy().tobytes())
    return h.hexdigest()

def inputs_for(encoder,texts,limits):
    tokens=encoder.tokenizer(texts,truncation=False,padding=True,return_tensors='pt',add_special_tokens=True)
    lengths=tokens['attention_mask'].sum(dim=1).tolist()
    assert all(n<=limit for n,limit in zip(lengths,limits)),(lengths,limits)
    return tokens.to('cuda:0')

def infer_batch(encoder,texts,limits):
    torch=encoder.torch
    with torch.inference_mode():
        output=encoder.flag.model(inputs_for(encoder,texts,limits),return_dense=True,return_sparse=False,return_colbert_vecs=False)['dense_vecs']
        return torch.nn.functional.normalize(output.float(),dim=-1)

def parity(encoder,peft):
    peft.eval();data=jsonl(CACHE/'data/queries.jsonl');archive=BASE/'vectors/dense-index-v1.npz'
    assert file_hash(archive)==read(BASE/'vectors/manifest.json')['sha256']
    with np.load(archive,allow_pickle=False) as vectors:
        ids=vectors['official_query_ids'].tolist();matrix=vectors['official_query_vectors']
    byid={row['id']:row for row in data}
    lengths=[len(encoder.tokenizer(q['query'],truncation=False)['input_ids']) for q in data]
    indices=sorted(set([0,1,int(np.argmax(lengths)),int(np.argmin(lengths))]));rows=[]
    for i in indices:
        q=byid[ids[i]];actual=infer_batch(encoder,[q['query']],[8192])[0].cpu().numpy();expected=matrix[i]
        error=float(np.max(np.abs(actual-expected)));cosine=float(np.dot(actual,expected))
        assert error<=3e-5 and cosine>=1-2e-6,(error,cosine)
        rows.append({'query_id':q['id'],'query_sha256':digest(q['query']),'maximum_absolute_error':error,'cosine':cosine})
    atomic_json(R/'baseline_parity.json',{'status':'passed','absolute_tolerance':3e-5,'minimum_cosine':1-2e-6,
        'reference':'Previously completed baseline lossless official-query vectors','fresh_reference_baseline_inference':False,
        'unmodified_peft_encoder_only':True,'adapter_zero_initialization':True,'queries':rows})

class Dataset:
    def __init__(self):
        self.skills={s['id']:s for s in jsonl(CACHE/'data/train-skills.jsonl')}
        self.queries={q['id']:q for q in jsonl(CACHE/'data/train-queries.jsonl')}
        self.documents=read(CACHE/'prepared/documents.json')
        self.examples=jsonl(CACHE/'prepared/examples.jsonl')
        self.train=[e for e in self.examples if e['split']=='train'];self.valid=[e for e in self.examples if e['split']=='validation']
        self.membership=read(STUDY/'manifests/split_membership.json')
        self.train_ids=self.membership['training_skill_ids'];self.val_ids=self.membership['validation_skill_ids']
        self.negatives={}

    def query(self,e):
        q=self.queries[e['query_id']]['query'];span=e['query_span'];assert digest(q)==e['query_sha256']
        return q[span['start']:span['end']]

    def passage(self,sid,index=0):
        doc=self.documents[sid];text=full_text(self.skills[sid]);span=doc['chunks'][index]
        part=text[span['start']:span['end']];assert digest(part)==span['text_sha256'];return part

    def triplet(self,e,visit,smoke=False):
        positive_index=(int(digest(e['query_id'])[:8],16)+visit//max(1,len(self.train)))%len(e['positives'])
        positive=e['positives'][positive_index];sid=positive['skill_id']
        if smoke:
            start=int(digest(e['query_id'])[:8],16)%len(self.train_ids)
            while self.train_ids[start] in e['all_positive_ids']:start=(start+1)%len(self.train_ids)
            negative=self.train_ids[start];kind='random'
        else:
            kind='hard' if visit%2 else 'random';negative=self.negatives[e['query_id']][kind+'_skill_id']
        assert negative not in e['all_positive_ids'] and negative in self.train_ids
        return [self.query(e),self.passage(sid,positive['chunk_index']),self.passage(negative)],kind

def freeze_protocol():
    protocol=read(BASE/'protocol.json')
    protocol.update({'experiment':'BGE-M3 LoRA on pinned SkillRet training split','baseline_git_revision':BASE_REV,
       'baseline_protocol_sha256':file_hash(BASE/'protocol.json'),'test_use':'Final evaluation only; overlap text audit with fixed quarantine permitted',
       'training':{'seed':42,'rank':8,'alpha':16,'lora_dropout':.05,'targets':'All 24 encoder layers attention query/value projections; verified 48 modules',
          'precision':'FP16 frozen backbone and activations; FP32 adapters and AdamW moments; dynamic gradient scaling',
          'query_max_tokens_including_specials':256,'passage_max_tokens_including_specials':384,
          'physical_query_groups':1,'passages_per_group':2,'negatives_per_group':1,'accumulation':16,
          'accumulation_adds_in_batch_negatives':False,'temperature':.05,'optimizer':'AdamW','learning_rate':1e-4,
          'weight_decay':.01,'gradient_clipping':1.,'gradient_checkpointing':'nonreentrant','maximum_epochs':2,
          'pilot_optimizer_steps':200,'validation_interval_optimizer_steps':100,'early_stopping_validation_patience':3,
          'warmup_optimizer_steps':20,'scheduler':'linear decay to zero over at most two eligible-query passes',
          'negative_mix':'Alternate train-only frozen-dense hard and deterministic random negatives, excluding every known positive',
          'positive_selection':'Train-only IDF-weighted lexical best of explicit source passages; no assumption all chunks positive',
          'validation_representation':'Fixed 128 training-derived heldout queries versus all heldout skills, first source passage <=384 tokens per skill; query <=256',
          'checkpoint_selection':'Highest validation NDCG@10 among trained checkpoints; tie earliest; official scores never select',
          'full_run_gate':'Pilot best NDCG@10 >= frozen validation NDCG@10 minus 0.01 and finite/stable optimizer steps',
          'smoke':'32 real eligible training queries, 64 updates, reset adapters and optimizer before main training',
          'dropout':'Pretrained encoder dropout retained during training; all dropout disabled for validation and final inference'},
       'training_source_manifest_sha256':file_hash(STUDY/'manifests/training_source.json'),
       'split_membership_sha256':file_hash(STUDY/'manifests/split_membership.json'),
       'positive_selection_sha256':file_hash(STUDY/'manifests/positive_passage_selection.jsonl'),
       'official_inference':'Exact baseline A/B/C source spans, raw queries, full 8192 capacity, max cosine and source-hash ties'})
    for config in CONFIGS:
        source=BASE/'results'/('documents-'+config+'.jsonl');destination=R/source.name
        if not destination.exists():destination.write_bytes(source.read_bytes())
    for name in ('candidate_pools.json','input_audit.json','query_audit.json'):
        target=R/name
        if not target.exists():target.write_bytes((BASE/'results'/name).read_bytes())
    path=STUDY/'protocol.json'
    if path.exists():assert read(path)==protocol,'Frozen protocol mismatch'
    else:atomic_json(path,protocol)
    return protocol

def step_loss(encoder,model,texts):
    torch=encoder.torch;tokens=inputs_for(encoder,texts,[256,384,384])
    cls=model(**tokens,return_dict=True).last_hidden_state[:,0].float()
    vectors=torch.nn.functional.normalize(cls,dim=-1)
    loss=contrastive_loss(vectors[:1],vectors[1:],torch.tensor([[True,False]],device='cuda'),temperature=.05)
    similarities=(vectors[0:1]@vectors[1:].T)[0]
    return loss,similarities,tokens['attention_mask'].sum().item()

def optimize(encoder,model,optimizer,scaler):
    torch=encoder.torch;scaler.unscale_(optimizer)
    params=[p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    norm=torch.nn.utils.clip_grad_norm_(params,1.0);previous=scaler.get_scale()
    scaler.step(optimizer);scaler.update();updated=scaler.get_scale()>=previous
    optimizer.zero_grad(set_to_none=True)
    assert torch.cuda.max_memory_reserved()<=7168*1024**2,'CUDA allocator reserve exceeded memory admission limit'
    encoder.monitor.check()
    return float(norm),updated

def smoke(encoder,model,data,initial):
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    torch=encoder.torch;examples=sorted(data.train,key=lambda e:e['query_id'])[:32]
    model.eval()
    def separation():
        rows=[]
        for e in examples:
            texts,_=data.triplet(e,0,smoke=True);v=infer_batch(encoder,texts,[256,384,384]);s=(v[:1]@v[1:].T)[0]
            rows.append({'query_id':e['query_id'],'positive':float(s[0]),'negative':float(s[1])})
        return rows
    before=separation();base_before=frozen_hash(model)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=200)
    model.train();rows=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for i in range(64):
        encoder.monitor.phase='training:smoke';encoder.monitor.check()
        texts,_=data.triplet(examples[i%32],0,smoke=True);loss,sim,tokens=step_loss(encoder,model,texts)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        adapter_grads=[p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in adapter_grads) and any(g.abs().sum()>0 for g in adapter_grads)
        norm,updated=optimize(encoder,model,optimizer,scaler)
        rows.append({'step':i+1,'loss':float(loss.detach()),'positive_cosine':float(sim[0].detach()),
          'negative_cosine':float(sim[1].detach()),'gradient_norm':norm,'optimizer_updated':updated,'gradient_scale':scaler.get_scale()})
        append(R/'smoke_steps.jsonl',rows[-1])
        if i==0:
            atomic_json(R/'training_feasibility.json',{'status':'passed' if updated and math.isfinite(norm) else 'failed',
                'physical_query_groups':1,'encoded_sequences':3,'forward_backward_optimizer_step':updated,
                'finite_loss':True,'finite_gradient_norm':math.isfinite(norm),'frozen_backbone_gradients_none':True,
                'adapter_gradients_present':True,'gradient_norm':norm,'loss':float(loss.detach()),
                'query_max_tokens':256,'document_max_tokens':384,'input_tokens_in_test':tokens,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'no_cpu_fallback':True})
    torch.cuda.synchronize();wall=time.perf_counter()-started;model.eval();after=separation()
    base_after=frozen_hash(model);assert base_before==base_after
    changed=any(not torch.equal(initial[k].to(v.device),v) for k,v in get_peft_model_state_dict(model).items())
    assert changed and all(row['optimizer_updated'] for row in rows)
    before_gap=np.mean([r['positive']-r['negative'] for r in before]);after_gap=np.mean([r['positive']-r['negative'] for r in after])
    assert after_gap>before_gap,'Functional smoke did not increase fixed-set cosine separation'
    atomic_json(R/'smoke.json',{'status':'passed','optimizer_steps':64,'real_training_queries':32,'wall_s':wall,
       'training_rows':rows,'before':before,'after':after,'mean_separation_before':float(before_gap),'mean_separation_after':float(after_gap),
       'frozen_parameter_sha256_before':base_before,'frozen_parameter_sha256_after':base_after,'adapter_updated':changed,
       'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
       'reset_before_main_training':True,'generalization_evidence':False})
    set_peft_model_state_dict(model,initial);del optimizer,scaler;torch.cuda.empty_cache()
    return base_before

def mine(encoder,model,data):
    torch=encoder.torch;output=CACHE/'prepared/negatives.jsonl'
    if output.exists():
        data.negatives={r['query_id']:r for r in jsonl(output)}
        assert set(data.negatives)=={e['query_id'] for e in data.train};return
    encoder.monitor.phase='mining:training_only';model.eval();started=time.perf_counter();vectors=[]
    with model.disable_adapter():
        for start in range(0,len(data.train_ids),8):
            batch=data.train_ids[start:start+8];encoder.monitor.check()
            vectors.append(infer_batch(encoder,[data.passage(sid) for sid in batch],[384]*len(batch)))
        matrix=torch.cat(vectors);del vectors
        ids=np.asarray(data.train_ids);tie=np.asarray([digest(sid) for sid in data.train_ids]);rows=[]
        for start in range(0,len(data.train),8):
            batch=data.train[start:start+8];encoder.monitor.check()
            q=infer_batch(encoder,[data.query(e) for e in batch],[256]*len(batch))
            scores=(q@matrix.T).cpu().numpy()
            for e,score in zip(batch,scores):
                order=np.lexsort((tie,-score));positives=set(e['all_positive_ids'])
                hard=next(str(ids[i]) for i in order if ids[i] not in positives)
                index=int(digest({'seed':42,'query':e['query_id'],'kind':'random'})[:12],16)%len(ids)
                while ids[index] in positives:index=(index+1)%len(ids)
                random_id=str(ids[index]);assert hard not in positives and random_id not in positives
                row={'query_id':e['query_id'],'random_skill_id':random_id,'hard_skill_id':hard,
                     'hard_score':float(score[np.where(ids==hard)[0][0]]),
                     'random_source_sha256':data.documents[random_id]['source_sha256'],
                     'hard_source_sha256':data.documents[hard]['source_sha256'],'known_positive_ids':e['all_positive_ids']}
                rows.append(row)
            if start%1000==0:print('MINING',start,'/',len(data.train),flush=True)
        del matrix
    with output.open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    (STUDY/'manifests/negative_selection.jsonl').write_bytes(output.read_bytes())
    data.negatives={r['query_id']:r for r in rows}
    atomic_json(R/'mining.json',{'queries':len(rows),'candidate_skills':len(ids),'batch_size':8,
       'frozen_base':True,'training_only':True,'known_positives_filtered':True,'wall_s':time.perf_counter()-started,
       'query_max_tokens':256,'representative_document_max_tokens':384,'all_examples_have_hard_and_random_negative':True,
       'negative_manifest_sha256':file_hash(output),'not_an_official_baseline_rerun':True})

def validate(encoder,model,data,step):
    torch=encoder.torch;encoder.monitor.phase='validation:training_derived';model.eval();started=time.perf_counter();vectors=[]
    with torch.inference_mode():
        for start in range(0,len(data.val_ids),8):
            batch=data.val_ids[start:start+8];encoder.monitor.check()
            vectors.append(infer_batch(encoder,[data.passage(sid) for sid in batch],[384]*len(batch)))
        matrix=torch.cat(vectors);rankings={};gold={};ties=np.asarray([data.documents[sid]['source_sha256'] for sid in data.val_ids])
        for start in range(0,len(data.valid),8):
            batch=data.valid[start:start+8];v=infer_batch(encoder,[data.query(e) for e in batch],[256]*len(batch))
            scores=(v@matrix.T).cpu().numpy()
            for e,score in zip(batch,scores):
                order=np.lexsort((ties,-score))[:50];qid=e['query_id'];rankings[qid]=[data.val_ids[i] for i in order]
                gold[qid]={sid:1 for sid in e['all_positive_ids']}
        del matrix,v,vectors
    metrics=evaluate(gold,rankings)
    row={'optimizer_step':step,'wall_s':time.perf_counter()-started,'candidate_skills':len(data.val_ids),
         'queries':len(data.valid),'metrics':metrics['aggregate'],'utc':utc()}
    atomic_json(R/f'validation-step{step:05d}.json',{'summary':row,'evaluation':metrics,'rankings':rankings})
    append(R/'validation_curve.jsonl',row);print('VALIDATION',step,row['metrics']['NDCG@10'],flush=True)
    return row

def save_checkpoint(model,optimizer,scheduler,scaler,step,groups,epoch):
    import torch
    folder=CACHE/'checkpoints'/f'step-{step:05d}';folder.mkdir(parents=True,exist_ok=True)
    model.peft_config['default'].base_model_name_or_path='BAAI/bge-m3'
    model.peft_config['default'].revision=MODEL_REV
    model.save_pretrained(folder,safe_serialization=True)
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
       'step':step,'groups':groups,'epoch':epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
       'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},folder/'recovery.pt')
    return folder

def main(resume=None):
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    assert not (R/'training_summary.json').exists(),'Training already completed; use its selected adapter'
    if resume is None:assert not (R/'training_curve.jsonl').exists(),'Existing main-training journal: recover a checkpoint rather than restarting'
    protocol=freeze_protocol();data=Dataset();monitor=Monitor('training:loading');began=time.perf_counter()
    sources={str(p.relative_to(STUDY)):file_hash(p) for p in sorted((STUDY/'src').glob('*.py'))}
    atomic_json(R/('training_execution_source-'+digest(sources)[:16]+'.json'),sources)
    atomic_json(R/'training_execution_source.json',sources)
    (STUDY/'runtime-lock.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True))
    encoder=None;main_started=None
    try:
        encoder,model=training_encoder(monitor);torch=encoder.torch
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        if resume is None:
            parity(encoder,model)
            initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
            frozen_before=smoke(encoder,model,data,initial)
        else:
            from safetensors.torch import load_file
            resume=Path(resume).resolve();assert resume.parent==(CACHE/'checkpoints').resolve()
            state=torch.load(resume/'recovery.pt',map_location='cpu',weights_only=False)
            set_peft_model_state_dict(model,load_file(str(resume/'adapter_model.safetensors')))
            frozen_before=read(R/'smoke.json')['frozen_parameter_sha256_before']
            assert frozen_hash(model)==frozen_before
            attempt=R/'attempts'/('resume-step-'+str(state['step'])+'-'+str(time.time_ns()));attempt.mkdir(parents=True)
            for name in ('training_curve.jsonl','validation_curve.jsonl','training_failure.json'):
                path=R/name
                if path.exists():
                    (attempt/name).write_bytes(path.read_bytes())
                    if name.endswith('.jsonl'):
                        rows=[r for r in jsonl(path) if r['optimizer_step']<=state['step']]
                        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        mine(encoder,model,data)
        frozen_validation=validate(encoder,model,data,0) if resume is None else read(R/'validation-step00000.json')['summary']
        baseline_ndcg=frozen_validation['metrics']['NDCG@10']
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        planned=math.ceil(2*len(data.train)/16)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/20,1.)*max(0.,(planned-step)/max(1,planned-20)))
        scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=200)
        best=-math.inf;best_step=None;best_folder=None;stale=0;step=0;groups=0;overflows=0;stop_reason='maximum_two_epochs'
        tokens_total=0;full_gate=None;final_folder=None
        if resume is not None:
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
            step,groups=state['step'],state['groups'];random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            trained=[r for r in jsonl(R/'validation_curve.jsonl') if r['optimizer_step']>0]
            if trained:
                winner=max(trained,key=lambda r:(r['metrics']['NDCG@10'],-r['optimizer_step']))
                best=winner['metrics']['NDCG@10'];best_step=winner['optimizer_step'];best_folder=CACHE/'checkpoints'/f'step-{best_step:05d}'
                stale=sum(r['optimizer_step']>best_step for r in trained)
            tokens_total=sum(r['input_tokens'] for r in jsonl(R/'training_curve.jsonl'))
            overflows=len(jsonl(R/'overflow_attempts.jsonl')) if (R/'overflow_attempts.jsonl').exists() else 0
            full_gate=read(R/'pilot_gate.json')['proceed_to_full'] if (R/'pilot_gate.json').exists() else None
        main_started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        start_epoch=groups//len(data.train);start_position=groups%len(data.train)
        for epoch in range(start_epoch,2):
            order=list(range(len(data.train)));random.Random(42+epoch).shuffle(order);model.train()
            losses=[];gaps=[];cycle_started=time.perf_counter();pending_tokens=0
            for position,index in enumerate(order):
                if epoch==start_epoch and position<start_position:continue
                monitor.phase='training:pilot' if step<200 else 'training:full';monitor.check()
                texts,kind=data.triplet(data.train[index],groups)
                loss,sim,tokens=step_loss(encoder,model,texts);assert torch.isfinite(loss)
                scaler.scale(loss/16).backward();losses.append(float(loss.detach()));gaps.append(float((sim[0]-sim[1]).detach()))
                groups+=1;pending_tokens+=tokens
                if groups%16 and position+1<len(order):continue
                if len(losses)<16:
                    for p in model.parameters():
                        if p.grad is not None:p.grad.mul_(16/len(losses))
                norm,updated=optimize(encoder,model,optimizer,scaler)
                if not updated:
                    overflows+=1;assert overflows<=3,'Repeated FP16 gradient overflow'
                    append(R/'overflow_attempts.jsonl',{'groups':groups,'scale':scaler.get_scale(),'gradient_norm':norm if math.isfinite(norm) else None,'finite':math.isfinite(norm)})
                    losses=[];gaps=[];pending_tokens=0;cycle_started=time.perf_counter();continue
                step+=1;scheduler.step();tokens_total+=pending_tokens
                elapsed=time.perf_counter()-cycle_started
                append(R/'training_curve.jsonl',{'optimizer_step':step,'epoch':epoch+1,'groups_seen':groups,
                   'loss':float(np.mean(losses)),'mean_cosine_separation':float(np.mean(gaps)),
                   'gradient_norm':norm,'learning_rate':optimizer.param_groups[0]['lr'],'gradient_scale':scaler.get_scale(),
                   'step_wall_s':elapsed,'query_groups_this_step':len(losses),'input_tokens':pending_tokens,
                   'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'utc':utc()})
                losses=[];gaps=[];pending_tokens=0
                if step%10==0:print('TRAIN',step,'groups',groups,'loss',float(loss.detach()),flush=True)
                if step%100==0:
                    result=validate(encoder,model,data,step);value=result['metrics']['NDCG@10']
                    final_folder=save_checkpoint(model,optimizer,scheduler,scaler,step,groups,epoch)
                    if value>best+1e-12:best=value;best_step=step;best_folder=final_folder;stale=0
                    else:stale+=1
                    if step==200:
                        full_gate=best>=baseline_ndcg-.01
                        atomic_json(R/'pilot_gate.json',{'completed_optimizer_steps':step,'frozen_validation_ndcg10':baseline_ndcg,
                            'best_trained_validation_ndcg10':best,'maximum_tolerated_pilot_drop':.01,'proceed_to_full':full_gate,
                            'test_scores_used':False})
                        if not full_gate:stop_reason='pilot_validation_gate_failed';break
                    if step>=200 and stale>=3:stop_reason='validation_early_stopping';break
                    model.train()
                cycle_started=time.perf_counter()
            if stop_reason!='maximum_two_epochs':break
        if step%100:
            result=validate(encoder,model,data,step);final_folder=save_checkpoint(model,optimizer,scheduler,scaler,step,groups,epoch)
            if result['metrics']['NDCG@10']>best:best=result['metrics']['NDCG@10'];best_step=step;best_folder=final_folder
        frozen_after=frozen_hash(model);assert frozen_before==frozen_after
        assert best_folder and final_folder
        selection={'adapter_relative_path':str(best_folder.relative_to(CACHE)),'optimizer_step':best_step,
           'adapter_sha256':file_hash(best_folder/'adapter_model.safetensors'),'validation_ndcg10':best,
           'validation_candidate_manifest_sha256':file_hash(STUDY/'manifests/split_membership.json'),
           'selection_metric':'training-derived validation NDCG@10','official_test_inspected':False}
        atomic_json(CACHE/'selection.json',selection);atomic_json(R/'selected_adapter.json',selection)
        invocation_wall=time.perf_counter()-main_started
        append(R/'training_invocations.jsonl',{'status':'completed','main_wall_s':invocation_wall,'resumed':resume is not None,'utc':utc()})
        wall=sum(r['main_wall_s'] for r in jsonl(R/'training_invocations.jsonl'))
        active_wall=sum(r['step_wall_s'] for r in jsonl(R/'training_curve.jsonl'))
        atomic_json(R/'training_summary.json',{'status':'completed','main_optimizer_steps':step,'main_query_groups':groups,
           'eligible_training_queries':len(data.train),'maximum_epochs':2,'effective_query_passes':groups/len(data.train),
           'stop_reason':stop_reason,'full_training_gate_passed':full_gate,'main_wall_s':wall,
           'pipeline_wall_s':time.perf_counter()-began,'optimizer_steps_per_second':step/wall,'query_groups_per_second':groups/wall,
           'active_input_tokens':tokens_total,'input_tokens_per_main_wall_second':tokens_total/wall,'gradient_overflow_attempts':overflows,
           'active_optimizer_group_wall_s':active_wall,'input_tokens_per_active_group_second':tokens_total/active_wall,
           'resumed_invocation':resume is not None,'main_wall_s_this_invocation':invocation_wall,
           'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
           'best_step':best_step,'best_validation_ndcg10':best,'frozen_validation_ndcg10':baseline_ndcg,
           'frozen_parameter_sha256_before':frozen_before,'frozen_parameter_sha256_after':frozen_after,
           'final_adapter_relative_path':str(final_folder.relative_to(CACHE)),'adapter_bytes':(best_folder/'adapter_model.safetensors').stat().st_size,
           'official_test_used_for_selection':False,'smoke_updates_discarded':64})
        print('TRAINING_COMPLETE',step,'best',best_step,best,'stop',stop_reason,flush=True)
    except Exception as error:
        if main_started is not None:append(R/'training_invocations.jsonl',{'status':'interrupted_or_failed','main_wall_s':time.perf_counter()-main_started,'resumed':resume is not None,'utc':utc()})
        atomic_json(R/'training_failure.json',{'exception':type(error).__name__,'message':public(str(error)),
          'phase':monitor.phase,'utc':utc(),'wall_s':time.perf_counter()-began})
        raise
    finally:monitor.close()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path)
    main(parser.parse_args().resume)
