#!/usr/bin/env python3
"""Run only on RunPod. Staged/resumable official MetaEncoder evaluation."""
import argparse, csv, hashlib, importlib.util, json, os, platform, random, subprocess, sys, time, traceback
from pathlib import Path
MODEL='facebook/meta-encoder-v1'
MODEL_REV='ed04ebd4defb8e837b5eaea164143177272696ac'
DATASET='ThakiCloud/SKILLRET'
DATA_REV='6583d7d2ed07644d0fb8938ed8178f3a7dc42a12'
DOC_INST='Represent the document for retrieval.'
QUERY_INST='Represent the query for retrieving relevant documents.'
SEED=42

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def rec(root,event,**kw):
    p=root/'results/events.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps({'time':time.time(),'event':event,**kw})+'\n')
def setup(root):
    from huggingface_hub import hf_hub_download,snapshot_download
    t=time.time();paths={}
    for kind in ['skills','queries','qrels']:
        p=hf_hub_download(DATASET,f'data/{kind}/test.jsonl',repo_type='dataset',revision=DATA_REV)
        paths[kind]=p
    dump(root/'manifests/dataset_files.json',{k:{'path':p,'sha256':sha(p)} for k,p in paths.items()})
    dump(root/'manifests/dataset_paths.json',paths)
    rec(root,'dataset_download',seconds=time.time()-t)
    t=time.time();p=str(root/"model")
    dump(root/'manifests/model.json',{'id':MODEL,'revision':MODEL_REV,'path':p})
    rec(root,'model_download',seconds=time.time()-t)
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=open(root/'manifests/pip-freeze.txt','w'),check=True)
    subprocess.run(['nvidia-smi'],stdout=open(root/'manifests/nvidia-smi.txt','w'),check=True)
    dump(root/'manifests/runtime.json',{'platform':platform.platform(),'python':sys.version,'cpu_count':os.cpu_count(),'start_time':time.time()})
    dump(root/'manifests/config.json',{'model':MODEL,'model_revision':MODEL_REV,'dataset':DATASET,'dataset_revision':DATA_REV,'document_instruction':DOC_INST,'query_instruction':QUERY_INST,'seed':SEED,'chunk_protocol':'Nonoverlapping character substrings bounded by exact formatted token count <=16384. L2 normalize mean of chunk vectors. All characters preserved. Separate truncated-first-chunk arm.','tie_break':'descending cosine; stable original catalog row order'})

def load_data(root):
    paths=json.loads((root/'manifests/dataset_paths.json').read_text())
    return {k:lines(p) for k,p in paths.items()}

def model_load(root):
    import torch
    p=json.loads((root/'manifests/model.json').read_text())['path'];sys.path.insert(0,p)
    from modeling_metaencoder import MetaEncoder
    t=time.time();m=MetaEncoder.from_pretrained(p,device_map='cuda',dtype=torch.bfloat16,attn_implementation='sdpa')
    torch.cuda.synchronize();rec(root,'model_load',seconds=time.time()-t,dimension=m.embedding_dim,peak_memory=torch.cuda.max_memory_allocated())
    return m

def token_count(m,text,instruction):
    msg=([{'role':'system','content':instruction}] if instruction else [])+[{'role':'user','content':[{'type':'text','text':text}]}]
    rendered=m.processor.apply_chat_template(msg,tokenize=False)
    return len(m.processor.tokenizer(rendered,add_special_tokens=False)['input_ids'])

def chunks(m,text,instruction,limit):
    # Exact original character slices, no token decode loss. Bisection only for oversized texts.
    if token_count(m,text,instruction)<=limit:return [text]
    out=[];pos=0
    while pos<len(text):
        lo=1;hi=len(text)-pos;best=0
        while lo<=hi:
            mid=(lo+hi)//2
            if token_count(m,text[pos:pos+mid],instruction)<=limit:best=mid;lo=mid+1
            else:hi=mid-1
        if not best:raise ValueError('Instruction exceeds token limit')
        out.append(text[pos:pos+best]);pos+=best
    assert ''.join(out)==text
    return out

def encode_stage(root,m,ids,texts,kind,instruction,limit=16384,batch=1,max_seconds=None):
    import numpy as np,torch
    out=root/'artifacts';out.mkdir(exist_ok=True);ck=root/'checkpoints'/kind;ck.mkdir(parents=True,exist_ok=True)
    start=time.time();audit=[]
    for i,(ident,text) in enumerate(zip(ids,texts)):
        p=ck/f'{i:06d}.npz';meta=ck/f'{i:06d}.json'
        if p.exists() and meta.exists():
            a=np.load(p);assert a['aggregate'].shape==(m.embedding_dim,) and np.isfinite(a['aggregate']).all();saved=json.loads(meta.read_text());assert saved['id']==ident and saved['input_sha256']==hashlib.sha256(text.encode()).hexdigest();audit.append(saved);continue
        if max_seconds is not None and time.time()-start>max_seconds:break
        n=token_count(m,text,instruction);cs=chunks(m,text,instruction,limit)
        t=time.time();torch.cuda.reset_peak_memory_stats()
        try:
            v=m.encode(cs,instruction=instruction,batch_size=batch);torch.cuda.synchronize()
            arr=v.detach().cpu().float().numpy();avg=arr.mean(0);avg/=np.linalg.norm(avg)
            assert arr.shape==(len(cs),m.embedding_dim) and np.isfinite(arr).all()
            assert np.allclose(np.linalg.norm(arr,axis=1),1,atol=1e-4)
            tmp=ck/f'{i:06d}.tmp.npz';np.savez(tmp,chunks=arr,aggregate=avg.astype('float32'),truncated=arr[0]);tmp.replace(p)
            row={'row':i,'id':ident,'input_sha256':hashlib.sha256(text.encode()).hexdigest(),'tokens':n,'chunk_tokens':[token_count(m,c,instruction) for c in cs],'chunk_count':len(cs),'seconds':time.time()-t,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'status':'ok'}
        except Exception as e:
            row={'row':i,'id':ident,'status':'failed','error':str(e),'seconds':time.time()-t};rec(root,'embedding_failure',**row);raise
        dump(meta,row);audit.append(row)
        if (i+1)%10==0:rec(root,'embedding_progress',kind=kind,completed=i+1,total=len(ids),seconds=time.time()-start)
    dump(root/f'manifests/{kind}_token_audit.json',audit)
    if len(audit)==len(ids):
        arr=np.stack([np.load(ck/f'{i:06d}.npz')['aggregate'] for i in range(len(ids))]).astype('float32')
        trunc=np.stack([np.load(ck/f'{i:06d}.npz')['truncated'] for i in range(len(ids))]).astype('float32')
        chunk=np.concatenate([np.load(ck/f'{i:06d}.npz')['chunks'] for i in range(len(ids))]).astype('float32')
        np.save(out/f'{kind}.npy',arr);np.save(out/f'{kind}_truncated.npy',trunc);np.save(out/f'{kind}_chunks.npy',chunk)
        dump(out/f'{kind}_ids.json',ids)
        offset=0;mapping=[]
        for row in audit:mapping.append({'id':row['id'],'offset':offset,'count':row['chunk_count']});offset+=row['chunk_count']
        dump(out/f'{kind}_chunks_mapping.json',mapping)
        rec(root,'embedding_complete',kind=kind,count=len(ids),shape=list(arr.shape),seconds=sum(r['seconds'] for r in audit))
    return len(audit)

def validate_data(root,data):
    expected={'skills':6006,'queries':4392,'qrels':7187}
    hashes={'skills':'1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81','queries':'4d07a35f40e597f8468c6dab480c95a1f9b40ecbde80274081ba2e7c643d0f6d','qrels':'edbfae0b8109de460d54e27e6235b1b47316f34956744db0571ad1b4ddb860af'}
    paths=json.loads((root/'manifests/dataset_paths.json').read_text())
    for k,n in expected.items():assert len(data[k])==n,(k,len(data[k]));assert sha(paths[k])==hashes[k],k
    ss=[str(x['id']) for x in data['skills']];qs=[str(x['id']) for x in data['queries']]
    assert len(set(ss))==len(ss) and len(set(qs))==len(qs)
    pairs=[(str(x['query_id']),str(x['skill_id'])) for x in data['qrels']]
    assert len(set(pairs))==7187 and all(q in set(qs) and s in set(ss) for q,s in pairs)
    assert all(x['relevance']>0 for x in data['qrels'])
    assert set(q for q,s in pairs)==set(qs)
    dump(root/'manifests/dataset_validation.json',{'counts':expected,'sha256':hashes,'unique_ids_and_pairs':True})

def texts(data):
    return [str(x['id']) for x in data['skills']],[' | '.join(x[k].strip() for k in ('name','description','skill_md')) for x in data['skills']],[str(x['id']) for x in data['queries']],[x['query'] for x in data['queries']]

def labels_for(data):
    labels={str(x['id']):{} for x in data['queries']}
    for x in data['qrels']:labels[str(x['query_id'])][str(x['skill_id'])]=x['relevance']
    return labels

def evaluate_rankings(root,name,labels,rankings):
    from metrics_existing import evaluate
    result=evaluate(labels,rankings)
    for k in (20,50):
        for row in result['per_query'].values():row[f'Recall@{k}']=row[f'Coverage@{k}']
        result['aggregate'][f'Recall@{k}']=result['aggregate'][f'Coverage@{k}']
    result['by_required_skills']={str(n):evaluate({q:g for q,g in labels.items() if len(g)==n},rankings)['aggregate'] for n in (1,2,3) if any(len(g)==n for g in labels.values())}
    dump(root/f'results/{name}_metrics.json',result)
    return result

def retrieval(root,data):
    import numpy as np
    sid,st,qid,qt=texts(data);labels=labels_for(data)
    d=np.load(root/'artifacts/documents.npy');q=np.load(root/'artifacts/queries.npy')
    assert d.shape==(6006,6656) and q.shape==(4392,6656)
    summary=[]
    for variant in ['full','truncated']:
        dd=d if variant=='full' else np.load(root/'artifacts/documents_truncated.npy')
        for dim in ([256,512,1024,2048,4096,6656] if variant=='full' else [6656]):
            ds=dd[:,:dim].copy();qs=q[:,:dim].copy();ds/=np.linalg.norm(ds,axis=1,keepdims=True);qs/=np.linalg.norm(qs,axis=1,keepdims=True)
            t=time.perf_counter();sim=qs@ds.T;search=time.perf_counter()-t
            order=np.argsort(-sim,axis=1,kind='stable')[:,:100]
            name=f'{variant}_{dim}';rankings={ident:[sid[j] for j in order[i]] for i,ident in enumerate(qid)}
            result=evaluate_rankings(root,name,labels,rankings)
            with (root/f'results/{name}_top100.jsonl').open('w') as f, (root/f'results/{name}_top50.jsonl').open('w') as f50:
                for i,ident in enumerate(qid):
                    hits=[{'skill_id':sid[j],'score':float(sim[i,j])} for j in order[i]]
                    f.write(json.dumps({'query_id':ident,'hits':hits})+'\n');f50.write(json.dumps({'query_id':ident,'hits':hits[:50]})+'\n')
            np.save(root/f'artifacts/documents_{name}.npy',ds);np.save(root/f'artifacts/queries_{name}.npy',qs)
            summary.append({'configuration':name,'dimension':dim,'query_count':len(qid),'search_seconds':search,'storage_bytes':ds.nbytes+qs.nbytes,'official_dimension_feature':dim==6656,**result['aggregate']})
    dump(root/'results/retrieval_summary.json',summary)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['setup','validate','smoke','documents','queries','retrieval','native','decisions','figures','manifest','compare']);p.add_argument('--root',type=Path,required=True);p.add_argument('--batch',type=int,default=1);p.add_argument('--max-seconds',type=float);p.add_argument('--limit',type=int,default=200);a=p.parse_args();r=a.root;r.mkdir(parents=True,exist_ok=True)
    if a.stage=='setup':setup(r);validate_data(r,load_data(r));return
    data=load_data(r);validate_data(r,data);sid,st,qid,qt=texts(data)
    if a.stage=='validate':return
    if a.stage=='retrieval':retrieval(r,data);return
    if a.stage=='compare':compare(r,data);return
    if a.stage=='manifest':
        dump(r/'manifests/artifacts.json',{str(f.relative_to(r)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in (r/'artifacts').glob('*') if f.is_file()});return
    if a.stage=='figures':figures(r);return
    m=model_load(r)
    if a.stage=='smoke':
        encode_stage(r,m,sid[:2],st[:2],'smoke_documents',DOC_INST,batch=a.batch,max_seconds=a.max_seconds);encode_stage(r,m,qid[:2],qt[:2],'smoke_queries',QUERY_INST,batch=a.batch,max_seconds=a.max_seconds)
    elif a.stage in ('documents','queries'):
        isdoc=a.stage=='documents';encode_stage(r,m,sid if isdoc else qid,st if isdoc else qt,a.stage,DOC_INST if isdoc else QUERY_INST,batch=a.batch,max_seconds=a.max_seconds)
    elif a.stage=='native':native(r,data,m,a.limit)
    elif a.stage=='decisions':decisions(r,data,m,a.limit,a.max_seconds)

def native(root,data,m,limit):
    import numpy as np,torch
    sid,st,qid,qt=texts(data);d=np.load(root/'artifacts/documents.npy');q=np.load(root/'artifacts/queries.npy');corpus=torch.from_numpy(d).to(m.model.device);rows=[];rankings={}
    for i in range(min(limit,len(qid))):
        t=time.perf_counter();hits=m.retrieve(qt[i],corpus,top_k=100,query_instruction=QUERY_INST);torch.cuda.synchronize();elapsed=time.perf_counter()-t
        manual=np.argsort(-(q[i]@d.T),kind='stable')[:100].tolist();actual=[h.index for h in hits]
        rows.append({'query_id':qid[i],'seconds':elapsed,'same_top100_order':actual==manual,'same_top100_set':set(actual)==set(manual),'hits':[{'skill_id':sid[h.index],'score':h.score} for h in hits]});rankings[qid[i]]=[sid[j] for j in actual]
    dump(root/'results/native_retrieve.json',{'query_count':len(rows),'includes_query_encoding':True,'corpus_cached':True,'rows':rows,'p50_seconds':float(np.quantile([x['seconds'] for x in rows],.5)),'p95_seconds':float(np.quantile([x['seconds'] for x in rows],.95))});evaluate_rankings(root,'native_retrieve',{k:v for k,v in labels_for(data).items() if k in rankings},rankings)

def decisions(root,data,m,limit,max_seconds):
    import numpy as np
    from metrics_existing import deterministic_subset
    sid,st,qid,qt=texts(data);docs=dict(zip(sid,st));queries=dict(zip(qid,qt));labels=labels_for(data)
    ranks={x['query_id']:[h['skill_id'] for h in x['hits']] for x in lines(root/'results/full_6656_top100.jsonl')}
    chosen=deterministic_subset(qid,limit);dump(root/'manifests/decision_query_ids.json',chosen)
    questions={'binary':{'type':'noul','instructions':'Given the user request and this candidate skill, determine whether the skill would help an agent fulfill the request.'},'ordinal':{'type':'score','instructions':'Rate how useful the candidate skill is for fulfilling the user request.','criteria':['Not relevant','Weakly related','Partially useful','Relevant and useful','Strongly relevant to fulfilling the request']}}
    dump(root/'manifests/decision_questions.json',questions);start=time.time();ck=root/'checkpoints/decisions';ck.mkdir(parents=True,exist_ok=True);completed=[]
    for qi in chosen:
        target=ck/(hashlib.sha256(qi.encode()).hexdigest()+'.json')
        if target.exists():completed.append(json.loads(target.read_text()));continue
        records=[]
        for si in ranks[qi][:50]:
            if max_seconds and time.time()-start>max_seconds:break
            state={'request':queries[qi],'candidate_skill':docs[si]};t=time.perf_counter()
            # Official decision API has no truncation. Retain full text, record failures honestly.
            try:out=m.decide(state,questions);records.append({'skill_id':si,'seconds':time.perf_counter()-t,'output':out,'status':'ok'})
            except Exception as e:records.append({'skill_id':si,'seconds':time.perf_counter()-t,'status':'failed','error':str(e)})
        row={'query_id':qi,'candidates':records,'choice':{}}
        if len(records)==50:
            for k in (5,10,20):
                if max_seconds and time.time()-start>max_seconds:break
                spec={'choose':{'type':'choice','instructions':'Choose the skill most relevant and useful for fulfilling the user request.','criteria':{s:docs[s] for s in ranks[qi][:k]}}};t=time.perf_counter()
                try:row['choice'][str(k)]={'output':m.decide(queries[qi],spec),'seconds':time.perf_counter()-t}
                except Exception as e:row['choice'][str(k)]={'error':str(e),'seconds':time.perf_counter()-t}
        dump(target,row);completed.append(row)
        if max_seconds and time.time()-start>max_seconds:break
    # Preserve the intended pilot denominator including unfinished/missing queries.
    pilot_labels={q:labels[q] for q in chosen};summary=[]
    for mode in ('binary','ordinal'):
        for k in (10,20,50):
            rankings={};tp=fp=fn=0
            for row in completed:
                rr=[x for x in row['candidates'][:k] if x['status']=='ok']
                score=lambda x:float(x['output']['questions'][mode]['probabilities']['true']) if mode=='binary' else float(x['output']['questions'][mode]['value'])
                rankings[row['query_id']]=[x['skill_id'] for x in sorted(rr,key=score,reverse=True)]
                if mode=='binary':
                    for x in rr:
                        yes=x['output']['questions']['binary']['value'];gold=x['skill_id'] in labels[row['query_id']];tp+=bool(yes and gold);fp+=bool(yes and not gold);fn+=bool(not yes and gold)
            result=evaluate_rankings(root,f'decide_{mode}_{k}',pilot_labels,rankings)
            summary.append({'mode':mode,'candidate_count':k,'intended_queries':len(chosen),'attempted_queries':len(completed),'metrics':result['aggregate'],'classification_precision':tp/(tp+fp) if tp+fp else None,'classification_recall':tp/(tp+fn) if tp+fn else None,'classification_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'classification_scope':'successfully scored candidate pairs only; retrieval coverage separately'})
    dump(root/'results/decision_summary.json',summary)
    choice=[]
    for k in (5,10,20):
        for n in (1,2,3):
            subset=[q for q in chosen if len(labels[q])==n];good=covered=attempted=0
            for q in subset:
                covered+=any(s in labels[q] for s in ranks[q][:k]);row=next((r for r in completed if r['query_id']==q),{});answer=row.get('choice',{}).get(str(k),{}).get('output',{}).get('questions',{}).get('choose')
                if answer:attempted+=1;good+=answer['value'] in labels[q]
            choice.append({'candidate_count':k,'required_skills':n,'denominator':len(subset),'attempted':attempted,'top1_accuracy':good/len(subset) if subset else None,'conditional_accuracy':good/covered if covered else None,'covered_queries':covered})
    dump(root/'results/choice_summary.json',choice)
    # Secondary calibration uses first half of fixed hash-ordered pilot; held-out remainder only.
    fitids=set(chosen[:len(chosen)//2]);fit=[];fl=[];held=[]
    for row in completed:
        for x in row['candidates']:
            if x['status']!='ok':continue
            a=x['output']['questions']['binary'];y=x['skill_id'] in labels[row['query_id']]
            if row['query_id'] in fitids:fit.append(a);fl.append(y)
            else:held.append((a,y))
    if fit and held:
        old=m.temperature;temp=m.calibrate(fit,fl);res={'fit_query_ids':sorted(fitids),'fit_pairs':len(fit),'heldout_pairs':len(held),'temperature':temp,'original_temperature':old,'heldout':{}}
        for label,t in [('uncalibrated',old),('calibrated',temp)]:
            ys=np.array([y for a,y in held]);ps=[]
            for a,y in held:
                keys=list(a['scores']);v=np.array(list(a['scores'].values()))/t;v=np.exp(v-v.max());v/=v.sum();ps.append(v[keys.index('true')])
            ps=np.array(ps);bins=[];ece=0
            for j in range(10):
                mask=(ps>=j/10)&(ps<((j+1)/10) if j<9 else ps<=1)
                if mask.any():confidence=float(ps[mask].mean());frequency=float(ys[mask].mean());ece+=mask.mean()*abs(confidence-frequency);bins.append({'count':int(mask.sum()),'confidence':confidence,'frequency':frequency})
            res['heldout'][label]={'ECE':float(ece),'Brier':float(((ps-ys)**2).mean()),'NLL':float(-(ys*np.log(np.clip(ps,1e-12,1))+(1-ys)*np.log(np.clip(1-ps,1e-12,1))).mean()),'bins':bins}
        dump(root/'results/calibration.json',res);m.temperature=old

def figures(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((root/'results/retrieval_summary.json').read_text());out=root/'figures';out.mkdir(exist_ok=True)
    rows=[r for r in rows if r['configuration'].startswith('full')]
    for metric in ('NDCG@10','Recall@10','Completeness@10'):
        fig,ax=plt.subplots();ax.plot([r['dimension'] for r in rows],[r[metric] for r in rows],marker='o');ax.set(xlabel='Dimensions (exploratory prefix truncation except full)',ylabel=metric,title='MetaEncoder v1 / SkillRet');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/(metric.replace('@','_')+'_dimension.png'),dpi=180);plt.close(fig)
    with (out/'dimension_data.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def compare(root,data):
    from metrics_existing import paired_bootstrap
    path=Path(os.environ.get('BASELINE_RANKINGS','/workspace/mia5150/experiments/benchmarks/skillret-embedding-jev-fusion/results/dense_top50.jsonl'))
    source=lines(path);rankings={x['query_id']:[h['skill_id'] for h in x['candidates']] for x in source}
    labels=labels_for(data);assert set(rankings)==set(labels)
    baseline=evaluate_rankings(root,'historical_baseline',labels,rankings)
    treatment=json.loads((root/'results/full_6656_metrics.json').read_text())
    dump(root/'results/paired_bootstrap.json',paired_bootstrap(baseline,treatment,n_resamples=1000,seed=42))
    dump(root/'manifests/baseline_source.json',{'path':str(path),'sha256':sha(path),'scope':'Existing saved rankings, no baseline inference rerun; historical latency incomparable'})

if __name__=='__main__':main()
