#!/usr/bin/env python3
"""RunPod-only deadline-bounded continuation with parallel validated checkpoint preload."""
import argparse,datetime,hashlib,json,time,os,signal,subprocess,sys
from pathlib import Path
import runner2 as r
import secondary_runner
rec=r.rec;dump=r.dump;token_count=r.token_count;chunks=r.chunks
def encode_stage(root,m,ids,texts,kind,instruction,limit=16384,batch=4,max_seconds=None):
    import numpy as np,torch
    out=root/'artifacts';out.mkdir(exist_ok=True);ck=root/'checkpoints'/kind;ck.mkdir(parents=True,exist_ok=True)
    start=time.time();audit={};pending=[];counts={};cache={}
    from concurrent.futures import ThreadPoolExecutor
    def existing(i):
        p=ck/f'{i:06d}.npz';meta=ck/f'{i:06d}.json'
        if not p.exists() or not meta.exists():return None
        saved=json.loads(meta.read_text())
        assert saved['id']==ids[i] and saved['input_sha256']==hashlib.sha256(texts[i].encode()).hexdigest()
        assert saved.get('instruction',instruction)==instruction and saved.get('token_limit',limit)==limit
        with np.load(p) as z:a={k:z[k] for k in ('aggregate','truncated','chunks')}
        assert a['aggregate'].shape==(m.embedding_dim,) and a['truncated'].shape==(m.embedding_dim,)
        assert a['chunks'].shape==(saved['chunk_count'],m.embedding_dim)
        assert all(np.isfinite(v).all() for v in a.values())
        assert np.allclose(np.linalg.norm(a['chunks'],axis=1),1,atol=1e-4)
        assert np.isclose(np.linalg.norm(a['aggregate']),1,atol=1e-4)
        return i,saved,a
    with ThreadPoolExecutor(max_workers=16) as pool:
        for result in pool.map(existing,range(len(ids))):
            if result is not None:i,saved,a=result;audit[i]=saved;cache[i]=a
    rec(root,'resume_validation',kind=kind,validated=len(audit),seconds=time.time()-start,workers=16)
    def path(i):return ck/f'{i:06d}.npz'
    def meta(i):return ck/f'{i:06d}.json'
    def save(i,arr,n,cs,elapsed,group_count,peak):
        avg=arr.mean(0);avg/=np.linalg.norm(avg)
        assert arr.shape==(len(cs),m.embedding_dim) and np.isfinite(arr).all()
        assert np.allclose(np.linalg.norm(arr,axis=1),1,atol=1e-4)
        tmp=ck/f'{i:06d}.tmp.npz';np.savez(tmp,chunks=arr,aggregate=avg.astype('float32'),truncated=arr[0]);tmp.replace(path(i))
        row={'row':i,'id':ids[i],'input_sha256':hashlib.sha256(texts[i].encode()).hexdigest(),'instruction':instruction,'token_limit':limit,'tokens':n,'chunk_tokens':[token_count(m,c,instruction) for c in cs],'chunk_count':len(cs),'seconds':elapsed/group_count,'latency_basis':'batch_elapsed_divided_by_input_count' if group_count>1 else 'single_input','batch_seconds':elapsed,'batch_input_count':group_count,'peak_gpu_bytes':peak,'status':'ok'}
        dump(meta(i),row);audit[i]=row;cache[i]={'aggregate':avg.astype('float32'),'truncated':arr[0],'chunks':arr}
    def flush():
        if not pending:return
        group=list(pending);pending.clear();t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        try:
            v=m.encode([texts[i] for i in group],instruction=instruction,batch_size=len(group));torch.cuda.synchronize();arr=v.detach().cpu().float().numpy();elapsed=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()
            for j,i in enumerate(group):save(i,arr[j:j+1],counts[i],[texts[i]],elapsed,len(group),peak)
            rec(root,'embedding_batch',kind=kind,count=len(group),seconds=elapsed,tokens=sum(counts[i] for i in group),peak_gpu_bytes=peak)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(group)==1:raise
            rec(root,'batch_oom_retry',kind=kind,count=len(group))
            for i in group:
                pending.append(i);flush()
    for i,(ident,text) in enumerate(zip(ids,texts)):
        if i in audit:continue
        if max_seconds is not None and time.time()-start>max_seconds:flush();break
        n=token_count(m,text,instruction);counts[i]=n
        if n<=2048:
            pending.append(i)
            if len(pending)>=max(1,batch):flush()
        else:
            flush();cs=chunks(m,text,instruction,limit);t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            try:
                # One long chunk at a time, even when caller requests a larger short-input batch.
                v=m.encode(cs,instruction=instruction,batch_size=1);torch.cuda.synchronize();arr=v.detach().cpu().float().numpy();save(i,arr,n,cs,time.perf_counter()-t,1,torch.cuda.max_memory_allocated())
            except Exception as e:rec(root,'embedding_failure',kind=kind,row=i,id=ident,error=str(e));raise
        if len(audit)%20==0:rec(root,'embedding_progress',kind=kind,completed=len(audit),total=len(ids),seconds=time.time()-start)
    flush();ordered=[audit[i] for i in sorted(audit)];dump(root/f'manifests/{kind}_token_audit.json',ordered)
    if len(audit)==len(ids):
        arrays=[];trunc=[];chunk_arrays=[];mapping=[];offset=0
        for i in range(len(ids)):
            a=cache[i];arrays.append(a['aggregate']);trunc.append(a['truncated']);chunk_arrays.append(a['chunks'])
            mapping.append({'id':ids[i],'offset':offset,'count':audit[i]['chunk_count']});offset+=audit[i]['chunk_count']
        arr=np.stack(arrays).astype('float32');np.save(out/f'{kind}.npy',arr);np.save(out/f'{kind}_truncated.npy',np.stack(trunc).astype('float32'));np.save(out/f'{kind}_chunks.npy',np.concatenate(chunk_arrays).astype('float32'))
        dump(out/f'{kind}_ids.json',ids);dump(out/f'{kind}_chunks_mapping.json',mapping)
        rec(root,'embedding_complete',kind=kind,count=len(ids),shape=list(arr.shape),seconds=sum(r['seconds'] for r in ordered))
    return len(audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--deadline',required=True,help='UTC ISO time, e.g. 2026-11-13T03:00:00+00:00');p.add_argument('--batch',type=int,default=4);p.add_argument('--secondary-seconds',type=float,default=180);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    deadline=datetime.datetime.fromisoformat(a.deadline.replace('Z','+00:00')).timestamp()
    def left():return deadline-time.time()
    if left()<=0:return
    if not a.worker:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],start_new_session=True)
        try:child.wait(timeout=max(.001,left()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            r.rec(a.root,'hard_deadline_stop',deadline=a.deadline)
        return
    root=a.root;data=r.load_data(root);r.validate_data(root,data);sid,st,qid,qt=r.texts(data);m=r.model_load(root)
    # Leave120sec for retrieval/metrics/manifests. Checkpoint reads count against time.
    for ids,tx,kind,instruction in [(sid,st,'documents',r.DOC_INST),(qid,qt,'queries',r.QUERY_INST)]:
        if left()<=120:return
        n=encode_stage(root,m,ids,tx,kind,instruction,batch=a.batch,max_seconds=max(.001,left()-120))
        if n!=len(ids):return
    r.retrieval(root,data)
    if left()>120:r.native(root,data,m,20,max_seconds=min(40,left()-100))
    if left()>100:secondary_runner.decisions(root,data,m,200,min(a.secondary_seconds,left()-90))
    if left()>75:r.compare(root,data)
    if left()>45:
        import report_figures;report_figures.main(root)
        import final_diagnostics;final_diagnostics.main(root)
    r.dump(root/'manifests/artifacts.json',{str(f.relative_to(root)):{'bytes':f.stat().st_size,'sha256':r.sha(f)} for f in (root/'artifacts').glob('*') if f.is_file()})
    r.rec(root,'continuation_finished',remaining_seconds=left())
if __name__=='__main__':main()
