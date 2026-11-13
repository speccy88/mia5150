#!/usr/bin/env python3
"""RunPod-only, resumable decision breadth-first extension; not primary retrieval."""
import argparse,hashlib,json,time
from pathlib import Path
import runner2 as r

def decisions(root,data,m,limit=200,max_seconds=1000):
    import numpy as np
    from metrics_existing import deterministic_subset
    sid,st,qid,qt=r.texts(data);docs=dict(zip(sid,st));queries=dict(zip(qid,qt));labels=r.labels_for(data)
    ranks={x['query_id']:[h['skill_id'] for h in x['hits']] for x in r.lines(root/'results/full_6656_top100.jsonl')}
    chosen=deterministic_subset(qid,limit);pilot_labels={q:labels[q] for q in chosen};start=time.time();ck=root/'checkpoints/decisions_v2';ck.mkdir(parents=True,exist_ok=True)
    questions={'binary':{'type':'noul','instructions':'Given the user request and this candidate skill, determine whether the skill would help an agent fulfill the request.'},'ordinal':{'type':'score','instructions':'Rate how useful the candidate skill is for fulfilling the user request.','criteria':['Not relevant','Weakly related','Partially useful','Relevant and useful','Strongly relevant to fulfilling the request']}}
    r.dump(root/'manifests/decision_v2_protocol.json',{'questions':questions,'query_ids':chosen,'schedule':'choice k5/10/20 on first4 fixed queries, then candidate-rank breadth-first across all pilot queries to10,20,50; binary and ordinal combined API call','fit_query_ids':chosen[::2],'heldout_query_ids':chosen[1::2],'budget_seconds':max_seconds})
    def alive():return time.time()-start<max_seconds
    def file(q,s):return ck/(hashlib.sha256((q+'|'+s).encode()).hexdigest()+'.json')
    # Initial bounded sample exercises every legitimate interface without pretending full pilot coverage.
    for k in (5,10,20):
        for q in chosen[:4]:
            p=file(q,'choice'+str(k))
            if p.exists() or not alive():continue
            spec={'choose':{'type':'choice','instructions':'Choose the skill most relevant and useful for fulfilling the user request.','criteria':{s:docs[s] for s in ranks[q][:k]}}};t=time.perf_counter()
            try:row={'query_id':q,'mode':'choice','candidate_count':k,'output':m.decide(queries[q],spec),'status':'ok'}
            except Exception as e:row={'query_id':q,'mode':'choice','candidate_count':k,'status':'failed','error':str(e)}
            row['seconds']=time.perf_counter()-t;r.dump(p,row)
    # All queries receive rank1 before rank2, preventing a partial pilot from being one long query.
    for rank in range(50):
        for q in chosen:
            if not alive():break
            s=ranks[q][rank];p=file(q,s)
            if p.exists():continue
            t=time.perf_counter()
            try:row={'query_id':q,'skill_id':s,'dense_rank':rank+1,'mode':'pair','output':m.decide({'request':queries[q],'candidate_skill':docs[s]},questions),'status':'ok'}
            except Exception as e:row={'query_id':q,'skill_id':s,'dense_rank':rank+1,'mode':'pair','status':'failed','error':str(e)}
            row['seconds']=time.perf_counter()-t;r.dump(p,row)
        if not alive():break
    allrows=[json.loads(p.read_text()) for p in ck.glob('*.json')];pairs=[x for x in allrows if x['mode']=='pair'];choice=[x for x in allrows if x['mode']=='choice'];summary=[]
    for k in (10,20,50):
        pool={q:ranks[q][:k] for q in chosen};coverage=r.evaluate_rankings(root,f'v2_candidate_pool_{k}',pilot_labels,pool)
        for mode in ('binary','ordinal'):
            rankings={};selection={};tp=fp=fn=0;lat=[];completed=0;scored=0
            for q in chosen:
                rr=[x for x in pairs if x['query_id']==q and x['dense_rank']<=k and x['status']=='ok'];scored+=len(rr);completed+=len(rr)==k;lat.extend(x['seconds'] for x in rr)
                score=lambda x:float(x['output']['questions'][mode]['probabilities']['true']) if mode=='binary' else float(x['output']['questions'][mode]['value'])
                rr.sort(key=lambda x:(-score(x),x['dense_rank']));rankings[q]=[x['skill_id'] for x in rr]
                if mode=='binary':
                    selection[q]=[x['skill_id'] for x in rr if x['output']['questions']['binary']['value']]
                    for x in rr:
                        yes=x['output']['questions']['binary']['value'];gold=x['skill_id'] in labels[q];tp+=bool(yes and gold);fp+=bool(yes and not gold);fn+=bool(not yes and gold)
            result=r.evaluate_rankings(root,f'v2_decide_{mode}_{k}',pilot_labels,rankings)
            if mode=='binary':r.evaluate_rankings(root,f'v2_selection_{k}',pilot_labels,selection)
            summary.append({'mode':mode,'candidate_count':k,'query_count':len(chosen),'fully_scored_queries':completed,'successful_pairs':scored,'expected_pairs':len(chosen)*k,'metrics':result['aggregate'],'coverage':coverage['aggregate'],'classification_precision':tp/(tp+fp) if tp+fp else None,'classification_recall':tp/(tp+fn) if tp+fn else None,'classification_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'classification_denominator':'successful candidate pairs only','latency_p50':float(np.quantile(lat,.5)) if lat else None,'latency_p95':float(np.quantile(lat,.95)) if lat else None,'latency_note':'combined binary+ordinal call, not independent costs'})
    r.dump(root/'results/decision_v2_summary.json',summary);r.dump(root/'results/decision_v2_raw.json',allrows)
    choice_summary=[]
    for k in (5,10,20):
        for n in (1,2,3):
            qs=[q for q in chosen if len(labels[q])==n];rr=[x for x in choice if x['candidate_count']==k and x['query_id'] in qs and x['status']=='ok'];good=sum(x['output']['questions']['choose']['value'] in labels[x['query_id']] for x in rr);covered=sum(any(s in labels[q] for s in ranks[q][:k]) for q in qs);attempted_covered=sum(any(s in labels[x['query_id']] for s in ranks[x['query_id']][:k]) for x in rr)
            choice_summary.append({'candidate_count':k,'required_skills':n,'intended_queries':len(qs),'successful_queries':len(rr),'top1_accuracy_full_denominator':good/len(qs) if qs else None,'accuracy_successful_only':good/len(rr) if rr else None,'conditional_accuracy_full_denominator':good/covered if covered else None,'conditional_accuracy_successful_only':good/attempted_covered if attempted_covered else None,'note':'unfinished queries remain zero in full-denominator accuracy; no MRR from one choice'})
    r.dump(root/'results/choice_v2_summary.json',choice_summary)
    fitids=set(chosen[::2]);fit=[];fl=[];held=[]
    for x in pairs:
        if x['status']!='ok':continue
        a=x['output']['questions']['binary'];y=x['skill_id'] in labels[x['query_id']]
        if x['query_id'] in fitids:fit.append(a);fl.append(y)
        else:held.append((a,y))
    if fit and held:
        old=m.temperature;temp=m.calibrate(fit,fl);res={'fit_query_ids':chosen[::2],'heldout_query_ids':chosen[1::2],'fit_pairs':len(fit),'heldout_pairs':len(held),'temperature':temp,'original_temperature':old,'heldout':{}}
        for label,t in [('uncalibrated',old),('calibrated',temp)]:
            ys=np.array([y for a,y in held]);ps=[]
            for a,y in held:
                keys=list(a['scores']);v=np.array(list(a['scores'].values()))/t;v=np.exp(v-v.max());v/=v.sum();ps.append(v[keys.index('true')])
            ps=np.array(ps);bins=[];ece=0
            for j in range(10):
                mask=(ps>=j/10)&((ps<(j+1)/10) if j<9 else (ps<=1))
                if mask.any():conf=float(ps[mask].mean());freq=float(ys[mask].mean());ece+=mask.mean()*abs(conf-freq);bins.append({'count':int(mask.sum()),'confidence':conf,'frequency':freq})
            res['heldout'][label]={'ECE':float(ece),'Brier':float(((ps-ys)**2).mean()),'NLL':float(-(ys*np.log(np.clip(ps,1e-12,1))+(1-ys)*np.log(np.clip(1-ps,1e-12,1))).mean()),'bins':bins}
        r.dump(root/'results/calibration_v2.json',res);m.temperature=old
    r.rec(root,'secondary_complete',wall_seconds=time.time()-start,attempted_pairs=len(pairs),attempted_choices=len(choice))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=1000);p.add_argument('--limit',type=int,default=200);a=p.parse_args();data=r.load_data(a.root);r.validate_data(a.root,data);m=r.model_load(a.root);decisions(a.root,data,m,a.limit,a.max_seconds)
if __name__=='__main__':main()
