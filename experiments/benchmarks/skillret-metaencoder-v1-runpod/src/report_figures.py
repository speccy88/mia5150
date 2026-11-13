"""Run only on RunPod. Eight figure panels; unavailable data is explicitly labelled."""
import argparse,csv,json
from pathlib import Path

def main(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'figures';out.mkdir(exist_ok=True)
    def read(name,default):
        p=root/'results'/name;return json.loads(p.read_text()) if p.exists() else default
    retrieval=read('retrieval_summary.json',[]);decisions=read('decision_v2_summary.json',[]);native=read('native_retrieve_metrics.json',{});baseline=read('historical_baseline_metrics.json',{})
    full=next((x for x in retrieval if x['configuration']=='full_6656'),None)
    caps=[]
    if full:caps.append({'name':'encode full (4392)','metrics':full})
    if native:caps.append({'name':f"native ({native['query_count']})",'metrics':native['aggregate']})
    for x in decisions:
        if x['candidate_count']==10:caps.append({'name':x['mode']+f" k10 ({x['fully_scored_queries']}/{x['query_count']} complete)",'metrics':x['metrics']})
    def fig(name,rows,draw):
        f,ax=plt.subplots(figsize=(10,5.5))
        if rows:draw(ax)
        else:ax.text(.5,.5,'Unavailable: experiment not completed or data not recorded',ha='center',va='center',wrap=True);ax.set_axis_off()
        f.tight_layout();f.savefig(out/(name+'.png'),dpi=180);plt.close(f);(out/(name+'.json')).write_text(json.dumps(rows,indent=2))
        if rows and isinstance(rows,list) and isinstance(rows[0],dict):
            keys=sorted(set().union(*(x.keys() for x in rows)))
            with (out/(name+'.csv')).open('w') as h:
                w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows])
    fig('01_ndcg_capability',caps,lambda ax:(ax.bar([x['name'] for x in caps],[x['metrics']['NDCG@10'] for x in caps]),ax.set(ylabel='NDCG@10',title='Scope differs: pilot missing outputs count as zero'),ax.tick_params(axis='x',rotation=15)))
    def drawrc(ax):
        xs=list(range(len(caps)));ax.bar([x-.18 for x in xs],[x['metrics']['Recall@10'] for x in caps],width=.36,label='Recall@10');ax.bar([x+.18 for x in xs],[x['metrics']['Completeness@10'] for x in caps],width=.36,label='Completeness@10');ax.set_xticks(xs,[x['name'] for x in caps],rotation=15);ax.legend();ax.set_title('Declared full/pilot denominators, not matched-scope improvement')
    fig('02_recall_completeness',caps,drawrc)
    dims=[x for x in retrieval if x['configuration'].startswith('full')]
    fig('03_dimension',dims,lambda ax:(ax.plot([x['dimension'] for x in dims],[x['NDCG@10'] for x in dims],marker='o'),ax.set(xlabel='Dimension (prefix truncation exploratory)',ylabel='NDCG@10')))
    changes=[]
    for x in decisions:
        base=read(f"v2_candidate_pool_{x['candidate_count']}_metrics.json",{})
        if base:changes.append({'mode':x['mode'],'k':x['candidate_count'],'delta_NDCG10_full_pilot_denominator':x['metrics']['NDCG@10']-base['aggregate']['NDCG@10'],'fully_scored_queries':x['fully_scored_queries'],'pilot_denominator':x['query_count']})
    fig('04_decision_changes',changes,lambda ax:(ax.bar([x['mode']+' k'+str(x['k']) for x in changes],[x['delta_NDCG10_full_pilot_denominator'] for x in changes]),ax.axhline(0,color='black',linewidth=.8),ax.set(ylabel='ΔNDCG@10 vs candidate-pool baseline',title='Incomplete scoring penalized; not clean model-effect estimate')))
    by=read('full_6656_metrics.json',{}).get('by_required_skills',{});groups=[{'required_skills':k,**v} for k,v in by.items()]
    fig('05_required_skills',groups,lambda ax:(ax.bar([x['required_skills'] for x in groups],[x['NDCG@10'] for x in groups]),ax.set(xlabel='Required skills',ylabel='NDCG@10')))
    perf=[]
    for kind in ('documents','queries'):
        p=root/f'manifests/{kind}_token_audit.json'
        if p.exists():
            rows=json.loads(p.read_text());perf.extend({'kind':kind,'id':x['id'],'seconds':x['seconds'],'peak_gpu_gib':x['peak_gpu_bytes']/2**30} for x in rows if x['status']=='ok')
    fig('06_latency_memory',perf,lambda ax:(ax.scatter([x['seconds'] for x in perf],[x['peak_gpu_gib'] for x in perf],s=3,alpha=.3),ax.set(xlabel='Input seconds (batch-amortized when batched)',ylabel='Peak allocated GPU GiB')))
    cal=read('calibration_v2.json',{});cr=[{'state':k,**b} for k,v in cal.get('heldout',{}).items() for b in v['bins']]
    def drawcal(ax):
        for state in ('uncalibrated','calibrated'):
            rows=[x for x in cr if x['state']==state];ax.plot([x['confidence'] for x in rows],[x['frequency'] for x in rows],marker='o',label=state)
        ax.plot([0,1],[0,1],'--',color='gray');ax.legend();ax.set(xlabel='Predicted relevance probability',ylabel='Held-out observed frequency')
    fig('07_reliability',cr,drawcal)
    # Use measured compute time, not fabricated billing allocation.
    cost=[]
    if full:cost.append({'configuration':'full native dimensions','measured_search_seconds':full['search_seconds'],'NDCG@10':full['NDCG@10']})
    cost.extend({'configuration':str(x['dimension'])+' dims','measured_search_seconds':x['search_seconds'],'NDCG@10':x['NDCG@10']} for x in dims if x['dimension']!=6656)
    fig('08_quality_compute',cost,lambda ax:(ax.scatter([x['measured_search_seconds'] for x in cost],[x['NDCG@10'] for x in cost]),ax.set(xlabel='Measured cached similarity-search seconds (not billed USD)',ylabel='NDCG@10',title='Shared inference excluded; actual billing reported separately')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
