"""Scientific figures from regenerated CSV/JSON only; no inferred missing points."""
import csv,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=pathlib.Path(__file__).resolve().parent;OUT=HERE/'report'
METHODS=['listing','bm25','vector','domain_vector','hybrid'];N=[10,100,1000,6006]
LABELS={'listing':'Full listing','bm25':'BM25','vector':'Vector','domain_vector':'Domain vector','hybrid':'Hybrid RRF60'}
COLORS=dict(zip(METHODS,['#8f559e','#2379a5','#24967e','#d77724','#c24c65']))
def main():
    data=json.loads((OUT/'summary.json').read_text());primary=[r for r in data['selection'] if r['scope']=='primary200_seed42']
    lookup={(r['method'],r['size']):r for r in primary}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold','savefig.facecolor':'white'})
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.8,6.3));values=np.array([[lookup[(m,n)]['official_prevalence_weighted_exact'] for n in N] for m in METHODS]);image=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    for i,m in enumerate(METHODS):
        for j,n in enumerate(N):
            row=lookup[(m,n)];text=f"{values[i,j]:.1%}\n{row['exact_successes']}/{row['planned']} exact\n{row['valid']} valid"
            if row['context_overflow']==row['planned']:text+='\ncontext overflow'
            ax.text(j,i,text,ha='center',va='center',color='white' if values[i,j]>.52 else '#102739',fontsize=10)
    ax.set_xticks(range(4),[f'{n:,}' for n in N]);ax.set_yticks(range(5),[LABELS[m] for m in METHODS]);ax.set_xlabel('Skills in the gold-conditioned catalog');ax.set_title('Final skill-set selection on 200 exposed queries\nQwen3-0.6B Q8, seed42, official-prevalence estimate',pad=16)
    fig.colorbar(image,ax=ax,label='Exact-set accuracy estimate',shrink=.85);fig.text(.02,.015,'All planned cases remain in the denominator. Counts show raw successes. Listing is unranked metadata exposure.',fontsize=9);fig.tight_layout(rect=(0,.04,1,1));save(fig,'selection-heatmap')
    for metric,name,title,ylabel,logy in [('tokenized_logical_input_mean_including_preflight','input-growth','Tokenized input growth','Mean logical prompt tokens, including overflow preflight',True),('fresh_valid_e2e_mean_s','latency','Measured successful workflow latency','Fresh valid end-to-end seconds',True),('retrieval_recall5','candidate-recall','Candidate retrieval on the matched primary panel','Macro Recall@5',False)]:
        fig,ax=plt.subplots(figsize=(9.8,5.5))
        for method in METHODS:
            y=[lookup[(method,n)].get(metric) for n in N];y=[np.nan if v is None else v for v in y]
            if all(np.isnan(y)):continue
            ax.plot(N,y,marker='o',lw=2,label=LABELS[method],color=COLORS[method])
        ax.set_xscale('log');ax.set_xticks(N,[f'{n:,}' for n in N]);ax.set_xlabel('Catalog size');ax.set_ylabel(ylabel);ax.set_title(title+'\n200 exposed queries, one common0.6B selector');ax.grid(alpha=.2);ax.legend(frameon=False)
        if logy:ax.set_yscale('log')
        else:ax.set_ylim(0,1.03)
        note='Overflow tokens describe preflight input, not model-consumed tokens.' if name=='input-growth' else ('Startup and failures excluded. Missing listing points are explicit context failures.' if name=='latency' else 'Listing coverage is1 by construction and has no semantic ranking; omitted from Recall@5.')
        fig.text(.02,.015,note,fontsize=9);fig.tight_layout(rect=(0,.045,1,1));save(fig,name)
    paired=data['paired']
    if paired:
        fig,ax=plt.subplots(figsize=(10.5,5.7));ys=np.arange(len(paired))[::-1]
        for y,row in zip(ys,paired):
            ax.plot([100*row['bonferroni8_nominal_low'],100*row['bonferroni8_nominal_high']],[y,y],color='#b6c1c8',lw=5)
            ax.plot([100*row['ci95_low'],100*row['ci95_high']],[y,y],color='#2379a5',lw=3);ax.plot(100*row['weighted_exact_difference'],y,'o',color='#143e56')
        ax.axvline(0,color='#596b77',ls='--',lw=1);ax.set_yticks(ys,[f"{r['contrast']}   N={r['size']:,}" for r in paired]);ax.set_xlabel('Difference in exact-set accuracy, percentage points');ax.set_title('Paired query-level differences\nStratified bootstrap,200 queries,2,000 resamples');ax.grid(axis='x',alpha=.2);fig.text(.02,.015,'Blue: nominal95% intervals. Gray: nominal Bonferroni8 percentile intervals. Exposed exploratory evaluation.',fontsize=9);fig.tight_layout(rect=(0,.04,1,1));save(fig,'paired-differences')
    print('Figures regenerated from saved aggregate data')
if __name__=='__main__':main()
