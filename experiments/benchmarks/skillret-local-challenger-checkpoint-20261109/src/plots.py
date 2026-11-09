"""Publication figures from verified actual measurements only."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS=['#536878','#007f86','#cb6e17','#6b51a1','#b24351']
def main():
 folder=STUDY/'figures';folder.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':300,'svg.fonttype':'none'})
 metrics={}
 for p in (STUDY/'results/final').glob('*.metrics.json'):metrics[p.name[:-13]]=read(p)['aggregate']
 metrics={'archived-base':read(OLD/'results/baseline_skillret_06b.json')['evaluation']['aggregate'],**metrics};core=[('archived-base','Archived\nreference'),('adapted-query','Query\nLoRA'),('mixture-query','Query\nmixture'),('released-reranker','Released\nreranker'),('trained-reranker','Reranker\nLoRA'),('trained-combined','Combined\nLoRA')];names=[key for key,label in core if key in metrics];labels=[label for key,label in core if key in metrics]
 def finish(fig,name):fig.tight_layout();fig.savefig(folder/(name+'.png'),bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(14,4.5))
 for ax,key in zip(axs,['NDCG@10','Recall@10','MRR@10']):
  bars=ax.bar(range(len(names)),[metrics[n][key] for n in names],color=[COLORS[i%len(COLORS)] for i in range(len(names))]);ax.set_xticks(range(len(names)),labels,fontsize=8);ax.set_ylim(0,1);ax.set_title(key);ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
 fig.suptitle('Complete official SkillRet: 4,392 queries × 6,006 skills');finish(fig,'01-full-test-quality')
 paired=read(STUDY/'results/paired_final_comparisons.json');entries=list(paired);fig,ax=plt.subplots(figsize=(10,max(3,len(entries)*.65)))
 for i,n in enumerate(entries):
  r=paired[n];d=r['ndcg10_delta']*100;lo,hi=[v*100 for v in r['ci95']];ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',color=COLORS[i%5],capsize=4)
 ax.axvline(0,color='grey',lw=1);ax.set_yticks(range(len(entries)),[n.replace('-',' ') for n in entries]);ax.set_xlabel('NDCG@10 change (percentage points), paired 95% interval');ax.set_title('10,000 question-bootstrap samples; fixed selected checkpoints');finish(fig,'02-paired-changes')
 fig,ax=plt.subplots(figsize=(10,5))
 trials=sorted({p.parent for p in (STUDY/'results/query').glob('*/selection-*.json')})
 for i,trial in enumerate(trials):
  points=[]
  for p in sorted(trial.glob('selection-*.json')):
   r=read(p);points.append((int(p.stem.split('-')[-1]),max(v['aggregate']['NDCG@10'] for v in r['mixtures'].values())))
  ax.plot([x for x,y in points],[y for x,y in points],'-o',label=trial.name,color=COLORS[i%5])
 ax.axhline(read(STUDY/'results/frozen-selection.json')['evaluation']['aggregate']['NDCG@10'],ls='--',color='grey',label='frozen embedding');ax.set(xlabel='Optimizer step',ylabel='Selection NDCG@10',title='Training-origin selection; all 10,123 skills');ax.legend();finish(fig,'03-query-development')
 fig,ax=plt.subplots(figsize=(10,5));sc=read(STUDY/'results/scaling.json')
 for i,name in enumerate(sorted({x['condition'] for x in sc})):
  records=sorted([x for x in sc if x['condition']==name],key=lambda x:x['library_size']);ax.plot([x['library_size'] for x in records],[x['evaluation']['aggregate']['NDCG@10'] for x in records],'--s' if 'combined' in name else '-o',label=name,color=COLORS[i%5])
 ax.set_xscale('log');ax.set(xlabel='Exact saved library size',ylabel='NDCG@10',title='Secondary exposed 20-question panel; membership varies with size');ax.legend();finish(fig,'04-scaling')
 lat=read(STUDY/'results/latency-end-to-end20.json');frozen=read(STUDY/'results/latency-end-to-end20-frozen.json');fig,ax=plt.subplots(figsize=(8,4.5));position=np.arange(2)
 for offset,result,label,color in [(-.18,frozen,'Frozen reference',COLORS[0]),(.18,lat,'Locked primary',COLORS[1])]:
  bars=ax.bar(position+offset,[result['median_seconds'],result['p95_seconds']],.34,label=label,color=color);ax.bar_label(bars,fmt='%.3f s',padding=3)
 ax.set_xticks(position,['Median','p95']);ax.margins(y=.17);ax.legend();ax.set(ylabel='Seconds',title='Actual latency on the same 20 uncached queries\n'+('Primary includes model loading and switching' if lat['includes_model_loading_and_switching'] else 'Single encoders resident; startup measured separately'));finish(fig,'05-actual-latency')
 fig,axs=plt.subplots(1,2,figsize=(12,4.5))
 for ax,key in zip(axs,['Completeness@10','Recall@10']):
  bars=ax.bar(range(len(names)),[metrics[n][key] for n in names],color=[COLORS[i%5] for i in range(len(names))]);ax.set_xticks(range(len(names)),labels,fontsize=8);ax.set_ylim(0,1);ax.set_title(key);ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
 finish(fig,'06-recall-and-completeness')
 fig,ax=plt.subplots(figsize=(10,5))
 trials=sorted({p.parent for p in (STUDY/'results/reranker').glob('*/selection-*.summary.json')})
 for i,trial in enumerate(trials):
  points=[]
  for p in sorted(trial.glob('selection-*.summary.json')):
   r=read(p);points.append((int(p.name.split('-')[1].split('.')[0]),max(v['evaluation']['aggregate']['NDCG@10'] for v in r.values())))
  ax.plot([x for x,y in points],[y for x,y in points],'-o',label=trial.name,color=COLORS[i%5])
 ax.set(xlabel='Optimizer step',ylabel='End-to-end selection NDCG@10',title='Reranker selection, no gold candidate insertion');ax.legend();finish(fig,'07-reranker-development')
 print('Saved7PNG+SVG figures',flush=True)
if __name__=='__main__':main()
