"""Measure released-reranker policies on training-derived selection only."""
from train_reranker import *
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','released1024','released2048','full-coverage'],default='prepare');p.add_argument('--budget',type=int,choices=[1024,2048],default=2048);a=p.parse_args()
 if a.mode=='prepare':prepare_renderer();return
 m,tok=load('reranker')
 if a.mode=='released1024':score_panel(m,tok,'selection','released1024/selection.jsonl',1024,50)
 if a.mode=='released2048':score_panel(m,tok,'selection','released2048/selection.jsonl',2048,50)
 if a.mode=='full-coverage':
  score_panel(m,tok,'selection','released-fullcoverage/selection.jsonl',a.budget,20,True)
  from scipy.stats import spearmanr
  records=rows(STUDY/'results/reranker/released-fullcoverage/selection.jsonl');correlations=[];counts=[]
  for record in records:
   n=[len(windows) for windows in record['selections']];counts.extend(n)
   if len(set(n))>1 and len(set(record['reranker_scores']))>1:correlations.append(float(spearmanr(n,record['reranker_scores']).statistic))
  save(STUDY/'results/reranker/full-coverage-cost-bias.json',{'skill_window_counts':summary(counts),'within_query_window_count_score_spearman_mean':float(np.mean(correlations)) if correlations else None,'queries_with_defined_correlation':len(correlations),'median_query_rerank_seconds':float(np.median([r['timing']['rerank'] for r in records])),'limitation':'Association, not a causal estimate of document-length bias. Skill-level max offers longer skills more scoring opportunities; no individual window is labeled positive.'})
if __name__=='__main__':main()
