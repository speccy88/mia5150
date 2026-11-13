from pathlib import Path
import runner2 as r
import secondary_runner
root=Path('/workspace/metaencoder')
r.rec(root,'bounded_decision_continuation_start',budget_seconds=3600,query_limit=200,schedule='complete_top10_first_then_remaining_breadth_first')
data=r.load_data(root)
m=r.model_load(root)
secondary_runner.decisions(root,data,m,200,3600)
r.rec(root,'bounded_decision_continuation_end')
