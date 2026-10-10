"""Freeze small exploration cohort; no inference, downloads or API calls."""
from pathlib import Path
import json
from probe_contract import digest
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
queries=json.loads((root/'manifests/scorer-queries.json').read_text())[8:20]
retrieval=json.loads((root/'results/skillret_embed_06-header-0.json').read_text())['rankings']
panel={'queries':queries,'candidate_ids':{q['id']:retrieval[q['id']][:20] for q in queries},'exposure':'Already-exposed public development, exploratory only; excludes the prior8-query GGUF subset. Not untouched final evaluation.','model':'jev-1.13.0','hypotheses':['Native Noul relevance','Ordinal Score rubric'],'calls_max':12,'questions_per_call':40,'source_dataset_revision':'6583d7d2ed07644d0fb8938ed8178f3a7dc42a12','controls':'Reuse saved dense/native reranker outputs on identical20 pools; no rerunning baselines.'}
panel['protocol_sha256']=digest(panel);(out/'exploration-freeze.json').write_text(json.dumps(panel,indent=2))
replay=json.loads((root/'manifests/lora-replay-clean-examples.json').read_text());assert replay['required_positive_pairs']==replay['retained_positive_pairs']==91 and replay['all_positives_preserved']
(out/'replay-status.json').write_text(json.dumps({'frozen_examples_sha256':digest(replay['examples']),'required_positive_pairs':91,'status':'Corrected protocol retained, lower priority under latest diversity preference; no repeated training started.'},indent=2))
print(json.dumps({'queries':len(queries),'candidate_count':20,'protocol_sha256':panel['protocol_sha256'],'paid_calls':0,'training_runs':0}))
