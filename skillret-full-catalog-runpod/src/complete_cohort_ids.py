"""Add exact group-ID manifests to a selected training cohort on RunPod."""
from pathlib import Path
import argparse
import json
import os

from data_stage import read_jsonl

def main(root: Path):
    path=root/"data/training_cohort.json"
    cohort=json.loads(path.read_text())
    selected=set(cohort["selected_training_query_ids"])
    excluded=set(cohort["excluded_training_query_ids"])
    rows=read_jsonl(root/"data/train_queries.jsonl")
    cohort["selected_training_group_ids"]=sorted({r["group"] for r in rows if r["id"] in selected})
    cohort["excluded_training_group_ids"]=sorted({r["group"] for r in rows if r["id"] in excluded})
    assert len(cohort["selected_training_group_ids"])==cohort["selected_training_groups"]
    assert not set(cohort["selected_training_group_ids"]) & set(cohort["excluded_training_group_ids"])
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cohort,ensure_ascii=False,indent=2)+"\n")
    os.replace(tmp,path)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args()
    main(a.root)
